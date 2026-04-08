"""
FastAPI REST API 層 — 跨平台商品比對系統
==========================================
將原本 matcher_app.py (Streamlit) 裡的業務邏輯抽成獨立 API，
讓 Streamlit、Line Bot、React 等任意前端都能呼叫。

啟動方式：
  uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

API 文件：
  http://localhost:8000/docs   (Swagger UI)
  http://localhost:8000/redoc  (ReDoc)
"""

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 延遲 import（避免啟動時就載入 torch / sentence_transformers 拖慢速度）
# ─────────────────────────────────────────────────────────────────────────────
def _import_heavy():
    """第一次呼叫時才 import 重型套件。"""
    import ollama
    import torch
    from sentence_transformers import SentenceTransformer

    return ollama, torch, SentenceTransformer


def _get_database():
    from database import get_cached_results, init_db, save_search_results

    return get_cached_results, init_db, save_search_results


def _get_scrapers():
    from product_scraper import fetch_products_for_momo, fetch_products_for_pchome

    return fetch_products_for_momo, fetch_products_for_pchome

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 應用程式
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    """應用啟動時初始化資料庫、預載模型與 Ollama。"""
    _, init_db, _ = _get_database()
    init_db()

    # ── 預熱 SentenceTransformer ──
    print("⏳ 正在預載 SentenceTransformer 模型...")
    t0 = time.time()
    get_model()
    print(f"✅ SentenceTransformer 模型已載入 ({time.time() - t0:.1f}s)")

    # ── 預熱 Ollama（觸發一次極短推論讓模型載入 GPU）──
    print(f"⏳ 正在預熱 Ollama ({OLLAMA_MODEL})...")
    t0 = time.time()
    try:
        ollama_mod, _, _ = _import_heavy()
        ollama_mod.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"temperature": 0, "num_predict": 1},
        )
        print(f"✅ Ollama 預熱完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"⚠️ Ollama 預熱失敗（服務可能未啟動）: {e}")

    yield


app = FastAPI(
    title="跨平台商品比對 API",
    description=(
        "提供商品搜尋、跨平台比對、快取查詢等功能的 REST API。\n\n"
        "**架構**：Selenium 爬蟲 → multilingual-e5-large 向量比對 → Ollama LLM 驗證"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# 全域設定 & 模型載入
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join("models", "models20-multilingual-e5-large_fold_1"),
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.739465"))

_model = None


def get_model():
    """延遲載入 SentenceTransformer 模型（單例）。"""
    global _model
    if _model is None:
        _, _, SentenceTransformer = _import_heavy()
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"找不到模型路徑: {MODEL_PATH}")
        _model = SentenceTransformer(MODEL_PATH)
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# 背景任務：工作追蹤
# ─────────────────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}  # job_id → {status, keyword, result, error, ...}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models — Request / Response
# ─────────────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100, description="搜尋關鍵字")
    max_products: int = Field(default=50, ge=1, le=100, description="每平台最大抓取商品數")
    use_cache: bool = Field(default=True, description="是否優先使用快取")
    cache_max_age_hours: int = Field(default=24, ge=1, le=168, description="快取有效時數")
    direction: Literal["momo_to_pchome", "pchome_to_momo"] = Field(
        default="momo_to_pchome", description="比對方向"
    )


class ProductOut(BaseModel):
    id: int | str
    title: str
    price: float | None = None
    platform: str
    image: str | None = Field(default=None, alias="image_url")
    url: str | None = Field(default=None, alias="product_url")
    sku: str | None = None

    model_config = {"populate_by_name": True}


class MatchPair(BaseModel):
    momo_title: str
    pchome_title: str
    similarity: float
    momo_price: float | None = None
    pchome_price: float | None = None


class MatchResult(BaseModel):
    is_match: bool
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class SimilarityHit(BaseModel):
    source_id: str
    source_title: str
    target_id: str
    target_title: str
    similarity: float
    target_price: float | None = None
    target_url: str | None = None
    target_image: str | None = None


class SearchResult(BaseModel):
    keyword: str
    from_cache: bool
    momo_count: int
    pchome_count: int
    momo_products: list[dict]
    pchome_products: list[dict]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "scraping", "matching", "completed", "failed"]
    keyword: str
    progress: str | None = None
    momo_current: int = 0
    momo_total: int = 0
    pchome_current: int = 0
    pchome_total: int = 0
    result: SearchResult | None = None
    error: str | None = None


class StatsOut(BaseModel):
    total_searches: int
    unique_keywords: int
    top_keywords: list[dict]
    current_online: int
    peak_users: int
    peak_timestamp: str | None


class CacheStatusOut(BaseModel):
    keyword: str
    cached: bool
    momo_count: int
    pchome_count: int


# ─────────────────────────────────────────────────────────────────────────────
# 輔助函式
# ─────────────────────────────────────────────────────────────────────────────
def _prepare_text(title: str, platform: str) -> str:
    return ("query: " if platform == "momo" else "passage: ") + str(title)


def _compute_similarities(
    model,
    momo_products: list[dict],
    pchome_products: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
    direction: str = "momo_to_pchome",
) -> list[SimilarityHit]:
    """
    計算語意相似度，回傳超過門檻的配對清單。
    direction 決定 source/target 方向，但 embedding 固定以 momo=query, pchome=passage。
    """
    _, torch, _ = _import_heavy()

    if not momo_products or not pchome_products:
        return []

    momo_texts = [_prepare_text(p["title"], "momo") for p in momo_products]
    pchome_texts = [_prepare_text(p["title"], "pchome") for p in pchome_products]

    momo_embs = model.encode(momo_texts, convert_to_tensor=True, batch_size=32).cpu()
    pchome_embs = model.encode(pchome_texts, convert_to_tensor=True, batch_size=32).cpu()

    momo_embs = torch.nn.functional.normalize(momo_embs, p=2, dim=1)
    pchome_embs = torch.nn.functional.normalize(pchome_embs, p=2, dim=1)

    sim_matrix = torch.mm(momo_embs, pchome_embs.T).numpy()

    hits: list[SimilarityHit] = []
    if direction == "momo_to_pchome":
        for m_idx, momo in enumerate(momo_products):
            for p_idx, pchome in enumerate(pchome_products):
                score = float(sim_matrix[m_idx][p_idx])
                if score >= threshold:
                    hits.append(
                        SimilarityHit(
                            source_id=str(momo.get("id", m_idx)),
                            source_title=momo["title"],
                            target_id=str(pchome.get("id", p_idx)),
                            target_title=pchome["title"],
                            similarity=round(score, 6),
                            target_price=pchome.get("price"),
                            target_url=pchome.get("url") or pchome.get("product_url"),
                            target_image=pchome.get("image") or pchome.get("image_url", ""),
                        )
                    )
    else:
        for p_idx, pchome in enumerate(pchome_products):
            for m_idx, momo in enumerate(momo_products):
                score = float(sim_matrix[m_idx][p_idx])
                if score >= threshold:
                    hits.append(
                        SimilarityHit(
                            source_id=str(pchome.get("id", p_idx)),
                            source_title=pchome["title"],
                            target_id=str(momo.get("id", m_idx)),
                            target_title=momo["title"],
                            similarity=round(score, 6),
                            target_price=momo.get("price"),
                            target_url=momo.get("url") or momo.get("product_url"),
                            target_image=momo.get("image") or momo.get("image_url", ""),
                        )
                    )
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits


def _llm_verify_batch(
    pairs: list[MatchPair],
    direction: str = "momo_to_pchome",
) -> list[MatchResult]:
    """
    呼叫 Ollama LLM 批次驗證配對結果。
    """
    if not pairs:
        return []

    if direction == "momo_to_pchome":
        platform_a, platform_b = "MOMO", "PChome"
    else:
        platform_a, platform_b = "PChome", "MOMO"

    prompt = "判斷以下商品配對是否為相同商品。\n\n"
    prompt += "**規則**：\n"
    prompt += "1. 品牌、型號、規格、容量、數量必須完全一致\n"
    prompt += "2. 顏色不同 = 相同商品\n"
    prompt += "3. 其他差異（容量、數量、規格、版本）= 不同商品\n"
    prompt += "4. 不要使用價格來判斷\n\n"

    for i, p in enumerate(pairs, 1):
        prompt += f"【配對 {i}】\n"
        prompt += f"  商品 A ({platform_a})：{p.momo_title}\n"
        prompt += f"  商品 B ({platform_b})：{p.pchome_title}\n"
        prompt += f"  相似度：{p.similarity:.4f}\n\n"

    prompt += (
        f"請回傳純 JSON 陣列（{len(pairs)} 個元素）：\n"
        '[{"is_match": true/false, "confidence": "high/medium/low", "reasoning": "簡短說明"}]'
    )

    try:
        ollama_mod, _, _ = _import_heavy()
        response = ollama_mod.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        text = response["message"]["content"].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        raw = json.loads(text)
        results = []
        for item in raw:
            results.append(
                MatchResult(
                    is_match=bool(item.get("is_match", False)),
                    confidence=item.get("confidence", "low"),
                    reasoning=item.get("reasoning", ""),
                )
            )
        # 數量不符時補齊
        while len(results) < len(pairs):
            results.append(
                MatchResult(is_match=False, confidence="low", reasoning="LLM 回傳數量不足")
            )
        return results[: len(pairs)]
    except Exception as e:
        return [
            MatchResult(is_match=False, confidence="low", reasoning=f"LLM 錯誤: {e}")
            for _ in pairs
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 背景任務函式
# ─────────────────────────────────────────────────────────────────────────────
def _run_search_job(
    job_id: str,
    keyword: str,
    max_products: int,
    use_cache: bool,
    cache_max_age_hours: int,
) -> None:
    """背景執行搜尋任務：快取查詢 → 爬蟲 → 儲存 DB。"""
    get_cached_results, _, save_search_results = _get_database()
    fetch_products_for_momo, fetch_products_for_pchome = _get_scrapers()

    def _momo_progress(current, total, message):
        _jobs[job_id]["momo_current"] = current
        _jobs[job_id]["momo_total"] = total
        _jobs[job_id]["progress"] = f"MOMO: {message}"

    def _pchome_progress(current, total, message):
        _jobs[job_id]["pchome_current"] = current
        _jobs[job_id]["pchome_total"] = total
        _jobs[job_id]["progress"] = f"PChome: {message}"

    try:
        _jobs[job_id]["status"] = "scraping"
        _jobs[job_id]["progress"] = "查詢快取中..."

        # 1. 快取
        if use_cache:
            cached_momo, cached_pchome = get_cached_results(keyword, cache_max_age_hours)
            if cached_momo is not None and cached_pchome is not None:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = SearchResult(
                    keyword=keyword,
                    from_cache=True,
                    momo_count=len(cached_momo),
                    pchome_count=len(cached_pchome),
                    momo_products=cached_momo,
                    pchome_products=cached_pchome,
                )
                return

        # 2. 爬蟲（並行 + 進度回報）
        _jobs[job_id]["progress"] = "正在同時爬取 MOMO 與 PChome 商品..."
        _jobs[job_id]["momo_total"] = max_products
        _jobs[job_id]["pchome_total"] = max_products

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_momo = executor.submit(
                fetch_products_for_momo,
                keyword, max_products,
                progress_callback=_momo_progress,
            )
            future_pchome = executor.submit(
                fetch_products_for_pchome,
                keyword, max_products,
                progress_callback=_pchome_progress,
            )
            momo_products = future_momo.result()
            pchome_products = future_pchome.result()

        _jobs[job_id]["momo_current"] = len(momo_products)
        _jobs[job_id]["pchome_current"] = len(pchome_products)

        # 3. 存 DB
        _jobs[job_id]["progress"] = "儲存至資料庫..."
        save_search_results(keyword, momo_products, pchome_products)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = SearchResult(
            keyword=keyword,
            from_cache=False,
            momo_count=len(momo_products),
            pchome_count=len(pchome_products),
            momo_products=momo_products,
            pchome_products=pchome_products,
        )

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ═════════════════════════════════════════════════════════════════════════════
# API 端點
# ═════════════════════════════════════════════════════════════════════════════


# ──── 搜尋（同步：快取命中時直接回傳） ────────────────────────────────────
@app.post(
    "/api/search",
    response_model=SearchResult,
    summary="搜尋商品（快取優先）",
    tags=["搜尋"],
)
def search_products(req: SearchRequest):
    """
    查詢快取是否有該關鍵字的結果：
    - **快取命中**：直接回傳商品清單。
    - **快取未命中**：啟動爬蟲 → 比對 → 存 DB → 回傳。

    若爬蟲耗時較長（30 s+），建議改用 `POST /api/search/async`。
    """
    get_cached_results, _, save_search_results = _get_database()
    fetch_products_for_momo, fetch_products_for_pchome = _get_scrapers()

    if req.use_cache:
        cached_momo, cached_pchome = get_cached_results(
            req.keyword, req.cache_max_age_hours
        )
        if cached_momo is not None and cached_pchome is not None:
            return SearchResult(
                keyword=req.keyword,
                from_cache=True,
                momo_count=len(cached_momo),
                pchome_count=len(cached_pchome),
                momo_products=cached_momo,
                pchome_products=cached_pchome,
            )

    # 無快取 → 爬蟲
    momo_products = fetch_products_for_momo(req.keyword, req.max_products)
    pchome_products = fetch_products_for_pchome(req.keyword, req.max_products)
    save_search_results(req.keyword, momo_products, pchome_products)

    return SearchResult(
        keyword=req.keyword,
        from_cache=False,
        momo_count=len(momo_products),
        pchome_count=len(pchome_products),
        momo_products=momo_products,
        pchome_products=pchome_products,
    )


# ──── 搜尋（非同步：立即回 job_id，背景執行爬蟲） ─────────────────────────
@app.post(
    "/api/search/async",
    response_model=JobStatus,
    summary="非同步搜尋（背景爬蟲）",
    tags=["搜尋"],
)
def search_products_async(req: SearchRequest, bg: BackgroundTasks):
    """
    建立背景搜尋任務，立即回傳 `job_id`。
    前端可用 `GET /api/jobs/{job_id}` 輪詢進度。
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "keyword": req.keyword,
        "progress": "已排入佇列",
        "momo_current": 0,
        "momo_total": 0,
        "pchome_current": 0,
        "pchome_total": 0,
        "result": None,
        "error": None,
    }
    bg.add_task(
        _run_search_job,
        job_id,
        req.keyword,
        req.max_products,
        req.use_cache,
        req.cache_max_age_hours,
    )
    return JobStatus(job_id=job_id, status="pending", keyword=req.keyword, progress="已排入佇列")


# ──── 查詢工作狀態 ──────────────────────────────────────────────────────────
@app.get(
    "/api/jobs/{job_id}",
    response_model=JobStatus,
    summary="查詢背景任務狀態",
    tags=["搜尋"],
)
def get_job(job_id: str):
    """輪詢 `POST /api/search/async` 回傳的背景任務進度。"""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到該任務")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        keyword=job["keyword"],
        progress=job.get("progress"),
        momo_current=job.get("momo_current", 0),
        momo_total=job.get("momo_total", 0),
        pchome_current=job.get("pchome_current", 0),
        pchome_total=job.get("pchome_total", 0),
        result=job.get("result"),
        error=job.get("error"),
    )


# ──── 快取狀態查詢 ──────────────────────────────────────────────────────────
@app.get(
    "/api/cache/{keyword}",
    response_model=CacheStatusOut,
    summary="查詢某關鍵字是否有快取",
    tags=["快取"],
)
def check_cache(keyword: str, max_age_hours: int = Query(24, ge=1, le=168)):
    """檢查指定關鍵字是否在 DB 快取中（不觸發爬蟲）。"""
    get_cached_results, _, _ = _get_database()
    cached_momo, cached_pchome = get_cached_results(keyword, max_age_hours)
    if cached_momo is not None and cached_pchome is not None:
        return CacheStatusOut(
            keyword=keyword,
            cached=True,
            momo_count=len(cached_momo),
            pchome_count=len(cached_pchome),
        )
    return CacheStatusOut(keyword=keyword, cached=False, momo_count=0, pchome_count=0)


# ──── Stage 1: 語意相似度計算 ───────────────────────────────────────────────
@app.post(
    "/api/match/similarity",
    response_model=list[SimilarityHit],
    summary="Stage 1：語意相似度計算",
    tags=["比對"],
)
def compute_similarity(req: SearchRequest):
    """
    對指定關鍵字的商品做 Stage 1 向量比對。

    1. 先從快取取得 MOMO / PChome 商品
    2. 用 multilingual-e5-large 計算 cosine similarity
    3. 回傳超過門檻的配對清單
    """
    get_cached_results, _, _ = _get_database()
    cached_momo, cached_pchome = get_cached_results(
        req.keyword, req.cache_max_age_hours
    )
    if cached_momo is None or cached_pchome is None:
        raise HTTPException(
            status_code=404,
            detail=f"關鍵字 '{req.keyword}' 尚無快取資料，請先呼叫 POST /api/search",
        )

    model = get_model()
    hits = _compute_similarities(
        model, cached_momo, cached_pchome, direction=req.direction
    )
    return hits


# ──── Stage 2: LLM 驗證 ────────────────────────────────────────────────────
@app.post(
    "/api/match/verify",
    response_model=list[MatchResult],
    summary="Stage 2：LLM 精確驗證",
    tags=["比對"],
)
def verify_matches(
    pairs: list[MatchPair],
    direction: Literal["momo_to_pchome", "pchome_to_momo"] = Query("momo_to_pchome"),
):
    """
    將 Stage 1 篩出的候選配對送 Ollama LLM 做最終判定。

    傳入配對清單，回傳對應的 is_match / confidence / reasoning。
    """
    if len(pairs) > 50:
        raise HTTPException(status_code=400, detail="單次最多驗證 50 組配對")
    return _llm_verify_batch(pairs, direction=direction)


# ──── 全流程（搜尋 + Stage 1 + Stage 2） ──────────────────────────────────
@app.post(
    "/api/match/full",
    summary="完整比對流程（搜尋 → Stage 1 → Stage 2）",
    tags=["比對"],
)
def full_match_pipeline(req: SearchRequest):
    """
    一鍵完成整條 pipeline：

    1. 搜尋商品（快取優先，否則爬蟲）
    2. Stage 1：向量語意篩選
    3. Stage 2：LLM 精確驗證
    4. 回傳最終確認配對結果
    """
    # Step 1: 搜尋
    search_result = search_products(req)

    # Step 2: Stage 1 相似度
    model = get_model()
    hits = _compute_similarities(
        model, search_result.momo_products, search_result.pchome_products,
        direction=req.direction,
    )

    if not hits:
        return {
            "keyword": req.keyword,
            "from_cache": search_result.from_cache,
            "momo_count": search_result.momo_count,
            "pchome_count": search_result.pchome_count,
            "stage1_candidates": 0,
            "stage2_verified": 0,
            "matches": [],
        }

    # Step 3: Stage 2 LLM 驗證
    pairs = [
        MatchPair(
            momo_title=h.source_title,
            pchome_title=h.target_title,
            similarity=h.similarity,
        )
        for h in hits
    ]
    verifications = _llm_verify_batch(pairs, direction=req.direction)

    # 合併結果
    matches = []
    for hit, verification in zip(hits, verifications):
        if verification.is_match:
            matches.append(
                {
                    "momo_title": hit.source_title,
                    "pchome_title": hit.target_title,
                    "similarity": hit.similarity,
                    "confidence": verification.confidence,
                    "reasoning": verification.reasoning,
                    "target_price": hit.target_price,
                    "target_url": hit.target_url,
                }
            )

    return {
        "keyword": req.keyword,
        "from_cache": search_result.from_cache,
        "momo_count": search_result.momo_count,
        "pchome_count": search_result.pchome_count,
        "stage1_candidates": len(hits),
        "stage2_verified": len(matches),
        "matches": matches,
    }


# ──── 統計資料 ──────────────────────────────────────────────────────────────
@app.get(
    "/api/stats",
    response_model=StatsOut,
    summary="系統統計資訊",
    tags=["統計"],
)
def get_stats():
    """
    回傳搜尋紀錄統計與在線用戶峰值。

    資料來源：search_logs.json + user_peak.json
    """
    # 搜尋紀錄
    log_file = Path("search_logs.json")
    logs: list[dict] = []
    if log_file.exists():
        try:
            logs = json.loads(log_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass

    keyword_counts: dict[str, int] = {}
    for entry in logs:
        kw = entry.get("keyword", "")
        if kw:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # 用戶峰值
    peak_file = Path("user_peak.json")
    peak_data = {"peak_users": 0, "peak_timestamp": None, "current_online": 0}
    if peak_file.exists():
        try:
            peak_data = json.loads(peak_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass

    return StatsOut(
        total_searches=len(logs),
        unique_keywords=len(keyword_counts),
        top_keywords=[{"keyword": k, "count": c} for k, c in top_keywords],
        current_online=peak_data.get("current_online", 0),
        peak_users=peak_data.get("peak_users", 0),
        peak_timestamp=peak_data.get("peak_timestamp"),
    )


# ──── 健康檢查 ──────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["系統"])
def health_check():
    """回傳 API 是否正常運作。"""
    model_ready = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_path": MODEL_PATH,
        "ollama_model": OLLAMA_MODEL,
        "timestamp": datetime.now().isoformat(),
    }
