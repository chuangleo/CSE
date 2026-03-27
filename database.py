"""
資料庫模組 - 商品比對系統的關聯性資料庫
===========================================
使用 MySQL 儲存爬蟲結果，提供快取查詢以避免對相同關鍵字重複爬蟲。

資料表結構：
  - keywords         : 搜尋關鍵字（原始 + 正規化）
  - products         : 各平台商品資訊（upsert by product_url）
  - keyword_products : 關鍵字 ↔ 商品 的中間表（含排名 & 抓取時間）
"""

import os
import re
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta

import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
from rapidfuzz import fuzz
from pypinyin import lazy_pinyin, Style

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 連線池
# ─────────────────────────────────────────────────────────────────────────────
_pool: "pooling.MySQLConnectionPool | None" = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="cse_pool",
            pool_size=5,
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "cse"),
            charset="utf8mb4",
            autocommit=False,
        )
    return _pool


@contextmanager
def get_conn():
    """取得連線（用完自動 commit/rollback 並歸還連線池）"""
    conn = _get_pool().get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# DDL：建立資料表
# ─────────────────────────────────────────────────────────────────────────────
_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS keywords (
        id           INT          AUTO_INCREMENT PRIMARY KEY,
        keyword_raw  VARCHAR(255) NOT NULL,
        keyword_norm VARCHAR(255) NOT NULL,
        status       VARCHAR(50)  NOT NULL DEFAULT 'active',
        created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                           ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_keyword_norm (keyword_norm)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS products (
        id            INT           AUTO_INCREMENT PRIMARY KEY,
        platform      VARCHAR(20)   NOT NULL,
        sku           VARCHAR(255)  NOT NULL DEFAULT '',
        title         TEXT          NOT NULL,
        price         DECIMAL(12,2),
        image_url     TEXT,
        product_url   VARCHAR(2048) NOT NULL DEFAULT '',
        first_seen_at DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_seen_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                             ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_product_url (product_url(512)),
        KEY idx_platform_sku (platform, sku(100))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS keyword_products (
        keyword_id  INT      NOT NULL,
        product_id  INT      NOT NULL,
        rank_no     INT      NOT NULL DEFAULT 0,
        captured_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (keyword_id, product_id),
        CONSTRAINT fk_kp_keyword FOREIGN KEY (keyword_id)
            REFERENCES keywords(id)  ON DELETE CASCADE,
        CONSTRAINT fk_kp_product FOREIGN KEY (product_id)
            REFERENCES products(id)  ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def init_db() -> None:
    """建立資料表（若已存在則跳過），在應用程式啟動時呼叫一次即可。"""
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            for stmt in _STATEMENTS:
                cursor.execute(stmt)
            cursor.close()
        print("✅ 資料庫初始化完成（keywords / products / keyword_products）")
    except Exception as e:
        print(f"⚠️ 資料庫初始化失敗（將以無快取模式繼續）: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 工具：關鍵字正規化
# ─────────────────────────────────────────────────────────────────────────────
def normalize_keyword(keyword: str) -> str:
    """
    全形 → 半形（NFKC）、轉小寫、去除所有空白。
    用於確保「iPhone 15」「iphone15」「ＩＰｈｏｎｅ１５」視為同一鍵。
    """
    keyword = unicodedata.normalize("NFKC", keyword)
    keyword = keyword.lower().strip()
    return re.sub(r"\s+", "", keyword)


# ─────────────────────────────────────────────────────────────────────────────
# 工具：模糊比對輔助
# ─────────────────────────────────────────────────────────────────────────────
_BOPO_TONES = str.maketrans("", "", "ˊˇˋ˙")  # 去除四聲符號


def _to_bopomofo(text: str) -> str:
    """將中文轉成無聲調注音符號，英數字保留原樣，用於注音相似度比對。

    例：'PS5遊戲' → 'PS5ㄧㄡㄒㄧ'
    """
    parts = lazy_pinyin(text, style=Style.BOPOMOFO)
    return "".join(p.translate(_BOPO_TONES) for p in parts)


def _fuzzy_score(query_norm: str, knorm: str) -> float:
    """
    三段式評分，專門處理商品關鍵字的中英文錯字：

    1. 數字序列必須完全一致，否則直接 0 分
       → iphone15 vs iphone16：['15'] ≠ ['16'] → 0 分，不命中
    2. 中文部分用注音相似度：同音字視為相同
       → 遊系 vs 遊戲：注音皆為 'ㄧㄡㄒㄧ' → 100 分
    3. 非中文非數字（英文品牌/型號）用字元相似度
       → ps vs ps → 100 分

    最終分數 = 中文注音分 × 0.6 + 英文字元分 × 0.4
    """
    # 數字序列完全一致才繼續
    if re.findall(r"\d+", query_norm) != re.findall(r"\d+", knorm):
        return 0.0

    def _split(text: str) -> tuple[str, str]:
        no_digits = re.sub(r"\d+", "", text)
        cn     = "".join(re.findall(r"[\u4e00-\u9fff]+", no_digits))
        non_cn = re.sub(r"[\u4e00-\u9fff]+", "", no_digits)
        return cn, non_cn

    cn_a, noncn_a = _split(query_norm)
    cn_b, noncn_b = _split(knorm)

    # 純英文（雙方都沒有中文字）→ 直接用字元相似度，避免中文權重白送導致門檻失真
    if not cn_a and not cn_b:
        return fuzz.ratio(noncn_a, noncn_b)

    cn_score    = fuzz.ratio(_to_bopomofo(cn_a), _to_bopomofo(cn_b)) if (cn_a or cn_b) else 100.0
    noncn_score = fuzz.ratio(noncn_a, noncn_b)                        if (noncn_a or noncn_b) else 100.0

    return cn_score * 0.6 + noncn_score * 0.4


def _fuzzy_find_keyword_id(
    conn,
    query_norm: str,
    threshold: float = 85.0,
) -> int | None:
    """
    從 keywords 表撈出所有 norm，用 _fuzzy_score 找最相似的 keyword id。
    分數低於門檻則回傳 None。
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyword_norm FROM keywords")
    rows = cursor.fetchall()  # [(id, norm), ...]
    cursor.close()

    if not rows:
        return None

    best_id, best_score = None, 0.0

    for kid, knorm in rows:
        score = _fuzzy_score(query_norm, knorm)
        if score > best_score:
            best_score = score
            best_id = kid

    if best_score >= threshold:
        print(f"🔍 模糊比對命中（相似度 {best_score:.0f}%）")
        return best_id
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 關鍵字：upsert
# ─────────────────────────────────────────────────────────────────────────────
def upsert_keyword(keyword_raw: str) -> int:
    """
    取得或建立 keyword 記錄，回傳 id。
    - 若 keyword_norm 已存在，更新 keyword_raw 並回傳現有 id。
    - 若不存在，插入新記錄並回傳新 id。
    """
    norm = normalize_keyword(keyword_raw)
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO keywords (keyword_raw, keyword_norm)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE
                keyword_raw = VALUES(keyword_raw),
                updated_at  = NOW()
            """,
            (keyword_raw, norm),
        )
        cursor.execute(
            "SELECT id FROM keywords WHERE keyword_norm = %s", (norm,)
        )
        row = cursor.fetchone()
        cursor.close()
    return row[0]


# ─────────────────────────────────────────────────────────────────────────────
# 商品：單筆 upsert（在已開啟的連線中執行）
# ─────────────────────────────────────────────────────────────────────────────
def _upsert_single_product(conn, product: dict) -> int | None:
    """
    （內部使用）在傳入的連線中 upsert 一筆商品，回傳 product.id。
    若商品沒有 product_url 則跳過（回傳 None）。
    """
    platform  = product.get("platform", "")
    sku       = (product.get("sku") or "").strip()
    title     = product.get("title", "")
    price     = product.get("price")
    image_url = (product.get("image_url") or product.get("image") or "").strip()
    url       = (product.get("url") or product.get("product_url") or "").strip()

    if not url:
        return None

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO products (platform, sku, title, price, image_url, product_url)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            sku          = IF(VALUES(sku) != '', VALUES(sku), sku),
            title        = VALUES(title),
            price        = VALUES(price),
            image_url    = VALUES(image_url),
            last_seen_at = NOW()
        """,
        (platform, sku, title, price, image_url, url),
    )
    cursor.execute(
        "SELECT id FROM products WHERE product_url = %s", (url[:512],)
    )
    row = cursor.fetchone()
    cursor.close()
    return row[0] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# 批次儲存：爬蟲結果 → keywords + products + keyword_products
# ─────────────────────────────────────────────────────────────────────────────
def save_search_results(
    keyword_raw: str,
    momo_products: list,
    pchome_products: list,
) -> None:
    """
    將爬蟲結果儲存至資料庫（若商品已存在則更新，不重複插入）。

    Args:
        keyword_raw    : 使用者輸入的原始關鍵字
        momo_products  : fetch_products_for_momo() 的回傳值（list of dict）
        pchome_products: fetch_products_for_pchome() 的回傳值（list of dict）
    """
    try:
        keyword_id = upsert_keyword(keyword_raw)

        with get_conn() as conn:
            all_ranked = (
                [(rank, p) for rank, p in enumerate(momo_products or [], start=1)]
                + [(rank, p) for rank, p in enumerate(pchome_products or [], start=1)]
            )

            for rank_no, product in all_ranked:
                product_id = _upsert_single_product(conn, product)
                if product_id is None:
                    continue

                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO keyword_products (keyword_id, product_id, rank_no, captured_at)
                    VALUES (%s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        rank_no     = VALUES(rank_no),
                        captured_at = NOW()
                    """,
                    (keyword_id, product_id, rank_no),
                )
                cursor.close()

        print(
            f"✅ DB 儲存完成: keyword='{keyword_raw}' "
            f"（MOMO: {len(momo_products or [])}, PChome: {len(pchome_products or [])}）"
        )
    except Exception as e:
        # 儲存失敗不影響主程式正常顯示
        print(f"⚠️ DB 儲存失敗（不影響搜尋結果）: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 快取查詢：相同關鍵字且快取未過期，直接回傳商品
# ─────────────────────────────────────────────────────────────────────────────
def get_cached_results(
    keyword_raw: str,
    max_age_hours: int = 24,
) -> tuple[list | None, list | None]:
    """
    查詢快取的搜尋結果。

    若該關鍵字在 max_age_hours 小時內已被爬蟲過，
    直接回傳 (momo_products, pchome_products)，可跳過爬蟲步驟。
    否則回傳 (None, None)。

    Args:
        keyword_raw   : 使用者輸入的原始關鍵字
        max_age_hours : 快取有效時間（小時），預設 24

    Returns:
        (momo_list, pchome_list) 若快取有效
        (None, None)             若無快取或已過期
    """
    try:
        norm = normalize_keyword(keyword_raw)

        with get_conn() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT k.id, MAX(kp.captured_at) AS last_captured
                FROM   keywords k
                JOIN   keyword_products kp ON kp.keyword_id = k.id
                WHERE  k.keyword_norm = %s
                GROUP  BY k.id
                """,
                (norm,),
            )
            meta = cursor.fetchone()

            # 精確查無 → 模糊比對 fallback
            if not meta:
                fuzzy_id = _fuzzy_find_keyword_id(conn, norm)
                if fuzzy_id is not None:
                    cursor2 = conn.cursor(dictionary=True)
                    cursor2.execute(
                        """
                        SELECT id, MAX(kp.captured_at) AS last_captured
                        FROM   keywords
                        JOIN   keyword_products kp ON kp.keyword_id = keywords.id
                        WHERE  keywords.id = %s
                        GROUP  BY keywords.id
                        """,
                        (fuzzy_id,),
                    )
                    meta = cursor2.fetchone()
                    cursor2.close()

            cursor.close()

        if not meta:
            return None, None  # 從未爬蟲過

        age = datetime.now() - meta["last_captured"]
        if age > timedelta(hours=max_age_hours):
            print(f"⏰ 快取過期（{age.seconds // 3600}h {(age.seconds % 3600) // 60}m 前），需重新爬蟲")
            return None, None

        keyword_id = meta["id"]

        with get_conn() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT p.platform,
                       p.sku,
                       p.title,
                       p.price,
                       p.image_url   AS image,
                       p.image_url,
                       p.product_url AS url,
                       kp.rank_no
                FROM   keyword_products kp
                JOIN   products p ON p.id = kp.product_id
                WHERE  kp.keyword_id = %s
                ORDER  BY p.platform, kp.rank_no
                """,
                (keyword_id,),
            )
            rows = cursor.fetchall()
            cursor.close()

        momo_products   = [r for r in rows if r["platform"] == "momo"]
        pchome_products = [r for r in rows if r["platform"] == "pchome"]

        # 補齊 id 欄位
        for rank, p in enumerate(momo_products, start=1):
            p["id"] = rank

        for rank, p in enumerate(pchome_products, start=1):
            p["id"] = rank

        remaining_hours = max_age_hours - age.seconds // 3600
        print(
            f"✅ 使用 DB 快取: keyword='{keyword_raw}'，"
            f"快取剩餘約 {remaining_hours}h "
            f"（MOMO: {len(momo_products)}, PChome: {len(pchome_products)}）"
        )
        return momo_products, pchome_products

    except Exception as e:
        print(f"⚠️ 快取查詢失敗（將重新爬蟲）: {e}")
        return None, None

