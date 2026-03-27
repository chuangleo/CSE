# Cross-Platform Product Matcher

跨平台商品比對系統，使用兩階段 AI 架構自動找出 MOMO 與 PChome 上的相同商品並提供比價結果。

## 專案簡介

使用者輸入關鍵字後，系統同步爬取兩大電商的商品清單，以微調過的多語言句向量模型進行語意相似度初篩，再由 Gemini LLM 精確驗證，最後將比對結果與快取資料存入 MySQL，呈現於 Streamlit 網頁介面。

## 系統架構

```
使用者輸入關鍵字
       │
       ▼
┌─────────────────────────────┐
│      MySQL 快取查詢          │  ← 24h 內同一關鍵字直接回傳快取
└──────────┬──────────────────┘
           │ 快取 miss
           ▼
┌──────────────────────────────────────────┐
│            並行爬蟲（Selenium）            │
│   MOMO ──────────────── PChome           │
│   最多 3 組同時執行，共 6 個 Chrome        │
└──────────┬───────────────────────────────┘
           │ 各取最多 100 筆商品
           ▼
┌──────────────────────────────────────────┐
│  Stage 1：向量語意篩選                    │
│  multilingual-e5-large（fine-tuned）      │
│  cosine similarity ≥ 0.739 → 進入 Stage 2 │
└──────────┬───────────────────────────────┘
           │ 候選配對
           ▼
┌──────────────────────────────────────────┐
│  Stage 2：LLM 精確驗證                   │
│  Google Gemini — 最多 3 個並行請求       │
└──────────┬───────────────────────────────┘
           │
           ▼
   MySQL 儲存 + Streamlit 呈現結果
```

## 技術棧

| 類別 | 技術 |
|---|---|
| 網頁介面 | Streamlit |
| 爬蟲 | Selenium WebDriver（Chrome headless）|
| 向量模型 | `multilingual-e5-large`（自行 fine-tune）|
| LLM 驗證 | Google Gemini API |
| 資料庫 | MySQL 8（連線池 + upsert 快取）|
| 資料處理 | Pandas、NumPy、PyTorch |
| 相似度計算 | scikit-learn cosine similarity |
| 模糊比對 | RapidFuzz（關鍵字正規化）|
| 套件管理 | uv |

## 主要功能

- **並行爬蟲**：最多 3 組爬蟲同時執行，含取消與重試機制
- **DB 快取**：相同關鍵字 24 小時內直接回傳資料庫快取，避免重複爬蟲
- **雙階段比對**：向量粗篩 + LLM 精確驗證，兼顧速度與準確度
- **LLM 佇列**：最多 3 個並行 Gemini 請求，超過自動排隊
- **使用者追蹤**：Session 管理與尖峰人數統計
- **關鍵字正規化**：全形 → 半形、去空白，確保「iPhone 15」與「ｉｐｈｏｎｅ１５」視為同一關鍵字

## 資料庫設計

三張資料表分工儲存，商品本體不重複：

```
keywords          keyword_products        products
────────          ────────────────        ────────
id (PK)      ──► keyword_id (FK)          id (PK)
keyword_raw      product_id   (FK) ◄──── platform
keyword_norm     rank_no                  sku
created_at       captured_at              title
                                          price
                                          product_url
```

- `keywords`：記錄關鍵字（原始 + 正規化）
- `products`：商品本體，以 `product_url` 作唯一鍵 upsert
- `keyword_products`：中間表，記錄關鍵字 ↔ 商品關聯與搜尋排名

## 快速開始

### 環境需求

- Python 3.10+
- MySQL 8.0+
- Google Chrome
- Gemini API Key（[免費申請](https://aistudio.google.com/app/apikey)）

### 安裝

```bash
git clone https://github.com/chuangleo/CSE.git
cd CSE

# 建議使用 uv
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 設定環境變數

複製範本並填入設定：

```bash
cp .env.example .env
```

`.env` 內容：

```env
GEMINI_API_KEY=你的_API_金鑰
GEMINI_MODEL=gemini-2.5-flash
MODEL_PATH=models/models20-multilingual-e5-large_fold_1

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的密碼
MYSQL_DATABASE=cse
```

### 啟動

```bash
uv run streamlit run matcher_app.py
```

開啟瀏覽器訪問 `http://localhost:8501`

## 檔案結構

```
CSE/
├── matcher_app.py            # Streamlit 主程式（UI + 爬蟲排程 + LLM 佇列）
├── product_scraper.py        # Selenium 爬蟲（MOMO / PChome）
├── similarity_calculator.py  # Stage 1 向量相似度計算
├── database.py               # MySQL 連線池、upsert、快取查詢
├── models/                   # fine-tuned multilingual-e5-large 模型（Git LFS）
├── pyproject.toml            # 專案依賴（uv 管理）
├── requirements.txt          # pip 相容依賴清單
└── .env.example              # 環境變數範本
```

## 注意事項

- `.env` 已加入 `.gitignore`，API Key 不會上傳至版本控制
- 爬蟲已加入隨機延遲，遵守基本爬蟲禮儀
- Gemini API 為免費方案，請注意每分鐘請求配額


