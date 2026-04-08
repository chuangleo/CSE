"""
import_popular_products.py
=========================
將 data/popular_product.json 的爬蟲結果批次匯入資料庫。

資料來源：
  - data/popular_product.json  → 各關鍵字的商品清單
  - data/popular_product_title.json → 熱門搜尋詞（供顯示，不強制依賴）

執行方式：
  uv run python import_popular_products.py
  # 或直接
  python import_popular_products.py
"""

import json
from collections import defaultdict
from pathlib import Path

from database import save_search_results

DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_FILE = DATA_DIR / "popular_product.json"


def load_products() -> dict[str, dict[str, list]]:
    """
    讀取 popular_product.json，回傳：
      { search_title: { "momo": [...], "pchome": [...], "other": [...] } }
    """
    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        products: list[dict] = json.load(f)

    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for item in products:
        keyword = item.get("search_title", "").strip()
        platform = item.get("platform", "").strip().lower()
        if not keyword:
            continue
        grouped[keyword][platform].append(item)

    return grouped


def main() -> None:
    print(f"📂 讀取 {PRODUCTS_FILE} ...")
    grouped = load_products()

    total_keywords = len(grouped)
    print(f"🔑 共 {total_keywords} 個關鍵字，開始匯入...\n")

    for idx, (keyword, platforms) in enumerate(grouped.items(), start=1):
        momo_products = platforms.get("momo", [])
        pchome_products = platforms.get("pchome", [])

        print(
            f"[{idx}/{total_keywords}] 關鍵字: 「{keyword}」"
            f"  MOMO: {len(momo_products)} 筆  PChome: {len(pchome_products)} 筆"
        )
        save_search_results(keyword, momo_products, pchome_products)

    print("\n✅ 全部匯入完成！")


if __name__ == "__main__":
    main()
