"""
Streamlit 主程式 — 跨平台商品比對系統
========================================
架構說明：
  1. 使用者在網頁輸入關鍵字，系統優先查詢 MySQL 快取（database.py）。
  2. 快取 miss 時，啟動並行 Selenium 爬蟲（product_scraper.py），
     最多 3 組同時爬取（= 6 個 Chrome headless），超過排隊等待。
  3. 爬蟲完成後進入雙階段比對：
       Stage 1：multilingual-e5-large 向量餘弦相似度初篩（similarity_calculator.py）
       Stage 2：Ollama LLM（qwen2.5:14b）精確驗證，最多 3 個並行請求，超過佇列。
  4. 比對結果與爬蟲資料存入 MySQL，下次相同關鍵字直接從快取回傳。

並行控制：
  - 爬蟲：SCRAPERS_FILE（JSON 跨 Session 共享狀態）
  - LLM  ：LLM_REQUESTS_FILE（JSON 跨 Session 共享狀態）
  - 使用者追蹤：USERS_FILE（JSON，記錄 Session 與尖峰人數）
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import ollama
import os
import json
import time
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from product_scraper import fetch_products_for_momo, fetch_products_for_pchome, save_to_csv
from similarity_calculator import calculate_all_similarities
from dotenv import load_dotenv
from database import init_db, save_search_results, get_cached_results

# ============= 全局線程鎖（用於搜尋記錄） =============
log_lock = threading.Lock()

# ============= LLM 佇列系統（跨進程版本）=============
MAX_CONCURRENT_LLM_REQUESTS = 3  # 最多同時處理3個LLM請求（與爬蟲並行數一致）
LLM_REQUESTS_FILE = "active_llm_requests.json"  # LLM 請求狀態文件
llm_queue_lock = threading.Lock()  # LLM 佇列文件訪問鎖
llm_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_LLM_REQUESTS, thread_name_prefix="LLM_Worker")

# ============= 用戶峰值追蹤系統 =============
users_lock = threading.Lock()  # 線程鎖
USER_TIMEOUT = 300  # 用戶超時時間（秒），超過此時間視為離線
USERS_FILE = "active_users.json"  # 用戶追蹤文件

# ============= 爬蟲佇列系統（跨進程版本）=============
MAX_CONCURRENT_SCRAPERS = 3  # 最多同時 3 組爬蟲（= 6 個 Chrome）
SCRAPERS_FILE = "active_scrapers.json"  # 爬蟲狀態文件
scraper_queue_lock = threading.Lock()  # 文件訪問鎖

# 載入環境變數
load_dotenv()

# ============= 頁面配置 =============
st.set_page_config(
    page_title="商品比價系統",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= 搜尋記錄功能 =============
def log_search_query(keyword, user_session_id, momo_count=0, pchome_count=0):
    """
    記錄用戶搜尋詞到 JSON 文件（線程安全版本）
    
    Args:
        keyword: 搜尋關鍵字
        user_session_id: 用戶 Session ID
        momo_count: MOMO 搜尋結果數量
        pchome_count: PChome 搜尋結果數量
    """
    log_file = "search_logs.json"
    
    try:
        print(f"🔍 log_search_query 被調用: keyword={keyword}, user={user_session_id}")
        
        # 使用鎖確保線程安全
        with log_lock:
            print(f"🔒 獲得鎖，準備寫入文件: {log_file}")
            
            # 讀取現有記錄
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    print(f"📖 讀取到 {len(logs)} 筆現有記錄")
                except json.JSONDecodeError:
                    logs = []
                    print("⚠️ JSON 解析失敗，創建新列表")
            else:
                logs = []
                print("📝 文件不存在，創建新列表")
            
            # 添加新記錄
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_session_id": user_session_id,
                "keyword": keyword,
                "momo_results": momo_count,
                "pchome_results": pchome_count
            }
            logs.append(log_entry)
            print(f"➕ 添加新記錄，現在共 {len(logs)} 筆")
            
            # 寫入文件
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"💾 成功寫入文件: {log_file}")
    
    except Exception as e:
        # 靜默處理錯誤，不影響主程式
        print(f"❌ 記錄搜尋失敗: {e}")
        import traceback
        traceback.print_exc()

# ============= 用戶峰值追蹤功能 =============
def update_user_peak(user_session_id, action='join'):
    """
    更新用戶峰值記錄（跨進程版本，使用文件存儲）
    
    Args:
        user_session_id: 用戶 Session ID
        action: 'join' 首次加入 / 'update' 更新活動時間 / 'leave' 離開
    """
    peak_file = "user_peak.json"
    
    try:
        with users_lock:
            current_time = time.time()
            
            # 從文件讀取當前在線用戶
            if os.path.exists(USERS_FILE):
                try:
                    with open(USERS_FILE, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        active_users = json.loads(content) if content else {}
                except (json.JSONDecodeError, ValueError):
                    active_users = {}
            else:
                active_users = {}
            
            # 清理超時用戶（超過 USER_TIMEOUT 秒未活動）
            timeout_users = [uid for uid, last_time in active_users.items() 
                           if current_time - last_time > USER_TIMEOUT]
            for uid in timeout_users:
                del active_users[uid]
                print(f"⏱️ 用戶超時移除: {uid[:8]}...")
            
            # 更新當前在線用戶
            if action == 'join':
                is_new = user_session_id not in active_users
                active_users[user_session_id] = current_time
                if is_new:
                    print(f"👤 新用戶加入: {user_session_id[:8]}...")
            elif action == 'update':
                # 僅更新活動時間，不打印訊息（避免刷屏）
                active_users[user_session_id] = current_time
            elif action == 'leave':
                if user_session_id in active_users:
                    del active_users[user_session_id]
                    print(f"👋 用戶離開: {user_session_id[:8]}...")
            
            # 寫回文件
            with open(USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(active_users, f, ensure_ascii=False, indent=2)
            
            current_online = len(active_users)
            user_list = [uid[:8] + "..." for uid in list(active_users.keys())[:3]]
            print(f"📊 當前在線人數: {current_online} | 在線用戶: {user_list}")
            
            # 讀取現有峰值記錄
            if os.path.exists(peak_file):
                try:
                    with open(peak_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            peak_data = json.loads(content)
                        else:
                            peak_data = {"peak_users": 0, "peak_timestamp": None, "current_online": 0}
                except (json.JSONDecodeError, ValueError):
                    peak_data = {"peak_users": 0, "peak_timestamp": None, "current_online": 0}
            else:
                peak_data = {"peak_users": 0, "peak_timestamp": None, "current_online": 0}
            
            # 更新當前在線人數
            peak_data["current_online"] = current_online
            
            # 檢查是否創造新高峰
            if current_online > peak_data.get("peak_users", 0):
                peak_data["peak_users"] = current_online
                peak_data["peak_timestamp"] = datetime.now().isoformat()
                print(f"🎉 新的峰值紀錄！{current_online} 人同時在線")
            
            # 寫入文件
            with open(peak_file, 'w', encoding='utf-8') as f:
                json.dump(peak_data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"❌ 更新用戶峰值失敗: {e}")
        import traceback
        traceback.print_exc()

# ============= 爬蟲佇列管理函數（跨進程版本）=============
def try_acquire_scraper_slot(user_id):
    """
    嘗試獲取爬蟲位置（基於文件的跨進程版本）
    
    Returns:
        tuple: (success: bool, current_active: int, queue_position: int)
    """
    with scraper_queue_lock:
        # 讀取當前爬蟲狀態
        if os.path.exists(SCRAPERS_FILE):
            try:
                with open(SCRAPERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        # 清理超時的爬蟲（超過 10 分鐘視為異常，強制釋放）
        current_time = time.time()
        timeout_scrapers = [uid for uid, start_time in data["active"].items() 
                           if current_time - start_time > 600]
        for uid in timeout_scrapers:
            del data["active"][uid]
            print(f"⏱️ 爬蟲超時強制釋放: {uid[:8]}...")
        
        active_count = len(data["active"])
        
        # 確保 waiting 列表存在
        if "waiting" not in data:
            data["waiting"] = []
        
        # 如果有空位
        if active_count < MAX_CONCURRENT_SCRAPERS:
            # 檢查等待隊列
            if len(data["waiting"]) > 0:
                # 如果有人在等待，只允許隊列第一個人獲取
                if data["waiting"][0] == user_id:
                    # 我是第一個，可以獲取
                    data["active"][user_id] = current_time
                    data["waiting"].pop(0)  # 從隊列移除
                    
                    # 寫回文件
                    with open(SCRAPERS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ 隊列第一位獲得爬蟲權限: {user_id[:8]}...")
                    return True, len(data["active"]), 0
                else:
                    # 不是第一個，繼續等待
                    queue_position = data["waiting"].index(user_id) + 1 if user_id in data["waiting"] else 0
                    
                    # 寫回文件
                    with open(SCRAPERS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    return False, active_count, queue_position
            else:
                # 沒有人等待，直接獲取
                data["active"][user_id] = current_time
                
                # 寫回文件
                with open(SCRAPERS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 直接獲得爬蟲權限: {user_id[:8]}...")
                return True, len(data["active"]), 0
        
        # 沒有空位，加入等待列表
        if user_id not in data["waiting"]:
            data["waiting"].append(user_id)
        
        queue_position = data["waiting"].index(user_id) + 1
        
        # 寫回文件
        with open(SCRAPERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return False, active_count, queue_position

def release_scraper_slot(user_id):
    """釋放爬蟲位置"""
    with scraper_queue_lock:
        if os.path.exists(SCRAPERS_FILE):
            try:
                with open(SCRAPERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        # 從活躍列表移除
        if user_id in data["active"]:
            del data["active"][user_id]
            print(f"🔓 釋放爬蟲位置: {user_id[:8]}... (剩餘: {len(data['active'])}/{MAX_CONCURRENT_SCRAPERS})")
        
        # 從等待列表移除（如果存在）
        if user_id in data.get("waiting", []):
            data["waiting"].remove(user_id)
        
        # 寫回文件
        with open(SCRAPERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return len(data["active"])

def get_queue_status(user_id):
    """獲取當前隊列狀態"""
    with scraper_queue_lock:
        if os.path.exists(SCRAPERS_FILE):
            try:
                with open(SCRAPERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        active_count = len(data["active"])
        queue_position = 0
        if user_id in data.get("waiting", []):
            queue_position = data["waiting"].index(user_id) + 1
        
        return active_count, queue_position

# ============= LLM 佇列管理函數 =============
def acquire_llm_slot(request_id, user_id):
    """獲取 LLM 處理位置
    
    Returns:
        tuple: (success, active_count, queue_position)
            success: 是否成功獲取位置
            active_count: 當前活躍的 LLM 請求數
            queue_position: 在隊列中的位置（0表示不在隊列中）
    """
    with llm_queue_lock:
        # 讀取當前 LLM 請求狀態
        if os.path.exists(LLM_REQUESTS_FILE):
            try:
                with open(LLM_REQUESTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        # 清理超時的 LLM 請求（超過 5 分鐘視為異常，強制釋放）
        current_time = time.time()
        timeout_requests = [rid for rid, info in data["active"].items() 
                           if current_time - info["start_time"] > 300]
        for rid in timeout_requests:
            del data["active"][rid]
            print(f"⏱️ LLM請求超時強制釋放: {rid[:8]}...")
        
        active_count = len(data["active"])
        
        # 確保 waiting 列表存在
        if "waiting" not in data:
            data["waiting"] = []
        
        # 如果有空位
        if active_count < MAX_CONCURRENT_LLM_REQUESTS:
            # 檢查等待隊列
            if len(data["waiting"]) > 0:
                # 如果有人在等待，只允許隊列第一個人獲取
                if data["waiting"][0] == request_id:
                    # 我是第一個，可以獲取
                    data["active"][request_id] = {
                        "start_time": current_time,
                        "user_id": user_id
                    }
                    data["waiting"].pop(0)  # 從隊列移除
                    
                    # 寫回文件
                    with open(LLM_REQUESTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ 隊列第一位獲得LLM權限: {request_id[:8]}... (用戶: {user_id[:8]}...)")
                    return True, len(data["active"]), 0
                else:
                    # 不是第一個，繼續等待
                    queue_position = data["waiting"].index(request_id) + 1 if request_id in data["waiting"] else 0
                    
                    # 寫回文件
                    with open(LLM_REQUESTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    return False, active_count, queue_position
            else:
                # 沒有人等待，直接獲取
                data["active"][request_id] = {
                    "start_time": current_time,
                    "user_id": user_id
                }
                
                # 寫回文件
                with open(LLM_REQUESTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 直接獲得LLM權限: {request_id[:8]}... (用戶: {user_id[:8]}...)")
                return True, len(data["active"]), 0
        
        # 沒有空位，加入等待列表
        if request_id not in data["waiting"]:
            data["waiting"].append(request_id)
        
        queue_position = data["waiting"].index(request_id) + 1
        
        # 寫回文件
        with open(LLM_REQUESTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"⏳ LLM請求排隊中: {request_id[:8]}... 位置: {queue_position}/{active_count+len(data['waiting'])}")
        return False, active_count, queue_position

def release_llm_slot(request_id):
    """釋放 LLM 處理位置"""
    with llm_queue_lock:
        if os.path.exists(LLM_REQUESTS_FILE):
            try:
                with open(LLM_REQUESTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        # 從活躍列表移除
        if request_id in data["active"]:
            del data["active"][request_id]
            print(f"🔓 釋放LLM位置: {request_id[:8]}... (剩餘: {len(data['active'])}/{MAX_CONCURRENT_LLM_REQUESTS})")
        
        # 從等待列表移除（如果存在）
        if request_id in data.get("waiting", []):
            data["waiting"].remove(request_id)
        
        # 寫回文件
        with open(LLM_REQUESTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return len(data["active"])

def get_llm_queue_status(request_id):
    """獲取 LLM 隊列狀態"""
    with llm_queue_lock:
        if os.path.exists(LLM_REQUESTS_FILE):
            try:
                with open(LLM_REQUESTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = {"active": {}, "waiting": []}
        else:
            data = {"active": {}, "waiting": []}
        
        active_count = len(data["active"])
        queue_position = 0
        if request_id in data.get("waiting", []):
            queue_position = data["waiting"].index(request_id) + 1
        
        return active_count, queue_position

# ============= 全域樣式設計 (CSS) =============
st.markdown("""
    <style>
    /* 引入 Google Fonts: Noto Sans TC */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
        color: #1a1a1a;
    }

    .stApp {
        background-color: #f5f6f7;
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7ea;
    }

    .stTextInput>div>div>input {
        font-size: 15px !important;
        color: #1a1a1a !important;
        font-weight: 500 !important;
        background-color: #ffffff !important;
        border: 1px solid #d1d5da !important;
        padding: 10px 14px !important;
        border-radius: 4px !important;
    }

    .stTextInput>div>div>input::placeholder {
        color: #aaaaaa !important;
        font-weight: 400 !important;
    }

    .stTextInput>div>div>input:focus {
        border-color: #0066cc !important;
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.15) !important;
        color: #1a1a1a !important;
    }

    .stButton>button {
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid transparent;
        box-shadow: none;
        transition: background 0.15s, border-color 0.15s;
    }
    .stButton>button:hover {
        transform: none;
        box-shadow: none;
    }
    
    button[kind="primary"] {
        background: #0066cc !important;
        border-color: #0066cc !important;
        color: #ffffff !important;
    }

    button[kind="primary"]:hover {
        background: #0052a3 !important;
        border-color: #0052a3 !important;
    }

    .stButton>button {
        color: #333333;
    }

    button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #d1d5da !important;
        color: #333333 !important;
    }

    button[kind="secondary"]:hover {
        background: #f5f5f5 !important;
    }

    .product-card {
        background: #ffffff;
        border-radius: 6px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 16px;
        border: 1px solid #e5e7ea;
        transition: box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .product-card::before {
        display: none;
    }
    .product-card:hover {
        transform: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #c8cdd4;
    }
    .product-card:hover::before {
        display: none;
    }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 10px;
        box-shadow: none;
        transition: none;
        letter-spacing: 0;
    }
    .badge:hover {
        transform: none;
        box-shadow: none;
    }
    .badge-momo {
        background: #fff0f0;
        color: #cc0000;
        border: 1px solid #ffcccc;
    }
    .badge-pchome {
        background: #fff5e5;
        color: #c05800;
        border: 1px solid #ffd090;
    }

    .price-tag {
        font-family: 'Noto Sans TC', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #cc0000;
        margin: 8px 0;
    }
    .price-symbol {
        font-size: 0.85rem;
        color: #888888;
        font-weight: 400;
    }

    .match-result-container {
        background: #f5fdf7;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        box-shadow: none;
        padding: 16px;
        margin-top: 16px;
    }
    
    .ai-reasoning-box {
        background: #f7f8fa;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 14px;
        border-left: 4px solid #0066cc;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #333333;
        box-shadow: none;
        transition: none;
    }
    .ai-reasoning-box:hover {
        border-left-width: 4px;
        box-shadow: none;
    }

    .stProgress > div > div > div > div {
        background-image: none;
        background-color: #0066cc;
    }
    
    .img-container {
        width: 100%;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: #f8f8f8;
        border-radius: 4px;
        margin-bottom: 14px;
        border: 1px solid #eeeeee;
        box-shadow: none;
        transition: none;
    }
    .img-container:hover {
        border-color: #eeeeee;
        box-shadow: none;
    }
    .img-container img {
        max-height: 100%;
        max-width: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        display: block;
        transition: none;
    }
    .img-container:hover img {
        transform: none;
    }

    .product-grid-card {
        border-radius: 6px;
        padding: 14px;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
        height: 100%;
        border: 1px solid #e5e7ea;
        transition: box-shadow 0.2s;
    }
    .product-grid-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .product-grid-card .img-wrapper {
        width: 100%;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background-color: #f8f8f8;
        border-radius: 4px;
        margin-bottom: 10px;
        border: 1px solid #eeeeee;
    }

    .product-grid-card .img-wrapper img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }

    .comparison-card {
        padding: 20px;
        display: flex;
        align-items: start;
        gap: 20px;
        margin-bottom: 16px;
        border-radius: 6px;
        border: 1px solid #e5e7ea;
        transition: box-shadow 0.2s;
        position: relative;
    }
    .comparison-card::after {
        display: none;
    }
    .comparison-card:hover {
        transform: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .comparison-card:hover::after {
        display: none;
    }

    .comparison-card .img-section {
        width: 130px;
        flex-shrink: 0;
        text-align: center;
    }

    .comparison-card .content-section {
        flex-grow: 1;
        min-width: 0;
    }

    /* 手機響應式設計 */
    @media only screen and (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem !important;
            max-width: 100% !important;
        }

        h1 {
            font-size: 1.8rem !important;
            line-height: 1.3 !important;
            color: #1a1a1a !important;
            font-weight: 700 !important;
        }
        h2 {
            font-size: 1.5rem !important;
            line-height: 1.3 !important;
            color: #1a1a1a !important;
            font-weight: 700 !important;
        }
        h3 {
            font-size: 1.25rem !important;
            line-height: 1.3 !important;
            color: #1a1a1a !important;
            font-weight: 700 !important;
        }
        h4 {
            font-size: 1.05rem !important;
            line-height: 1.4 !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }

        .product-card {
            padding: 16px !important;
            border-radius: 6px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
            background: #ffffff !important;
            border: 1px solid #e5e7ea !important;
        }

        .product-grid-card {
            padding: 14px !important;
            margin-bottom: 14px !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
            background: #ffffff !important;
            border: 1px solid #e5e7ea !important;
        }

        .product-grid-card .img-wrapper {
            height: 160px !important;
            border-radius: 4px !important;
            background: #f8f8f8 !important;
            border: 1px solid #eeeeee !important;
        }

        .comparison-card {
            flex-direction: column !important;
            padding: 14px !important;
            gap: 12px !important;
        }

        .comparison-card .img-section {
            width: 100% !important;
            max-width: 180px !important;
            margin: 0 auto !important;
        }

        .comparison-card .content-section {
            width: 100% !important;
        }

        .img-container {
            height: 160px !important;
            border-radius: 4px !important;
            background: #f8f8f8 !important;
            border: 1px solid #eeeeee !important;
        }

        .stButton>button {
            font-size: 1rem !important;
            padding: 12px 20px !important;
            white-space: normal !important;
            height: auto !important;
            min-height: 44px !important;
            font-weight: 600 !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }

        .stButton>button:active {
            transform: none !important;
        }

        button[kind="primary"] {
            color: #ffffff !important;
            background: #0066cc !important;
            border-color: #0066cc !important;
        }

        button[kind="secondary"] {
            color: #333333 !important;
            background: #ffffff !important;
            border: 1px solid #d1d5da !important;
        }

        .stButton>button:not([kind="primary"]):not([kind="secondary"]) {
            background: #ffffff !important;
            color: #333333 !important;
            border: 1px solid #d1d5da !important;
            font-weight: 600 !important;
        }

        .price-tag {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #cc0000 !important;
        }

        .ai-reasoning-box {
            padding: 14px 16px !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            color: #333333 !important;
            background: #f7f8fa !important;
            border-left: 4px solid #0066cc !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }

        .badge {
            font-size: 0.8rem !important;
            padding: 3px 10px !important;
            font-weight: 700 !important;
            border-radius: 3px !important;
            box-shadow: none !important;
        }

        a {
            font-size: 0.9rem !important;
            color: #0066cc !important;
            font-weight: 600 !important;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }

        .stTextInput>div>div>input {
            font-size: 16px !important;
            color: #1a1a1a !important;
            font-weight: 500 !important;
            background-color: #ffffff !important;
            border: 1px solid #d1d5da !important;
            border-radius: 4px !important;
            padding: 12px 14px !important;
            -webkit-text-fill-color: #1a1a1a !important;
        }

        .stTextInput>div>div>input:focus {
            border-color: #0066cc !important;
            box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.15) !important;
            color: #1a1a1a !important;
            -webkit-text-fill-color: #1a1a1a !important;
        }

        .stRadio > label {
            font-size: 1rem !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }

        .stRadio > div > label {
            color: #1a1a1a !important;
            font-weight: 500 !important;
        }

        label {
            font-size: 0.95rem !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        p {
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            color: #333333 !important;
        }

        span {
            font-size: 0.9rem !important;
            color: #333333 !important;
        }

        strong, b {
            font-weight: 700 !important;
            color: #1a1a1a !important;
        }

        /* 對話框（Dialog）優化 */
        [data-testid="stDialog"] {
            background-color: #ffffff !important;
        }

        /* 對話框背景遮罩 */
        [data-testid="stDialog"]::before {
            background-color: rgba(255, 255, 255, 0.95) !important;
        }

        /* 對話框內容區域 */
        [data-testid="stDialog"] > div {
            background-color: #ffffff !important;
        }

        [data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            background-color: #ffffff !important;
        }
        
        /* 對話框關閉按鈕優化 */
        [data-testid="stDialog"] button[aria-label*="Close"],
        [data-testid="stDialog"] button[aria-label*="close"],
        [data-testid="stDialog"] button[kind="header"],
        [data-testid="stDialog"] [data-testid="baseButton-header"],
        [data-testid="stDialog"] button[class*="baseButton-header"] {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #4a5568 !important;
            width: 40px !important;
            height: 40px !important;
            border-radius: 50% !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
            border: 2px solid #e2e8f0 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            opacity: 1 !important;
            backdrop-filter: blur(8px) !important;
        }
        
        [data-testid="stDialog"] button[aria-label*="Close"]:hover,
        [data-testid="stDialog"] button[aria-label*="close"]:hover,
        [data-testid="stDialog"] button[kind="header"]:hover,
        [data-testid="stDialog"] [data-testid="baseButton-header"]:hover,
        [data-testid="stDialog"] button[class*="baseButton-header"]:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            transform: scale(1.08) !important;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
            border-color: transparent !important;
        }
        
        [data-testid="stDialog"] button[aria-label*="Close"]:active,
        [data-testid="stDialog"] button[aria-label*="close"]:active,
        [data-testid="stDialog"] button[kind="header"]:active,
        [data-testid="stDialog"] [data-testid="baseButton-header"]:active,
        [data-testid="stDialog"] button[class*="baseButton-header"]:active {
            transform: scale(0.96) !important;
        }
        
        /* 關閉按鈕的 SVG 圖標優化 */
        [data-testid="stDialog"] button[aria-label*="Close"] svg,
        [data-testid="stDialog"] button[aria-label*="close"] svg,
        [data-testid="stDialog"] button[kind="header"] svg,
        [data-testid="stDialog"] [data-testid="baseButton-header"] svg,
        [data-testid="stDialog"] button[class*="baseButton-header"] svg {
            width: 18px !important;
            height: 18px !important;
            stroke-width: 2.5px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stDialog"] button[aria-label*="Close"]:hover svg,
        [data-testid="stDialog"] button[aria-label*="close"]:hover svg,
        [data-testid="stDialog"] button[kind="header"]:hover svg,
        [data-testid="stDialog"] [data-testid="baseButton-header"]:hover svg,
        [data-testid="stDialog"] button[class*="baseButton-header"]:hover svg {
            stroke-width: 3px !important;
            color: white !important;
            fill: white !important;
        }

        [data-testid="stDialog"] h1,
        [data-testid="stDialog"] h2,
        [data-testid="stDialog"] h3,
        [data-testid="stDialog"] h4 {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        [data-testid="stDialog"] p,
        [data-testid="stDialog"] span,
        [data-testid="stDialog"] div {
            color: #1a202c !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
        }

        [data-testid="stDialog"] a {
            color: #2c5282 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            text-decoration: underline !important;
        }

        /* 對話框內的商品卡片 */
        [data-testid="stDialog"] .product-card {
            background: #ffffff !important;
            border: 2px solid #e2e8f0 !important;
        }

        [data-testid="stDialog"] .product-card h4 {
            color: #000000 !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            line-height: 1.5 !important;
        }

        [data-testid="stDialog"] .price-tag {
            color: #c53030 !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
        }

        [data-testid="stDialog"] strong {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        /* 對話框內的比對結果卡片 */
        [data-testid="stDialog"] .comparison-card {
            background: #ffffff !important;
        }

        [data-testid="stDialog"] .comparison-card h4 {
            color: #000000 !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }

        [data-testid="stDialog"] .ai-reasoning-box {
            background-color: #f0f4f8 !important;
            color: #1a202c !important;
            font-size: 1rem !important;
            border-left: 4px solid #3182ce !important;
        }

        [data-testid="stDialog"] .ai-reasoning-box strong {
            color: #2c5282 !important;
            font-weight: 700 !important;
        }

        [data-testid="stDialog"] .ai-reasoning-box span {
            color: #000000 !important;
            font-weight: 500 !important;
        }

        /* 對話框內的徽章 */
        [data-testid="stDialog"] .badge {
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }

        /* 對話框內所有文字強制黑色 */
        [data-testid="stDialog"] * {
            color: #1a202c !important;
        }

        [data-testid="stDialog"] h1,
        [data-testid="stDialog"] h2,
        [data-testid="stDialog"] h3,
        [data-testid="stDialog"] h4,
        [data-testid="stDialog"] strong {
            color: #000000 !important;
        }
        

    /* 超小螢幕（iPhone SE 等） */
    @media only screen and (max-width: 375px) {
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        h4 { font-size: 1rem !important; }

        .product-card { padding: 12px !important; }
        .product-grid-card { padding: 10px !important; }
        .product-grid-card .img-wrapper { height: 140px !important; }
        .img-container { height: 140px !important; }
        .stButton>button { font-size: 0.9rem !important; min-height: 40px !important; }
        .ai-reasoning-box { font-size: 0.85rem !important; padding: 10px 12px !important; }
        .badge { font-size: 0.75rem !important; padding: 2px 8px !important; }
        .price-tag { font-size: 1.25rem !important; }
    }

    /* 平板尺寸 */
    @media only screen and (min-width: 769px) and (max-width: 1024px) {
        .main .block-container { padding: 1.5rem !important; }
        .product-card { padding: 18px !important; }
        .img-container { height: 180px !important; }
        .product-grid-card .img-wrapper { height: 160px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ============= 安全配置：從環境變數或 Streamlit secrets 載入 =============
def get_api_key():
    """
    安全地獲取 API Key
    優先順序：Streamlit Secrets > 環境變數 > 側邊欄輸入
    """
    # 1. 嘗試從 Streamlit Secrets 讀取（部署到 Streamlit Cloud 時使用）
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # 2. 嘗試從環境變數讀取（本地開發時使用）
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        return api_key
    
    # 3. 如果都沒有，返回 None（稍後會要求用戶輸入）
    return None

# Ollama 配置
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join("models", "models20-multilingual-e5-large_fold_1"))

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"找不到模型路徑：{path}")
        return None
    return SentenceTransformer(path)

def load_local_data():
    """載入本地預設資料（僅用於初始化示例）"""
    # 先嘗試從根目錄讀取
    momo_path = "momo.csv"
    pchome_path = "pchome.csv"
    
    # 如果根目錄沒有，再試 dataset/test/
    if not os.path.exists(momo_path):
        momo_path = os.path.join("dataset", "test", "momo.csv")
        pchome_path = os.path.join("dataset", "test", "pchome.csv")
    
    try:
        # 直接讀取 CSV，使用第一行作為表頭
        momo_df = pd.read_csv(momo_path, sep=',')
        pchome_df = pd.read_csv(pchome_path, sep=',')
        
        # 移除 dtype=str，讓 pandas 自動推斷類型
        # 確保價格欄位是數值型
        if 'price' in momo_df.columns:
            momo_df['price'] = pd.to_numeric(momo_df['price'], errors='coerce')
        if 'price' in pchome_df.columns:
            pchome_df['price'] = pd.to_numeric(pchome_df['price'], errors='coerce')
            
        return momo_df, pchome_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

def calculate_similarities_in_memory(momo_df, pchome_df, model, direction="momo_to_pchome"):
    """在內存中計算相似度（不寫入文件）
    
    Args:
        momo_df: MOMO 商品資料
        pchome_df: PChome 商品資料
        model: 語意模型
        direction: 比對方向，"momo_to_pchome" 或 "pchome_to_momo"
    """
    if momo_df.empty or pchome_df.empty:
        return {}
    
    try:
        # 準備文本
        momo_texts = [prepare_text(title, 'momo') for title in momo_df['title']]
        pchome_texts = [prepare_text(title, 'pchome') for title in pchome_df['title']]
        
        # 計算嵌入向量
        momo_embeddings = get_batch_embeddings(model, momo_texts)
        pchome_embeddings = get_batch_embeddings(model, pchome_texts)
        
        # 計算相似度
        similarities = {}
        threshold = 0.739465
        
        if direction == "momo_to_pchome":
            # MOMO → PChome（預設）
            for idx, momo_row in momo_df.iterrows():
                momo_id = str(momo_row['id'])
                momo_emb = momo_embeddings[idx].unsqueeze(0)
                
                # 計算與所有 PChome 商品的相似度
                cos_similarities = torch.nn.functional.cosine_similarity(
                    momo_emb, pchome_embeddings, dim=1
                ).cpu().numpy()
                
                # 找出超過門檻的商品
                matches = []
                for pchome_idx, score in enumerate(cos_similarities):
                    if score >= threshold:
                        pchome_row = pchome_df.iloc[pchome_idx]
                        matches.append({
                            'target_id': str(pchome_row['id']),
                            'target_title': pchome_row['title'],
                            'target_price': pchome_row.get('price'),
                            'target_image': pchome_row.get('image', ''),
                            'target_url': pchome_row.get('url', ''),
                            'similarity': float(score)
                        })
                
                # 按相似度排序
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                similarities[momo_id] = matches
        else:
            # PChome → MOMO
            for idx, pchome_row in pchome_df.iterrows():
                pchome_id = str(pchome_row['id'])
                pchome_emb = pchome_embeddings[idx].unsqueeze(0)
                
                # 計算與所有 MOMO 商品的相似度
                cos_similarities = torch.nn.functional.cosine_similarity(
                    pchome_emb, momo_embeddings, dim=1
                ).cpu().numpy()
                
                # 找出超過門檻的商品
                matches = []
                for momo_idx, score in enumerate(cos_similarities):
                    if score >= threshold:
                        momo_row = momo_df.iloc[momo_idx]
                        matches.append({
                            'target_id': str(momo_row['id']),
                            'target_title': momo_row['title'],
                            'target_price': momo_row.get('price'),
                            'target_image': momo_row.get('image', ''),
                            'target_url': momo_row.get('url', ''),
                            'similarity': float(score)
                        })
                
                # 按相似度排序
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                similarities[pchome_id] = matches
        
        return similarities
    except Exception as e:
        st.error(f"計算相似度時發生錯誤: {e}")
        return {}

def prepare_text(title, platform):
    return ("query: " if platform == 'momo' else "passage: ") + str(title)

def get_single_embedding(model, text):
    return model.encode([text], convert_to_tensor=True).cpu()

def get_batch_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=True).cpu()

def _llm_call_worker(prompt, request_id, user_id):
    """線程安全的LLM調用工作函數（帶佇列系統）
    
    Args:
        prompt: 提示文字
        request_id: 請求ID（用於追蹤）
        user_id: 用戶ID（用於追蹤）
    
    Returns:
        str: Ollama 回應文字
    """
    import threading
    thread_id = threading.current_thread().name
    
    try:
        # 等待獲取 LLM 處理位置
        max_wait_time = 300  # 最多等待 5 分鐘
        wait_start = time.time()
        
        while True:
            success, active_count, queue_pos = acquire_llm_slot(request_id, user_id)
            
            if success:
                # 獲得位置，開始處理
                print(f"\n🤖 LLM請求開始處理: {request_id[:8]}...")
                print(f"   線程: {thread_id}")
                print(f"   用戶: {user_id[:8]}...")
                print(f"   當前並行: {active_count}/{MAX_CONCURRENT_LLM_REQUESTS}")
                print(f"   時間: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                break
            
            # 檢查是否超時
            if time.time() - wait_start > max_wait_time:
                print(f"❌ LLM請求等待超時: {request_id[:8]}...")
                raise TimeoutError(f"等待LLM處理位置超時（超過{max_wait_time}秒）")
            
            # 顯示等待狀態
            if queue_pos > 0:
                print(f"⏳ LLM請求等待中: {request_id[:8]}... 隊列位置: {queue_pos}")
            
            # 等待一小段時間後重試
            time.sleep(2)
        
        # 執行LLM調用
        start_time = time.time()
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        duration = time.time() - start_time
        
        print(f"✅ LLM請求完成: {request_id[:8]}... (耗時: {duration:.2f}秒)")
        
        return response['message']['content']
    
    except Exception as e:
        print(f"❌ LLM請求失敗: {request_id[:8]}... - {str(e)}")
        raise
    
    finally:
        # 釋放 LLM 處理位置
        remaining = release_llm_slot(request_id)
        print(f"🔓 LLM資源已釋放，剩餘活躍: {remaining}/{MAX_CONCURRENT_LLM_REQUESTS}")

def gemini_verify_match(momo_title, pchome_title, similarity_score, momo_price=0, pchome_price=0):
    prompt = f"""判斷以下兩個商品是否為相同的產品。

商品 A (MOMO)：{momo_title}
商品 B (PChome)：{pchome_title}

**重要！顏色不同必須視為相同商品！**
- 如果品牌、型號、規格、容量、數量相同，但顏色不同 → **必須回答 is_match=true**
- 理由必須寫：「相同商品（顏色不同）」

**判斷規則**：
1. 品牌、型號、規格、容量、數量必須完全一致
2. **顏色不同 = 相同商品（一定要回答 is_match=true）**
3. 其他差異（容量、數量、規格、版本等）= 不同商品
4. 不要使用價格來判斷

請回傳 JSON 格式：
{{
    "is_match": true,  // 顏色不同時必須是 true
    "confidence": "high"/"medium"/"low",
    "reasoning": "相同商品（顏色不同）"  // 顏色不同時必須用這個格式
}}
"""
    try:
        # 為本次請求生成唯一ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # 獲取用戶ID（如果在 session_state 中）
        user_id = st.session_state.get('session_id', 'unknown')
        
        # 使用線程池提交LLM調用（帶佇列控制）
        future = llm_executor.submit(
            _llm_call_worker,
            OLLAMA_MODEL,
            [{'role': 'user', 'content': prompt}],
            {'temperature': 0.1},
            request_id,
            user_id
        )
        
        # 等待結果
        response = future.result()
        
        text = response['message']['content'].strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"is_match": False, "confidence": "low", "reasoning": f"API 錯誤: {str(e)}"}

def gemini_verify_batch(match_pairs, direction="momo_to_pchome"):
    """批次驗證商品配對（一次處理一個來源商品的所有候選商品）
    
    Args:
        match_pairs: list of dict, 每個 dict 包含 {'momo_title', 'pchome_title', 'momo_price', 'pchome_price', 'similarity'}
        direction: 比對方向，"momo_to_pchome" 或 "pchome_to_momo"
    
    Returns:
        list of dict: 每個結果包含 {'is_match', 'confidence', 'reasoning'}
    """
    if not match_pairs:
        return []
    
    # 根據比對方向設定平台名稱
    if direction == "momo_to_pchome":
        platform_a = "MOMO"
        platform_b = "PChome"
    else:
        platform_a = "PChome"
        platform_b = "MOMO"
    
    # 構建批次 prompt
    prompt = f"""判斷「一個 {platform_a} 商品」與「多個 {platform_b} 候選商品」的配對。

**重要**：
- 獨立判斷每組配對，不受其他結果影響
- 可能 0 個匹配、1 個或多個匹配

**重要！顏色不同必須視為相同商品！**
- 如果品牌、型號、規格、容量、數量相同，但顏色不同 → **必須回答 is_match=true**

**判斷規則**：
1. 品牌、型號、規格、容量、數量必須完全一致
2. **顏色不同 = 相同商品（一定要回答 is_match=true）**
3. 其他差異（容量、數量、規格、版本等）= 不同商品
4. 不要使用價格來判斷

---

"""
    
    # 添加每組商品配對
    for i, pair in enumerate(match_pairs, 1):
        prompt += f"""【配對 {i}】
商品 A ({platform_a})：{pair['momo_title']}
商品 B ({platform_b})：{pair['pchome_title']}
第一階段相似度：{pair['similarity']:.4f}

"""
    
    prompt += f"""請針對以上 {len(match_pairs)} 組商品配對，分別判斷並回傳純 JSON 陣列格式：
[
    {{"is_match": true/false, "confidence": "high/medium/low", "reasoning": "簡短說明（如：相同商品、包裝數不同等）"}},
    {{"is_match": true/false, "confidence": "high/medium/low", "reasoning": "簡短說明（如：相同商品、包裝數不同等）"}},
    ...
]

請確保陣列中有 {len(match_pairs)} 個結果，順序對應上述配對順序。"""
    
    try:
        # 為本次請求生成唯一ID
        import uuid
        request_id = str(uuid.uuid4())
        
        print(f"\n📤 提交 LLM 請求: {request_id[:8]}...")
        print(f"   候選商品數: {len(match_pairs)}")
        print(f"   時間: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # 獲取用戶ID
        user_id = st.session_state.get('session_id', 'unknown')
        
        # 使用線程池提交LLM調用（允許並行）
        future = llm_executor.submit(
            _llm_call_worker,
            prompt,
            request_id,
            user_id
        )
        
        print(f"📝 請求已提交至線程池: {request_id[:8]}...")
        
        # 等待結果
        text = future.result()
        
        print(f"📬 收到 LLM 回應: {request_id[:8]}...")
        
        text = text.strip()
        
        # 解析 JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        results = json.loads(text)
        
        # 確保返回正確數量的結果
        if len(results) != len(match_pairs):
            # 如果數量不匹配，返回預設錯誤結果
            print(f"⚠️ AI 返回結果數量不正確: 預期 {len(match_pairs)} 個，實際收到 {len(results)} 個")
            return [{"is_match": False, "confidence": "low", "reasoning": f"AI 返回結果數量錯誤（預期 {len(match_pairs)} 個，收到 {len(results)} 個）"} for _ in match_pairs]
        
        return results
    
    except json.JSONDecodeError as e:
        # JSON 解析錯誤
        print(f"❌ JSON 解析失敗: {str(e)}")
        return [{"is_match": False, "confidence": "low", "reasoning": "AI 回應格式錯誤（JSON 解析失敗）"} for _ in match_pairs]
    
    except Exception as e:
        # 其他錯誤
        print(f"❌ 處理 AI 回應時發生錯誤: {str(e)}")
        return [{"is_match": False, "confidence": "low", "reasoning": f"處理錯誤: {str(e)}"} for _ in match_pairs]

# ============= 初始化資料庫 =============
init_db()

# ============= 初始化 Session State =============
if 'momo_df' not in st.session_state:
    # 嘗試載入示例數據，如果沒有就用空 DataFrame
    momo_df, pchome_df = load_local_data()
    st.session_state.momo_df = momo_df
    st.session_state.pchome_df = pchome_df
if 'scraping_done' not in st.session_state:
    st.session_state.scraping_done = False
if 'similarities' not in st.session_state:
    st.session_state.similarities = {}
if 'user_session_id' not in st.session_state:
    # 為每個用戶生成唯一 ID（使用 UUID 確保絕對唯一性）
    import uuid
    st.session_state.user_session_id = str(uuid.uuid4())
    print(f"🆕 創建新用戶 ID: {st.session_state.user_session_id}")
    # 只在首次創建時標記為新用戶加入
    update_user_peak(st.session_state.user_session_id, 'join')
else:
    # 已存在的用戶，只更新活動時間（不打印新用戶加入訊息）
    update_user_peak(st.session_state.user_session_id, 'update')
if 'cancel_search' not in st.session_state:
    st.session_state.cancel_search = False
if 'is_searching' not in st.session_state:
    st.session_state.is_searching = False

# ============= 搜尋商品函數 =============
def handle_product_search(keyword, model, momo_progress_placeholder, momo_status_placeholder, pchome_progress_placeholder, pchome_status_placeholder):
    """處理商品搜尋的函數（多用戶安全版本 + 並行爬取 + 進度條 + 跨進程佇列系統）"""
    if not keyword:
        st.error("請填寫商品名稱！")
        return False
    
    # 設置搜尋狀態
    st.session_state.is_searching = True
    st.session_state.cancel_search = False
    
    # 固定參數
    max_products = 100
    
    # ========== 佇列系統：嘗試獲取爬蟲位置 ==========
    user_id = st.session_state.user_session_id
    acquired = False
    
    try:
        # 輪詢直到獲取位置
        while not acquired:
            success, active_count, queue_pos = try_acquire_scraper_slot(user_id)
            
            if success:
                acquired = True
                print(f"🚀 開始爬蟲，當前活躍爬蟲: {active_count}/{MAX_CONCURRENT_SCRAPERS}")
                with momo_status_placeholder:
                    st.success(f"已取得搜尋權限！開始搜尋...（當前 {active_count}/{MAX_CONCURRENT_SCRAPERS} 組）")
                time.sleep(1)
                break
            
            # 沒有獲取到，顯示排隊訊息
            if queue_pos > 0:
                with momo_status_placeholder:
                    st.info(f"系統目前有 {active_count}/{MAX_CONCURRENT_SCRAPERS} 組用戶正在搜尋，您在第 {queue_pos} 位，請稍候...")
                with pchome_status_placeholder:
                    st.info("等待中...")
            
            # 檢查用戶是否取消
            if st.session_state.cancel_search:
                release_scraper_slot(user_id)  # 確保從等待列表移除
                st.warning("搜尋已取消")
                st.session_state.is_searching = False
                return False
            
            # 等待 2 秒後重試
            time.sleep(2)

        # ========== DB 快取查詢：命中則跳過爬蟲 ==========
        cached_momo, cached_pchome = get_cached_results(keyword, max_age_hours=24)
        if cached_momo is not None and cached_pchome is not None:
            release_scraper_slot(user_id)
            acquired = False  # 不需要爬蟲，標記已釋放
            momo_status_placeholder.success(f"已使用快取資料！找到 {len(cached_momo)} 件 MOMO 商品")
            pchome_status_placeholder.success(f"已使用快取資料！找到 {len(cached_pchome)} 件 PChome 商品")

            momo_df_cache = pd.DataFrame(cached_momo)
            pchome_df_cache = pd.DataFrame(cached_pchome)
            # image 欄位已由 DB 查詢直接回傳，不需要 rename
            st.session_state.momo_df = momo_df_cache
            st.session_state.pchome_df = pchome_df_cache

            st.markdown("---")
            st.markdown("### 正在分析商品...")
            calc_progress = st.progress(0, text="處理中，請稍候...")
            try:
                calc_progress.progress(30, text="找尋相似產品中...")                
                st.session_state.similarities = calculate_similarities_in_memory(
                    st.session_state.momo_df,
                    st.session_state.pchome_df,
                    model,
                    direction=st.session_state.get('match_direction', 'momo_to_pchome')
                )
                calc_progress.progress(100, text="完成！")
                time.sleep(0.3)
                calc_progress.empty()
                st.success("商品資料準備完成，現在可以選擇商品進行比價。")
                log_search_query(
                    keyword=keyword,
                    user_session_id=st.session_state.user_session_id,
                    momo_count=len(st.session_state.momo_df),
                    pchome_count=len(st.session_state.pchome_df)
                )
                time.sleep(1)
                st.session_state.is_searching = False
                st.rerun()
            except Exception as e:
                calc_progress.empty()
                st.error(f"計算相似度時發生錯誤: {e}")
            st.session_state.is_searching = False
            return True

        # ========== 無快取，正常爬蟲 ==========
        # 使用多線程和隊列
        import threading
        import queue
        
        # 創建隊列來傳遞進度信息
        momo_queue = queue.Queue()
        pchome_queue = queue.Queue()
        
        # 存儲結果的容器
        results = {'momo': None, 'pchome': None}
        
        # 使用線程安全的標誌來控制取消（避免在子線程中訪問 session_state）
        cancel_flag = {'value': False}
        
        # 使用 Event 來同步爬蟲啟動時機
        momo_ready = threading.Event()
        
        # 取消檢查函數
        def is_cancelled():
            return cancel_flag['value']
        
        def fetch_momo():
            try:
                # 定義回調函數 - 將進度放入隊列
                def momo_callback(current, total, message):
                    momo_queue.put({'current': current, 'total': total, 'message': message})
                    # 當 MOMO 開始實際抓取數據時，通知 PChome 可以啟動
                    if not momo_ready.is_set() and current > 0:
                        momo_ready.set()
                
                results['momo'] = fetch_products_for_momo(keyword, max_products, momo_callback, is_cancelled)
                momo_queue.put({'done': True})  # 標記完成
            except Exception as e:
                results['momo'] = []
                momo_queue.put({'error': str(e)})
                momo_ready.set()  # 即使失敗也要釋放鎖，避免死鎖
        
        def fetch_pchome():
            try:
                # 等待 MOMO 開始工作，但最多等待 10 秒避免死鎖
                momo_ready.wait(timeout=10)
                
                # 定義回調函數 - 將進度放入隊列
                def pchome_callback(current, total, message):
                    pchome_queue.put({'current': current, 'total': total, 'message': message})
                
                results['pchome'] = fetch_products_for_pchome(keyword, max_products, pchome_callback, is_cancelled)
                pchome_queue.put({'done': True})  # 標記完成
            except Exception as e:
                results['pchome'] = []
                pchome_queue.put({'error': str(e)})
        
        # 創建並啟動線程
        momo_thread = threading.Thread(target=fetch_momo, daemon=True)
        pchome_thread = threading.Thread(target=fetch_pchome, daemon=True)
        
        momo_thread.start()
        # PChome 線程會等待 MOMO 實際開始工作後再繼續
        pchome_thread.start()
        
        # 輪詢隊列並更新 UI
        momo_done = False
        pchome_done = False
        
        while not (momo_done and pchome_done):
            # 檢查是否被取消（同步 session_state 到 cancel_flag）
            if st.session_state.cancel_search:
                cancel_flag['value'] = True
                print("❌ 用戶取消搜尋")
                momo_status_placeholder.warning("搜尋已被取消")
                pchome_status_placeholder.warning("搜尋已被取消")
                st.session_state.is_searching = False
                return False
            
            # 更新 MOMO 進度
            if not momo_done:
                try:
                    momo_data = momo_queue.get_nowait()
                    if 'done' in momo_data:
                        momo_done = True
                    elif 'error' in momo_data:
                        momo_status_placeholder.error(f"錯誤: {momo_data['error']}")
                        momo_done = True
                    elif 'current' in momo_data:
                        progress = min(momo_data['current'] / momo_data['total'], 1.0)
                        momo_progress_placeholder.progress(progress)
                        momo_status_placeholder.info(momo_data['message'])
                except queue.Empty:
                    pass
            
            # 更新 PChome 進度
            if not pchome_done:
                try:
                    pchome_data = pchome_queue.get_nowait()
                    if 'done' in pchome_data:
                        pchome_done = True
                    elif 'error' in pchome_data:
                        pchome_status_placeholder.error(f"錯誤: {pchome_data['error']}")
                        pchome_done = True
                    elif 'current' in pchome_data:
                        progress = min(pchome_data['current'] / pchome_data['total'], 1.0)
                        pchome_progress_placeholder.progress(progress)
                        pchome_status_placeholder.info(pchome_data['message'])
                except queue.Empty:
                    pass
            
            # 短暫休眠避免過度輪詢
            time.sleep(0.1)
        
        # 等待線程完全結束
        momo_thread.join(timeout=1)
        pchome_thread.join(timeout=1)
        
        # 清除進度條
        momo_progress_placeholder.empty()
        pchome_progress_placeholder.empty()
        
        # 處理 MOMO 結果
        momo_products = results['momo']
        if momo_products:
            momo_status_placeholder.success(f"找到 {len(momo_products)} 件商品")
            # 直接轉換為 DataFrame 存入 session state
            st.session_state.momo_df = pd.DataFrame(momo_products)
            # 重命名 image_url 為 image（匹配顯示代碼的欄位名稱）
            if 'image_url' in st.session_state.momo_df.columns:
                st.session_state.momo_df.rename(columns={'image_url': 'image'}, inplace=True)
            if 'price' in st.session_state.momo_df.columns:
                st.session_state.momo_df['price'] = pd.to_numeric(st.session_state.momo_df['price'], errors='coerce')
        else:
            momo_status_placeholder.warning("沒有找到相關商品")
            st.session_state.momo_df = pd.DataFrame()
        
        # 處理 PChome 結果
        pchome_products = results['pchome']
        if pchome_products:
            pchome_status_placeholder.success(f"找到 {len(pchome_products)} 件商品")
            # 直接轉換為 DataFrame 存入 session state
            st.session_state.pchome_df = pd.DataFrame(pchome_products)
            # 重命名 image_url 為 image（匹配顯示代碼的欄位名稱）
            if 'image_url' in st.session_state.pchome_df.columns:
                st.session_state.pchome_df.rename(columns={'image_url': 'image'}, inplace=True)
            if 'price' in st.session_state.pchome_df.columns:
                st.session_state.pchome_df['price'] = pd.to_numeric(st.session_state.pchome_df['price'], errors='coerce')
        else:
            pchome_status_placeholder.warning("沒有找到相關商品")
            st.session_state.pchome_df = pd.DataFrame()
        
        st.markdown("---")
        
        if not st.session_state.momo_df.empty and not st.session_state.pchome_df.empty:
            st.success("搜尋完成！")
            
            # 在內存中計算相似度（不寫入文件）
            st.markdown("---")
            st.markdown("### 正在分析商品...")
            
            calc_progress = st.progress(0, text="處理中，請稍候...")
            
            try:
                calc_progress.progress(30, text="找尋相似產品中...")
                # 在內存中計算相似度，傳入比對方向
                st.session_state.similarities = calculate_similarities_in_memory(
                    st.session_state.momo_df,
                    st.session_state.pchome_df,
                    model,
                    direction=st.session_state.get('match_direction', 'momo_to_pchome')
                )
                
                calc_progress.progress(100, text="完成！")
                time.sleep(0.3)
                calc_progress.empty()
                
                st.success("商品資料準備完成，現在可以選擇商品進行比價。")
                
                # 儲存至資料庫（異步，不阻塞 UI）
                threading.Thread(
                    target=save_search_results,
                    args=(keyword, results.get('momo') or [], results.get('pchome') or []),
                    daemon=True
                ).start()

                # 記錄搜尋（在 rerun 之前）
                print(f"📝 正在記錄搜尋: {keyword}")
                log_search_query(
                    keyword=keyword,
                    user_session_id=st.session_state.user_session_id,
                    momo_count=len(st.session_state.momo_df),
                    pchome_count=len(st.session_state.pchome_df)
                )
                print(f"✅ 搜尋記錄完成")
                
                time.sleep(1)
                st.rerun()
                    
            except Exception as e:
                calc_progress.empty()
                st.error(f"計算相似度時發生錯誤: {e}")
        else:
            st.error("搜尋失敗，請重試")
        
        # 重置搜尋狀態
        st.session_state.is_searching = False
        st.session_state.cancel_search = False
        
        return True
    except Exception as e:
        st.error(f"搜尋過程發生錯誤: {e}")
        st.session_state.is_searching = False
        st.session_state.cancel_search = False
        return False
    
    finally:
        # ========== 釋放爬蟲佇列資源 ==========
        if acquired:
            remaining = release_scraper_slot(user_id)
            print(f"🔓 爬蟲資源已釋放，剩餘活躍: {remaining}/{MAX_CONCURRENT_SCRAPERS}")

# ============= UI 介面 =============

# 頁首區塊
col_header_left, col_header_right = st.columns([3, 1])

with col_header_left:
    st.markdown("# 商品跨平台比價")
    st.markdown("#### 在 MOMO 與 PChome 之間搜尋相同商品並比較價格")

with col_header_right:
    # 搜尋欄在右上角
    with st.form("search_form", clear_on_submit=False):
        # 比對方向選擇
        match_direction = st.radio(
            "比對方向",
            options=["momo_to_pchome", "pchome_to_momo"],
            format_func=lambda x: "MOMO → PChome" if x == "momo_to_pchome" else "PChome → MOMO",
            horizontal=True,
            label_visibility="collapsed"
        )
        search_keyword = st.text_input("商品名稱", placeholder="例如：dyson 吸塵器", label_visibility="collapsed")
        search_button = st.form_submit_button("搜尋", use_container_width=True, type="primary")

# 處理搜尋（在主畫面中間顯示進度）
if search_button and search_keyword:
    # 儲存比對方向到 session state
    st.session_state.match_direction = match_direction
    
    # 創建置中的進度顯示區域
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 使用 3:6:3 比例，讓進度條在中間，兩側留白
    _, center_col, _ = st.columns([2, 8, 2])
    
    with center_col:
        st.markdown("""
            <div style='text-align: center; padding: 24px 0 16px 0; border-bottom: 1px solid #e5e7ea; margin-bottom: 16px;'>
                <h3 style='color: #1a1a1a; margin: 0; font-size: 1.3rem; font-weight: 700;'>
                    正在搜尋商品中，請稍候...
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # 取消按鈕
        if st.button("取消搜尋", use_container_width=True, type="secondary"):
            st.session_state.cancel_search = True
            st.warning("正在取消搜尋...")
            time.sleep(0.5)
            st.rerun()
        
        # 進度條區域（根據比對方向調整順序）
        if match_direction == "momo_to_pchome":
            # MOMO 在左，PChome 在右
            prog_col1, prog_col2 = st.columns(2)
            
            with prog_col1:
                st.markdown("""
                    <div style='text-align: center; padding: 12px; background: #fff0f0; border: 1px solid #ffcccc; border-radius: 4px; margin-bottom: 10px;'>
                        <h4 style='color: #cc0000; margin: 0; font-size: 15px; font-weight: 700;'>MOMO</h4>
                    </div>
                """, unsafe_allow_html=True)
                momo_progress = st.empty()
                momo_status = st.empty()
            
            with prog_col2:
                st.markdown("""
                    <div style='text-align: center; padding: 12px; background: #fff5e5; border: 1px solid #ffd090; border-radius: 4px; margin-bottom: 10px;'>
                        <h4 style='color: #c05800; margin: 0; font-size: 15px; font-weight: 700;'>PChome</h4>
                    </div>
                """, unsafe_allow_html=True)
                pchome_progress = st.empty()
                pchome_status = st.empty()
        else:
            # PChome 在左，MOMO 在右
            prog_col1, prog_col2 = st.columns(2)
            
            with prog_col1:
                st.markdown("""
                    <div style='text-align: center; padding: 12px; background: #fff5e5; border: 1px solid #ffd090; border-radius: 4px; margin-bottom: 10px;'>
                        <h4 style='color: #c05800; margin: 0; font-size: 15px; font-weight: 700;'>PChome</h4>
                    </div>
                """, unsafe_allow_html=True)
                pchome_progress = st.empty()
                pchome_status = st.empty()
            
            with prog_col2:
                st.markdown("""
                    <div style='text-align: center; padding: 12px; background: #fff0f0; border: 1px solid #ffcccc; border-radius: 4px; margin-bottom: 10px;'>
                        <h4 style='color: #cc0000; margin: 0; font-size: 15px; font-weight: 700;'>MOMO</h4>
                    </div>
                """, unsafe_allow_html=True)
                momo_progress = st.empty()
                momo_status = st.empty()
    
    # 需要先載入模型
    temp_model = load_model(MODEL_PATH)
    if temp_model:
        # 使用剛創建的 placeholder 執行搜尋
        handle_product_search(search_keyword, temp_model, momo_progress, momo_status, pchome_progress, pchome_status)

st.markdown("---")

# ============= 比對模式（唯一頁面）=============
# 載入資料
momo_df = st.session_state.momo_df
pchome_df = st.session_state.pchome_df

# 載入資源
with st.spinner("系統準備中，請稍候..."):
    model = load_model(MODEL_PATH)

if model is None:
    st.stop()

# ============= 檢查商品資料 =============
if momo_df.empty and pchome_df.empty:
    st.warning("目前系統中還沒有商品資料，請點擊上方「搜尋」按鈕來新增商品。")
    st.stop()
elif momo_df.empty:
    st.warning("目前 MOMO 購物網沒有商品資料，請搜尋商品以新增資料。")
    st.stop()
elif pchome_df.empty:
    st.warning("目前 PChome 購物網沒有商品資料，請搜尋商品以新增資料。")
    st.stop()

# 所有 MOMO 商品（不分類別）
momo_products_in_query = momo_df.reset_index(drop=True)
pchome_candidates_pool = pchome_df.reset_index(drop=True)

# 固定相似度門檻為 0.739465
threshold = 0.739465

# 初始化選中的商品索引
if 'selected_product_index' not in st.session_state:
    st.session_state.selected_product_index = None
if 'dialog_open' not in st.session_state:
    st.session_state.dialog_open = False
if 'dialog_key' not in st.session_state:
    st.session_state.dialog_key = 0

# ============= 比對結果 Dialog 函數 =============
@st.dialog("商品比價結果", width="large")
def show_comparison_dialog(selected_product_row, dialog_key):
    """顯示商品比對結果"""
    
    # 記錄比對開始時間（整個dialog的開始）
    dialog_start_time = time.time()
    
    # 第一步：完全清空對話框內容
    clear_placeholder = st.empty()
    with clear_placeholder:
        st.markdown("")
    
    # 使用商品ID和dialog_key組合作為唯一標識
    unique_key = f"{selected_product_row.get('id', 0)}_{dialog_key}_{int(time.time() * 1000)}"
    
    # 清空佔位符
    clear_placeholder.empty()
    
    # 創建一個全新的容器來包裹所有內容
    main_container = st.container(key=f"dialog_main_{unique_key}")
    
    with main_container:
        # 使用兩欄布局顯示比對結果
        col_main_left, col_main_right = st.columns([1, 2], gap="large")
        
        # --- 左側：顯示選中的商品 ---
        with col_main_left:
            st.markdown("### 選中的商品")
            
            # 根據比對方向決定顯示的平台標籤
            match_direction = st.session_state.get('match_direction', 'momo_to_pchome')
            if match_direction == 'momo_to_pchome':
                platform_badge = "MOMO 購物網"
                badge_class = "badge-momo"
            else:
                platform_badge = "PChome 購物網"
                badge_class = "badge-pchome"
            
            # 顯示選中商品的詳細卡片
            price = selected_product_row.get('price')
            if pd.isna(price) or price is None:
                price_str = "價格未提供"
            else:
                price_str = f"NT$ {price:,.0f}"
            
            st.markdown(f"""
            <div class="product-card" style="position: relative; overflow: visible;">
                <div class="badge {badge_class}" style="margin-bottom: 15px;">{platform_badge}</div>
                <div class="img-container">
                    <img src="{selected_product_row.get('image', '')}" 
                         alt="{selected_product_row['title'][:50]}" 
                         loading="lazy"
                         onerror="this.onerror=null; this.src='https://via.placeholder.com/200x200?text=無法載入圖片';">
                </div>
                <h4 style="margin-top:14px; line-height:1.6; color:#1a1a1a; font-weight:700; font-size:1.05rem;">{selected_product_row['title']}</h4>
                <div style="background: #f8f8f8; padding: 14px; border-radius: 4px; margin-top: 14px; border: 1px solid #e5e7ea;">
                    <div style="color: #888888; font-size: 0.8rem; font-weight: 500; margin-bottom: 4px;">售價</div>
                    <div class="price-tag">NT$ {price_str}</div>
                </div>
                <div style="color:#333333; font-size:0.9rem; margin-top:12px; line-height:1.8; background: #f8f8f8; padding: 12px 14px; border-radius: 4px; border: 1px solid #e5e7ea;">
                    <div style="margin-bottom: 6px;"><strong style="color:#1a1a1a;">ID:</strong> <span style="color: #555;">{selected_product_row.get('id', 'N/A')}</span></div>
                    <div><strong style="color:#1a1a1a;">SKU:</strong> <span style="color: #555;">{selected_product_row.get('sku', 'N/A')}</span></div>
                </div>
                <a href="{selected_product_row.get('url', '#')}" target="_blank" 
                   style="display:block; text-align:center; margin-top:16px; background: #0066cc; color: white; padding:12px; border-radius:4px; text-decoration:none; font-weight:600; font-size:0.95rem;">
                   開啟商品頁面
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        # 設定變數以便後續比對邏輯使用
        is_valid_selection = True
        is_new_selection = True  # 每次進入比對頁面都視為新選擇
        
        # --- 右側：Action & Results ---
        with col_main_right:
            # 根據比對方向顯示不同的標題
            match_direction = st.session_state.get('match_direction', 'momo_to_pchome')
            target_platform = "PChome" if match_direction == 'momo_to_pchome' else "MOMO"
            
            # 建立固定的標題
            st.markdown(f"### 在 {target_platform} 尋找相同商品")
            progress_container = st.empty()
            
            # 清空區域標記
            clear_marker = st.empty()
            with clear_marker:
                st.markdown("")  # 空白標記用於分隔
            
            # 自動開始比對（當選擇新商品時）
            if is_valid_selection and is_new_selection:
                
                product_id = str(selected_product_row['id'])
                
                # 直接使用預計算的相似度資料
                stage1_matches_list = []
                
                if st.session_state.similarities and product_id in st.session_state.similarities:
                    stage1_matches_list = st.session_state.similarities[product_id]
                
                # 檢查第一階段結果，如果沒有找到則立即顯示
                if not stage1_matches_list:
                    st.warning(f"在 {target_platform} 沒有找到相似的商品")
                    st.info(f"建議：請選擇其他商品再試一次，或直接到 {target_platform} 網站手動搜尋")
                else:
                    candidates_to_verify = stage1_matches_list
                    
                    # 一次性處理所有候選商品
                    verified_results = []
                    
                    # 建立進度條顯示比對進度
                    overall_progress = st.progress(0, text="正在使用 AI 分析所有候選商品...")
                    
                    # 顯示並行處理狀態
                    with llm_queue_lock:
                        if os.path.exists(LLM_REQUESTS_FILE):
                            try:
                                with open(LLM_REQUESTS_FILE, 'r') as f:
                                    llm_data = json.load(f)
                                    current_parallel = len(llm_data.get("active", []))
                            except:
                                current_parallel = 0
                        else:
                            current_parallel = 0
                    
                    if current_parallel > 0:
                        st.info(f"系統狀態：目前有 {current_parallel} 個用戶正在使用 AI 比對功能，您的請求將並行處理")
                    
                    # 檢查候選商品數量，設定最大限制
                    MAX_CANDIDATES_PER_CALL = 50
                    
                    if len(candidates_to_verify) > MAX_CANDIDATES_PER_CALL:
                        st.warning(f"找到 {len(candidates_to_verify)} 個候選商品，數量較多，將使用前 {MAX_CANDIDATES_PER_CALL} 個進行比對")
                        candidates_to_verify = candidates_to_verify[:MAX_CANDIDATES_PER_CALL]
                    
                    # 準備所有配對資料（包含價格資訊）
                    all_pairs = [
                        {
                            'momo_title': selected_product_row['title'],
                            'momo_price': float(selected_product_row.get('price', 0)),
                            'pchome_title': match['target_title'],
                            'pchome_price': float(match.get('target_price', 0)),
                            'similarity': match['similarity']
                        }
                        for match in candidates_to_verify
                    ]
                    
                    # 記錄開始時間
                    stage2_start_time = time.time()
                    
                    # 分批處理（每批最多 10 個）
                    BATCH_SIZE = 10
                    all_results = []
                    processed_count = 0
                    
                    # 創建進度顯示容器
                    batch_progress_container = st.empty()
                    
                    # 創建標籤頁（tabs）來分類顯示結果
                    tab_matched, tab_unmatched = st.tabs(["比對成功", "未比對"])
                    
                    # 在每個標籤頁內創建容器
                    with tab_matched:
                        matched_container = st.container()
                    
                    with tab_unmatched:
                        unmatched_container = st.container()
                    
                    for i in range(0, len(all_pairs), BATCH_SIZE):
                        batch = all_pairs[i:i + BATCH_SIZE]
                        batch_candidates = candidates_to_verify[i:i + BATCH_SIZE]
                        batch_num = i // BATCH_SIZE + 1
                        total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
                        
                        # 只有在多於一個批次時才顯示批次處理進度
                        if total_batches > 1:
                            with batch_progress_container:
                                st.info(f"處理批次 {batch_num}/{total_batches}（{len(batch)} 個產品）")
                        
                        print(f"🔄 處理批次 {batch_num}/{total_batches}（{len(batch)} 個產品）")
                        
                        # 呼叫 LLM 處理這批資料
                        batch_results = gemini_verify_batch(batch, direction=match_direction)
                        all_results.extend(batch_results)
                        
                        # 立即輸出這批結果（分別放入匹配和未匹配容器）
                        for match, result in zip(batch_candidates, batch_results):
                            processed_count += 1
                            
                            # 記錄到 verified_results（用於後續統計）
                            verified_results.append({
                                'match': match,
                                'result': result,
                                'is_match': result.get('is_match', False)
                            })
                            
                            # 根據結果顯示不同樣式
                            if result.get('is_match'):
                                card_style = "background: #f6fff9;"
                                border_gradient = "background: #28a745;"
                                icon = "比對成功"
                                icon_badge = "background: #28a745; color: white; padding: 4px 12px; border-radius: 3px; font-size: 0.8rem; font-weight: 700;"
                                target_container = matched_container
                            else:
                                card_style = "background: #fff8f8;"
                                border_gradient = "background: #cc0000;"
                                icon = "未比對"
                                icon_badge = "background: #f0f0f0; color: #666; padding: 4px 12px; border-radius: 3px; font-size: 0.8rem; font-weight: 700;"
                                target_container = unmatched_container

                            # 結果卡片渲染（放入對應的容器）
                            with target_container:
                                st.markdown(f"""
                                <div class="product-card comparison-card" style="{card_style} position: relative; overflow: hidden;">
                                    <div style="position: absolute; left: 0; top: 0; bottom: 0; width: 6px; {border_gradient}"></div>
                                    <div class="img-section" style="padding-left: 10px;">
                                        <div class="badge badge-pchome" style="margin-bottom: 10px;">{target_platform}</div>
                                        <div style="background: white; border-radius: 12px; padding: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                                            <img src="{match.get('target_image', '')}" 
                                                 alt="{match['target_title'][:30]}"
                                                 loading="lazy"
                                                 style="width: 100%; height: auto; max-height: 130px; border-radius: 8px; object-fit: contain; display: block;" 
                                                 onerror="this.onerror=null; this.src='https://via.placeholder.com/130x130?text=無法載入圖片';">
                                        </div>
                                    </div>
                                    <div class="content-section">
                                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                                            <h4 style="margin: 0; font-size: 1.15rem; color: #1a202c; word-wrap: break-word; flex: 1; min-width: 200px; font-weight: 700; line-height: 1.5;">{match['target_title']}</h4>
                                            <span style="{icon_badge}">{icon}</span>
                                        </div>
                                        <div style="margin-top: 14px; display: flex; gap: 15px; font-size: 1rem; color: #1a202c; flex-wrap: wrap; align-items: center; background: white; padding: 12px 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                                            <span style="font-size: 0.9rem; color: #718096;">價格</span>
                                            <strong style="color: #c53030; font-size: 1.3rem; font-weight: 800;">NT$ {match.get('target_price', 0) if match.get('target_price') and not pd.isna(match.get('target_price')) else '價格未提供'}</strong>
                                        </div>
                                        <div class="ai-reasoning-box">
                                            <strong style="color: #2c5282; font-size: 1rem;">AI 判斷理由</strong>
                                            <p style="color: #1a202c; margin: 8px 0 0 0; line-height: 1.7;">{result.get('reasoning', '無詳細理由')}</p>
                                        </div>
                                        <div style="margin-top: 16px; text-align: right;">
                                            <a href="{match.get('target_url', '#')}" target="_blank" 
                                               style="display: inline-block; background: #0066cc; color: white; text-decoration: none; font-size: 0.9rem; font-weight: 600; padding: 9px 18px; border-radius: 4px; transition: background 0.15s;">
                                               查看商品詳情
                                            </a>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # 清除批次進度顯示
                    batch_progress_container.empty()
                    
                    # 記錄結束時間
                    stage2_end_time = time.time()
                    stage2_duration = stage2_end_time - stage2_start_time
                    
                    # 統計配對成功數量
                    matched_count = sum(1 for r in verified_results if r['is_match'])
                    
                    # 計算比對總耗時（從dialog開始到現在）
                    total_comparison_time = time.time() - dialog_start_time
                    
                    # 記錄性能數據到 JSON
                    performance_log = {
                        "timestamp": datetime.now().isoformat(),
                        "user_session_id": st.session_state.user_session_id,
                        "source_product_id": str(selected_product_row.get('id', 'N/A')),
                        "source_product_title": selected_product_row['title'],
                        "match_direction": match_direction,
                        "total_comparison_time_seconds": round(total_comparison_time, 3),
                        "stage2_llm_duration_seconds": round(stage2_duration, 3),
                        "total_candidates_tested": len(candidates_to_verify),
                        "matched_count": matched_count
                    }
                    
                    # 寫入詳細性能日誌（追加模式）
                    performance_file = "stage2_performance.json"
                    try:
                        if os.path.exists(performance_file):
                            try:
                                with open(performance_file, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                    if content:
                                        performance_logs = json.loads(content)
                                    else:
                                        performance_logs = []
                            except (json.JSONDecodeError, ValueError):
                                # 文件損壞或為空，重新創建
                                performance_logs = []
                        else:
                            performance_logs = []
                        
                        performance_logs.append(performance_log)
                        
                        with open(performance_file, 'w', encoding='utf-8') as f:
                            json.dump(performance_logs, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"❌ 記錄性能數據失敗: {e}")
                    
                    # 寫入 Session 比對時間記錄（專門用於追蹤每個session的比對時間）
                    session_times_file = "session_comparison_times.json"
                    try:
                        # 讀取現有的session記錄
                        if os.path.exists(session_times_file):
                            try:
                                with open(session_times_file, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                    if content:
                                        session_data = json.loads(content)
                                    else:
                                        session_data = {}
                            except (json.JSONDecodeError, ValueError):
                                session_data = {}
                        else:
                            session_data = {}
                        
                        # 為當前session創建或更新記錄
                        user_id = st.session_state.user_session_id
                        if user_id not in session_data:
                            session_data[user_id] = {
                                "session_id": user_id,
                                "first_comparison_time": datetime.now().isoformat(),
                                "total_comparisons": 0,
                                "comparison_records": []
                            }
                        
                        # 添加本次比對記錄（只記錄LLM處理時間）
                        comparison_record = {
                            "timestamp": datetime.now().isoformat(),
                            "product_id": str(selected_product_row.get('id', 'N/A')),
                            "product_title": selected_product_row['title'][:50] + "...",  # 限制長度
                            "match_direction": match_direction,
                            "llm_processing_time_seconds": round(stage2_duration, 3),  # 只記錄LLM處理時間
                            "candidates_tested": len(candidates_to_verify),
                            "matches_found": matched_count
                        }
                        
                        session_data[user_id]["comparison_records"].append(comparison_record)
                        session_data[user_id]["total_comparisons"] = len(session_data[user_id]["comparison_records"])
                        session_data[user_id]["last_comparison_time"] = datetime.now().isoformat()
                        
                        # 計算session統計數據（基於LLM處理時間）
                        records = session_data[user_id]["comparison_records"]
                        if records:
                            session_data[user_id]["average_llm_time"] = round(
                                sum(r["llm_processing_time_seconds"] for r in records) / len(records), 3
                            )
                            session_data[user_id]["total_llm_time"] = round(
                                sum(r["llm_processing_time_seconds"] for r in records), 3
                            )
                        
                        # 寫入文件
                        with open(session_times_file, 'w', encoding='utf-8') as f:
                            json.dump(session_data, f, ensure_ascii=False, indent=2)
                        
                        print(f"✅ Session LLM處理時間已記錄: {user_id[:8]}... - {stage2_duration:.2f}秒")
                    
                    except Exception as e:
                        print(f"❌ 記錄Session比對時間失敗: {e}")
                    
                    # 清除進度條
                    overall_progress.empty()
                    
                    # 顯示最終統計結果
                    st.markdown("---")
                    
                    verified_count = matched_count
                    
                    if verified_count == 0:
                        st.info(f"已檢查所有候選商品，但沒有找到完全相同的商品。")
                    else:
                        st.success(f"完成！在 {target_platform} 找到 {verified_count} 件相同商品")

# ============= 主內容區 =============

# 顯示完整商品網格
# 根據比對方向決定顯示哪個平台的商品
match_direction = st.session_state.get('match_direction', 'momo_to_pchome')

# 先設定變量（在 columns 外面）
if match_direction == 'momo_to_pchome':
    title_text = "## MOMO 購物網商品列表"
    source_platform = "MOMO"
    target_platform = "PChome"
    display_df = momo_products_in_query
else:
    title_text = "## PChome 購物網商品列表"
    source_platform = "PChome"
    target_platform = "MOMO"
    display_df = pchome_candidates_pool

# 檢查是否需要重新計算相似度（當切換方向後）
if (st.session_state.momo_df is not None and 
    st.session_state.pchome_df is not None and 
    not st.session_state.similarities):
    
    with st.spinner("正在重新計算相似度..."):
        try:
            # 重新計算相似度，使用當前的比對方向
            st.session_state.similarities = calculate_similarities_in_memory(
                st.session_state.momo_df,
                st.session_state.pchome_df,
                model,
                direction=match_direction
            )
            st.success("相似度計算完成！")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"計算相似度時發生錯誤: {e}")

# 添加切換比對方向的功能
col_title, col_switch = st.columns([3, 1])

with col_title:
    st.markdown(title_text)

with col_switch:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # 對齊標題
    if st.button("切換比對方向", use_container_width=True, help=f"目前：{source_platform} → {target_platform}"):
        # 切換方向
        new_direction = 'pchome_to_momo' if match_direction == 'momo_to_pchome' else 'momo_to_pchome'
        st.session_state.match_direction = new_direction
        
        # 清除舊的相似度計算結果，強制重新計算
        if 'similarities' in st.session_state:
            del st.session_state['similarities']
        
        st.success(f"已切換比對方向：{'PChome → MOMO' if new_direction == 'pchome_to_momo' else 'MOMO → PChome'}")
        st.rerun()

# 根據是否有相似商品分類
if st.session_state.similarities:
    # 分類商品：有相似商品 vs 無相似商品
    products_with_matches = []
    products_without_matches = []
    
    for idx, row in display_df.iterrows():
        product_id = str(row['id'])
        if product_id in st.session_state.similarities and st.session_state.similarities[product_id]:
            products_with_matches.append((idx, row))
        else:
            products_without_matches.append((idx, row))
    
    # 顯示有相似商品的部分
    if products_with_matches:
        st.markdown(f"### 找到相似商品（{len(products_with_matches)} 件）")
        st.markdown(f"以下商品在 {target_platform} 找到了相似的商品，點擊查看詳細比價")
        
        cols_per_row = 4
        for i in range(0, len(products_with_matches), cols_per_row):
            row_products = products_with_matches[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for col_idx, (prod_idx, row) in enumerate(row_products):
                with cols[col_idx]:
                    price = row.get('price')
                    if pd.isna(price) or price is None:
                        price_str = "價格未提供"
                    else:
                        price_str = f"NT$ {price:,.0f}"
                    
                    # 商品卡片 - 綠色邊框表示有匹配
                    st.markdown(f"""
                    <div class="product-grid-card" style="border: 2px solid #48bb78;">
                        <div class="img-wrapper">
                            <img src="{row.get('image', '')}" 
                                 alt="{row['title'][:50]}"
                                 loading="lazy"
                                 onerror="this.onerror=null; this.src='https://via.placeholder.com/200x200?text=無法載入圖片';">
                        </div>
                        <div style="flex: 1; font-size: 1rem; line-height: 1.5; margin-bottom: 10px; min-height: 60px; word-wrap: break-word; color: #1a202c; font-weight: 500;">{row['title']}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #38a169; margin: 10px 0;">{price_str}</div>
                        <div style="font-size: 0.9rem; color: #4a5568; margin-bottom: 12px; font-weight: 500;">
                            ID: {row.get('id', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 點擊按鈕 - 確保寬度一致
                    if st.button(
                        "查看比價",
                        key=f"view_comparison_{prod_idx}",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state.selected_product_index = prod_idx
                        st.session_state.dialog_open = True
                        st.session_state.dialog_key += 1
                        st.rerun()
        
        st.markdown("---")
    
    # 顯示無相似商品的部分
    if products_without_matches:
        st.markdown("### 未找到相似商品（{} 件）".format(len(products_without_matches)))
        st.markdown(f"以下商品在 {target_platform} 找到了相似的商品，點擊查看詳細比價")
        
        cols_per_row = 4
        for i in range(0, len(products_without_matches), cols_per_row):
            row_products = products_without_matches[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for col_idx, (prod_idx, row) in enumerate(row_products):
                with cols[col_idx]:
                    price = row.get('price')
                    if pd.isna(price) or price is None:
                        price_str = "價格未提供"
                    else:
                        price_str = f"NT$ {price:,.0f}"
                    
                    # 商品卡片 - 灰色邊框表示無匹配
                    st.markdown(f"""
                    <div class="product-grid-card" style="border: 2px solid #cbd5e0; opacity: 0.85;">
                        <div class="img-wrapper">
                            <img src="{row.get('image', '')}" 
                                 alt="{row['title'][:50]}"
                                 loading="lazy"
                                 onerror="this.onerror=null; this.src='https://via.placeholder.com/200x200?text=無法載入圖片';">
                        </div>
                        <div style="flex: 1; font-size: 1rem; line-height: 1.5; margin-bottom: 10px; min-height: 60px; word-wrap: break-word; color: #2d3748; font-weight: 500;">{row['title']}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #718096; margin: 10px 0;">{price_str}</div>
                        <div style="font-size: 0.9rem; color: #4a5568; margin-bottom: 12px; font-weight: 500;">
                            ID: {row.get('id', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 點擊按鈕 - 確保寬度一致
                    if st.button(
                        "查看詳情",
                        key=f"view_comparison_{prod_idx}",
                        use_container_width=True
                    ):
                        st.session_state.selected_momo_index = prod_idx
                        st.session_state.dialog_open = True
                        st.session_state.dialog_key += 1
                        st.rerun()
else:
    # 如果還沒有相似度數據，顯示所有商品（初始狀態）
    st.markdown("點擊商品卡片以查看比價結果")
    
    cols_per_row = 4
    rows = [momo_products_in_query[i:i+cols_per_row] for i in range(0, len(momo_products_in_query), cols_per_row)]

    for row_products in rows:
        cols = st.columns(cols_per_row)
        for col_idx, (prod_idx, row) in enumerate(row_products.iterrows()):
            with cols[col_idx]:
                price = row.get('price')
                if pd.isna(price) or price is None:
                    price_str = "價格未提供"
                else:
                    price_str = f"NT$ {price:,.0f}"
                
                # 商品卡片
                card_html = f"""
                <div class="momo-grid-card" style="min-height: 450px; display: flex; flex-direction: column;">
                    <div class="momo-grid-badge">#{prod_idx+1}</div>
                    <div class="momo-grid-img-container">
                        <img src="{row.get('image', '')}" 
                             class="momo-grid-img"
                             onerror="this.onerror=null; this.src='https://via.placeholder.com/200x200?text=無法載入圖片';">
                    </div>
                    <div class="momo-grid-title" style="flex: 1; min-height: 60px;">{row['title']}</div>
                    <div class="momo-grid-price">{price_str}</div>
                    <div class="momo-grid-info" style="margin-bottom: 10px;">
                        ID: {row.get('id', 'N/A')}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # 點擊按鈕
                if st.button(
                    "查看比價",
                    key=f"view_comparison_{prod_idx}",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state.selected_product_index = prod_idx
                    st.session_state.dialog_open = True
                    st.session_state.dialog_key += 1
                    st.rerun()

# 檢查是否需要顯示 dialog
if st.session_state.dialog_open and st.session_state.selected_product_index is not None:
    # 根據比對方向選擇正確的商品資料源
    match_direction = st.session_state.get('match_direction', 'momo_to_pchome')
    if match_direction == 'momo_to_pchome':
        selected_product_row = momo_products_in_query.iloc[st.session_state.selected_product_index]
    else:
        selected_product_row = pchome_candidates_pool.iloc[st.session_state.selected_product_index]
    
    show_comparison_dialog(selected_product_row, st.session_state.dialog_key)
    # Dialog 關閉後清除狀態
    st.session_state.dialog_open = False
    st.session_state.selected_product_index = None
