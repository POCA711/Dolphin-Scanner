import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import re

def calculate_obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def is_rejection_candle(df_row):
    body = abs(df_row['Close'] - df_row['Open'])
    lower_wick = min(df_row['Open'], df_row['Close']) - df_row['Low']
    if body == 0: return lower_wick > 0 
    return lower_wick > (body * 2)

@st.cache_data(ttl=3600)
def fetch_and_scan(tickers, use_rejection):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    
    for i, symbol in enumerate(tickers):
        symbol = symbol.strip()
        if not symbol.endswith(".TW") and not symbol.endswith(".TWO"):
            symbol = f"{symbol}.TW"
            
        status_text.text(f"正在掃描: {symbol} ({i+1}/{total})")
        
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="6mo")
            if len(df) < 60: continue

            df['OBV'] = calculate_obv(df)
            df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
            
            recent_high = df['High'].tail(60).max()
            recent_low = df['Low'].tail(60).min()
            diff = recent_high - recent_low
            gp_top = recent_high - (0.618 * diff)
            gp_bottom = recent_high - (0.65 * diff)
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            in_gp = (gp_bottom * 0.985) <= current['Close'] <= (gp_top * 1.015)
            obv_ok = current['OBV'] > current['OBV_MA20']
            
            rejection_ok = True
            if use_rejection:
                rejection_ok = is_rejection_candle(current) or is_rejection_candle(prev)

            volume_ok = df['Volume'].tail(5).mean() > 1000

            if in_gp and obv_ok and rejection_ok and volume_ok:
                results.append({
                    "股票代碼": symbol.replace(".TW", ""),
                    "最新收盤價": round(current['Close'], 2),
                    "金色口袋 (0.618-0.65)": f"{round(gp_bottom, 2)} - {round(gp_top, 2)}",
                    "OBV 狀態": "🟢 流入中",
                    "長影線拒絕": "✅ 觸發" if is_rejection_candle(current) or is_rejection_candle(prev) else "無"
                })
        except Exception:
            pass
        
        progress_bar.progress((i + 1) / total)
        time.sleep(0.15)
        
    status_text.text("掃描完成！")
    return pd.DataFrame(results)

st.set_page_config(page_title="Dolphin 波段掃描器", layout="wide")
st.title("🐬 Dolphin 波段資金雷達 (台股版)")
st.markdown("結合 **OBV 資金流向** 與 **0.618 金色口袋** 的高勝率波段掃描工具。")

st.sidebar.header("⚙️ 1. 輸入或過濾股票代碼")

# --- 進階 CSV 功能區 ---
uploaded_file = st.sidebar.file_uploader("📂 上傳自選股清單 (CSV檔)", type=["csv"])
ticker_list = []

if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    found_tickers = list(set(re.findall(r'\b\d{4}\b', content)))
    st.sidebar.success(f"成功抓取 {len(found_tickers)} 檔股票代碼！")
    
    # 檢查並剔除下市股票按鈕
    if st.sidebar.button("🧹 幫我檢查並剔除下市股票"):
        with st.spinner("正在連線證交所資料庫比對狀態，請稍候..."):
            valid_tickers = []
            for t in found_tickers:
                try:
                    # 測試抓取近期資料，抓不到代表已下市或有誤
                    if not yf.Ticker(f"{t}.TW").history(period="5d").empty:
                        valid_tickers.append(t)
                    time.sleep(0.1)
                except:
                    pass
            
            # 提供下載乾淨清單的功能
            clean_df = pd.DataFrame({"股票代碼": valid_tickers})
            csv_data = clean_df.to_csv(index=False).encode('utf-8-sig') # 加上 sig 讓 Excel 打開不會亂碼
            
            st.sidebar.success(f"檢查完畢！發現了 {len(found_tickers) - len(valid_tickers)} 檔已下市或無效股票。")
            st.sidebar.download_button(
                label="📥 下載更新後的乾淨 CSV",
                data=csv_data,
                file_name="Updated_Dolphin_List.csv",
                mime="text/csv",
                type="primary"
            )
    ticker_list = found_tickers

else:
    default_tickers = "2330, 2317, 2454, 2308, 2382, 3231, 2603, 1513, 1519, 2376, 2357, 6235"
    ticker_input = st.sidebar.text_area("✍️ 手動輸入 (用逗號分隔)", value=default_tickers)
    ticker_list = [t.strip() for t in ticker_input.split(",") if t.strip()]

st.sidebar.header("⚙️ 2. 掃描過濾設定")
use_rejection = st.sidebar.toggle("啟用「長下影線拒絕」過濾", value=True)
st.sidebar.markdown("---")

if st.button("🚀 開始掃描", type="primary"):
    if not ticker_list:
        st.error("找不到任何股票代碼，請確認檔案內容或手動輸入！")
    else:
        st.info(f"即將掃描 {len(ticker_list)} 檔股票，請稍等...")
        with st.spinner("系統正在運算指標與比對區間..."):
            df_result = fetch_and_scan(ticker_list, use_rejection)
            if not df_result.empty:
                st.success(f"掃描完畢！共發現 {len(df_result)} 檔符合條件的標的。")
                st.dataframe(df_result, use_container_width=True)
            else:
                st.info("目前市場中沒有符合所有條件（進入金色口袋 + 資金流入）的標的。")