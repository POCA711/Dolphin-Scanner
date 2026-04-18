import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

# --- 核心邏輯 (與先前相同) ---
def calculate_obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def is_rejection_candle(df_row):
    body = abs(df_row['Close'] - df_row['Open'])
    lower_wick = min(df_row['Open'], df_row['Close']) - df_row['Low']
    if body == 0: return lower_wick > 0 
    return lower_wick > (body * 2)

# --- 介面與執行緒 ---
def run_scan():
    btn_scan.config(state=tk.DISABLED)
    txt_result.delete(1.0, tk.END)
    txt_result.insert(tk.END, "🚀 掃描啟動中，請稍候...\n\n")
    
    tickers_input = entry_tickers.get()
    use_rejection = var_rejection.get()
    
    ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not ticker_list:
        messagebox.showwarning("警告", "請輸入至少一檔股票代碼！")
        btn_scan.config(state=tk.NORMAL)
        return

    # 使用獨立執行緒避免視窗卡死
    threading.Thread(target=scan_process, args=(ticker_list, use_rejection), daemon=True).start()

def scan_process(tickers, use_rejection):
    results = []
    for i, symbol in enumerate(tickers):
        if not symbol.endswith(".TW"): symbol = f"{symbol}.TW"
        
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="6mo")
            if len(df) < 60: continue

            df['OBV'] = calculate_obv(df)
            df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
            
            recent_high = df['High'].tail(60).max()
            recent_low = df['Low'].tail(60).min()
            diff = recent_high - recent_low
            gp_top, gp_bottom = recent_high - (0.618 * diff), recent_high - (0.65 * diff)
            
            current, prev = df.iloc[-1], df.iloc[-2]
            
            in_gp = (gp_bottom * 0.985) <= current['Close'] <= (gp_top * 1.015)
            obv_ok = current['OBV'] > current['OBV_MA20']
            
            rejection_ok = True
            if use_rejection:
                rejection_ok = is_rejection_candle(current) or is_rejection_candle(prev)
            volume_ok = df['Volume'].tail(5).mean() > 1000

            if in_gp and obv_ok and rejection_ok and volume_ok:
                results.append(f"✅ {symbol.replace('.TW','')} | 收盤: {current['Close']:.2f} | 區間: {gp_bottom:.2f}-{gp_top:.2f}")
        except Exception:
            pass
            
    # 更新 UI
    txt_result.delete(1.0, tk.END)
    if results:
        txt_result.insert(tk.END, "🎯 發現符合條件的波段標的：\n" + "-"*40 + "\n")
        for res in results:
            txt_result.insert(tk.END, res + "\n")
    else:
        txt_result.insert(tk.END, "目前市場中沒有符合所有條件的標的。")
        
    btn_scan.config(state=tk.NORMAL)

# --- 建立 Windows 介面 ---
root = tk.Tk()
root.title("Dolphin 波段資金雷達")
root.geometry("500x450")

tk.Label(root, text="🐬 Dolphin 掃描器 (台股版)", font=("Arial", 16, "bold")).pack(pady=10)

tk.Label(root, text="輸入股票代碼 (用逗號分隔):").pack(anchor="w", padx=20)
entry_tickers = tk.Entry(root, width=60)
entry_tickers.insert(0, "2330, 2317, 2454, 2308, 2382, 3231, 2603")
entry_tickers.pack(padx=20, pady=5)

var_rejection = tk.BooleanVar(value=True)
chk_rejection = tk.Checkbutton(root, text="啟用「長下影線拒絕」過濾", variable=var_rejection)
chk_rejection.pack(anchor="w", padx=20, pady=5)

btn_scan = tk.Button(root, text="🚀 開始掃描", bg="lightblue", font=("Arial", 12, "bold"), command=run_scan)
btn_scan.pack(pady=10)

txt_result = scrolledtext.ScrolledText(root, width=55, height=12, font=("Consolas", 10))
txt_result.pack(padx=20, pady=10)

root.mainloop()