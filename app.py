import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import re
import json
import os

# ============================================================
#  Dolphin V4 — OBV 穿越 + 價格帶分級 + 大盤保護 (台股版)
#  級別依據：60-250元+2根前穿越=首選（457筆 WR64% PF3.7）
#  大盤保護：櫃買/加權 > 5MA
# ============================================================


@st.cache_data(ttl=86400)
def load_universe():
    for path in ["universe.json", os.path.join(os.path.dirname(__file__), "universe.json")]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


@st.cache_data(ttl=86400)
def load_stock_names():
    for path in ["stock_names.json", os.path.join(os.path.dirname(__file__), "stock_names.json")]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


# ============================================================
#  OBV 核心函數
# ============================================================

def calculate_obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()


def detect_obv_crossover(obv, obv_ma, obv_z=None, lookback=3):
    if len(obv) < lookback + 1 or obv_ma.isna().iloc[-lookback:].any():
        return {"crossed_up": False, "bars_since_cross": 999, "strength": 0, "z_score": 0}
    for i in range(1, lookback + 1):
        if obv.iloc[-i - 1] < obv_ma.iloc[-i - 1] and obv.iloc[-i] >= obv_ma.iloc[-i]:
            ma_val = obv_ma.iloc[-1]
            strength = ((obv.iloc[-1] - ma_val) / abs(ma_val) * 100) if ma_val != 0 else 0
            z = round(obv_z.iloc[-1], 2) if obv_z is not None and len(obv_z) > 0 else 0
            return {"crossed_up": True, "bars_since_cross": i - 1, "strength": round(strength, 2), "z_score": z}
    z = round(obv_z.iloc[-1], 2) if obv_z is not None and len(obv_z) > 0 else 0
    return {"crossed_up": False, "bars_since_cross": 999, "strength": 0, "z_score": z}


def detect_obv_divergence(df, obv, window=20):
    base = {"bullish_div": False, "div_strength": 0, "div_low_price": 0}
    if len(df) < window + 5:
        return base
    price_slice = df['Close'].iloc[-window:]
    price_min_idx = price_slice.idxmin()
    price_min_pos = price_slice.index.get_loc(price_min_idx)
    if price_min_pos < window // 3:
        return base
    first_half = price_slice.iloc[:price_min_pos]
    if len(first_half) < 3:
        return base
    prev_low_idx = first_half.idxmin()
    curr_low_price = price_slice[price_min_idx]
    if curr_low_price > first_half[prev_low_idx] * 1.01:
        return base
    obv_at_prev = obv[prev_low_idx]
    obv_at_curr = obv[price_min_idx]
    if obv_at_curr > obv_at_prev:
        ds = ((obv_at_curr - obv_at_prev) / abs(obv_at_prev) * 100) if obv_at_prev != 0 else 0
        ds = round(abs(ds), 2)
        if ds < 2.0:
            return {"bullish_div": False, "div_strength": ds, "div_low_price": float(curr_low_price)}
        return {"bullish_div": True, "div_strength": ds, "div_low_price": float(curr_low_price)}
    return base


def detect_obv_slope(obv, short_period=5, long_period=20):
    if len(obv) < long_period + 1:
        return {"accelerating": False}
    oc = obv.replace([np.inf, -np.inf], np.nan).dropna()
    if len(oc) < long_period + 1:
        return {"accelerating": False}
    ss = (oc.iloc[-1] - oc.iloc[-short_period]) / short_period
    ls = (oc.iloc[-1] - oc.iloc[-long_period]) / long_period
    return {"accelerating": (ss > 0) and (ss > ls * 1.5)}


def is_rejection_candle(row, direction="bullish"):
    body = abs(row['Close'] - row['Open'])
    if body == 0:
        body = 0.001
    if direction == "bullish":
        return (min(row['Open'], row['Close']) - row['Low']) > (body * 2)
    return (row['High'] - max(row['Open'], row['Close'])) > (body * 2)


def validate_golden_pocket(df, min_struct_pct=8.0):
    if len(df) < 60:
        return {"in_gp": False}
    tail = df.tail(60)
    rh, rl = tail['High'].max(), tail['Low'].min()
    sp = (rh - rl) / rl * 100
    if sp < min_struct_pct:
        return {"in_gp": False}
    if tail.index.get_loc(tail['High'].idxmax()) >= tail.index.get_loc(tail['Low'].idxmin()):
        return {"in_gp": False}
    d = rh - rl
    gt, gb = rh - 0.618 * d, rh - 0.65 * d
    ig = gb <= df['Close'].iloc[-1] <= gt
    gc = (gt + gb) / 2
    dv = abs(df['Close'].iloc[-1] - gc) / (gt - gb) * 100 if gt != gb else 999
    return {"in_gp": ig, "gp_range": f"{gb:.2f}-{gt:.2f}", "deviation": round(dv, 2), "struct_pct": round(sp, 2)}


def compute_score(cross_info, div_info, slope_info, gp_info, has_rejection, vol_ratio):
    s = 0

    # === 穿越是唯一主訊號 (0-25) ===
    # 156筆數據: 2根前 WR66.7%/+2.36% > 1根前 57.1%/+2.11% > 0根前 59.0%/+0.99%
    # 2根前 = 穿越已確認+回踩，最穩；0根前 = 追高，最差
    if cross_info["crossed_up"]:
        s += 15
        if cross_info["bars_since_cross"] == 2: s += 10   # 最佳：確認+回踩
        elif cross_info["bars_since_cross"] == 1: s += 7   # 次佳
        # 0根前不加分

    # === 背離降為加分項 (0-10) ===
    # 156筆: 背離30筆 WR43.3% 平均-0.23%，穿越126筆 WR61.1% 平均+2.01%
    # 背離單獨無效，搭配穿越才加分
    if div_info["bullish_div"]:
        ds = div_info["div_strength"]
        if cross_info["crossed_up"]:
            s += 10 if ds <= 50 else 3   # 穿越+背離=確認，加分
        else:
            s += 3   # 純背離，象徵性給分

    # === 斜率加分 (0-15) ===
    if slope_info.get("accelerating"):
        s += 15 if cross_info["crossed_up"] else 5

    # === Z-score (0-10) ===
    z = abs(cross_info.get("z_score", 0))
    if z >= 2.5: s += 10
    elif z >= 2.0: s += 7
    elif z >= 1.5: s += 4

    # === GP / 拒絕 / 量比 ===
    if gp_info.get("in_gp"):
        s += 10
        if gp_info.get("deviation", 999) < 30: s += 5
    if has_rejection: s += 5
    if vol_ratio >= 3.0: s += 10
    elif vol_ratio >= 2.0: s += 7
    elif vol_ratio >= 1.5: s += 4

    return min(s, 100)


def prepare_obv_data(df):
    """準備 OBV 相關欄位"""
    df['OBV'] = calculate_obv(df)
    df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
    od = df['OBV'] - df['OBV_MA20']
    df['OBV_Z'] = od / od.rolling(window=28).std()
    df['OBV_Z'] = df['OBV_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ============================================================
#  即時掃描
# ============================================================

def scan_single_stock(symbol, use_gp, use_rejection, min_struct_pct, max_runup, name_map=None):
    try:
        df = yf.Ticker(symbol).history(period="6mo")
        if len(df) < 60:
            return None
        # 清理 NaN 收盤價（yfinance 偶爾最後一根不完整）
        df = df.dropna(subset=['Close'])
        if len(df) < 60:
            return None
        df = prepare_obv_data(df)

        cur, prv = df.iloc[-1], df.iloc[-2]
        avg_vol = df['Volume'].tail(5).mean()
        lots = avg_vol / 1000
        if lots < 2000:
            return None

        vol_20d = df['Volume'].tail(20).mean()
        vr = (avg_vol / vol_20d) if vol_20d > 0 else 0

        cross = detect_obv_crossover(df['OBV'], df['OBV_MA20'], df['OBV_Z'], lookback=3)
        div = detect_obv_divergence(df, df['OBV'], window=20)
        slope = detect_obv_slope(df['OBV'])

        runup = 0.0
        if div["bullish_div"] and div["div_low_price"] > 0:
            try:
                runup = round((cur['Close'] - div["div_low_price"]) / div["div_low_price"] * 100, 1)
                if np.isnan(runup):
                    runup = 0.0
            except Exception:
                runup = 0.0
            if runup > max_runup:
                return None

        above_ma = cur['OBV'] > cur['OBV_MA20']
        # 穿越是唯一主訊號，純背離無法單獨通過
        if not cross["crossed_up"]:
            if not (slope.get("accelerating") and above_ma and vr >= 2.0):
                return None

        gp = {"in_gp": False}
        if use_gp:
            gp = validate_golden_pocket(df, min_struct_pct)
        rej = False
        if use_rejection:
            rej = is_rejection_candle(cur) or is_rejection_candle(prv)

        sc = compute_score(cross, div, slope, gp, rej, vr)
        if sc < 15:
            return None

        sigs = []
        bars_since = cross['bars_since_cross'] if cross["crossed_up"] else 99
        if cross["crossed_up"]:
            fresh_label = "⚡" if bars_since == 2 else ""
            sigs.append(f"穿越↑({bars_since}根前){fresh_label}")
        if div["bullish_div"]:
            sigs.append(f"底背離({div['div_strength']:.1f}%)")
        if slope.get("accelerating"): sigs.append("斜率加速")
        if above_ma and not cross["crossed_up"]: sigs.append("MA上方")

        cs = symbol.replace(".TWO", "").replace(".TW", "")
        nm = (name_map or {}).get(cs, "")
        price = round(cur['Close'], 2)

        # === 級別判定（依據 457 筆實戰數據） ===
        # 60-250元 + 2根前: WR64% PF3.66 → 首選
        # 60-250元 + 0/1根前, 或 非甜蜜區+2根前 → 可選
        # <30元 或 無穿越 → 觀望
        in_sweet = 60 <= price <= 250
        is_best_cross = bars_since == 2

        if in_sweet and is_best_cross:
            tier = "🔴 首選"
        elif in_sweet or is_best_cross:
            tier = "🟡 可選"
        elif price < 30:
            tier = "⚪ 觀望"
        else:
            tier = "🟡 可選"

        return {
            "股票": f"{cs} {nm}" if nm else cs,
            "級別": tier,
            "評分": sc,
            "收盤": price,
            "OBV訊號": " | ".join(sigs),
            "Z": round(cross.get("z_score", 0), 1),
            "離背離低點": f"+{runup}%" if div["bullish_div"] else "-",
            "均量(張)": f"{lots:,.0f}",
            "量比": f"{vr:.1f}x",
        }
    except Exception:
        return None


# ============================================================
#  績效追蹤：歷史訊號回測
# ============================================================


# ============================================================
#  Streamlit UI
# ============================================================
st.set_page_config(page_title="Dolphin V4", layout="wide")
st.title("🐬 Dolphin V4 — OBV 波段掃描 + 績效追蹤")

universe = load_universe()
name_map = load_stock_names()
display_names = name_map if name_map else {k: v.get("name", "") for k, v in universe.items()}

tab_scan, tab_perf = st.tabs(["🔍 即時掃描", "📊 績效追蹤"])

# ====================
#  Tab 1: 即時掃描
# ====================
with tab_scan:
    st.markdown("以 **OBV 穿越**為主訊號 + **價格帶×穿越新鮮度**分級 + **大盤保護**。457筆數據驅動。")

    st.sidebar.header("⚙️ 1. 股票池")
    pool = st.sidebar.radio("來源", ["📦 內建清單", "📂 上傳", "✍️ 手動"], index=0)

    tickers = []
    if pool.startswith("📦"):
        if universe:
            tickers = list(universe.keys())
            st.sidebar.success(f"內建 {len(tickers)} 檔（已排除傳產）")
        else:
            st.sidebar.error("universe.json 未載入")
    elif pool.startswith("📂"):
        uf = st.sidebar.file_uploader("CSV/XLSX", type=["csv", "xlsx", "xls"])
        if uf:
            try:
                if uf.name.endswith('.csv'):
                    try: ct = uf.getvalue().decode("utf-8")
                    except: ct = uf.getvalue().decode("cp950", errors="ignore")
                else:
                    ct = pd.read_excel(uf).to_string()
                tickers = list(set(re.findall(r'\b\d{4}\b', ct)))
                if tickers: st.sidebar.success(f"{len(tickers)} 檔")
            except Exception as e:
                st.sidebar.error(str(e))
    else:
        inp = st.sidebar.text_area("逗號分隔", "2330, 2317, 2454, 3231, 1513, 1519")
        tickers = [t.strip() for t in inp.split(",") if t.strip()]

    st.sidebar.header("⚙️ 2. 過濾")
    use_market_filter = st.sidebar.toggle("櫃買指數 > 5MA（大盤保護）", value=True,
                                           help="櫃買指數收在5日均線以下時不出訊號，避免中小股全面被倒貨的日子")
    use_gp = st.sidebar.toggle("Golden Pocket", value=False)
    use_rej = st.sidebar.toggle("下影線拒絕", value=False)
    msp = st.sidebar.slider("GP結構%", 5.0, 15.0, 8.0, 0.5) if use_gp else 8.0
    min_sc = st.sidebar.slider("最低評分", 15, 60, 30, 5)
    max_ru = st.sidebar.slider("背離後最大漲幅%", 5, 50, 15, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""**V4 級別（457筆驗證）**
- 🔴 首選：60-250元 + 2根前⚡
- 🟡 可選：甜蜜區或最佳穿越（擇一）
- ⚪ 觀望：低價股或弱組合
    """)

    if st.button("🚀 開始掃描", type="primary"):
        if not tickers:
            st.error("沒有股票")
        else:
            # === 大盤過濾：櫃買指數 > 5MA ===
            market_ok = True
            tpex_status = ""
            if use_market_filter:
                tpex_close = None
                tpex_ma5 = None
                tpex_label = ""

                # 嘗試多個 ticker：櫃買指數 → 櫃買50 ETF → 加權指數(fallback)
                for ticker, label in [("^TWOTCI", "櫃買指數"), ("006201.TW", "櫃買50ETF"), ("^TWII", "加權指數")]:
                    try:
                        tdf = yf.Ticker(ticker).history(period="1mo")
                        if len(tdf) >= 5:
                            tdf.index = tdf.index.tz_localize(None)
                            tpex_close = tdf['Close'].iloc[-1]
                            tpex_ma5 = tdf['Close'].tail(5).mean()
                            tpex_label = label
                            break
                    except Exception:
                        continue

                if tpex_close is not None and tpex_ma5 is not None:
                    tpex_diff = (tpex_close - tpex_ma5) / tpex_ma5 * 100
                    if tpex_close > tpex_ma5:
                        tpex_status = f"🟢 {tpex_label} {tpex_close:.2f} > 5MA {tpex_ma5:.2f} ({tpex_diff:+.2f}%)"
                        st.success(tpex_status)
                    else:
                        tpex_status = f"🔴 {tpex_label} {tpex_close:.2f} < 5MA {tpex_ma5:.2f} ({tpex_diff:+.2f}%)"
                        st.error(f"{tpex_status} — 中小股環境偏空，今日訊號暫停")
                        market_ok = False
                else:
                    st.warning("大盤指數資料取得失敗，跳過大盤過濾")

            if not market_ok:
                st.info("大盤保護啟動，今日不出訊號。如果要強制掃描，關閉 sidebar 的「櫃買指數 > 5MA」開關。")
            else:
                st.info(f"掃描 {len(tickers)} 檔...")
                results, prog, stat = [], st.progress(0), st.empty()
                for i, t in enumerate(tickers):
                    t = t.strip()
                    sym = t + (universe.get(t, {}).get("suffix", ".TW") if not t.endswith((".TW", ".TWO")) else "")
                    if t.endswith((".TW", ".TWO")): sym = t
                    c = t.replace(".TWO", "").replace(".TW", "")
                    stat.text(f"掃描: {c} {display_names.get(c, '')} ({i+1}/{len(tickers)})")
                    r = scan_single_stock(sym, use_gp, use_rej, msp, max_ru, display_names)
                    if r and r["評分"] >= min_sc: results.append(r)
                    prog.progress((i + 1) / len(tickers))
                    time.sleep(0.12)
                stat.text("完成！")

                if results:
                    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
                    dfr = pd.DataFrame(results)
                    dfr.insert(0, "掃描日", today_str)
                    if tpex_status:
                        dfr.insert(1, "大盤狀態", tpex_status)

                    # 按級別排序：首選 > 可選 > 觀望，同級別按評分
                    tier_order = {"🔴 首選": 0, "🟡 可選": 1, "⚪ 觀望": 2}
                    dfr["_sort"] = dfr["級別"].map(tier_order).fillna(3)
                    dfr = dfr.sort_values(["_sort", "評分"], ascending=[True, False]).drop(columns="_sort").reset_index(drop=True)
                    dfr.index += 1

                    # 統計
                    n_top = (dfr["級別"] == "🔴 首選").sum()
                    n_mid = (dfr["級別"] == "🟡 可選").sum()
                    n_low = (dfr["級別"] == "⚪ 觀望").sum()
                    st.success(f"共 {len(dfr)} 檔：🔴首選 {n_top} / 🟡可選 {n_mid} / ⚪觀望 {n_low}")

                    # 分層顯示
                    for tier_name, tier_desc in [
                        ("🔴 首選", "60-250元 + 2根前穿越｜WR 64% PF 3.7（457筆驗證）"),
                        ("🟡 可選", "甜蜜區非最佳穿越，或最佳穿越非甜蜜區"),
                        ("⚪ 觀望", "低價股或弱訊號組合"),
                    ]:
                        sub = dfr[dfr["級別"] == tier_name]
                        if not sub.empty:
                            st.subheader(f"{tier_name} — {len(sub)} 檔")
                            st.caption(tier_desc)
                            st.dataframe(sub, use_container_width=True)

                    fname = f"Dolphin_V4_{today_str}.csv"
                    st.download_button("📥 下載（記得存檔，績效追蹤要用）", dfr.to_csv(index=False).encode('utf-8-sig'), fname, "text/csv")
                else:
                    st.info("沒有符合條件的標的")


# ====================
#  Tab 2: 績效追蹤
# ====================
with tab_perf:
    st.markdown("""
    ### 訊號績效追蹤
    上傳之前的掃描結果 CSV → 自動抓**隔日開盤價**當進場價 → 算到今天的報酬。
    可以同時上傳多份不同日期的 CSV，一次看所有歷史訊號的表現。
    """)

    perf_files = st.file_uploader(
        "上傳 Dolphin 掃描結果 CSV（可多選）",
        type=["csv"],
        accept_multiple_files=True,
    )

    if perf_files and st.button("📊 查看績效", type="primary"):
        # 1. 讀取所有上傳的 CSV，提取 掃描日 + 股票代碼
        all_records = []
        for f in perf_files:
            try:
                df_up = pd.read_csv(f, encoding='utf-8-sig')
                if "股票" not in df_up.columns:
                    st.warning(f"{f.name} 缺少「股票」欄位，跳過")
                    continue

                # 取得掃描日
                if "掃描日" in df_up.columns:
                    scan_date = str(df_up["掃描日"].iloc[0]).strip()
                else:
                    # 嘗試從檔名取得日期 (Dolphin_V3_2026-04-25.csv)
                    import re as _re
                    m = _re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
                    if m:
                        scan_date = m.group(1)
                    else:
                        st.warning(f"{f.name} 沒有掃描日資訊，跳過")
                        continue

                for _, row in df_up.iterrows():
                    code = str(row["股票"]).split()[0].strip()
                    if not code.isdigit():
                        continue
                    all_records.append({
                        "code": code,
                        "scan_date": scan_date,
                        "scan_score": row.get("評分", ""),
                        "scan_signal": row.get("OBV訊號", row.get("OBV 訊號", "")),
                        "scan_close": row.get("收盤", row.get("最新收盤", "")),
                        "display": str(row["股票"]),
                    })
            except Exception as e:
                st.warning(f"{f.name} 讀取失敗: {e}")

        if not all_records:
            st.error("沒有有效的掃描紀錄")
        else:
            st.info(f"共 {len(all_records)} 筆訊號，來自 {len(perf_files)} 份掃描...")

            results = []
            prog = st.progress(0)
            stat = st.empty()

            for i, rec in enumerate(all_records):
                code = rec["code"]
                scan_date = rec["scan_date"]
                stat.text(f"追蹤: {rec['display']} (掃描日 {scan_date}) ({i+1}/{len(all_records)})")

                try:
                    suffix = universe.get(code, {}).get("suffix", ".TW")
                    symbol = code + suffix

                    # 從掃描日前 3 天開始抓，確保拿到隔日開盤
                    start = (pd.Timestamp(scan_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
                    df_price = yf.Ticker(symbol).history(start=start)
                    if df_price.empty:
                        continue

                    # yfinance 回傳帶時區的 index，統一轉成 tz-naive 避免比較失敗
                    df_price.index = df_price.index.tz_localize(None)

                    # 找掃描日之後的第一個交易日 = 進場日
                    scan_ts = pd.Timestamp(scan_date)
                    future = df_price[df_price.index > scan_ts]
                    if len(future) < 1:
                        continue

                    entry_date = future.index[0]
                    entry_price = future['Open'].iloc[0]
                    if entry_price <= 0:
                        continue

                    # 最新收盤
                    latest_close = df_price['Close'].iloc[-1]
                    latest_date = df_price.index[-1]
                    hold_days = (latest_date - entry_date).days
                    ret_pct = (latest_close - entry_price) / entry_price * 100

                    # 期間最高/最低（計算 MDD）
                    hold_period = df_price.loc[entry_date:]
                    max_price = hold_period['High'].max()
                    min_price = hold_period['Low'].min()
                    mfe = (max_price - entry_price) / entry_price * 100
                    mae = (min_price - entry_price) / entry_price * 100

                    results.append({
                        "股票": rec["display"],
                        "掃描日": scan_date,
                        "評分": rec["scan_score"],
                        "訊號": rec["scan_signal"],
                        "進場日": entry_date.strftime("%Y-%m-%d"),
                        "進場價": round(entry_price, 2),
                        "現價": round(latest_close, 2),
                        "持有天數": hold_days,
                        "報酬%": round(ret_pct, 2),
                        "最大獲利%": round(mfe, 1),
                        "最大回撤%": round(mae, 1),
                        "結果": "✅ 獲利" if ret_pct > 0 else "❌ 虧損",
                    })
                except Exception:
                    pass

                prog.progress((i + 1) / len(all_records))
                time.sleep(0.1)

            stat.text("追蹤完成！")

            if results:
                dfp = pd.DataFrame(results)

                # === 整體統計 ===
                st.subheader("📈 整體績效")
                rets = dfp["報酬%"]
                wins = (rets > 0).sum()
                losses = (rets <= 0).sum()
                total = len(rets)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("訊號數", total)
                c2.metric("勝率", f"{wins / total * 100:.1f}%")
                c3.metric("平均報酬", f"{rets.mean():.2f}%")
                c4.metric("最大獲利", f"+{rets.max():.1f}%")
                c5.metric("最大虧損", f"{rets.min():.1f}%")

                # 獲利因子
                gross_win = rets[rets > 0].sum()
                gross_loss = abs(rets[rets < 0].sum())
                pf = f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "∞"
                st.caption(f"中位數: {rets.median():.2f}% ｜ 獲利因子: {pf} ｜ 勝{wins} 負{losses}")

                # === 按訊號類型 ===
                if "訊號" in dfp.columns:
                    st.subheader("📊 按訊號類型")

                    # 判斷包含哪些訊號
                    dfp["有背離"] = dfp["訊號"].str.contains("背離", na=False)
                    dfp["有穿越"] = dfp["訊號"].str.contains("穿越", na=False)

                    type_groups = {
                        "有背離": dfp[dfp["有背離"]],
                        "有穿越(無背離)": dfp[dfp["有穿越"] & ~dfp["有背離"]],
                        "其他": dfp[~dfp["有穿越"] & ~dfp["有背離"]],
                    }

                    for label, sub in type_groups.items():
                        if len(sub) < 1:
                            continue
                        v = sub["報酬%"]
                        w = (v > 0).sum()
                        st.markdown(f"**{label}**（{len(sub)} 筆）— 勝率 {w/len(sub)*100:.1f}%，平均 {v.mean():.2f}%，中位數 {v.median():.2f}%")

                    dfp.drop(columns=["有背離", "有穿越"], inplace=True, errors="ignore")

                # === 按掃描日 ===
                if dfp["掃描日"].nunique() > 1:
                    st.subheader("📅 按掃描日")
                    for sd in sorted(dfp["掃描日"].unique()):
                        sub = dfp[dfp["掃描日"] == sd]
                        v = sub["報酬%"]
                        w = (v > 0).sum()
                        st.markdown(f"**{sd}**（{len(sub)} 筆）— 勝率 {w/len(sub)*100:.1f}%，平均 {v.mean():.2f}%")

                # === 個股明細 ===
                st.subheader("📋 個股明細")
                st.dataframe(dfp.sort_values("報酬%", ascending=False).reset_index(drop=True), use_container_width=True)

                st.download_button("📥 下載績效", dfp.to_csv(index=False).encode('utf-8-sig'), "Dolphin_V3_Performance.csv", "text/csv")
            else:
                st.info("無法取得任何股票的價格資料")
