import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import re

# ============================================================
#  Dolphin V2 — OBV 核心波段掃描器 (台股版)
#  核心邏輯：OBV 穿越 + OBV 底背離 + OBV 斜率加速
#  輔助邏輯：Golden Pocket 結構驗證 + 下影線拒絕
# ============================================================

# --- 傳產代碼範圍（用於過濾） ---
TRADITIONAL_SECTORS = {
    "水泥": (1101, 1199),
    "食品": (1201, 1299),
    "塑膠": (1301, 1399),
    "紡織": (1401, 1499),
    "電纜": (1601, 1699),
    "化學": (1701, 1799),
    "玻璃陶瓷": (1801, 1899),
    "造紙": (1901, 1999),
    "鋼鐵": (2001, 2099),
    "橡膠": (2101, 2199),
    "汽車": (2201, 2299),
    "航運": (2601, 2699),
    "觀光": (2701, 2799),
    "金融保險": (2801, 2899),
    "百貨貿易": (2901, 2999),
    "其他": (9900, 9999),
}


def is_traditional(code_str: str) -> str | None:
    """回傳產業名稱，如果是傳產的話；否則回傳 None"""
    try:
        code = int(code_str.replace(".TW", "").replace(".TWO", ""))
    except ValueError:
        return None
    for sector, (lo, hi) in TRADITIONAL_SECTORS.items():
        if lo <= code <= hi:
            return sector
    return None


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """標準 OBV 計算"""
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()


def detect_obv_crossover(obv: pd.Series, obv_ma: pd.Series, lookback: int = 3) -> dict:
    """
    偵測 OBV 穿越 MA 的狀態
    回傳:
      - crossed_up: 最近 lookback 根內 OBV 從下往上穿越 MA
      - bars_since_cross: 穿越後經過幾根 K 線
      - strength: 穿越後 OBV 超出 MA 的幅度 (%)
    """
    if len(obv) < lookback + 1 or obv_ma.isna().iloc[-lookback:].any():
        return {"crossed_up": False, "bars_since_cross": 999, "strength": 0}

    for i in range(1, lookback + 1):
        idx_now = -i
        idx_prev = -i - 1
        if obv.iloc[idx_prev] < obv_ma.iloc[idx_prev] and obv.iloc[idx_now] >= obv_ma.iloc[idx_now]:
            bars_since = i - 1
            ma_val = obv_ma.iloc[-1]
            strength = ((obv.iloc[-1] - ma_val) / abs(ma_val) * 100) if ma_val != 0 else 0
            return {"crossed_up": True, "bars_since_cross": bars_since, "strength": round(strength, 2)}

    return {"crossed_up": False, "bars_since_cross": 999, "strength": 0}


def detect_obv_divergence(df: pd.DataFrame, obv: pd.Series, window: int = 20) -> dict:
    """
    偵測 OBV 底背離：價格創近期新低，但 OBV 沒有創新低
    這是最強的進場訊號之一 — 主力在低點吸貨
    """
    if len(df) < window + 5:
        return {"bullish_div": False, "div_strength": 0}

    price_slice = df['Close'].iloc[-window:]
    obv_slice = obv.iloc[-window:]

    # 找價格的兩個低點
    price_min_idx = price_slice.idxmin()
    price_min_pos = price_slice.index.get_loc(price_min_idx)

    # 價格最低點必須在後半段（近期才創低）
    if price_min_pos < window // 3:
        return {"bullish_div": False, "div_strength": 0}

    # 前半段的最低價
    first_half = price_slice.iloc[:price_min_pos]
    if len(first_half) < 3:
        return {"bullish_div": False, "div_strength": 0}

    prev_low_idx = first_half.idxmin()
    prev_low_price = first_half[prev_low_idx]
    curr_low_price = price_slice[price_min_idx]

    # 價格必須創更低（或接近等低）
    if curr_low_price > prev_low_price * 1.01:
        return {"bullish_div": False, "div_strength": 0}

    # OBV 在對應位置：後面的低點 OBV > 前面的低點 OBV = 背離
    obv_at_prev = obv[prev_low_idx]
    obv_at_curr = obv[price_min_idx]

    if obv_at_curr > obv_at_prev:
        div_strength = ((obv_at_curr - obv_at_prev) / abs(obv_at_prev) * 100) if obv_at_prev != 0 else 0
        div_strength = round(abs(div_strength), 2)
        # 背離強度 < 2% = OBV 幾乎沒差，不算真正背離
        if div_strength < 2.0:
            return {"bullish_div": False, "div_strength": div_strength}
        return {"bullish_div": True, "div_strength": div_strength}

    return {"bullish_div": False, "div_strength": 0}


def detect_obv_slope(obv: pd.Series, short_period: int = 5, long_period: int = 20) -> dict:
    """
    OBV 斜率加速偵測
    短期 OBV 變化率 vs 長期 OBV 變化率
    """
    if len(obv) < long_period + 1:
        return {"accelerating": False, "short_slope": 0, "long_slope": 0}

    obv_clean = obv.replace([np.inf, -np.inf], np.nan).dropna()
    if len(obv_clean) < long_period + 1:
        return {"accelerating": False, "short_slope": 0, "long_slope": 0}

    short_slope = (obv_clean.iloc[-1] - obv_clean.iloc[-short_period]) / short_period
    long_slope = (obv_clean.iloc[-1] - obv_clean.iloc[-long_period]) / long_period

    # 短期斜率 > 長期斜率 且都為正 = 資金加速流入
    accelerating = (short_slope > 0) and (short_slope > long_slope * 1.5)

    return {
        "accelerating": accelerating,
        "short_slope": round(short_slope, 2),
        "long_slope": round(long_slope, 2),
    }


def validate_golden_pocket(df: pd.DataFrame, min_struct_pct: float = 8.0) -> dict:
    """
    Golden Pocket 結構驗證
    改善：加入結構幅度門檻、確認回撤方向
    """
    if len(df) < 60:
        return {"in_gp": False}

    tail = df.tail(60)
    recent_high = tail['High'].max()
    recent_low = tail['Low'].min()
    high_idx = tail['High'].idxmax()
    low_idx = tail['Low'].idxmin()
    current_price = df['Close'].iloc[-1]

    # 結構幅度檢查：波段幅度必須 >= min_struct_pct%
    struct_pct = (recent_high - recent_low) / recent_low * 100
    if struct_pct < min_struct_pct:
        return {"in_gp": False, "reason": "結構幅度不足"}

    # 確認是回撤結構：高點在低點之前 (從高往低拉回)
    high_pos = tail.index.get_loc(high_idx)
    low_pos = tail.index.get_loc(low_idx)

    if high_pos >= low_pos:
        # 高點在低點之後 = 上升趨勢，不是回撤，GP 無意義
        return {"in_gp": False, "reason": "非回撤結構"}

    diff = recent_high - recent_low
    gp_top = recent_high - (0.618 * diff)
    gp_bottom = recent_high - (0.65 * diff)

    in_gp = gp_bottom <= current_price <= gp_top

    # 計算距離 GP 中心的偏離度
    gp_center = (gp_top + gp_bottom) / 2
    deviation = abs(current_price - gp_center) / (gp_top - gp_bottom) * 100 if gp_top != gp_bottom else 999

    return {
        "in_gp": in_gp,
        "gp_range": f"{round(gp_bottom, 2)} - {round(gp_top, 2)}",
        "deviation": round(deviation, 2),
        "struct_pct": round(struct_pct, 2),
    }


def is_rejection_candle(row, direction="bullish") -> bool:
    """
    改進版拒絕 K 線偵測
    bullish: 下影線 > 實體 2 倍（買盤在下方撐住）
    bearish: 上影線 > 實體 2 倍
    """
    body = abs(row['Close'] - row['Open'])
    if body == 0:
        body = 0.001  # 避免除以零

    if direction == "bullish":
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        return lower_wick > (body * 2)
    else:
        upper_wick = row['High'] - max(row['Open'], row['Close'])
        return upper_wick > (body * 2)


def compute_score(cross_info, div_info, slope_info, gp_info, has_rejection, vol_ratio) -> int:
    """
    綜合評分（0-100）
    核心原則：穿越或背離是「主訊號」，斜率加速是「加分項」
    單獨斜率加速 + MA上方 不夠格，必須有主訊號才能拿到有意義的分數
    """
    score = 0
    has_main_signal = cross_info["crossed_up"] or div_info["bullish_div"]

    # --- OBV 穿越 (0-25 分) --- 主訊號
    if cross_info["crossed_up"]:
        score += 15
        if cross_info["bars_since_cross"] == 0:
            score += 10  # 今天剛穿越，最新鮮
        elif cross_info["bars_since_cross"] == 1:
            score += 5

    # --- OBV 底背離 (0-30 分) --- 最強主訊號
    if div_info["bullish_div"]:
        score += 20
        if div_info["div_strength"] > 10:
            score += 10
        elif div_info["div_strength"] > 5:
            score += 5

    # --- OBV 斜率加速 (0-15 分) ---
    # 有主訊號時才給滿分，單獨觸發只給 5 分（不足以過門檻）
    if slope_info["accelerating"]:
        if has_main_signal:
            score += 15  # 加分項：主訊號 + 斜率 = 強確認
        else:
            score += 5   # 單獨斜率 = 弱訊號，不足以單獨立案

    # --- Golden Pocket (0-15 分) ---
    if gp_info.get("in_gp"):
        score += 10
        if gp_info.get("deviation", 999) < 30:
            score += 5

    # --- 拒絕 K 線 (0-5 分) ---
    if has_rejection:
        score += 5

    # --- 成交量倍數 (0-10 分) ---
    if vol_ratio >= 3.0:
        score += 10
    elif vol_ratio >= 2.0:
        score += 7
    elif vol_ratio >= 1.5:
        score += 4

    return min(score, 100)


def scan_single_stock(symbol: str, use_gp: bool, use_rejection: bool, min_struct_pct: float) -> dict | None:
    """掃描單一股票，回傳結果 dict 或 None"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if len(df) < 60:
            return None

        df['OBV'] = calculate_obv(df)
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # === 基本量能門檻 ===
        # yfinance .TW 的 Volume 單位是「股」，台股 1 張 = 1000 股
        avg_vol_shares = df['Volume'].tail(5).mean()
        avg_vol_lots = avg_vol_shares / 1000
        if avg_vol_lots < 500:  # 5日均量 < 500 張，流動性不足
            return None

        vol_20d = df['Volume'].tail(20).mean()
        vol_ratio = (avg_vol_shares / vol_20d) if vol_20d > 0 else 0

        # === OBV 核心訊號 ===
        obv = df['OBV']
        obv_ma = df['OBV_MA20']

        cross_info = detect_obv_crossover(obv, obv_ma, lookback=3)
        div_info = detect_obv_divergence(df, obv, window=20)
        slope_info = detect_obv_slope(obv, short_period=5, long_period=20)

        # OBV 基本條件：必須有「主訊號」（穿越或背離）
        # 單獨「斜率加速 + MA上方」不夠格進入候選
        obv_above_ma = current['OBV'] > current['OBV_MA20']
        has_main_signal = cross_info["crossed_up"] or div_info["bullish_div"]
        has_slope = slope_info["accelerating"]

        if not has_main_signal:
            # 沒有主訊號：只有 斜率+MA上方+放量 才勉強保留
            if not (has_slope and obv_above_ma and vol_ratio >= 2.0):
                return None

        # === Golden Pocket（可選） ===
        gp_info = {"in_gp": False, "gp_range": "-", "deviation": 999, "struct_pct": 0}
        if use_gp:
            gp_info = validate_golden_pocket(df, min_struct_pct)

        # === 拒絕 K 線（可選） ===
        has_rejection = False
        if use_rejection:
            has_rejection = is_rejection_candle(current, "bullish") or is_rejection_candle(prev, "bullish")

        # === 評分 ===
        score = compute_score(cross_info, div_info, slope_info, gp_info, has_rejection, vol_ratio)

        # 最低分數門檻：至少要有一個明確訊號
        if score < 15:
            return None

        # === 組裝 OBV 訊號標籤 ===
        obv_signals = []
        if cross_info["crossed_up"]:
            obv_signals.append(f"穿越↑ ({cross_info['bars_since_cross']}根前)")
        if div_info["bullish_div"]:
            obv_signals.append(f"底背離 ({div_info['div_strength']:.1f}%)")
        if slope_info["accelerating"]:
            obv_signals.append("斜率加速")
        if obv_above_ma and not cross_info["crossed_up"]:
            obv_signals.append("MA上方")

        clean_symbol = symbol.replace(".TW", "").replace(".TWO", "")

        return {
            "股票代碼": clean_symbol,
            "評分": score,
            "最新收盤": round(current['Close'], 2),
            "OBV 訊號": " | ".join(obv_signals) if obv_signals else "MA上方",
            "5日均量(張)": f"{avg_vol_lots:,.0f}",
            "量比(5/20)": f"{vol_ratio:.2f}x",
            "金色口袋": gp_info.get("gp_range", "-") if gp_info.get("in_gp") else "-",
            "結構幅度%": gp_info.get("struct_pct", 0) if gp_info.get("in_gp") else "-",
            "下影線拒絕": "✅" if has_rejection else "-",
        }

    except Exception:
        return None


# ============================================================
#  Streamlit UI
# ============================================================
st.set_page_config(page_title="Dolphin V2.1 OBV 波段掃描器", layout="wide")
st.title("🐬 Dolphin V2.1 — OBV 核心波段掃描器")
st.markdown("以 **OBV 資金流向**為核心（穿越 / 底背離 / 斜率加速），單獨斜率不成立，必須有主訊號。")

# --- Sidebar ---
st.sidebar.header("⚙️ 1. 股票池")

uploaded_file = st.sidebar.file_uploader("📂 上傳自選股清單 (CSV / XLSX)", type=["csv", "xlsx", "xls"])
ticker_list = []

if uploaded_file is not None:
    content = ""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                content = uploaded_file.getvalue().decode("utf-8")
            except Exception:
                content = uploaded_file.getvalue().decode("cp950", errors="ignore")
        else:
            df_upload = pd.read_excel(uploaded_file)
            content = df_upload.to_string()

        found_tickers = list(set(re.findall(r'\b\d{4}\b', content)))

        if found_tickers:
            st.sidebar.success(f"讀取到 {len(found_tickers)} 檔股票代碼")
        else:
            st.sidebar.error("檔案中沒有找到 4 位數股票代碼")

        ticker_list = found_tickers
    except Exception as e:
        st.sidebar.error(f"檔案讀取失敗: {e}")
else:
    default_tickers = "2330, 2317, 2454, 2308, 2382, 3231, 2603, 1513, 1519, 2376, 2357, 6235"
    ticker_input = st.sidebar.text_area("✍️ 手動輸入 (逗號分隔)", value=default_tickers)
    ticker_list = [t.strip() for t in ticker_input.split(",") if t.strip()]

# --- 傳產過濾 ---
st.sidebar.header("⚙️ 2. 產業過濾")
filter_traditional = st.sidebar.toggle("排除傳產股", value=True,
                                       help="排除水泥/食品/塑膠/紡織/電纜/化學/玻璃/造紙/鋼鐵/橡膠/汽車/航運/觀光/金融/百貨")

if filter_traditional and ticker_list:
    before_count = len(ticker_list)
    ticker_list = [t for t in ticker_list if is_traditional(t) is None]
    removed = before_count - len(ticker_list)
    if removed > 0:
        st.sidebar.info(f"已過濾 {removed} 檔傳產股，剩餘 {len(ticker_list)} 檔")

# --- 掃描參數 ---
st.sidebar.header("⚙️ 3. 訊號過濾")
use_gp = st.sidebar.toggle("啟用 Golden Pocket 過濾", value=False,
                            help="只保留價格落在 0.618-0.65 回撤區間的標的")
use_rejection = st.sidebar.toggle("啟用下影線拒絕過濾", value=False,
                                  help="要求最近 2 根 K 線出現多方拒絕形態")
min_struct_pct = st.sidebar.slider("GP 最小結構幅度 (%)", 5.0, 15.0, 8.0, 0.5,
                                   help="波段高低差佔價格的最小百分比，避免盤整區間的假 Fib") if use_gp else 8.0

min_score = st.sidebar.slider("最低評分門檻", 15, 60, 30, 5,
                               help="低於此分數的標的不會顯示（建議 30 以上）")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**評分邏輯（V2.1）**
- OBV 穿越：15-25分（主訊號）
- OBV 底背離：20-30分（最強主訊號）
- OBV 斜率加速：有主訊號+15 / 單獨僅+5
- Golden Pocket：10-15分
- 下影線拒絕：5分
- 量能倍數：4-10分

*必須有穿越或背離才算有效訊號*
""")

# --- 驗證清單按鈕 ---
st.sidebar.markdown("---")
if st.sidebar.button("🧹 檢查並剔除無效/下市股票"):
    if not ticker_list:
        st.sidebar.error("沒有股票代碼可檢查")
    else:
        progress = st.sidebar.progress(0)
        valid_tickers = []
        total = len(ticker_list)
        for i, t in enumerate(ticker_list):
            sym = f"{t}.TW" if not t.endswith((".TW", ".TWO")) else t
            try:
                if not yf.Ticker(sym).history(period="5d").empty:
                    valid_tickers.append(t)
                time.sleep(0.1)
            except Exception:
                pass
            progress.progress((i + 1) / total)

        removed_count = total - len(valid_tickers)
        clean_df = pd.DataFrame({"股票代碼": valid_tickers})
        csv_data = clean_df.to_csv(index=False).encode('utf-8-sig')

        st.sidebar.success(f"清除 {removed_count} 筆無效資料，剩餘 {len(valid_tickers)} 檔")
        st.sidebar.download_button(
            label="📥 下載乾淨清單",
            data=csv_data,
            file_name="Clean_Dolphin_V2.csv",
            mime="text/csv",
            type="primary",
        )

# --- 主掃描 ---
if st.button("🚀 開始掃描", type="primary"):
    if not ticker_list:
        st.error("找不到任何股票代碼")
    else:
        st.info(f"即將掃描 {len(ticker_list)} 檔股票...")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(ticker_list)

        for i, t in enumerate(ticker_list):
            t = t.strip()
            symbol = t if t.endswith((".TW", ".TWO")) else f"{t}.TW"
            status_text.text(f"掃描中: {symbol} ({i+1}/{total})")

            result = scan_single_stock(symbol, use_gp, use_rejection, min_struct_pct)
            if result and result["評分"] >= min_score:
                results.append(result)

            progress_bar.progress((i + 1) / total)
            time.sleep(0.12)

        status_text.text("掃描完成！")

        if results:
            df_result = pd.DataFrame(results).sort_values("評分", ascending=False).reset_index(drop=True)
            df_result.index += 1  # 排名從 1 開始

            st.success(f"掃描完畢！共 {len(df_result)} 檔符合條件，按評分排序：")

            # 分層顯示
            tier_a = df_result[df_result["評分"] >= 50]
            tier_b = df_result[(df_result["評分"] >= 30) & (df_result["評分"] < 50)]
            tier_c = df_result[df_result["評分"] < 30]

            if not tier_a.empty:
                st.subheader(f"🔴 A 級（≥50分）— {len(tier_a)} 檔")
                st.dataframe(tier_a, use_container_width=True)

            if not tier_b.empty:
                st.subheader(f"🟡 B 級（30-49分）— {len(tier_b)} 檔")
                st.dataframe(tier_b, use_container_width=True)

            if not tier_c.empty:
                st.subheader(f"⚪ C 級（<30分）— {len(tier_c)} 檔")
                st.dataframe(tier_c, use_container_width=True)

            # 下載結果
            csv_out = df_result.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載掃描結果 CSV", csv_out, "Dolphin_V2_Results.csv", "text/csv")
        else:
            st.info("目前沒有符合條件的標的。試著降低評分門檻或關閉 GP/拒絕過濾。")