import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import re
import json
import os

# ============================================================
#  Dolphin V3 — OBV 核心波段掃描器 + 績效追蹤 (台股版)
#  核心邏輯：OBV 穿越 + OBV 底背離 + OBV 斜率加速 + Z-score
#  新增：內建股票池、歷史訊號績效追蹤
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
    hm = cross_info["crossed_up"] or div_info["bullish_div"]
    if cross_info["crossed_up"]:
        s += 15
        if cross_info["bars_since_cross"] == 0: s += 10
        elif cross_info["bars_since_cross"] == 1: s += 5
    if div_info["bullish_div"]:
        s += 20
        if div_info["div_strength"] > 10: s += 10
        elif div_info["div_strength"] > 5: s += 5
    if slope_info.get("accelerating"):
        s += 15 if hm else 5
    z = abs(cross_info.get("z_score", 0))
    if z >= 2.5: s += 10
    elif z >= 2.0: s += 7
    elif z >= 1.5: s += 4
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
            runup = round((cur['Close'] - div["div_low_price"]) / div["div_low_price"] * 100, 1)
            if runup > max_runup:
                return None

        above_ma = cur['OBV'] > cur['OBV_MA20']
        has_main = cross["crossed_up"] or div["bullish_div"]
        if not has_main:
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
        if cross["crossed_up"]: sigs.append(f"穿越↑({cross['bars_since_cross']}根前)")
        if div["bullish_div"]: sigs.append(f"底背離({div['div_strength']:.1f}%)")
        if slope.get("accelerating"): sigs.append("斜率加速")
        if above_ma and not cross["crossed_up"]: sigs.append("MA上方")

        cs = symbol.replace(".TW", "").replace(".TWO", "")
        nm = (name_map or {}).get(cs, "")

        return {
            "股票": f"{cs} {nm}" if nm else cs,
            "評分": sc,
            "收盤": round(cur['Close'], 2),
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

def backtest_stock(symbol, name_map=None, hold_days=[5, 10, 20]):
    """
    對單一股票滾動偵測歷史訊號，隔日開盤進場，追蹤 N 日報酬
    """
    try:
        df = yf.Ticker(symbol).history(period="1y")
        if len(df) < 120:
            return []
        df = prepare_obv_data(df)

        if df['Volume'].tail(20).mean() / 1000 < 2000:
            return []

        cs = symbol.replace(".TW", "").replace(".TWO", "")
        nm = (name_map or {}).get(cs, "")
        display = f"{cs} {nm}" if nm else cs
        max_hold = max(hold_days)

        signals = []
        prev_cross = False
        prev_div = False

        for idx in range(60, len(df) - max_hold - 1):
            sub = df.iloc[:idx + 1]
            obv_s, obv_ma_s, obv_z_s = sub['OBV'], sub['OBV_MA20'], sub['OBV_Z']

            cr = detect_obv_crossover(obv_s, obv_ma_s, obv_z_s, lookback=1)
            dv = detect_obv_divergence(sub, obv_s, window=20)

            new_cross = cr["crossed_up"] and not prev_cross
            new_div = dv["bullish_div"] and not prev_div
            prev_cross = cr["crossed_up"]
            prev_div = dv["bullish_div"]

            if not (new_cross or new_div):
                continue

            entry_idx = idx + 1
            if entry_idx >= len(df):
                continue
            ep = df['Open'].iloc[entry_idx]
            if ep <= 0:
                continue

            st_list = []
            if new_cross: st_list.append("穿越")
            if new_div: st_list.append("背離")

            rec = {
                "股票": display,
                "訊號日": df.index[idx].strftime("%Y-%m-%d"),
                "類型": "+".join(st_list),
                "進場價": round(ep, 2),
            }
            for d in hold_days:
                ei = entry_idx + d
                if ei < len(df):
                    rec[f"+{d}日%"] = round((df['Close'].iloc[ei] - ep) / ep * 100, 2)
                else:
                    rec[f"+{d}日%"] = None
            signals.append(rec)

        return signals
    except Exception:
        return []


# ============================================================
#  Streamlit UI
# ============================================================
st.set_page_config(page_title="Dolphin V3", layout="wide")
st.title("🐬 Dolphin V3 — OBV 波段掃描 + 績效追蹤")

universe = load_universe()
name_map = load_stock_names()
display_names = name_map if name_map else {k: v.get("name", "") for k, v in universe.items()}

tab_scan, tab_perf = st.tabs(["🔍 即時掃描", "📊 績效追蹤"])

# ====================
#  Tab 1: 即時掃描
# ====================
with tab_scan:
    st.markdown("以 **OBV 資金流向**為核心（穿越 / 底背離 / 斜率加速 / Z-score）")

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
    use_gp = st.sidebar.toggle("Golden Pocket", value=False)
    use_rej = st.sidebar.toggle("下影線拒絕", value=False)
    msp = st.sidebar.slider("GP結構%", 5.0, 15.0, 8.0, 0.5) if use_gp else 8.0
    min_sc = st.sidebar.slider("最低評分", 15, 60, 30, 5)
    max_ru = st.sidebar.slider("背離後最大漲幅%", 5, 50, 15, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**V3** 穿越15-25 / 背離20-30 / 斜率+15|+5 / Z 4-10 / GP 10-15 / 拒絕5 / 量比4-10")

    if st.button("🚀 開始掃描", type="primary"):
        if not tickers:
            st.error("沒有股票")
        else:
            st.info(f"掃描 {len(tickers)} 檔...")
            results, prog, stat = [], st.progress(0), st.empty()
            for i, t in enumerate(tickers):
                t = t.strip()
                sym = t + (universe.get(t, {}).get("suffix", ".TW") if not t.endswith((".TW", ".TWO")) else "")
                if t.endswith((".TW", ".TWO")): sym = t
                c = t.replace(".TW", "").replace(".TWO", "")
                stat.text(f"掃描: {c} {display_names.get(c, '')} ({i+1}/{len(tickers)})")
                r = scan_single_stock(sym, use_gp, use_rej, msp, max_ru, display_names)
                if r and r["評分"] >= min_sc: results.append(r)
                prog.progress((i + 1) / len(tickers))
                time.sleep(0.12)
            stat.text("完成！")

            if results:
                dfr = pd.DataFrame(results).sort_values("評分", ascending=False).reset_index(drop=True)
                dfr.index += 1
                st.success(f"{len(dfr)} 檔符合條件")
                for label, lo, hi, emoji in [("A ≥50", 50, 999, "🔴"), ("B 30-49", 30, 50, "🟡"), ("C <30", 0, 30, "⚪")]:
                    sub = dfr[(dfr["評分"] >= lo) & (dfr["評分"] < hi)] if hi < 999 else dfr[dfr["評分"] >= lo]
                    if not sub.empty:
                        st.subheader(f"{emoji} {label} — {len(sub)} 檔")
                        st.dataframe(sub, use_container_width=True)
                st.download_button("📥 下載", dfr.to_csv(index=False).encode('utf-8-sig'), "Dolphin_V3_Results.csv", "text/csv")
            else:
                st.info("沒有符合條件的標的")


# ====================
#  Tab 2: 績效追蹤
# ====================
with tab_perf:
    st.markdown("""
    ### 歷史訊號回測
    用 1 年的資料，滾動偵測所有 OBV 穿越和底背離訊號，
    以**隔日開盤進場**，追蹤 +5 / +10 / +20 日報酬。
    不用每天手動記錄 — 跑一次就知道訊號品質。
    """)

    bt_mode = st.radio("回測來源", ["✍️ 手動輸入", "📂 上傳掃描結果"], horizontal=True)

    bt_tickers = []
    if bt_mode.startswith("✍️"):
        bt_inp = st.text_area("回測股票（逗號分隔，建議5-20檔）", "2330, 2317, 2454, 3231, 1513, 2308")
        bt_tickers = [t.strip() for t in bt_inp.split(",") if t.strip()]
    else:
        bt_uf = st.file_uploader("上傳 Dolphin 結果 CSV", type=["csv"])
        if bt_uf:
            try:
                btdf = pd.read_csv(bt_uf, encoding='utf-8-sig')
                if "股票" in btdf.columns:
                    bt_tickers = [str(x).split()[0] for x in btdf["股票"] if str(x).strip()]
                    st.success(f"載入 {len(bt_tickers)} 檔")
            except Exception as e:
                st.error(str(e))

    if st.button("📊 開始回測", type="primary"):
        if not bt_tickers:
            st.error("請輸入股票")
        else:
            all_sigs, prog, stat = [], st.progress(0), st.empty()
            for i, t in enumerate(bt_tickers):
                t = t.strip()
                sym = t + (universe.get(t, {}).get("suffix", ".TW") if not t.endswith((".TW", ".TWO")) else "")
                if t.endswith((".TW", ".TWO")): sym = t
                c = t.replace(".TW", "").replace(".TWO", "")
                stat.text(f"回測: {c} {display_names.get(c, '')} ({i+1}/{len(bt_tickers)})")
                all_sigs.extend(backtest_stock(sym, display_names))
                prog.progress((i + 1) / len(bt_tickers))
                time.sleep(0.15)
            stat.text("完成！")

            if all_sigs:
                dfs = pd.DataFrame(all_sigs)

                # 整體統計
                st.subheader("📈 整體績效")
                stats = {}
                for col in ["+5日%", "+10日%", "+20日%"]:
                    v = dfs[col].dropna()
                    if len(v) > 0:
                        stats[col] = {
                            "訊號數": int(len(v)),
                            "勝率": f"{(v > 0).mean() * 100:.1f}%",
                            "平均報酬": f"{v.mean():.2f}%",
                            "中位數": f"{v.median():.2f}%",
                            "最大獲利": f"+{v.max():.1f}%",
                            "最大虧損": f"{v.min():.1f}%",
                            "獲利因子": f"{v[v > 0].sum() / abs(v[v < 0].sum()):.2f}" if (v < 0).any() else "∞",
                        }
                if stats:
                    st.dataframe(pd.DataFrame(stats).T, use_container_width=True)

                # 按類型
                st.subheader("📊 按訊號類型")
                for st_type in sorted(dfs["類型"].unique()):
                    sub = dfs[dfs["類型"] == st_type]
                    if len(sub) < 2: continue
                    st.markdown(f"**{st_type}**（{len(sub)} 筆）")
                    ts = {}
                    for col in ["+5日%", "+10日%", "+20日%"]:
                        v = sub[col].dropna()
                        if len(v) > 0:
                            ts[col] = {
                                "勝率": f"{(v > 0).mean() * 100:.1f}%",
                                "平均": f"{v.mean():.2f}%",
                                "中位數": f"{v.median():.2f}%",
                            }
                    if ts:
                        st.dataframe(pd.DataFrame(ts).T, use_container_width=True)

                # 個股明細
                st.subheader("📋 訊號明細")
                st.dataframe(dfs.sort_values("訊號日", ascending=False).reset_index(drop=True), use_container_width=True)

                st.download_button("📥 下載回測", dfs.to_csv(index=False).encode('utf-8-sig'), "Dolphin_V3_Backtest.csv", "text/csv")
            else:
                st.info("沒有歷史訊號（可能均量不足或無穿越/背離）")
