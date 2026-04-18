"""
在本地跑一次，產生 stock_names.json，推到 GitHub repo
用法：python generate_names.py
"""
import requests
import json

name_map = {}

# TWSE 上市
print("抓取 TWSE 上市...")
resp = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=30)
data = resp.json()
sample = data[0]
code_key = next(k for k in sample if "代號" in k)
name_key = next(k for k in sample if "簡稱" in k)
for item in data:
    c = str(item.get(code_key, "")).strip()
    n = str(item.get(name_key, "")).strip()
    if c and n:
        name_map[c] = n
print(f"  TWSE: {len(name_map)} 檔")

# TPEx 上櫃
print("抓取 TPEx 上櫃...")
resp = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", timeout=30)
data = resp.json()
sample = data[0]
code_key = next(k for k in sample if "代號" in k or "Code" in k)
name_key = next(k for k in sample if "簡稱" in k or "Name" in k)
before = len(name_map)
for item in data:
    c = str(item.get(code_key, "")).strip()
    n = str(item.get(name_key, "")).strip()
    if c and n:
        name_map[c] = n
print(f"  TPEx: {len(name_map) - before} 檔")

with open("stock_names.json", "w", encoding="utf-8") as f:
    json.dump(name_map, f, ensure_ascii=False, indent=2)

print(f"\n完成！stock_names.json ({len(name_map)} 檔)")
print("請把這個檔案推到你的 GitHub repo")
