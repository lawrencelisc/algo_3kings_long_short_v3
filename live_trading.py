#!/usr/bin/env python3
"""
實盤交易啟動腳本（取代以 simulate_trading.py 跑 live 的需求）

- 強制 SIMULATION_MODE=false（覆寫 .env 內同名字段）
- 從 .env 載入 BYBIT_API_KEY / BYBIT_SECRET 等
- 呼叫 prototype_long_short_v3.main()（真實下單）

模擬請仍用: python simulate_trading.py
"""

import os
import sys

# 先載入 .env，再強制實盤（避免 .env 誤留 SIMULATION_MODE=true）
env_file = '.env'
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
else:
    print("❌ 找不到 .env，實盤請在專案目錄建立 .env 並設定 BYBIT_API_KEY / BYBIT_SECRET")
    sys.exit(1)

os.environ['SIMULATION_MODE'] = 'false'
# 避免沿用模擬用變數
os.environ.pop('SIM_BALANCE', None)

print("=" * 70)
print("🔴 加密貨幣交易系統 — 實盤模式（LIVE）")
print("=" * 70)
print("版本: V3 Long/Short")
print("模式: 真實下單 · Bybit USDT 線性合約")
print("日誌: result/live_long_short_log.csv（見 prototype 內 LOG_FILE）")
print("=" * 70)

try:
    import ccxt
    import pandas as pd
    import numpy as np
    print("✅ 核心模塊檢查通過:")
    print(f"   ccxt: {ccxt.__version__}")
    print(f"   pandas: {pd.__version__}")
    print(f"   numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ 模塊缺失: {e}")
    print("請安裝: pip install ccxt pandas numpy python-dotenv requests")
    sys.exit(1)

api_key = os.environ.get('BYBIT_API_KEY', '').strip()
api_secret = os.environ.get('BYBIT_SECRET', '').strip()

if not api_key or not api_secret or api_key == 'your_bybit_api_key_here':
    print("❌ 實盤必須設定有效 BYBIT_API_KEY 與 BYBIT_SECRET（.env）")
    sys.exit(1)

print("✅ API 密鑰已讀取（長度已隱藏）")
print("=" * 70)
print("⚠️  實盤風險：程式會下單、改槓桿、設 TP/SL。請確認帳戶與 prototype 內風控參數。")
print("   建議：子帳戶、小額、IP 白名單、勿開提款權限。")
print("=" * 70)

try:
    print("\n🚀 正在啟動實盤系統（SIMULATION_MODE=false）...\n")
    import prototype_long_short_v3 as trading_system

    trading_system.main()

except KeyboardInterrupt:
    print("\n\n👋 實盤程式已手動終止；請到交易所核對持倉與掛單。")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ 啟動或運行失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
