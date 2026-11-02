# 台股資料與回測（輕量模組版）

```text
twbacktest/
├── providers.py    # Provider 介面、CSV／yfinance／TWSE／快取與切換
├── indicators.py   # SMA20／SMA60、VMA20、RSI、MACD
├── backtest.py     # DCA、智慧抄底、波段策略與績效計算
├── charts.py       # Swing／DCA 對照圖（英文圖例，適用無中文字型伺服器）
├── cli.py          # argparse、設定組裝與輸出
└── exceptions.py   # Provider 例外
run_backtest.py     # CLI 入口
```

所有 Provider 都輸出 `Date, Open, High, Low, Close, Volume, Ticker, Name, Provider`。
Router 按順序切換來源，成功的網路／本機結果會更新快取；所有來源失敗時會列出每一項原因。

```bash
# 優先使用已匯出的台灣 50 快照，找不到才依序用 Yahoo、TWSE、快取
python3 run_backtest.py --ticker 2317 --start 2020-01-01 --end 2026-10-30

# 波段策略：多頭排列下的帶量突破或 SMA20 回測反彈，並和 DCA 對照
python3 run_backtest.py --ticker 2454.TW --monthly-amount 15000 \
  --swing-initial-cash 300000 --swing-position-fraction 1.0 \
  --swing-volume-multiplier 1.3 \
  --providers local,yfinance,twse,cache
```

波段策略進場需 `Close > SMA60`、`SMA20 > SMA60`，再符合「帶量突破前 20 日高點」或「回測 SMA20 後量縮上漲」。收盤跌破 SMA20 或 SMA60 時全數出場。DCA 則維持每月月初固定金額買入，作為基準。

產物會寫入 `output/modular_backtest/`，包含 `{ticker}_swing_trades.csv`、`{ticker}_dca_trades.csv` 與比較圖；`--no-chart` 可跳過 matplotlib，適合 CI 或伺服器。
