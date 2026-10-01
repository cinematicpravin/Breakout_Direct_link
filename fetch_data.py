"""
Runs on GitHub Actions (or locally). Builds data.json for the Breakout Screener.

1. TradingView scanner -> NSE stocks passing the loose universe filters below
2. yfinance -> daily OHLC history for those stocks
3. Saves compact data.json (the HTML page runs the indicator on it)
"""

import json
import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from tradingview_screener import Query, col

# ---- Universe (edit these if you want a wider / narrower list) ----
MIN_PRICE = 30
MIN_MCAP = 5_000_000_000      # 5B INR
MIN_TURNOVER = 50_000_000     # 50M INR (price x volume)
BARS = 330                    # daily bars stored per stock
CHUNK = 100


def get_universe():
    _, df = (
        Query()
        .set_markets("india")
        .select("name", "close", "volume", "market_cap_basic", "price_52_week_low")
        .where(
            col("exchange") == "NSE",
            col("type") == "stock",
            col("close") > MIN_PRICE,
            col("market_cap_basic") > MIN_MCAP,
        )
        .limit(5000)
        .get_scanner_data()
    )
    df["turnover"] = df["close"] * df["volume"]
    df = df[df["turnover"] > MIN_TURNOVER].reset_index(drop=True)
    print(f"Universe: {len(df)} stocks")
    return df


def r2(x):
    return round(float(x), 2)


def main():
    uni = get_universe()
    meta = {row["name"]: row for _, row in uni.iterrows()}
    symbols = list(meta.keys())
    stocks = []

    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        tickers = [s + ".NS" for s in chunk]
        try:
            raw = yf.download(
                tickers, period="2y", interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception as e:
            print("chunk failed:", e)
            continue

        for s in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    d = raw[s + ".NS"]
                else:
                    d = raw
                d = d.dropna(subset=["Open", "High", "Low", "Close"]).tail(BARS)
                if len(d) < 60:
                    continue
                m = meta[s]
                stocks.append({
                    "s": s,
                    "p": r2(m["close"]),
                    "m": r2(m["market_cap_basic"] / 1e9),
                    "to": r2(m["turnover"] / 1e6),
                    "lo": r2(m["price_52_week_low"]) if not math.isnan(m["price_52_week_low"]) else 0,
                    "d": [int(x.strftime("%Y%m%d")) for x in d.index],
                    "o": [r2(x) for x in d["Open"]],
                    "h": [r2(x) for x in d["High"]],
                    "l": [r2(x) for x in d["Low"]],
                    "c": [r2(x) for x in d["Close"]],
                })
            except Exception:
                continue
        print(f"  {min(i + CHUNK, len(symbols))}/{len(symbols)}")

    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "stocks": stocks}
    with open("data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved data.json with {len(stocks)} stocks")


if __name__ == "__main__":
    main()
