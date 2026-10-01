"""
Builds data.json for the Breakout Screener (runs on GitHub Actions).

Universe : ALL NSE equities from NSE's daily EQUITY_L.csv
           (falls back to the TradingView NSE list if NSE blocks the download)
Prices   : yfinance daily OHLC (last BARS bars)
Mkt cap  : TradingView scanner (0 if unknown)
"""

import io
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from tradingview_screener import Query, col

NSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
SERIES = {"EQ"}                    # add "BE" to include trade-for-trade stocks
MIN_PRICE = 10                     # drop penny stocks (INR)
MIN_AVG_TURNOVER = 10_000_000      # drop dead stocks: 20-day avg of price x volume (INR)
BARS = 330                         # daily bars stored per stock
CHUNK = 100


def nse_symbols():
    r = requests.get(
        NSE_URL,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip().isin(SERIES)]
    return [s.strip() for s in df["SYMBOL"]]


def tv_table():
    _, df = (
        Query()
        .set_markets("india")
        .select("name", "market_cap_basic")
        .where(col("exchange") == "NSE", col("type") == "stock")
        .limit(5000)
        .get_scanner_data()
    )
    return df


def download(tickers):
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers, period="2y", interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:
            print("  download error:", e)
        time.sleep(5 * (attempt + 1))
    return None


def r2(x):
    return round(float(x), 2)


def main():
    # market caps (and fallback symbol list) from TradingView
    mcap = {}
    tv_names = []
    try:
        tv = tv_table()
        mcap = dict(zip(tv["name"], tv["market_cap_basic"]))
        tv_names = list(tv["name"])
    except Exception as e:
        print("TradingView list failed:", e)

    try:
        symbols = nse_symbols()
        print(f"NSE EQUITY_L.csv: {len(symbols)} symbols")
    except Exception as e:
        print("NSE download failed, using TradingView list:", e)
        symbols = tv_names
        print(f"TradingView list: {len(symbols)} symbols")

    stocks = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        raw = download([s + ".NS" for s in chunk])
        if raw is None:
            print("  chunk skipped")
            continue

        for s in chunk:
            try:
                d = raw[s + ".NS"] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Open", "High", "Low", "Close"]).tail(BARS)
                if len(d) < 60:
                    continue
                close, vol = d["Close"], d["Volume"].fillna(0)
                to = close * vol
                if close.iloc[-1] < MIN_PRICE or to.tail(20).mean() < MIN_AVG_TURNOVER:
                    continue
                m = mcap.get(s, 0)
                stocks.append({
                    "s": s,
                    "p": r2(close.iloc[-1]),
                    "m": r2(m / 1e9) if m and not math.isnan(m) else 0,
                    "to": r2(to.iloc[-1] / 1e6),
                    "lo": r2(d["Low"].tail(252).min()),
                    "d": [int(x.strftime("%Y%m%d")) for x in d.index],
                    "o": [r2(x) for x in d["Open"]],
                    "h": [r2(x) for x in d["High"]],
                    "l": [r2(x) for x in d["Low"]],
                    "c": [r2(x) for x in close],
                })
            except Exception:
                continue
        print(f"  {min(i + CHUNK, len(symbols))}/{len(symbols)}  kept={len(stocks)}")

    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "stocks": stocks}
    with open("data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved data.json with {len(stocks)} stocks")


if __name__ == "__main__":
    main()
