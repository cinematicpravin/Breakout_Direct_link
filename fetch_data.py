"""
Builds data.json for the Breakout Screener (runs on GitHub Actions).

Universe : ALL NSE equities from NSE's daily EQUITY_L.csv
           (falls back to the TradingView NSE list if NSE blocks the download)
Prices   : yfinance daily OHLC (~13 years), then resampled to weekly + monthly
Mkt cap  : TradingView scanner (0 if unknown)
Output   : data_d.json (daily), data_w.json (weekly), data_m.json (monthly)
"""

import io
import json
import math
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf
from tradingview_screener import Query, col

NSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
SERIES = {"EQ"}                    # add "BE" to include trade-for-trade stocks
MIN_PRICE = 10                     # drop penny stocks (INR)
MIN_AVG_TURNOVER = 10_000_000      # drop dead stocks: 20-day avg of price x volume (INR)
BARS = {"d": 330, "w": 330, "m": 156}   # bars stored per stock for each timeframe
HISTORY_YEARS = 13                      # enough for 156 monthly bars
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
                tickers, start=(date.today() - timedelta(days=365 * HISTORY_YEARS)).isoformat(),
                interval="1d", group_by="ticker",
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


def resample(d, freq, bars):
    """Daily OHLC -> weekly ('W-FRI') or monthly ('M') bars, dated by the last trading day."""
    idx = d.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    g = d.assign(_k=idx.to_period(freq), _t=idx).groupby("_k")
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(),
        "Low": g["Low"].min(), "Close": g["Close"].last(), "T": g["_t"].last(),
    })
    return out.tail(bars)


def pack(meta, dates, f):
    return {
        **meta,
        "d": [int(x.strftime("%Y%m%d")) for x in dates],
        "o": [r2(x) for x in f["Open"]],
        "h": [r2(x) for x in f["High"]],
        "l": [r2(x) for x in f["Low"]],
        "c": [r2(x) for x in f["Close"]],
    }


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

    stocks = {"d": [], "w": [], "m": []}
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        raw = download([s + ".NS" for s in chunk])
        if raw is None:
            print("  chunk skipped")
            continue

        for s in chunk:
            try:
                d = raw[s + ".NS"] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.dropna(subset=["Open", "High", "Low", "Close"])
                if len(d) < 60:
                    continue
                close, vol = d["Close"], d["Volume"].fillna(0)
                to = close * vol
                if close.iloc[-1] < MIN_PRICE or to.tail(20).mean() < MIN_AVG_TURNOVER:
                    continue
                m = mcap.get(s, 0)
                meta = {
                    "s": s,
                    "p": r2(close.iloc[-1]),
                    "m": r2(m / 1e9) if m and not math.isnan(m) else 0,
                    "to": r2(to.iloc[-1] / 1e6),
                    "lo": r2(d["Low"].tail(252).min()),
                }
                dd = d.tail(BARS["d"])
                stocks["d"].append(pack(meta, dd.index, dd))

                w = resample(d, "W-FRI", BARS["w"])
                if len(w) >= 30:
                    stocks["w"].append(pack(meta, w["T"], w))
                mo = resample(d, "M", BARS["m"])
                if len(mo) >= 20:
                    stocks["m"].append(pack(meta, mo["T"], mo))
            except Exception:
                continue
        print(f"  {min(i + CHUNK, len(symbols))}/{len(symbols)}  kept={len(stocks['d'])}")

    updated = datetime.now(timezone.utc).isoformat(timespec="minutes")
    for k, lst in stocks.items():
        with open(f"data_{k}.json", "w") as f:
            json.dump({"updated": updated, "stocks": lst}, f, separators=(",", ":"))
        print(f"Saved data_{k}.json with {len(lst)} stocks")


if __name__ == "__main__":
    main()
