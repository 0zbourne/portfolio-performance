from pathlib import Path
import pandas as pd
import yfinance as yf
import requests

DATA_DIR = Path("data")

def get_sp500_daily(start: str, end: str) -> pd.DataFrame:
    """Get S&P 500 prices in GBP."""
    # Fetch SPY (USD)
    spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    spy = spy.reset_index()
    spy["date"] = pd.to_datetime(spy["Date"]).dt.tz_localize(None)
    spy = spy[["date", "Close"]].rename(columns={"Close": "close_usd"})
    
    # Fetch USD→GBP FX
    url = f"https://api.frankfurter.app/{start}..{end}"
    r = requests.get(url, params={"from": "USD", "to": "GBP"}, timeout=20)
    fx_data = r.json()["rates"]
    fx = pd.DataFrame.from_dict(fx_data, orient="index").rename_axis("date").reset_index()
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx.rename(columns={"GBP": "usd_gbp"})
    
    # Merge
    df = spy.merge(fx, on="date", how="left").ffill()
    df["close_gbp"] = df["close_usd"] * df["usd_gbp"]
    df["daily_ret"] = df["close_gbp"].pct_change().fillna(0)
    
    return df[["date", "close_usd", "usd_gbp", "close_gbp", "daily_ret"]]
