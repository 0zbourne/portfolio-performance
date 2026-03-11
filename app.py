import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import numpy_financial as npf

from perf import read_nav, daily_returns_twr, cumulative_return, max_drawdown, xirr
from perf import build_cash_flows
from sp500 import get_sp500_daily

# ---- Config ----
st.set_page_config(page_title="Portfolio Performance", layout="wide")
DATA = Path("data")
NAV_CSV = DATA / "nav_daily.csv"

# ---- T212 Auth ----
def _auth_header() -> dict:
    key = (os.getenv("T212_API_KEY") or "").strip() or str(st.secrets.get("T212_API_KEY", "")).strip()
    if not key:
        return {}
    if key.lower().startswith("apikey "):
        return {"Authorization": key, "Accept": "application/json"}
    return {"Authorization": f"Apikey {key}", "Accept": "application/json"}

# ---- Load Data ----
@st.cache_data
def load_nav():
    if NAV_CSV.exists():
        return read_nav(NAV_CSV)
    return None

@st.cache_data
def load_flows():
    return build_cash_flows(DATA / "transactions.json")

# ---- Main ----
st.title("Portfolio Performance")
st.caption("Am I beating the market?")

nav = load_nav()
flows = load_flows()

if nav is None or len(nav) == 0:
    st.error("No NAV data. Run backfill first.")
    st.stop()

# Date range
start_date = nav.index.min().strftime("%Y-%m-%d")
end_date = nav.index.max().strftime("%Y-%m-%d")

# ---- Your Returns ----
daily_returns = daily_returns_twr(nav, flows)
your_total = cumulative_return(daily_returns, start_date, end_date)
your_xirr = xirr(nav, flows)

# ---- S&P 500 Returns ----
sp500 = get_sp500_daily(start_date, end_date)
sp500_total = float((1 + sp500["daily_ret"]).prod() - 1)

# For S&P 500 XIRR, we need to simulate the same cash flows into SPY
# Simplified: use CAGR as approximation
sp500_cagr = (1 + sp500_total) ** (252 / len(sp500)) - 1

# ---- Delta ----
delta = your_total - sp500_total

# ---- Max Drawdown ----
drawdown = max_drawdown(nav)

# ---- Display ----
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Your Return", f"{your_total:.1%}", f"XIRR: {your_xirr:.1%}")

with col2:
    st.metric("S&P 500", f"{sp500_total:.1%}", f"CAGR: {sp500_cagr:.1%}")

with col3:
    color = "normal" if delta >= 0 else "inverse"
    st.metric("Delta", f"{delta:+.1%}", "Beat the market" if delta >= 0 else "Underperforming", delta_color=color)

st.divider()

col4, col5 = st.columns(2)
with col4:
    st.metric("Max Drawdown", f"{drawdown:.1%}")
with col5:
    years = len(nav) / 252
    st.metric("Period", f"{years:.1f} years")

# ---- NAV Chart ----
st.subheader("NAV vs S&P 500 (rebased to 100)")

# Rebase both to 100
plot_nav = (nav / nav.iloc[0] * 100).reset_index()
plot_nav.columns = ["date", "Portfolio"]
plot_nav["date"] = pd.to_datetime(plot_nav["date"]).dt.normalize()

sp_rebased = (sp500["close_gbp"] / sp500["close_gbp"].iloc[0] * 100).reset_index()
sp_rebased.columns = ["date", "S&P 500"]
sp_rebased["date"] = pd.to_datetime(sp_rebased["date"]).dt.normalize()

# Merge for plotting
merged = plot_nav.merge(sp_rebased, on="date", how="inner")

import altair as alt
melted = merged.melt("date", var_name="Series", value_name="Index")

chart = alt.Chart(melted).mark_line().encode(
    x="date:T",
    y=alt.Y("Index:Q", scale=alt.Scale(zero=False)),
    color="Series:N",
    tooltip=["date:T", "Series:N", alt.Tooltip("Index:Q", format=".1f")]
).properties(height=350)

st.altair_chart(chart, width="stretch")

# ---- Sidebar: Backfill Trigger ----
with st.sidebar:
    st.header("Backfill")
    if st.button("Run NAV Backfill"):
        from jobs.backfill import backfill_nav_from_orders
        try:
            backfill_nav_from_orders(start="2025-01-01")
            st.success("Backfill complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
