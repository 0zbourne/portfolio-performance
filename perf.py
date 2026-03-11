from pathlib import Path
import numpy as np
import pandas as pd
import json
import numpy_financial as npf

DATA_DIR = Path("data")

# ---- NAV Reading ----
def read_nav(path: Path = DATA_DIR / "nav_daily.csv") -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "nav_gbp"]).sort_values("date")
    s = pd.Series(df["nav_gbp"].values, index=df["date"].values, name="nav_gbp")
    s.index.name = "date"
    return s

# ---- Cash Flows ----
def build_cash_flows(transactions_json_path: Path) -> pd.DataFrame:
    if not transactions_json_path.exists():
        return pd.DataFrame(columns=["date", "amount_gbp"])
    
    raw = json.loads(transactions_json_path.read_text(encoding="utf-8"))
    items = raw.get("items", raw)
    if not isinstance(items, list) or not items:
        return pd.DataFrame(columns=["date", "amount_gbp"])
    
    df = pd.json_normalize(items)
    
    action_col = next((c for c in df.columns if c.lower() in {"action", "type"}), None)
    amount_col = next((c for c in df.columns if "totalamount" in c.replace("_", "").lower() or c.lower() in {"total", "amount"}), None)
    date_col = next((c for c in df.columns if any(k in c.lower() for k in ("time", "date", "created", "settled"))), None)
    
    if not all([action_col, amount_col, date_col]):
        return pd.DataFrame(columns=["date", "amount_gbp"])
    
    df[action_col] = df[action_col].astype(str).str.lower()
    flows = df[df[action_col].str.contains("deposit|withdraw", na=False)].copy()
    if flows.empty:
        return pd.DataFrame(columns=["date", "amount_gbp"])
    
    flows["amount_gbp"] = pd.to_numeric(flows[amount_col], errors="coerce")
    flows.loc[flows[action_col].str.contains("withdraw"), "amount_gbp"] *= -1
    
    flows["date"] = pd.to_datetime(flows[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    return flows[["date", "amount_gbp"]]

# ---- Daily Returns (TWR) ----
def daily_returns_twr(nav: pd.Series, flows: pd.DataFrame | None = None) -> pd.DataFrame:
    if nav is None or len(nav) == 0:
        return pd.DataFrame(columns=["date", "r_port"])
    
    nav = pd.to_numeric(nav, errors="coerce").sort_index()
    nav.index = pd.to_datetime(nav.index).normalize()
    nav = nav.where(nav > 0)
    
    first_idx = nav.first_valid_index()
    if first_idx is None:
        return pd.DataFrame(columns=["date", "r_port"])
    nav = nav.loc[first_idx:]
    
    if flows is not None and not flows.empty:
        f = flows.copy()
        f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
        f = f.dropna(subset=["date"])
        f["amount_gbp"] = pd.to_numeric(f["amount_gbp"], errors="coerce")
        per_day = f.groupby("date")["amount_gbp"].sum().sort_index()
        cum = per_day.cumsum()
        cum_on_nav = cum.reindex(nav.index, method="ffill").fillna(0.0)
        flow_s = cum_on_nav.diff().fillna(cum_on_nav)
    else:
        flow_s = pd.Series(0.0, index=nav.index, dtype="float64")
    
    denom = nav.shift(1) + flow_s
    r = pd.Series(np.where(denom > 0, nav / denom - 1.0, 0.0), index=nav.index, dtype="float64")
    r.iloc[0] = 0.0
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    
    return pd.DataFrame({"date": r.index.astype("datetime64[ns]"), "r_port": r.values})

# ---- Cumulative Return ----
def cumulative_return(obj, start: str | None = None, end: str | None = None) -> float:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]
            s = pd.to_numeric(df["r_port"], errors="coerce")
        else:
            s = pd.to_numeric(df.select_dtypes(include=[np.number]).iloc[:, 0], errors="coerce")
    else:
        s = pd.to_numeric(pd.Series(obj), errors="coerce")
    
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() - 1.0)

# ---- Max Drawdown ----
def max_drawdown(nav: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    if nav is None or len(nav) == 0:
        return np.nan
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    return float(drawdown.min())

# ---- XIRR (Money-Weighted Return) ----
def xirr(nav: pd.Series, flows: pd.DataFrame) -> float:
    """
    Internal rate of return for irregular cash flows.
    If no external flows, calculates simple CAGR.
    """
    if nav is None or len(nav) < 2:
        return np.nan
    
    # Build cash flow series
    cash_flows = []
    
    # Add initial NAV as negative cash flow (what you "invested" at start)
    initial_date = nav.index[0]
    initial_nav = float(nav.iloc[0])
    
    # Add external cash flows if any
    if flows is not None and not flows.empty:
        for _, row in flows.iterrows():
            date = pd.to_datetime(row["date"])
            amount = float(row["amount_gbp"]) if pd.notna(row["amount_gbp"]) else 0.0
            # Deposit = negative (money out of your pocket)
            # Withdrawal = positive (money into your pocket)
            if amount != 0:
                cash_flows.append((date, -amount))
    
    # Add final NAV as positive cash flow (what you'd get if you sold everything)
    final_date = nav.index[-1]
    final_nav = float(nav.iloc[-1])
    cash_flows.append((pd.to_datetime(final_date), final_nav))
    
    if len(cash_flows) < 1:
        return np.nan
    
    # Sort by date
    cash_flows.sort(key=lambda x: x[0])
    
    # Use first cash flow date as base (usually initial investment)
    base_date = cash_flows[0][0]
    
    # If no external flows, just calculate CAGR
    if len(cash_flows) == 1:
        days = (final_date - initial_date).days
        if days <= 0:
            return np.nan
        years = days / 365.25
        total_return = final_nav / initial_nav - 1
        return (1 + total_return) ** (1 / years) - 1
    
    # XIRR calculation with multiple cash flows
    dates = [cf[0] for cf in cash_flows]
    amounts = [cf[1] for cf in cash_flows]
    
    try:
        years = [(d - base_date).days / 365.25 for d in dates]
        
        def npv(rate, amounts, years):
            return sum(a / (1 + rate) ** t for a, t in zip(amounts, years))
        
        # Binary search for IRR
        low, high = -0.99, 10.0
        for _ in range(100):
            mid = (low + high) / 2
            val = npv(mid, amounts, years)
            if abs(val) < 0.01:
                return mid
            if val > 0:
                low = mid
            else:
                high = mid
        return mid
    except:
        return np.nan
