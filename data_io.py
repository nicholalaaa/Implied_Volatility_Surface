# data_io.py
from datetime import timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple
from config import MAX_EXPIRIES, MIN_T_DAYS


def get_ticker(sym: str) -> yf.Ticker:
    return yf.Ticker(sym)


def fetch_options_and_spot(sym: str) -> Tuple[pd.DataFrame, float, pd.Timestamp]:
    tk = get_ticker(sym)
    today = pd.Timestamp("today").normalize()
    expirations = tk.options
    exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=MIN_T_DAYS)]
    exp_dates = sorted(exp_dates)[:MAX_EXPIRIES]

    rows = []
    for exp_date in exp_dates:
        try:
            oc = tk.option_chain(exp_date.strftime("%Y-%m-%d"))
        except Exception:
            continue
        calls = oc.calls.assign(otype="call")
        puts  = oc.puts.assign(otype="put")
        df = pd.concat([calls, puts], ignore_index=True)
        df["expirationDate"] = exp_date
        rows.append(df)
    opt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Failed to retrieve spot price for {sym}.")
    spot = float(hist["Close"].iloc[-1])

    # standardize columns to lowercase for downstream
    if not opt.empty:
        opt = opt.rename(columns=str.lower)
        opt["mid"] = (opt["bid"] + opt["ask"]) / 2.0
        opt["expirationdate"] = pd.to_datetime(opt["expirationdate"])
        opt["daystoexpiration"] = (opt["expirationdate"] - today).dt.days
        opt["timetoexpiration"] = opt["daystoexpiration"] / 365.0
    return opt, spot, today


def filter_options(
    df: pd.DataFrame,
    spot_price: float,
    otype: str,
    strike_band=(0.80, 1.20),
    max_opts_per_expiry=400,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    lo, hi = strike_band
    out = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
    out = out[out["otype"].str.lower() == otype.lower()]
    out = out[(out["strike"] >= spot_price * lo) & (out["strike"] <= spot_price * hi)]
    # keep > 1 week to avoid zero T
    out = out[out["timetoexpiration"] > MIN_T_DAYS / 365.0]
    # downsample per expiry to keep runtime bounded
    out = out.sort_values(["expirationdate", "strike"])
    out = out.groupby("expirationdate", group_keys=False).head(max_opts_per_expiry).reset_index(drop=True)
    return out
