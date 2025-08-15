# app.py
import numpy as np
import pandas as pd
import streamlit as st
from config import (
    GRID_RES, MAX_EXPIRIES, MAX_OPTS_PER_EXPIRY, STRIKE_BAND,
    IV_NEWTON_ITERS, IV_NEWTON_TOL, SIGMA0
)
from data_io import fetch_options_and_spot, filter_options
from iv_solve import implied_vol_newton
from plotting import build_surface


st.set_page_config(page_title="Implied Volatility Surface", layout="wide")
st.title("Implied Volatility Surface")

# Sidebar
st.sidebar.header("Model Parameters")
otype = st.sidebar.selectbox("Option Type", ("Call", "Put"))
r = st.sidebar.number_input("Risk-Free Rate (e.g., 0.015=1.5%)", value=0.015, format="%.4f")
q = st.sidebar.number_input("Dividend Yield (e.g., 0.013=1.3%)", value=0.013, format="%.4f")

st.sidebar.header("Visualization")
y_axis_option = st.sidebar.selectbox("Y-axis", ("Moneyness", "Strike"))
grid_res = st.sidebar.slider("Grid resolution", 20, 80, GRID_RES, 5)

st.sidebar.header("Ticker & Limits")
ticker_symbol = st.sidebar.text_input("Ticker", value="SPY", max_chars=10).upper()
max_expiries = st.sidebar.slider("Max expiries (nearest)", 3, 20, MAX_EXPIRIES, 1)
max_opts_per_expiry = st.sidebar.slider("Max options per expiry", 50, 1000, MAX_OPTS_PER_EXPIRY, 50)

st.sidebar.header("Strike Filter (% of Spot)")
min_pct = st.sidebar.number_input("Minimum %", 10.0, 499.0, STRIKE_BAND[0]*100, 1.0, format="%.1f")
max_pct = st.sidebar.number_input("Maximum %", 11.0, 500.0, STRIKE_BAND[1]*100, 1.0, format="%.1f")
if min_pct >= max_pct:
    st.sidebar.error("Minimum must be less than maximum.")
    st.stop()

# Fetch data
with st.spinner("Fetching option chains…"):
    opt, spot, today = fetch_options_and_spot(ticker_symbol)

if opt.empty:
    st.error("No option chains available.")
    st.stop()

# Filter
df = filter_options(
    opt, spot, otype,
    strike_band=(min_pct/100.0, max_pct/100.0),
    max_opts_per_expiry=max_opts_per_expiry
)

if df.empty:
    st.error("No option rows after filters.")
    st.stop()

# Solve IVs (vectorized)
with st.spinner("Solving implied volatilities…"):
    n = len(df)
    S = np.full(n, spot, float)
    K = df["strike"].to_numpy(float)
    T = df["timetoexpiration"].to_numpy(float)
    P = df["mid"].to_numpy(float)
    iv = implied_vol_newton(S, K, T, r, q, P, otype=otype.lower(),
                            max_iter=IV_NEWTON_ITERS, tol=IV_NEWTON_TOL, sigma0=SIGMA0)
    df = df.assign(impliedvolatility=iv)
    df = df[np.isfinite(df["impliedvolatility"])].copy()

if df.empty:
    st.error("No converged implied volatilities. Try widening filters.")
    st.stop()

# Add moneyness and plot
df["moneyness"] = df["strike"] / spot
try:
    fig = build_surface(df, y_axis=("strike" if y_axis_option == "Strike" else "moneyness"), grid_res=grid_res)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(str(e))

st.caption(
    f"Points used: {len(df):,} | Expiries: {df['expirationdate'].nunique():,} | "
    f"Spot: {spot:.2f} | r={r:.4f}, q={q:.4f}"
)
