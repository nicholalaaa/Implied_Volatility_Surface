from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.special import ndtr
import streamlit as st
import yfinance as yf


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # erf-based normal CDF (fast, no SciPy)
    return ndtr(x)


def bs_price(S, K, T, r, q, sigma, otype="call"):
    """
    Vectorized Black–Scholes price (arrays broadcastable).
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    if otype.lower() == "call":
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    else:
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    """
    Vectorized Black–Scholes vega (dPrice/dSigma).
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return S * np.exp(-q * T) * pdf * sqrtT


def intrinsic_value(S, K, T, r, q, otype="call"):
    """
    Present-value intrinsic under Black–Scholes (with r,q).
    """
    if otype.lower() == "call":
        return np.maximum(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        return np.maximum(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))


def implied_vol_newton_vectorized(
    S, K, T, r, q, price, otype="call",
    max_iter=25, tol=1e-4, sigma0=0.25
):
    """
    Vectorized Newton–Raphson IV solver.
    Inputs: 1D arrays (or broadcastable) of equal length.
    Returns: sigma array with np.nan where not converged.
    """
    n = price.shape[0]
    sigma = np.full(n, sigma0, dtype=float)

    # Validity mask: positive time, positive price, above intrinsic (slack)
    intr = intrinsic_value(S, K, T, r, q, otype=otype)
    valid = (T > 0) & (price > 0) & (price >= 0.999 * intr)

    # Iterate Newton only on valid points
    for _ in range(max_iter):
        idx = valid & np.isfinite(sigma)
        if not np.any(idx):
            break

        theo = bs_price(S[idx], K[idx], T[idx], r, q, sigma[idx], otype=otype)
        diff = theo - price[idx]
        done = np.abs(diff) < tol
        if np.all(done):
            valid[idx] = False
            continue

        vega = bs_vega(S[idx], K[idx], T[idx], r, q, sigma[idx])
        safe = vega > 1e-8
        step = np.zeros_like(diff)
        step[safe] = diff[safe] / vega[safe]

        # Damped update + clipping for stability
        sigma_new = sigma[idx] - step
        sigma[idx] = np.clip(sigma_new, 1e-6, 5.0)

        # Mark converged ones as done
        theo2 = bs_price(S[idx], K[idx], T[idx], r, q, sigma[idx], otype=otype)
        valid[idx] = np.abs(theo2 - price[idx]) >= tol

    # Non-converged -> NaN
    sigma[~np.isfinite(sigma)] = np.nan
    return sigma


st.set_page_config(page_title="Implied Volatility Surface", layout="wide")
st.title("Implied Volatility Surface")

st.sidebar.header("Model Parameters")
otype = st.sidebar.selectbox("Select Option Type:", ("Call", "Put"))
r = st.sidebar.number_input("Risk-Free Rate (e.g., 0.015 = 1.5%)", value=0.015, format="%.4f")
q = st.sidebar.number_input("Dividend Yield (e.g., 0.013 = 1.3%)", value=0.013, format="%.4f")

st.sidebar.header("Visualization")
y_axis_option = st.sidebar.selectbox("Select Y-axis:", ("Strike Price ($)", "Moneyness"))
grid_res = st.sidebar.slider("Grid resolution (per axis)", min_value=20, max_value=80, value=50, step=5)

st.sidebar.header("Ticker & Data Limits")
ticker_symbol = st.sidebar.text_input("Ticker", value="SPY", max_chars=10).upper()
max_expiries = st.sidebar.slider("Max expiries (nearest)", min_value=3, max_value=20, value=10, step=1)
max_opts_per_expiry = st.sidebar.slider("Max options per expiry", min_value=50, max_value=1000, value=400, step=50)

st.sidebar.header("Strike Filter (% of Spot)")
min_strike_pct = st.sidebar.number_input("Minimum %", min_value=10.0, max_value=499.0, value=80.0, step=1.0, format="%.1f")
max_strike_pct = st.sidebar.number_input("Maximum %", min_value=11.0, max_value=500.0, value=120.0, step=1.0, format="%.1f")
if min_strike_pct >= max_strike_pct:
    st.sidebar.error("Minimum percentage must be less than maximum percentage.")
    st.stop()


@st.cache_resource(show_spinner=False)
def cached_ticker(sym: str):
    return yf.Ticker(sym)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_and_spot(sym: str):
    tk = cached_ticker(sym)
    today = pd.Timestamp("today").normalize()
    try:
        expirations = tk.options
    except Exception as e:
        raise RuntimeError(f"Error fetching options for {sym}: {e}")

    # keep expiries at least 7 days out; take nearest N
    exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]
    exp_dates = sorted(exp_dates)[:max_expiries]

    rows = []
    for exp_date in exp_dates:
        try:
            oc = tk.option_chain(exp_date.strftime("%Y-%m-%d"))
        except Exception as e:
            # skip bad expiry
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

    return opt, spot, today


try:
    raw_options, spot_price, today = fetch_options_and_spot(ticker_symbol)
except Exception as e:
    st.error(str(e))
    st.stop()

if raw_options.empty:
    st.error("No option chains available after fetching.")
    st.stop()

df = raw_options.rename(columns=str.lower).copy()
df = df[(df["bid"] > 0) & (df["ask"] > 0)]
df["mid"] = (df["bid"] + df["ask"]) / 2.0
df["daystoexpiration"] = (pd.to_datetime(df["expirationdate"]) - today).dt.days
df["timetoexpiration"] = df["daystoexpiration"] / 365.0

# user filters
df = df[df["otype"].str.lower() == otype.lower()]
df = df[(df["strike"] >= spot_price * (min_strike_pct / 100.0)) &
        (df["strike"] <= spot_price * (max_strike_pct / 100.0))]
df = df[df["timetoexpiration"] > 7/365.0]  # keep > ~1 week to avoid near-zero T

# Downsample per expiry for speed (optional)
df["expirationDate"] = pd.to_datetime(df["expirationdate"])
df = df.sort_values(["expirationDate", "strike"])
df = df.groupby("expirationDate", group_keys=False).head(max_opts_per_expiry).reset_index(drop=True)

if df.empty:
    st.error("No option rows after filters/downsampling.")
    st.stop()

with st.spinner("Calculating implied volatility…"):
    n = len(df)
    S = np.full(n, spot_price, dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["timetoexpiration"].to_numpy(dtype=float)
    P = df["mid"].to_numpy(dtype=float)

    iv = implied_vol_newton_vectorized(S, K, T, r, q, P, otype=otype.lower(), max_iter=25, tol=1e-4, sigma0=0.25)
    df = df.assign(impliedVolatility=iv)
    df = df[np.isfinite(df["impliedVolatility"])].copy()

if df.empty:
    st.error("No converged implied volatilities (try wider strike band or farther expiries).")
    st.stop()

df["impliedVolatility"] *= 100.0  # percent
df["moneyness"] = df["strike"] / spot_price


if y_axis_option == "Strike Price ($)":
    Y = df["strike"].to_numpy(dtype=float)
    y_label = "Strike Price ($)"
else:
    Y = df["moneyness"].to_numpy(dtype=float)
    y_label = "Moneyness (Strike / Spot)"

X = df["timetoexpiration"].to_numpy(dtype=float)
Z = df["impliedvolatility"].to_numpy(dtype=float)

if len(X) < 10:
    st.error("Too few points to build a surface; broaden filters or increase expiries.")
    st.stop()

ti = np.linspace(X.min(), X.max(), grid_res)
yi = np.linspace(Y.min(), Y.max(), grid_res)
Tg, Yg = np.meshgrid(ti, yi)

Zi = griddata((X, Y), Z, (Tg, Yg), method="linear")
Zi = np.ma.array(Zi, mask=np.isnan(Zi))

fig = go.Figure(data=[go.Surface(
    x=Tg, y=Yg, z=Zi,
    colorscale="Viridis",
    colorbar_title="Implied Volatility (%)"
)])
fig.update_layout(
    title=f"Implied Volatility Surface — {ticker_symbol} ({otype.title()}s)",
    scene=dict(
        xaxis_title="Time to Expiration (years)",
        yaxis_title=y_label,
        zaxis_title="Implied Volatility (%)"
    ),
    autosize=False, width=900, height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(fig, use_container_width=True)

# Small summary
st.caption(
    f"Points used: {len(df):,} | Expiries: {df['expirationDate'].nunique():,} | "
    f"Spot: {spot_price:.2f} | r={r:.4f}, q={q:.4f}"
)
