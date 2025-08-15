# Implied Volatility Surface

An interactive Streamlit app that pulls listed options from Yahoo Finance, computes **implied volatilities** via a fast **vectorized Newton solver (analytic vega)**, and renders a **3D surface** (strike/moneyness × time-to-expiry × IV) with Plotly.

> Compute IVs quickly, explore smiles/term structures, and visualize the surface interactively.

---

## Features

- **Fast IV solver** — NumPy-vectorized Newton–Raphson with **analytic vega** (no autograd), damped updates, vega floor, and sigma clipping.  
- **Live market data** — Options chains and spot via `yfinance`, with lightweight caching.  
- **Flexible controls** — Calls/puts, risk-free rate `r`, dividend yield `q`, strike or moneyness axis, nearest-N expiries, and max options per expiry.  
- **Smooth visualization** — Interpolated surface using `scipy.interpolate.griddata` and a 3D Plotly chart.  
- **Robust filtering** — Intrinsic-value check and minimum time-to-expiry threshold to avoid pathological points.

---

## Quick Start

### Requirements
- Python **3.10–3.12** recommended

Install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt` (reference):
```
numpy
pandas
scipy
plotly
yfinance
streamlit
```

### Run locally
```bash
streamlit run implied_volatility_surface.py
```
Open the URL shown by Streamlit (usually http://localhost:8501).

### Deploy on Streamlit Community Cloud
- Set the **Main file path** to `implied_volatility_surface.py` in the app settings.
- (Optional) Add `runtime.txt` with your Python version, e.g., `python-3.11.9`.

---

## How to Use

1. **Ticker** — enter a symbol (default: `SPY`).  
2. **Option Type** — choose Calls or Puts.  
3. **Rates** — set **Risk-Free Rate** `r` and **Dividend Yield** `q`.  
4. **Strike Filter** — choose min/max % of spot (defaults 80%–120%).  
5. **Data Limits** — cap **nearest expiries** and **options per expiry** for speed.  
6. **Y-axis** — pick **Strike** or **Moneyness**.  
7. **Grid Resolution** — control surface density (higher → smoother but slower).

The app fetches option chains ≥ 7 days out, computes mid prices, solves implied vols in a vectorized pass, drops non-converged rows, and plots the surface.

---

## Numerical Details

- **Model**: Black–Scholes with continuous dividend yield.  
- **IV Solver**: Newton–Raphson on price with **analytic vega** (no autograd).
  - Initialization `σ₀ = 0.25` (configurable in code)  
  - Tolerance `1e-4` on absolute price error; max 25 iterations  
  - Guards: intrinsic-value check, **vega floor** (`1e-8`), sigma clipping to `[1e-6, 5]`
- **Normal CDF**: `scipy.special.ndtr` (avoids `np.erf` issues across NumPy versions).  
- **Interpolation**: `scipy.interpolate.griddata` (linear). Increase grid size for a denser mesh.

> **Tip:** Use **Moneyness** instead of raw strike for a more uniform surface across expiries.

---

## Performance Tips

- Keep **nearest expiries** to ~8–12 and **options/expiry** ≤ 400 for fast solves on broad ETFs.  
- Wider strike bands and very short-dated options can slow convergence; consider excluding expiries < ~1 week.  
- Plotting cost scales with grid resolution; 40–60 points per axis is usually smooth enough.

---

## Troubleshooting

- **`KeyError: 'daystoexpiration'`** — keep derived columns all lowercase after `rename(columns=str.lower)` (e.g., `daystoexpiration`, `timetoexpiration`, `impliedvolatility`).  
- **`AttributeError: module 'numpy' has no attribute 'erf'`** — use `scipy.special.ndtr` for the normal CDF (already in this code).  
- **Streamlit Cloud: “Main module file does not exist”** — set the Main file path to `implied_volatility_surface.py` or add a thin launcher with that exact name.

---

## Roadmap

- Cross-section plots (smile at fixed maturity; term structure at fixed moneyness)  
- Optional bracketed fallback (Brent/bisection) for low-vega cases  
- Pull `r` from a Treasury curve and `q` from dividend history  
- SVI/SSVI or SABR smoothing and calibration

---

## License

MIT — see `LICENSE` (add one if missing).

---

## Acknowledgments

Built with **NumPy**, **pandas**, **SciPy**, **Plotly**, **yfinance**, and **Streamlit**. Thanks to the open-source community.
