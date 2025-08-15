# iv_solve.py
import numpy as np
from bs import bs_price, bs_vega, intrinsic_value
from config import IV_NEWTON_ITERS, IV_NEWTON_TOL, SIGMA0, VEGA_FLOOR


def implied_vol_newton(
    S, K, T, r, q, price, otype="call",
    max_iter=IV_NEWTON_ITERS, tol=IV_NEWTON_TOL, sigma0=SIGMA0
):
    """
    Vectorized Newton–Raphson implied volatility.
    Inputs: 1D numpy arrays (broadcastable) of equal length.
    Returns: sigma array; np.nan where invalid or not converged.
    """
    n = price.shape[0]
    sigma = np.full(n, sigma0, dtype=float)

    intr = intrinsic_value(S, K, T, r, q, otype=otype)
    valid = (T > 0) & (price > 0) & (price >= 0.999 * intr)

    for _ in range(max_iter):
        idx = valid & np.isfinite(sigma)
        if not np.any(idx):
            break
        theo = bs_price(S[idx], K[idx], T[idx], r, q, sigma[idx], otype=otype)
        diff = theo - price[idx]
        if np.all(np.abs(diff) < tol):
            valid[idx] = False
            continue
        vega = bs_vega(S[idx], K[idx], T[idx], r, q, sigma[idx])
        safe = vega > VEGA_FLOOR
        step = np.zeros_like(diff)
        step[safe] = diff[safe] / vega[safe]
        sigma[idx] = np.clip(sigma[idx] - step, 1e-6, 5.0)
        # continue until all residuals < tol or max_iter exhausted

    sigma[~np.isfinite(sigma)] = np.nan
    return sigma
