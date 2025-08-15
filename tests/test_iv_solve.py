# tests/test_iv_solve.py
import numpy as np
from bs import bs_price
from iv_solve import implied_vol_newton


def test_iv_recovers_sigma_on_synthetic_prices():
    np.random.seed(0)
    n = 100
    S = np.full(n, 100.0)
    K = np.linspace(80, 120, n)
    T = np.linspace(0.1, 1.0, n)
    r, q = 0.01, 0.00
    true_sigma = 0.25
    price = bs_price(S, K, T, r, q, true_sigma, otype="call")
    iv = implied_vol_newton(S, K, T, r, q, price, otype="call", max_iter=50, tol=1e-6, sigma0=0.2)
    assert np.nanmax(np.abs(iv - true_sigma)) < 1e-3
