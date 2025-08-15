# bs.py
import numpy as np
from scipy.special import ndtr  # Φ(x)


def norm_cdf(x):  # vectorized standard normal CDF
    return ndtr(x)


def bs_price(S, K, T, r, q, sigma, otype="call"):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    if otype.lower() == "call":
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return S * np.exp(-q * T) * pdf * sqrtT


def intrinsic_value(S, K, T, r, q, otype="call"):
    if otype.lower() == "call":
        return np.maximum(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    return np.maximum(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
