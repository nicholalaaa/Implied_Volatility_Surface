# config.py
GRID_RES = 50
MAX_EXPIRIES = 10
MAX_OPTS_PER_EXPIRY = 400
IV_NEWTON_ITERS = 25
IV_NEWTON_TOL = 1e-4
SIGMA0 = 0.25
VEGA_FLOOR = 1e-8
MIN_T_DAYS = 7               # ignore options expiring within a week
STRIKE_BAND = (0.80, 1.20)   # default moneyness band
