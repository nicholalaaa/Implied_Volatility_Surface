from datetime import timedelta
from jax import grad
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st
import yfinance as yf


st.title('Implied Volatility Surface')


def black_scholes(
    S: float,
    K: float,
    T: int,
    r: float,
    sigma: float,
    q: float = 0,
    otype: str = "call"
) -> float:

    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    if otype == "call":
        return S * jnp.exp(-q * T) * jnorm.cdf(d1, 0, 1) - K * jnp.exp(-r * T) * jnorm.cdf(d2, 0, 1)
    elif otype == "put":
        return -S * jnp.exp(-q * T) * jnorm.cdf(-d1, 0, 1) + K * jnp.exp(-r * T) * jnorm.cdf(-d2, 0, 1)


def solve_for_iv(
    S: float,
    K: float,
    T: int,
    r: float,
    price: float,
    sigma_guess: float = 0.5,
    q: float = 0,
    otype: str = "call",
    N_iter: int = 20,
    epsilon: float = 0.001,
    verbose: bool = False
) -> float:

    if T <= 0 or price <= 0:
        return jnp.nan

    converged = False
    # 1. Make a guess for the volatility
    sigma = sigma_guess
    for i in range(N_iter):

        # 2. Calculate the loss function
        loss_val = loss_func(S, K, T, r, sigma, price, q, otype=otype)

        if verbose:
            print("\nIteration: ", i)
            print("Current Error in Theoretical vs Market Price:")
            print(loss_val)

        # 3. Check if the loss is less than the tolerance, $\epsilon$

        # If yes, STOP!
        if abs(loss_val) < epsilon:
            converged = True
            break

        # If no, CONTINUE to step 4
        else:

            # 4. Calculate the gradient of the loss function
            loss_grad_val = loss_grad(S, K, T, r, sigma, price, q=q, otype=otype)

            if verbose:
                print("Gradient:", loss_grad_val)

            # 5. Update the volatility using the Newton-Raphson formula
            sigma = sigma - loss_val / loss_grad_val

            if verbose:
                print("New sigma: ", sigma)

    if not converged:
        return jnp.nan

    return sigma


def loss_func(
    S: float,
    K: float,
    T: int,
    r: float,
    sigma_guess: float,
    price: float,
    q: float = 0,
    otype: str = "call"
) -> float:

    # Price with the GUESS for the volatility
    theoretical_price = black_scholes(S, K, T, r, sigma_guess, q=q, otype=otype)

    # Actual price
    market_price = price

    # Loss is the difference between the theoretical price and the actual price
    # We want to MINIMIZE this loss!
    return theoretical_price - market_price


loss_grad = grad(loss_func, argnums=4)


st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the parameters for the Black-Scholes model.')

otype = st.sidebar.selectbox(
    'Select Option Type:',
    ('Call', 'Put')
)

r = st.sidebar.number_input(
    'Risk-Free Rate (e.g., 0.015 for 1.5%)',
    value=0.015,
    format="%.4f"
)

q = st.sidebar.number_input(
    'Dividend Yield (e.g., 0.013 for 1.3%)',
    value=0.013,
    format="%.4f"
)

st.sidebar.header('Visualization Parameters')
y_axis_option = st.sidebar.selectbox(
    'Select Y-axis:',
    ('Strike Price ($)', 'Moneyness')
)

st.sidebar.header('Ticker Symbol')
ticker_symbol = st.sidebar.text_input(
    'Enter Ticker Symbol',
    value='SPY',
    max_chars=10
).upper()

st.sidebar.header('Strike Price Filter Parameters')

min_strike_pct = st.sidebar.number_input(
    'Minimum Strike Price (% of Spot Price)',
    min_value=10.0,
    max_value=499.0,
    value=80.0,
    step=1.0,
    format="%.1f"
)

max_strike_pct = st.sidebar.number_input(
    'Maximum Strike Price (% of Spot Price)',
    min_value=11.0,
    max_value=500.0,
    value=120.0,
    step=1.0,
    format="%.1f"
)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()


ticker = yf.Ticker(ticker_symbol)

today = pd.Timestamp('today').normalize()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'Error fetching options for {ticker_symbol}: {e}')
    st.stop()

exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]

if not exp_dates:
    st.error(f'No available option expiration dates for {ticker_symbol}.')
else:
    option_data = []

    for exp_date in exp_dates:
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            if otype == 'Call':
                options = opt_chain.calls
            elif otype == 'Put':
                options = opt_chain.puts
        except Exception as e:
            st.warning(f'Failed to fetch option chain for {exp_date.date()}: {e}')
            continue

        options = options[(options['bid'] > 0) & (options['ask'] > 0)]

        for index, row in options.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })

    if not option_data:
        st.error('No option data available after filtering.')
    else:
        options_df = pd.DataFrame(option_data)

        try:
            spot_history = ticker.history(period='5d')
            if spot_history.empty:
                st.error(f'Failed to retrieve spot price data for {ticker_symbol}.')
                st.stop()
            else:
                spot_price = spot_history['Close'].iloc[-1]
        except Exception as e:
            st.error(f'An error occurred while fetching spot price data: {e}')
            st.stop()

        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

        options_df = options_df[
            (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
            (options_df['strike'] <= spot_price * (max_strike_pct / 100))
        ]

        options_df.reset_index(drop=True, inplace=True)

        with st.spinner('Calculating implied volatility...'):
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: solve_for_iv(
                    price=row['mid'],
                    S=spot_price,
                    K=row['strike'],
                    T=row['timeToExpiration'],
                    r=r,
                    q=q
                ), axis=1
            )
        options_df.dropna(subset=['impliedVolatility'], inplace=True)

        options_df['impliedVolatility'] *= 100

        options_df.sort_values('strike', inplace=True)

        options_df['moneyness'] = options_df['strike'] / spot_price

        if y_axis_option == 'Strike Price ($)':
            Y = options_df['strike']
            y_label = 'Strike Price ($)'
        else:
            Y = options_df['moneyness']
            y_label = 'Moneyness (Strike / Spot)'

        X = options_df['timeToExpiration'].values
        Z = options_df['impliedVolatility'].values

        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T, K = np.meshgrid(ti, ki)

        Zi = griddata((X, Y), Z, (T, K), method='linear')

        Zi = np.ma.array(Zi, mask=np.isnan(Zi))

        fig = go.Figure(data=[go.Surface(
            x=T, y=K, z=Zi,
            colorscale='Viridis',
            colorbar_title='Implied Volatility (%)'
        )])

        fig.update_layout(
            title=f'Implied Volatility Surface for {ticker_symbol} Options',
            scene=dict(
                xaxis_title='Time to Expiration (years)',
                yaxis_title=y_label,
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=900,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        st.plotly_chart(fig)
