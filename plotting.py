# plotting.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata


def build_surface(df: pd.DataFrame, y_axis="moneyness", grid_res=50):
    if y_axis == "strike":
        Y = df["strike"].to_numpy(float)
        y_label = "Strike Price ($)"
    else:
        if "moneyness" not in df.columns:
            raise ValueError("Missing 'moneyness' column.")
        Y = df["moneyness"].to_numpy(float)
        y_label = "Moneyness (Strike / Spot)"

    X = df["timetoexpiration"].to_numpy(float)
    Z = df["impliedvolatility"].to_numpy(float)

    if len(X) < 10:
        raise ValueError("Too few points to build a surface.")

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
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title=y_label,
            zaxis_title="Implied Volatility (%)"
        ),
        margin=dict(l=65, r=50, b=65, t=60),
        autosize=True,
        height=800
    )
    return fig
