import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import pandas as pd 


def plot_forecasts(df, pred, test_df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sales"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_df.index,
            y=pred.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    fig.update_layout(
        title="Actual vs Forecast", xaxis_title="Date", yaxis_title="Value"
    )
    fig.show()

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=df.index[-30:],
            y=df["sales"].iloc[-30:],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4"),
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=test_df.index,
            y=pred.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    fig1.update_layout(
        title="Actual vs Forecast - Reduced", xaxis_title="Date", yaxis_title="Value"
    )
    fig1.show()


def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    return np.sqrt(np.mean((log_pred - log_true) ** 2))


def plot_acf_plotly(time_series: pd.Series, nlags: int = 100):

    lags = list(range(nlags))
    acf_values = sm.tsa.acf(time_series, nlags=nlags)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=lags,
                y=acf_values,
                mode="markers+lines",
                marker=dict(
                    color="#2F97FF", size=15, line=dict(color="black", width=1)
                ),
            )
        ]
    )

    for lag, acf in zip(lags, acf_values):
        fig.add_shape(
            type="line", x0=lag, y0=0, x1=lag, y1=acf, line=dict(color="#007DA0")
        )

    fig.update_layout(
        title="ACF",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        xaxis=dict(tickmode="linear"),
    )

    return fig
