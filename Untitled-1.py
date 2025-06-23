
# Standard Library


import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)



import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # or ERROR
import warnings

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statsmodels
from statsmodels.graphics.tsaplots import month_plot, quarter_plot, plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Scikit-learn
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import ParameterGrid

# Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Parallel Processing
from joblib import Parallel, delayed

# Progress Bar
from tqdm.notebook import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")


def plot_predictions(train, test):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train["ds"][-30:],
            y=train["y"][-30:],
            name="Train",
            line=dict(color="#2A61EB"),
        )
    )

    test_connected = pd.concat([train.tail(1), test])

    fig.add_trace(
        go.Scatter(
            x=test_connected["ds"], y=test_connected["y"], name="Test", line=dict(color="#2456D2")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_connected["ds"],
            y=test_connected["yhat"],
            name="Forecast",
            line=dict(color="#159A20", dash = 'dash'),
        )
    )

    return fig 


df = pd.read_parquet("train.parquet")
df = df.drop(columns=["city", "state", "type", "transactions"])
df = df[[col for col in df.columns if not col.startswith("holiday")]]
df = df.sort_values(by=["store_nbr", "family", "date"])

df["dcoilwtico_interpolate"] = df["dcoilwtico_interpolate"].bfill()
df = df.rename(columns={"dcoilwtico_interpolate": "oil", "sales": "y", "date": "ds"})


periods = 16


holiday_df = pd.read_csv("holidays_events.csv")
holiday_df = holiday_df[["date", "description"]]
holiday_df = holiday_df.rename(columns={"date": "ds", "description": "holiday"})
holiday_df["lower_window"] = -2
holiday_df["upper_window"] = 0


x = df[["store_nbr", "family"]].drop_duplicates().sample(1)
store_nbr = x["store_nbr"].iloc[0]

family = x["family"].iloc[0]



df_sample = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)].sort_values('ds')
train, test = df_sample.iloc[:-periods], df_sample.iloc[-periods:]


def run_train(train, test, params):
    m = Prophet(
        seasonality_mode=params["seasonality_mode"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holiday_df,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
    )
    m.add_regressor("onpromotion")
    m.add_regressor("oil")
    # m.add_regressor("month")
    # m.add_regressor("day_of_month")
    # m.add_regressor("day_of_year")
    # m.add_regressor("week_of_month")
    # m.add_regressor("day_of_week")
    # m.add_regressor("year")
    # m.add_regressor("is_wknd")
    # m.add_regressor("quarter")
    # m.add_regressor("is_month_start")
    # m.add_regressor("is_month_end")
    # m.add_regressor("is_quarter_start")
    # m.add_regressor("is_quarter_end")
    # m.add_regressor("is_year_start")
    # m.add_regressor("is_year_end")

    m.fit(train)

    future = m.make_future_dataframe(periods=periods, include_history=False)
    future = future.merge(
        test[
            [
                "ds",
                "onpromotion",
                "oil",
                "month",
                "day_of_month",
                "day_of_year",
                "week_of_month",
                "day_of_week",
                "year",
                "is_wknd",
                "quarter",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "is_year_start",
                "is_year_end",
            ]
        ],
        on="ds",
        how="left",
    )

    forecast = m.predict(future)[["ds", "yhat"]]
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    test_score = test[["ds", "y"]].merge(forecast, on=["ds"])

    rmsle = round(root_mean_squared_log_error(test_score["y"], test_score["yhat"]), 2)
    rmse = round(root_mean_squared_error(test_score["y"], test_score["yhat"]), 2)

    # df_cv = cross_validation(
    #     model=m,
    #     period="30 days",
    #     initial="1200 days",
    #     horizon="16 days",
    #     parallel="threads",
    # )

    # performance = performance_metrics(df_cv).iloc[:, 1:].mean().T.round(2).to_dict()

    # df_p = performance_metrics(df_cv)
    # rmsle = df_p["rmsle"].mean()
 
    return {
            'rmse': rmse, 
            'rmsle': rmsle, 
            # 'cross_validation_result': performance
        }
    


import optuna 


def objective(trial):
    params = {
        "changepoint_prior_scale": trial.suggest_loguniform(
            "changepoint_prior_scale", 0.001, 0.5
        ),
        "seasonality_prior_scale": trial.suggest_loguniform(
            "seasonality_prior_scale", 0.001, 15
        ),
        "holidays_prior_scale": trial.suggest_loguniform(
            "holidays_prior_scale", 0.001, 15.0
        ),
        "seasonality_mode": trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        ),
        "growth": trial.suggest_categorical("growth", ["linear", "logistic"]),
    }

    result = run_train(train, test, params) 

    return result['rmsle']





study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)


# import joblib 
# from dask.distributed import Client, LocalCluster

# cluster = LocalCluster(n_workers=4, processes=True)  # Adjust based on CPU
# # Start local cluster
# client = Client(cluster)  # Uses all available cores

# print(client)

# with joblib.parallel_backend('dask'): 
#     study.optimize(objective, n_trials=100, show_progress_bar=True)

print(study.best_params, study.best_value)






