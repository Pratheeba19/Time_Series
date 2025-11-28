"""
Benchmark models for time series forecasting.
Includes:
- SARIMAX baseline (multivariate handled dimension-wise)
- Optional Prophet baseline (commented out)

Author: Your Name
"""

import numpy as np
import statsmodels.api as sm

# Optional: Prophet baseline
# from prophet import Prophet
# import pandas as pd


def sarimax_forecast(train_series, pred_len=16, order=(2, 0, 2)):
    """
    Train SARIMAX models per dimension and forecast forward.

    Args:
        train_series (np.ndarray): shape (N, D) multivariate series.
        pred_len (int): forecast horizon.
        order (tuple): ARIMA order.

    Returns:
        np.ndarray: forecast array of shape (pred_len, D)
    """
    N, D = train_series.shape
    forecasts = np.zeros((pred_len, D))

    for d in range(D):
        model = sm.tsa.SARIMAX(
            train_series[:, d],
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)

        predicted = result.get_forecast(pred_len)
        forecasts[:, d] = predicted.predicted_mean

    return forecasts


def sarimax_batch_forecast(X_last, full_series, pred_len=16):
    """
    Wrapper for batch SARIMAX forecasting.

    Args:
        X_last: unused (kept for API consistency)
        full_series (np.ndarray): historical data for fitting.
        pred_len (int): forecast horizon.

    Returns:
        np.ndarray: forecast array of shape (pred_len, D)
    """
    return sarimax_forecast(full_series, pred_len=pred_len)


# Optional Prophet version
"""
def prophet_forecast(train_series, pred_len=16):
    N, D = train_series.shape
    forecasts = np.zeros((pred_len, D))

    for d in range(D):
        df = pd.DataFrame({
            "ds": pd.date_range(start="2000-01-01", periods=N, freq="D"),
            "y": train_series[:, d]
        })

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=pred_len)
        output = model.predict(future)

        forecasts[:, d] = output["yhat"].values[-pred_len:]

    return forecasts
"""
