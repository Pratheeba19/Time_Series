# src/benchmarks.py
"""
Provide strong traditional benchmark: SARIMAX (via statsmodels).
We fit SARIMAX on each variable separately (or choose primary var) and produce multi-step forecasts.
"""
import numpy as np
import statsmodels.api as sm
from typing import Tuple

def sarimax_forecast(series: np.ndarray, steps: int, order=(2,0,2), seasonal_order=(0,0,0,0)) -> np.ndarray:
    """
    series: 1D array of historical values
    steps: forecast horizon
    returns: forecast array shape (steps,)
    """
    model = sm.tsa.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res.forecast(steps=steps)

# Optional: Prophet example (user must install prophet)
# from prophet import Prophet
# def prophet_forecast(df, steps):
#     # df must be DataFrame with columns ['ds', 'y']
#     m = Prophet()
#     m.fit(df)
#     future = m.make_future_dataframe(periods=steps, freq="D")
#     fcst = m.predict(future)
#     return fcst['yhat'][-steps:].values
