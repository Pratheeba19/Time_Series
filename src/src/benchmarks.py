from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def run_arima(series, order=(3,1,1)):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=50)
    return np.array(forecast)
