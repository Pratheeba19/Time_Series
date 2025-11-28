# Neural ODE vs Traditional Time Series Benchmarks
This document provides the analysis for:
- Dataset generation
- Neural ODE forecasting
- Uncertainty quantification using MC Dropout
- Baseline comparison with SARIMAX
- Metrics: RMSE, MAE, PICP, MPIW

---

## 1. Load Dataset & Model

```python
import torch
import numpy as np
from src.models import NeuralODEModel
from src.evaluate import mc_dropout_predict, predictive_interval_metrics
from data.generate_data import build_dataset

X, tX, y, tY = build_dataset(seq_len=64, pred_len=16)

model = NeuralODEModel(obs_dim=X.shape[-1])
model.load_state_dict(torch.load("checkpoint.pt")["model_state"])
model.eval()

t_pred = torch.linspace(0., 1., y.shape[1])
samples = mc_dropout_predict(
    model,
    X[:, -1, :],       # last observed timestep
    t_pred,
    mc_samples=50      # number of stochastic passes
)
samples.shape  # (50, batch, pred_len, D)

metrics = predictive_interval_metrics(y, samples)
print(metrics)

import matplotlib.pyplot as plt

i = 0   # sample index
mean_pred = samples.mean(axis=0)[i]
true = y[i]

plt.figure(figsize=(6,4))
plt.plot(true[:, 0], label="True")
plt.plot(mean_pred[:, 0], label="Neural ODE Mean Prediction")
plt.title("Neural ODE Forecast — Dimension 0")
plt.legend()
plt.show()

from src.benchmarks import sarimax_forecast

full_series = X.reshape(-1, X.shape[-1])   # flatten batch/sequence
sarimax_pred = sarimax_forecast(full_series, pred_len=16)

from src.evaluate import rmse, mae

print("SARIMAX RMSE:", rmse(y.mean(axis=0), sarimax_pred))
print("SARIMAX MAE:", mae(y.mean(axis=0), sarimax_pred))
