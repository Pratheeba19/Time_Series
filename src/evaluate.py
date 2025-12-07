# src/evaluate.py
"""
Evaluation script: loads checkpoint, runs Neural ODE on test loader, computes RMSE/MAE and PICP/MPIW,
and computes SARIMAX baseline for comparison.
"""
import torch
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.generate_data import generate_lorenz, create_dataloaders
from src.models import NeuralODEPredictor
from src.benchmarks import sarimax_forecast

def rmse(a, b):
    return sqrt(mean_squared_error(a.reshape(-1), b.reshape(-1)))

def picp_mpiw(pred_mean, pred_var, target, alpha=0.05):
    # Using normal approx: z for (1-alpha)
    z = 1.96  # ~95% interval
    std = np.sqrt(pred_var + 1e-8)
    lower = pred_mean - z * std
    upper = pred_mean + z * std
    inside = (target >= lower) & (target <= upper)
    picp = inside.mean()
    mpiw = (upper - lower).mean()
    return picp, mpiw

def run_neuralode(checkpoint='checkpoints/best.pth', seq_len=100, pred_horizon=10):
    ck = torch.load(checkpoint, map_location='cpu')
    cfg = ck.get('cfg', {})
    t, X = generate_lorenz((0, 80), dt=0.02)
    _, _, test_loader = create_dataloaders(X, seq_len=seq_len, pred_horizon=pred_horizon, batch_size=128)
    model = NeuralODEPredictor(dim=X.shape[1], pred_horizon=pred_horizon,
                               hidden=cfg.get('hidden', 128), dropout=cfg.get('dropout', 0.0))
    model.load_state_dict(ck['model_state'])
    model.eval()
    all_mean, all_var, all_y = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            last = xb[:, -1, :]
            t_span = torch.linspace(0.0, 1.0, pred_horizon + 1)
            mean, logvar = model(last, t_span)
            all_mean.append(mean.numpy())
            all_var.append(np.exp(logvar.numpy()))
            all_y.append(yb.numpy())
    pred_mean = np.concatenate(all_mean, axis=0)  # (N, H, D)
    pred_var = np.concatenate(all_var, axis=0)
    y = np.concatenate(all_y, axis=0)
    metrics = {
        'rmse': rmse(pred_mean, y),
        'mae': mean_absolute_error(pred_mean.reshape(-1), y.reshape(-1))
    }
    picp_val, mpiw_val = picp_mpiw(pred_mean, pred_var, y)
    metrics.update({'picp': float(picp_val), 'mpiw': float(mpiw_val)})
    return metrics, pred_mean, pred_var, y

def run_sarimax_baseline(pred_horizon=10):
    # Fit SARIMAX on first dimension only (for a simple baseline)
    t, X = generate_lorenz((0, 80), dt=0.02)
    series = X[:, 0]
    train_n = int(len(series) * 0.85)
    train = series[:train_n]
    test = series[train_n:]
    # forecast in chunks to compare with seq->horizon windows, but here we give aggregate metrics on the test series
    preds = sarimax_forecast(train, steps=len(test))
    return {'rmse': rmse(preds, test), 'mae': mean_absolute_error(preds, test)}

if __name__ == "__main__":
    ne_metrics, pred_mean, pred_var, y = run_neuralode()
    sar_metrics = run_sarimax_baseline()
    print("Neural ODE metrics:", ne_metrics)
    print("SARIMAX baseline (univariate):", sar_metrics)
