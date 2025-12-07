# data/generate_data.py
"""
Generate multivariate, non-linear (chaotic) time series using the Lorenz system.
Provide helper to create PyTorch DataLoaders for seq2seq forecasting.
"""
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_lorenz(t_span=(0.0, 100.0), dt=0.01, sigma=10.0, rho=28.0, beta=8/3, x0=None):
    if x0 is None:
        x0 = [1.0, 1.0, 1.0]
    def lorenz(t, s):
        x, y, z = s
        return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lorenz, [t_span[0], t_span[1]], x0, t_eval=t_eval, rtol=1e-8)
    return sol.t, sol.y.T  # (T, 3)

def create_dataloaders(X: np.ndarray, seq_len: int = 100, pred_horizon: int = 10,
                       batch_size: int = 64, split=(0.7,0.15,0.15)) -> Tuple[DataLoader,DataLoader,DataLoader]:
    """
    X: (T, D)
    Returns train, val, test DataLoaders where each sample is:
      input: (seq_len, D)
      target: (pred_horizon, D)
    """
    T, D = X.shape
    samples = []
    for i in range(T - seq_len - pred_horizon + 1):
        inp = X[i: i + seq_len]
        out = X[i + seq_len: i + seq_len + pred_horizon]
        samples.append((inp, out))
    inps = np.stack([s[0] for s in samples]).astype(np.float32)  # (N, seq_len, D)
    outs = np.stack([s[1] for s in samples]).astype(np.float32)  # (N, pred_horizon, D)
    N = inps.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    n1 = int(N * split[0]); n2 = n1 + int(N * split[1])
    def make_loader(i0, i1):
        x = torch.from_numpy(inps[idx[i0:i1]]).float()
        y = torch.from_numpy(outs[idx[i0:i1]]).float()
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
    return make_loader(0, n1), make_loader(n1, n2), make_loader(n2, N)

if __name__ == "__main__":
    t, X = generate_lorenz((0, 50), dt=0.02)
    print("Generated X shape:", X.shape)
    tr, va, te = create_dataloaders(X, seq_len=100, pred_horizon=10, batch_size=32)
    print("Train samples:", len(tr.dataset))
