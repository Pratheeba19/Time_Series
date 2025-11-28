"""Generate a synthetic multivariate chaotic time series (Lorenz) with irregular sampling.


Saves sequences as numpy arrays for training / evaluation in-memory (no files written by default).
"""
from scipy.integrate import solve_ivp
import numpy as np


SEED = 42
np.random.seed(SEED)




def lorenz(t, state, sigma=10., rho=28., beta=8./3.):
x, y, z = state
return [sigma*(y-x), x*(rho - z) - y, x*y - beta*z]




def generate_lorenz_series(total_length=2000, dt=0.01, noise_std=0.5):
t_span = (0, total_length * dt)
t_eval = np.linspace(t_span[0], t_span[1], total_length)
sol = solve_ivp(lambda t, y: lorenz(t, y), t_span, [1., 1., 1.], t_eval=t_eval, rtol=1e-8)
series = sol.y.T # shape: (total_length, 3)
# add small observation noise
series += np.random.normal(scale=noise_std, size=series.shape)
return t_eval, series




def build_dataset(seq_len=64, pred_len=16, stride=4, irregular_fraction=0.2):
"""Return X (seq_len, d) and y (pred_len, d) paired arrays.
We also randomly drop some observation times to create irregular sampling.
"""
t, series = generate_lorenz_series(total_length=4000)
N, D = series.shape
X_list, tX_list, y_list, tY_list = [], [], [], []
for start in range(0, N - seq_len - pred_len, stride):
seq = series[start : start + seq_len + pred_len]
t_seq = t[start : start + seq_len + pred_len]
# make irregular by removing some observation times from the observed part
mask = np.ones(len(seq), dtype=bool)
# only drop in the input section
remove_idx = np.random.choice(range(seq_len), size=int(seq_len*irregular_fraction), replace=False)
mask[remove_idx] = False
observed = seq[mask][:seq_len - len(remove_idx)] # observed input (variable length)
t_observed = t_seq[mask][:seq_len - len(remove_idx)]
# for simplicity in this repo we'll pad inputs to fixed length and provide an observation mask
X = seq[:seq_len]
y = seq[seq_len: seq_len + pred_len]
X_list.append(X)
tX_list.append(t_seq[:seq_len])
y_list.append(y)
tY_list.append(t_seq[seq_len: seq_len + pred_len])


X = np.stack(X_list)
tX = np.stack(tX_list)
y = np.stack(y_list)
tY = np.stack(tY_list)
return X, tX, y, tY




if __name__ == '__main__':
X, tX, y, tY = build_dataset()
print('X', X.shape, 'y', y.shape)
