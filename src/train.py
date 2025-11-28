"""Training loop for Neural ODE time series forecasting.


Provides training, checkpointing and a deterministic evaluation pass.
"""
import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


from .models import NeuralODEModel, HeteroscedasticGaussianLoss




def train_model(X, y, epochs=50, batch_size=64, lr=1e-3, device=None, save_path=None):
"""Train Neural ODE model on (X, y) pairs.


Args:
X: np.array (N, seq_len, obs_dim) input sequences
y: np.array (N, pred_len, obs_dim) targets
Returns:
model, training_history
"""
device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
N, seq_len, obs_dim = X.shape
pred_len = y.shape[1]
# use last observed frame as x_last baseline
x_last = torch.from_numpy(X[:, -1, :]).float()
t_pred = torch.linspace(0., 1., pred_len)


dataset = TensorDataset(torch.from_numpy(X[:, -1, :]).float(), torch.from_numpy(y).float())
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = NeuralODEModel(obs_dim=obs_dim).to(device)
criterion = HeteroscedasticGaussianLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)


history = {'loss': []}
for ep in range(epochs):
model.train()
running = 0.0
for xb, yb in loader:
xb = xb.to(device)
yb = yb.to(device)
opt.zero_grad()
mu, log_var = model(t_pred.to(device), xb)
loss = criterion(yb, mu, log_var)
loss.backward()
opt.step()
running += loss.item() * xb.size(0)
avg = running / len(dataset)
history['loss'].append(avg)
print(f'Epoch {ep+1}/{epochs} loss={avg:.6f}')
if save_path is not None:
torch.save({'model_state': model.state_dict(), 'epoch': ep}, save_path)
return model, history




if __name__ == '__main__':
from data.generate_data import build_dataset
X, tX, y, tY = build_dataset(seq_len=64, pred_len=16)
model, history = train_model(X, y, epochs=10, batch_size=128)
