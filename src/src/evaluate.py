"""Evaluation utilities: RMSE, MAE, PICP and MPIW for predictive intervals.


PICP: Prediction Interval Coverage Probability
MPIW: Mean Prediction Interval Width


Also includes an inference wrapper that performs MC dropout sampling.
"""
import numpy as np
import torch




def rmse(y, yhat):
return np.sqrt(((y - yhat) ** 2).mean())




def mae(y, yhat):
return np.abs(y - yhat).mean()




def mc_dropout_predict(model, x_last, t_pred, mc_samples=20, device=None):
"""Perform MC dropout by keeping dropout layers active during inference.


Returns posterior samples of shape (mc_samples, B, T, D)
"""
if device is None:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.train() # important: keep dropout on
x_last = torch.from_numpy(x_last).float().to(device)
t_pred = t_pred.to(device)
samples = []
with torch.no_grad():
for _ in range(mc_samples):
mu, log_var = model(t_pred, x_last)
# sample from predicted Gaussian
std = torch.exp(0.5 * log_var)
eps = torch.randn_like(mu)
s = mu + eps * std
samples.append(s.cpu().numpy())
return np.stack(samples, axis=0)




def predictive_interval_metrics(y_true, samples, alpha=0.05):
"""Compute PICP and MPIW for provided samples.


Args:
y_true: (B, T, D)
samples: (S, B, T, D)
Returns:
dict with rmse/mae/PICP/MPIW
"""
S = samples.shape[0]
lower = np.quantile(samples, alpha/2., axis=0)
upper = np.quantil
