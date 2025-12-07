# src/models.py
"""
Neural ODE predictor with heteroscedastic output (mean + log-variance).
Requires `torchdiffeq` for odeint. If torchdiffeq missing, raise helpful error.
"""
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except Exception as e:
    raise ImportError("torchdiffeq is required. Install with `pip install torchdiffeq`. Error: " + str(e))

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, t, x):
        # x shape: (B, D)
        return self.net(x)

class NeuralODEPredictor(nn.Module):
    """
    Takes last observed state (B, D) as initial condition, integrates forward over times t_span,
    and outputs (mean, logvar) of shape (B, pred_horizon, D).
    """
    def __init__(self, dim, pred_horizon=10, hidden=128, dropout=0.0, use_mc_dropout=False):
        super().__init__()
        self.dim = dim
        self.pred_horizon = pred_horizon
        self.odefunc = ODEFunc(dim, hidden=hidden, dropout=dropout)
        self.use_mc_dropout = use_mc_dropout
        # head maps latent state -> mean and logvar per horizon step (we apply head to each step)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim * 2)  # mean and logvar for the step
        )
    def forward(self, last_state, t_span):
        """
        last_state: (B, D)
        t_span: 1D tensor length pred_horizon+1 (including t0). We'll drop t0 after odeint.
        """
        # odeint returns (T, B, D)
        traj = odeint(self.odefunc, last_state, t_span)  # (T, B, D)
        traj = traj[1:]  # drop t0 -> (pred_horizon, B, D)
        traj = traj.permute(1, 0, 2)  # (B, pred_horizon, D)
        B, T, D = traj.shape
        head_in = traj.reshape(B * T, D)
        stats = self.head(head_in)  # (B*T, 2*D)
        stats = stats.view(B, T, 2, D)
        mean = stats[:, :, 0, :]
        logvar = stats[:, :, 1, :]
        return mean, logvar

def heteroscedastic_nll(pred_mean, pred_logvar, target):
    """
    pred_mean, pred_logvar, target: (B, T, D)
    Returns scalar NLL loss (mean over elements)
    """
    var = torch.exp(pred_logvar)
    se = (pred_mean - target) ** 2
    loss = 0.5 * (torch.log(var + 1e-8) + se / (var + 1e-8))
    return loss.mean()
