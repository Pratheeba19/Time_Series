"""Neural ODE model and utilities.


def forward(self, t, x):
return self.net(x)




class NeuralODEModel(nn.Module):
"""Encoder-decoder Neural ODE.


Workflow:
1. Encode last observed state(s) into latent initial condition z0
2. Integrate ODE dz/dt = f(t, z) over prediction horizon
3. Decode latent trajectory into observed-space predictions (mean and log-variance)
"""


def __init__(self, obs_dim: int, latent_dim: int = 32, ode_hidden: int = 64, dropout: float = 0.1):
super().__init__()
self.obs_dim = obs_dim
self.latent_dim = latent_dim
self.encoder = nn.Sequential(
nn.Linear(obs_dim, 64),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(64, latent_dim),
)
self.odefunc = ODEFunc(latent_dim, hidden=ode_hidden, dropout=dropout)
self.decoder = nn.Sequential(
nn.Linear(latent_dim, 64),
nn.ReLU(),
nn.Linear(64, obs_dim * 2), # predict mean and log-variance
)


def forward(self, t_pred: torch.Tensor, x_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
"""Forward pass.


Args:
t_pred: tensor of shape (T_pred,) containing time points to predict at (absolute or relative)
x_last: tensor of shape (B, obs_dim) the last observed state for each batch


Returns:
mean: (B, T_pred, obs_dim)
log_var: (B, T_pred, obs_dim)
"""
# encode
z0 = self.encoder(x_last) # (B, latent_dim)
# integrate
# odeint expects shape (T, B, latent_dim)
z_t = odeint(self.odefunc, z0, t_pred) # (T_pred, B, latent_dim)
z_t = z_t.permute(1, 0, 2) # (B, T_pred, latent_dim)
# decode
dec = self.decoder(z_t) # (B, T_pred, obs_dim*2)
dec = dec.view(dec.shape[0], dec.shape[1], self.obs_dim, 2)
mean = dec[..., 0]
log_var = dec[..., 1]
return mean, log_var




class HeteroscedasticGaussianLoss(nn.Module):
"""Negative log-likelihood for Gaussian with predicted variance.


Minimizes: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
"""


def __init__(self):
super().__init__()


def forward(self, y, mu, log_var):
return 0.5 * (log_var + ((y - mu) ** 2) / torch.exp(log_var)).mean()
