# src/train.py
"""
Train script for Neural ODE predictor.
Saves best checkpoint to checkpoints/best.pth
"""
import os
import torch
import yaml
from data.generate_data import generate_lorenz, create_dataloaders
from src.models import NeuralODEPredictor, heteroscedastic_nll

def train(cfg):
    t, X = generate_lorenz((0, 80), dt=0.02)
    train_loader, val_loader, test_loader = create_dataloaders(X, seq_len=cfg['seq_len'],
                                                               pred_horizon=cfg['pred_horizon'],
                                                               batch_size=cfg['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralODEPredictor(dim=X.shape[1], pred_horizon=cfg['pred_horizon'],
                               hidden=cfg['hidden'], dropout=cfg['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    os.makedirs('checkpoints', exist_ok=True)
    best_val = float('inf')
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            last = xb[:, -1, :]  # (B, D)
            t_span = torch.linspace(0.0, 1.0, cfg['pred_horizon'] + 1).to(device)
            mean, logvar = model(last, t_span)
            loss = heteroscedastic_nll(mean, logvar, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0); count += xb.size(0)
        avg_train = train_loss / count

        # validation
        model.eval()
        val_loss = 0.0; vcount = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                last = xb[:, -1, :]
                t_span = torch.linspace(0.0, 1.0, cfg['pred_horizon'] + 1).to(device)
                mean, logvar = model(last, t_span)
                loss = heteroscedastic_nll(mean, logvar, yb)
                val_loss += loss.item() * xb.size(0); vcount += xb.size(0)
        avg_val = val_loss / vcount
        print(f"Epoch {epoch+1}/{cfg['epochs']}  train={avg_train:.6f}  val={avg_val:.6f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, 'checkpoints/best.pth')

if __name__ == "__main__":
    cfg = dict(seq_len=100, pred_horizon=10, batch_size=64, hidden=128, dropout=0.1, lr=1e-3, epochs=30)
    train(cfg)
