import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train(model, data, epochs=50, lr=1e-3):
    X, Y = data
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
