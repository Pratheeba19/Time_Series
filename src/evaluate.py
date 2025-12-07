import torch
import matplotlib.pyplot as plt

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()

    plt.plot(Y.cpu().numpy(), label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.show()
