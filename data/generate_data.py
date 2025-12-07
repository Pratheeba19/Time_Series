import numpy as np
import pandas as pd

def generate_sine_wave(n=1000):
    x = np.linspace(0, 50, n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    return pd.DataFrame({"x": x, "y": y})

if __name__ == "__main__":
    df = generate_sine_wave()
    df.to_csv("synthetic_data.csv", index=False)
    print("Data generated: synthetic_data.csv")
