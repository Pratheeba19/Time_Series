# Analysis — Neural ODE vs SARIMAX

## 1. Dataset
- Generated Lorenz system with parameters: sigma=10, rho=28, beta=8/3, dt=0.02, tspan=(0,80)
- Train/Val/Test splits: 70/15/15 on sequence samples.

## 2. Model & Uncertainty
- Neural ODE: vector field modeled by a 2-layer MLP (hidden=128), integrated with `torchdiffeq.odeint`.
- Output head predicts mean and log-variance for heteroscedastic Gaussian NLL.
- Alternative uncertainty: MC Dropout (toggle dropout at inference & average predictions).

## 3. Training
- Loss: heteroscedastic NLL.
- Key hyperparams: seq_len=100, pred_horizon=10, lr=1e-3, epochs=30.

## 4. Evaluation metrics
- Accuracy: RMSE, MAE
- Uncertainty calibration: PICP (Prediction Interval Coverage Probability), MPIW (Mean Prediction Interval Width)
- Procedure: compute metrics over test windows, compare to SARIMAX univariate baseline

## 5. Results (paste numeric results here)
- Neural ODE: RMSE=..., MAE=..., PICP=..., MPIW=...
- SARIMAX baseline: RMSE=..., MAE=...

## 6. Figures to produce
- Example trajectories: true vs predicted means with 95% PI bands (for 3 random test samples)
- Calibration plot: empirical coverage vs nominal coverage
- RMSE/MAE bar chart comparing models

## 7. Interpretation & Conclusion
- Where Neural ODE outperforms (multi-step smoothness, handling continuous-time)
- Where it struggles (heavy noise, sharp discontinuities)
- Practical recommendations: use heteroscedastic loss + MC-dropout for better calibrated intervals; tune solver tolerance & latent size.

