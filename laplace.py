import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from diffprivlib.mechanisms import Laplace
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Utility metrics
# -----------------------------
def utility_metrics(X_clean, X_dp):
    X_clean = np.atleast_2d(X_clean)
    X_dp = np.atleast_2d(X_dp)

    mse = np.mean((X_clean - X_dp) ** 2)
    mae = np.mean(np.abs(X_clean - X_dp))

    corr = np.corrcoef(X_clean[:, 0], X_dp[:, 0])[0, 1]
    ks = ks_2samp(X_clean[:, 0], X_dp[:, 0]).statistic

    noise_power = np.mean((X_clean - X_dp) ** 2)
    signal_power = np.mean(X_clean ** 2) + 1e-12
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))

    return mse, mae, corr, ks, snr_db


# -----------------------------
# Privacy metric: linkage
# -----------------------------
def linkage_rate(X_clean, X_dp):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_clean)
    idx = nn.kneighbors(X_dp, return_distance=False).ravel()
    return np.mean(idx == np.arange(len(X_clean)))


# -----------------------------
# DP Laplace release
# -----------------------------
def dp_laplace(X_scaled, epsilon, clip_B=1.0):
    X = np.atleast_2d(X_scaled)
    X_clean = np.clip(X, -clip_B, clip_B)

    delta1 = 2 * clip_B * X_clean.shape[1]
    mech = Laplace(epsilon=epsilon, sensitivity=delta1)

    X_dp = np.zeros_like(X_clean)
    for i in range(len(X_clean)):
        X_dp[i, 0] = mech.randomise(float(X_clean[i, 0]))

    return X_clean, X_dp


# -----------------------------
# Run sweep
# -----------------------------
def run_sweep(X_scaled, epsilons, clip_B=1.0):
    rows = []

    for eps in epsilons:
        X_clean, laplace_dp = dp_laplace(X_scaled, eps, clip_B)

        mse, mae, corr, ks, snr = utility_metrics(X_clean, laplace_dp)
        link = linkage_rate(X_clean, laplace_dp)

        rows.append({
            "epsilon": eps,
            "mean_corr": corr,
            "snr_db": snr,
            "mse": mse,
            "mean_ks": ks,
            "linkage_rate": link
        })

    return pd.DataFrame(rows)


# -----------------------------
# MAIN
# -----------------------------
epsilons = [0.2, 0.5, 1.0, 2.0, 3.0, 3.3, 3.8]

results = run_sweep(
    X_scaled=X_scaled,   # your existing latent
    epsilons=epsilons,
    clip_B=1.0
)

print(results)



epsilon_final = 3.3

X_clean, laplace_dp = dp_laplace(
    X_scaled=X_scaled,
    epsilon=epsilon_final,
    clip_B=1.0
)



print("Laplace DP data shape:", laplace_dp.shape)
