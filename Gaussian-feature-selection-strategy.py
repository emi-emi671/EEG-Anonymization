import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from diffprivlib.mechanisms import GaussianAnalytic

# --- 0. Setup (Assuming X_norm exists) ---
# If you are testing this, ensure X_norm is defined. 
# Example: X_norm = np.random.rand(100, 10) 

# --- 1. Robust Clamping ---
# This prevents extreme outliers from exploding the sensitivity
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. PCA: Reduce Components ---
n_components = 3 
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_clamped)

# --- 3. Robust Scaling ---
# Using StandardScaler as suggested in the second snippet
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# --- 4. Calculate Real Sensitivity ---
# This ensures the noise scale matches the actual data range
local_sensitivity = np.max(X_scaled) - np.min(X_scaled)

# --- 5. Privacy Budget Allocation ---
epsilon_total = 9.99
delta = 1e-5
X_dp_scaled = np.zeros_like(X_scaled)

# --- 6. Apply DP per Component with 'Budget Pacing' ---
for j in range(n_components):
    # Logic: Give more budget (less noise) to the 1st principal component
    eps_j = 7.0 if j == 0 else (epsilon_total - 7.0) / (n_components - 1)
    
    # Use the calculated sensitivity instead of a hardcoded 1.0
    mechanism = GaussianAnalytic(epsilon=eps_j, delta=delta, sensitivity=local_sensitivity)
    
    for i in range(X_scaled.shape[0]):
        X_dp_scaled[i, j] = mechanism.randomise(X_scaled[i, j])

# --- 7. Reconstruction ---
X_dp_pca = scaler.inverse_transform(X_dp_scaled)
X_dp = pca.inverse_transform(X_dp_pca)

# --- 8. Verify Utility ---
# Measure Latent SNR (on the scaled/noisy data)
mse_latent = np.mean((X_scaled - X_dp_scaled)**2)
signal_power_latent = np.mean(X_scaled**2)
snr_latent = 10 * np.log10(signal_power_latent / mse_latent)

# Measure Global SNR (on the reconstructed data)
mse_global = np.mean((X_norm - X_dp)**2)
snr_global = 10 * np.log10(np.mean(X_norm**2) / mse_global)

print(f"New Latent SNR (dB): {snr_latent:.4f}")
print(f"Overall Global SNR (dB): {snr_global:.4f}")
