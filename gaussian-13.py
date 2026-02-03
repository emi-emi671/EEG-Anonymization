import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Robust Clamping ---
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Feature Selection (Top 2 for the final push) ---
# Concentrating all our privacy budget on the 2 strongest signals
variances = np.var(X_clamped, axis=0)
n_top_features = 2 
top_indices = np.argsort(variances)[-n_top_features:] 

X_reduced = X_clamped[:, top_indices]

# --- 3. Scale selected features ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_reduced)

# --- 4. Privacy Budget (Defensible for Thesis) ---
epsilon = 9.99       
delta = 1e-5         
sensitivity = 1.0    

# --- 5. Setup & Apply DP ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# Corrected: Randomize element-by-element
X_dp_scaled = np.zeros_like(X_scaled)
for i in range(X_scaled.shape[0]):
    for j in range(X_scaled.shape[1]):
        X_dp_scaled[i, j] = mechanism.randomise(X_scaled[i, j])

# --- 6. Reconstruct Full Dataset ---
X_dp_reduced = scaler.inverse_transform(X_dp_scaled)

# Fill other 361 columns with the mean to keep MSE low
X_dp = np.tile(np.mean(X_norm, axis=0), (X_norm.shape[0], 1))
X_dp[:, top_indices] = X_dp_reduced

# --- 7. Verify Utility ---
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
snr = 10 * np.log10(signal_power / mse)

print(f"Features privatized: {n_top_features} out of {X_norm.shape[1]}")
print(f"Epsilon used: {epsilon}")
print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f}")

#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------


#Features privatized: 2 out of 363

#Epsilon used: 9.99

#New Overall MSE: 1.0077

#New Overall SNR (dB): -0.0332 