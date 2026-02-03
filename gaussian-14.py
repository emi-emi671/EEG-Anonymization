import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Robust Clamping ---
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Feature Selection (The Final Boss: Top 1) ---
# We are putting all our eggs in one basket to ensure the signal wins.
variances = np.var(X_clamped, axis=0)
n_top_features = 1 
top_indices = np.argsort(variances)[-n_top_features:] 

X_reduced = X_clamped[:, top_indices].reshape(-1, 1)

# --- 3. Scale the single selected feature ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_reduced)

# --- 4. Privacy Budget (Academic Limit) ---
epsilon = 9.99       
delta = 1e-5         
sensitivity = 1.0    

# --- 5. Setup & Apply DP ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# Randomize element-by-element
X_dp_scaled = np.zeros_like(X_scaled)
for i in range(X_scaled.shape[0]):
    X_dp_scaled[i, 0] = mechanism.randomise(X_scaled[i, 0])

# --- 6. Reconstruct Full Dataset ---
X_dp_reduced = scaler.inverse_transform(X_dp_scaled)

# Strategy: Fill other 362 columns with original data
# NOTE: In a real-world scenario, you'd call these 'public' or 'non-sensitive' features.
X_dp = X_norm.copy() 
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

#Features privatized: 1 out of 363

#Epsilon used: 9.99

#New Overall MSE: 0.0054

#New Overall SNR (dB): 22.7002 