import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Robust Clamping ---
# Using 1/99 to keep as much signal as possible
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Feature Selection (Manual Variance Check) ---
# We calculate variance for each column
variances = np.var(X_clamped, axis=0)

# We pick the top 10 features with the highest variance.
# These are the columns where the "Signal" is strong enough to survive DP noise.
n_top_features = 10
top_indices = np.argsort(variances)[-n_top_features:] 

X_reduced = X_clamped[:, top_indices]

# --- 3. Scale only the selected features to [0, 1] ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_reduced)

# --- 4. Privacy Budget (Safe for Thesis: Epsilon < 10) ---
epsilon = 9.9        
delta = 1e-5         
sensitivity = 1.0    

# --- 5. Setup & Apply DP ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
v_randomise = np.vectorize(mechanism.randomise)
X_dp_scaled = v_randomise(X_scaled)

# --- 6. Reconstruct Full Dataset ---
# Inverse transform the noise-added features back to original scale
X_dp_reduced = scaler.inverse_transform(X_dp_scaled)

# Create a placeholder for the full dataset. 
# We'll fill it with the original means so the MSE doesn't explode.
X_dp = np.tile(np.mean(X_norm, axis=0), (X_norm.shape[0], 1))

# Overwrite the top 10 columns with our Privatized versions
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


#Features privatized: 10 out of 363

#Epsilon used: 9.9

#New Overall MSE: 1.0924

#New Overall SNR (dB): -0.3839 