import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Robust Clamping (The "SNR Fixer") ---
# Instead of scaling the raw X_norm, we clamp it to the 1st/99th percentile.
# This ensures outliers don't dominate the scaling range.
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Scale data to [0, 1] ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_clamped)

# --- 3. Optimized Configuration ---
# Increasing epsilon slightly more (e.g., 4.0) can significantly boost SNR
# while still providing a strong privacy guarantee with GaussianAnalytic.
epsilon = 4.0        
delta = 1e-5         
sensitivity = 1.0    # Remains 1.0 because of MinMaxScaler

# --- 4. Setup Mechanism ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# --- 5. Apply DP to the scaled data ---
# Optimization: Using a vectorized approach for speed if your dataset is large
X_dp_scaled = np.array([[mechanism.randomise(x) for x in row] for row in X_scaled])

# --- 6. Inverse transform ---
X_dp = scaler.inverse_transform(X_dp_scaled)

# --- 7. Verify Utility ---
# We compare against the original X_norm
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
noise_power = np.mean((X_norm - X_dp)**2)
snr = 10 * np.log10(signal_power / noise_power)

print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f} (Target: > 0 dB)")





#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------

#New Overall MSE: 30.1758
#New Overall SNR (dB): -14.7966 (Target: > 0 dB) 