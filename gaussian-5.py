import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Aggressive Clamping ---
# Tightening the bounds to 5/95 helps the SNR by ignoring more extreme outliers
lower_bounds = np.percentile(X_norm, 5, axis=0)
upper_bounds = np.percentile(X_norm, 95, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Scale data to [0, 1] ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_clamped)

# --- 3. Higher Privacy Budget (Epsilon) ---
# To reach SNR > 0, we need to increase epsilon. 
# 8.0 is a common upper bound for high-utility DP applications.
epsilon = 8.0        
delta = 1e-5         
sensitivity = 1.0    

# --- 4. Setup Mechanism ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# --- 5. Apply DP (Optimized) ---
# Instead of a nested loop, we use np.vectorize for efficiency
v_randomise = np.vectorize(mechanism.randomise)
X_dp_scaled = v_randomise(X_scaled)

# --- 6. Inverse transform ---
X_dp = scaler.inverse_transform(X_dp_scaled)

# --- 7. Verify Utility ---
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
noise_power = np.mean((X_norm - X_dp)**2)
snr = 10 * np.log10(signal_power / noise_power)

print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f}")

if snr < 0:
    print("SNR is still negative. Consider: Increasing epsilon further or reducing the number of features.")


#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------

#New Overall MSE: 2.6400

#New Overall SNR (dB): -4.2160

#SNR is still negative. Consider: Increasing epsilon further or reducing the number of features. 
