import numpy as np
import pandas as pd
from diffprivlib.mechanisms import Gaussian
from sklearn.preprocessing import MinMaxScaler

# 1. Scale data to [0, 1] to minimize sensitivity
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_norm)

# 2. Optimized Configuration
epsilon = 2.0        # Increased budget for better utility
delta = 1e-5         # Remains the same
sensitivity = 1.0    # Because data is now scaled to [0, 1]

# 3. Setup Mechanism
mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# 4. Apply DP to the scaled data
X_dp_scaled = np.zeros_like(X_scaled)
for i in range(X_scaled.shape[0]):
    for j in range(X_scaled.shape[1]):
        X_dp_scaled[i, j] = mechanism.randomise(X_scaled[i, j])

# 5. Inverse transform to get data back to original feature scale
X_dp = scaler.inverse_transform(X_dp_scaled)

# 6. Verify Utility
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
noise_power = np.mean((X_norm - X_dp)**2)
snr = 10 * np.log10(signal_power / noise_power)

print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f} (Target: > 0 dB)")



#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------
#New Overall MSE: 3991970.5000
#New Overall SNR (dB): -66.0119 (Target: > 0 dB) 