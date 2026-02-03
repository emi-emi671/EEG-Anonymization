import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler

# --- 1. Refined Clamping ---
# We use 2/98 percentiles instead of 5/95. 
# This preserves more of your actual "signal" while still cutting off the crazy outliers.
lower_bounds = np.percentile(X_norm, 2, axis=0)
upper_bounds = np.percentile(X_norm, 98, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Scale data to [0, 1] ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_clamped)

# --- 3. Privacy Budget Adjustment ---
# Moving to 12.0 to push the SNR over the 0dB threshold.
# In many industry applications, epsilon values between 8 and 14 are used 
# for complex datasets where element-wise noise is required.
epsilon = 12.0        
delta = 1e-5         
sensitivity = 1.0    

# --- 4. Setup Mechanism ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# --- 5. Apply DP (Optimized) ---
v_randomise = np.vectorize(mechanism.randomise)
X_dp_scaled = v_randomise(X_scaled)

# --- 6. Inverse transform ---
X_dp = scaler.inverse_transform(X_dp_scaled)

# --- 7. Verify Utility ---
# We calculate MSE and SNR against the original X_norm
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
noise_power = mse # Equivalent to mean((X_norm - X_dp)**2)
snr = 10 * np.log10(signal_power / noise_power)

print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f}")

if snr < 0:
    print("SNR is still negative. Suggestion: Try epsilon = 15.0 or check if X_norm has many zero-valued columns.")
else:
    print("Target Achieved: SNR is positive! The signal is now stronger than the noise.")
    
    
    
    
#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------


#New Overall MSE: 0.9570

#New Overall SNR (dB): 0.1908

#Target Achieved: SNR is positive! The signal is now stronger than the noise. 
