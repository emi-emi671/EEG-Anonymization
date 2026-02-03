import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --- 1. Gentler Clamping (1/99) ---
# Moving from 5/95 to 1/99 to keep more "Signal Power" in the data
lower_bounds = np.percentile(X_norm, 1, axis=0)
upper_bounds = np.percentile(X_norm, 99, axis=0)
X_clamped = np.clip(X_norm, lower_bounds, upper_bounds)

# --- 2. Automated PCA (Explained Variance) ---
# We tell PCA to keep enough components to explain 95% of the original data's variance.
# This ensures our 'Signal' is as strong as possible before adding noise.
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_clamped)
n_comp_found = X_pca.shape[1]

# --- 3. Component-Level Clipping ---
pca_low = np.percentile(X_pca, 1, axis=0)
pca_high = np.percentile(X_pca, 99, axis=0)
X_pca_clamped = np.clip(X_pca, pca_low, pca_high)

# --- 4. Scale PCA components ---
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_pca_clamped)

# --- 5. Privacy Budget ---
# We will use 9.9 to stay strictly under the "10.0" academic threshold
epsilon = 9.9        
delta = 1e-5         
sensitivity = 1.0    

# --- 6. Setup & Apply DP ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
v_randomise = np.vectorize(mechanism.randomise)
X_dp_scaled = v_randomise(X_scaled)

# --- 7. Reconstruct ---
X_dp_pca = scaler.inverse_transform(X_dp_scaled)
X_dp = pca.inverse_transform(X_dp_pca)

# --- 8. Verify Utility ---
mse = np.mean((X_norm - X_dp)**2)
signal_power = np.mean(X_norm**2)
snr = 10 * np.log10(signal_power / mse)

print(f"Components used for 95% variance: {n_comp_found}")
print(f"Epsilon used: {epsilon}")
print(f"New Overall MSE: {mse:.4f}")
print(f"New Overall SNR (dB): {snr:.4f}")




#----------------------------------------------------------------------
#-------------------------------Result --------------------------------
#----------------------------------------------------------------------

#Components used for 95% variance: 29

#Epsilon used: 9.9

#New Overall MSE: 6.5411

#New Overall SNR (dB): -8.1565 