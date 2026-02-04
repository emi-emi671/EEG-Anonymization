# --- 1. Use a more robust Scaler ---
# StandardScaler keeps the signal centered, which is better for Gaussian noise
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# --- 2. Calculate Real Sensitivity ---
# Instead of 1.0, we use the actual max range of the scaled data
# This is a critical step for Security/Privacy proofs
local_sensitivity = np.max(X_scaled) - np.min(X_scaled)

# --- 3. Apply Noise with 'Budget Pacing' ---
X_dp_scaled = np.zeros_like(X_scaled)
for j in range(n_components):
    # We give the first component (the biggest signal) the most budget
    # Epsilon for this component = 7.0, others share the rest
    eps_j = 7.0 if j == 0 else (9.99 - 7.0) / (n_components - 1)
    
    # We use GaussianAnalytic with the ACTUAL sensitivity of the data
    mechanism = GaussianAnalytic(epsilon=eps_j, delta=1e-5, sensitivity=local_sensitivity)
    
    for i in range(X_scaled.shape[0]):
        X_dp_scaled[i, j] = mechanism.randomise(X_scaled[i, j])

# --- 4. Re-measure Latent SNR ---
mse_latent = np.mean((X_scaled - X_dp_scaled)**2)
signal_power_latent = np.mean(X_scaled**2)
snr_latent = 10 * np.log10(signal_power_latent / mse_latent)

print(f"New Latent SNR (dB): {snr_latent:.4f}")