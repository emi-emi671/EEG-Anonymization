import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic

# --- 1. PREPARE DATA ---
# Let's assume 'data' is your original numpy array
# We clip it to a known range to control sensitivity
lower_b, upper_b = np.percentile(data, [1, 99]) # Automatic clipping at 1st and 99th percentile
clipped_data = np.clip(data, lower_b, upper_b)

# --- 2. DEFINE PARAMETERS ---
# If your data is large, sensitivity is the range of a single point
sensitivity = upper_b - lower_b 
epsilon = 1.0  # Start here, increase to 2.0 or 5.0 if SNR is still low
delta = 1e-5

# --- 3. SETUP MECHANISM ---
mechanism = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

# --- 4. APPLY DP ---
# We vectorize the application if it's an array
dp_data = np.array([mechanism.randomise(x) for x in clipped_data])

# --- 5. EVALUATE ---
mse = np.mean((data - dp_data)**2)
snr = 10 * np.log10(np.mean(data**2) / mse)

print(f"New MSE: {mse:.4f}")
print(f"New SNR (dB): {snr:.4f}")