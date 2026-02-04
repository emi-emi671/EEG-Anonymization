import numpy as np
import matplotlib.pyplot as plt

# Flatten in case shape is (n_samples, 1)
orig = X_scaled.flatten()
dp   = X_dp_scaled.flatten()

# Use a small subset so bars are readable
n_show = 30
orig_small = orig[:n_show]
dp_small   = dp[:n_show]

indices = np.arange(n_show)

############# bar chart for original signals ######


plt.figure(figsize=(9, 4))
plt.bar(indices, orig_small)
plt.title("Original Data (X_scaled) – Bar Chart")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.tight_layout()
plt.show()



############# bar chart for dp protected signals ######
plt.figure(figsize=(9, 4))
plt.bar(indices, dp_small)
plt.title("DP-Protected Data (X_dp_scaled) – Bar Chart")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.tight_layout()
plt.show()


############# bar chart for combined signals signals ######




plt.figure(figsize=(10, 4))
plt.bar(indices - width/2, orig_small, width, label="Original")
plt.bar(indices + width/2, dp_small,   width, label="DP Protected")

plt.title("Original vs DP Data – Bar Comparison")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()