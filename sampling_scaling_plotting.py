import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- CONFIG ---
data_path = "rmse_vs_samples_fixed_per_overlap.npz"
colors = ['blue', 'green', 'red', 'purple', 'orange']

# --- LOAD DATA ---
data = np.load(data_path, allow_pickle=True)
sample_counts_all = data["sample_counts"]
mean_rmse_matrix = data["mean_rmse_matrix"]
stderr_rmse_matrix = data["stderr_rmse_matrix"]
tau_values = data["tau_values"]

# --- FIT MODEL: RMSE ~ a * R^(-b) ---
def power_law(x, a, b):
    return a * x**(-b)

# Collect all points
all_R = np.concatenate(sample_counts_all)
all_rmse = np.concatenate(mean_rmse_matrix)
all_std = np.concatenate(stderr_rmse_matrix)

# Filter out any zero values to avoid log issues
mask = (all_R > 0) & (all_rmse > 0)
fit_R = all_R[mask]
fit_rmse = all_rmse[mask]
fit_weights = 1.0 / all_std[mask]  # Inverse variance weighting

# Perform curve fitting
popt, _ = curve_fit(power_law, fit_R, fit_rmse, sigma=1.0/fit_weights, absolute_sigma=True)
a_fit, b_fit = popt
print(f"Best fit: RMSE ≈ {a_fit:.3e} × R^(-{b_fit:.3f})")

# --- PLOT ---
plt.figure(figsize=(8, 6))

for idx, tau in enumerate(tau_values):
    R_vals = sample_counts_all[idx]
    rmse_vals = mean_rmse_matrix[idx]
    stderr_vals = stderr_rmse_matrix[idx]
    color = colors[idx % len(colors)]
    plt.plot(R_vals, rmse_vals, 'o-', label=fr"$\tau = {tau}$", color=color)
    plt.fill_between(R_vals,
                     np.array(rmse_vals) - np.array(stderr_vals),
                     np.array(rmse_vals) + np.array(stderr_vals),
                     alpha=0.3, color=color)

# Global fit curve
R_fit_plot = np.logspace(np.log10(min(all_R)), np.log10(max(all_R)), 500)
plt.plot(R_fit_plot, power_law(R_fit_plot, *popt),
         '--', color='black', linewidth=2, label=fr"Global Fit: $aR^{{-b}}$, $b={b_fit:.2f}$")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Samples per Overlap $R_k$")
plt.ylabel("$\epsilon$")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("sampling_error_with_global_fit.pdf")
plt.show()
