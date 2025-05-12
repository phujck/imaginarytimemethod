import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, expm
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

# Plotting settings
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)

# System parameters
Lx, Ly = 2, 3
N_2d = Lx * Ly
N_up = N_down = N_2d // 2
J, U, mu = 1.0, 2.0, 0.5

lambda_points = 200
tau_base = 0.5
n_nodes_base = 250
k_values = [1,2,4,8]
weight_threshold = 1e-10
samples_per_overlap_list = np.unique(np.logspace(2, 5, 10, dtype=int))  # 10 to 1000 log scale
num_runs = 10

# Construct Hamiltonian
s = np.arange(N_2d)
x = s % Lx
y = s // Lx
Tx = (x + 1) % Lx + Lx * y
Ty = x + Lx * ((y + 1) % Ly)
basis_2d = spinful_fermion_basis_general(N_2d, Nf=(N_up, N_down), double_occupancy=True)
hopping_left = [[-J, i, Tx[i]] for i in range(N_2d)] + [[-J, i, Ty[i]] for i in range(N_2d)]
hopping_right = [[+J, i, Tx[i]] for i in range(N_2d)] + [[+J, i, Ty[i]] for i in range(N_2d)]
potential = [[-mu, i] for i in range(N_2d)]
interaction = [[U, i, i] for i in range(N_2d)]
static = [["+-|", hopping_left], ["-+|", hopping_right], ["|+-", hopping_left], ["|-+", hopping_right],
          ["n|", potential], ["|n", potential], ["n|n", interaction]]
H = hamiltonian(static, [], basis=basis_2d, dtype=np.complex128).toarray()
E = np.linalg.eigvalsh(H)
E_min = E[0]

# Utilities
def truncated_hermgauss(N_total):
    i = np.arange(1, N_total)
    a = np.zeros(N_total)
    b = np.sqrt(i / 2)
    x_full, V = eigh_tridiagonal(a, b)
    w_full = (V[0, :] ** 2) * np.sqrt(np.pi)
    pos_mask = x_full > 0
    return x_full[pos_mask], w_full[pos_mask] * 2

def exp_h(lam, E, tau):
    w = np.exp(-tau * (E - lam) ** 2)
    return np.sum(E * w) / np.sum(w)

def stochastic_trace(A, R):
    N = A.shape[0]
    acc = 0.0 + 0.0j
    for _ in range(R):
        v = np.random.choice([1, -1], size=N) + 1j * np.random.choice([1, -1], size=N)
        v /= np.linalg.norm(v)
        acc += np.conj(v) @ (A @ v)
    return acc / R if R > 0 else 0.0

# Main sweep loop
colors = ['blue', 'green', 'red']
sample_counts_all = []
mean_rmse_matrix = []
stderr_rmse_matrix = []
Ms_all = []

for idx, k in enumerate(k_values):
    tau = k * tau_base
    n_nodes = k * n_nodes_base
    x_k, w_k = truncated_hermgauss(n_nodes)
    t_k = np.sqrt(tau) * x_k
    M_range = [i for i, w in enumerate(w_k) if w >= weight_threshold]
    lam_values = np.linspace(-abs(E_min), abs(E_min), lambda_points)
    exps = np.array([exp_h(l, E, tau) for l in lam_values])
    M = len(M_range)
    Ms_all.append(M)

    rmse_means = []
    rmse_stds = []
    for R in samples_per_overlap_list:
        rmse_runs = []
        for _ in range(num_runs):
            z_samp, O_samp = [], []
            for i in M_range:
                U_t = expm(-2j * t_k[i] * H)
                z_samp.append(stochastic_trace(U_t, R))
                O_samp.append(stochastic_trace(U_t @ H, R))

            def approx(lam):
                num = 0.0 + 0.0j
                den = 0.0 + 0.0j
                for j, i in enumerate(M_range):
                    w, t, z, O = w_k[i], t_k[i], z_samp[j], O_samp[j]
                    phase = np.exp(2j * lam * t)
                    num += w * 2 * np.real(phase * O)
                    den += w * 2 * np.real(phase * z)
                return np.real(num / den) if abs(den) > 1e-14 else np.nan

            val = np.array([approx(lam) for lam in lam_values])
            mask = ~np.isnan(val)
            err = np.sqrt(np.mean((val[mask] - exps[mask]) ** 2))
            rmse_runs.append(err / 200)

        rmse_means.append(np.mean(rmse_runs))
        rmse_stds.append(1.96 * np.std(rmse_runs) / np.sqrt(num_runs))

    sample_counts_all.append(samples_per_overlap_list)
    mean_rmse_matrix.append(rmse_means)
    stderr_rmse_matrix.append(rmse_stds)

# Save results
np.savez_compressed(
    "rmse_vs_samples_fixed_per_overlap.npz",
    k_values=np.array(k_values),
    M_values=np.array(Ms_all),
    tau_values=np.array([k * tau_base for k in k_values]),
    sample_counts=np.array(sample_counts_all),
    mean_rmse_matrix=np.array(mean_rmse_matrix),
    stderr_rmse_matrix=np.array(stderr_rmse_matrix)
)

# Plot
plt.figure(figsize=(8, 6))
for idx, k in enumerate(k_values):
    tau = k * tau_base
    batches = sample_counts_all[idx]
    mean_rmse = mean_rmse_matrix[idx]
    stderr = stderr_rmse_matrix[idx]
    color = colors[idx]
    plt.plot(batches, mean_rmse, marker='o', color=color, label=fr"$\tau = {tau}$")
    plt.fill_between(batches, np.array(mean_rmse) - np.array(stderr),
                     np.array(mean_rmse) + np.array(stderr),
                     color=color, alpha=0.3)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Samples per Overlap $R_k$")
plt.ylabel("RMSE")
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("sampling_error_fixed_per_overlap_loglog.pdf")
plt.show()
