import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, expm
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

# Plotting setup
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)

# Parameters
Lx = 2
Ly_values = [1,2,3]
mu = 0.5
U = 2.0
tau_base=0.5
n_nodes_base = 250
k_values=[2,4,8,16,32,64]
# tau_values=k*tau_base
weight_threshold = 1e-6

def truncated_hermgauss(N_total):
    i = np.arange(1, N_total)
    a = np.zeros(N_total)
    b = np.sqrt(i / 2)
    x_full, V = eigh_tridiagonal(a, b)
    w_full = (V[0, :] ** 2) * np.sqrt(np.pi)
    pos_mask = x_full > 0
    return x_full[pos_mask], w_full[pos_mask] * 2

results = {}

for Ly in Ly_values:
    N_2d = Lx * Ly
    N_up = N_down = N_2d // 2
    s = np.arange(N_2d)
    x = s % Lx
    y = s // Lx
    Tx = (x + 1) % Lx + Lx * y
    Ty = x + Lx * ((y + 1) % Ly)

    basis_2d = spinful_fermion_basis_general(N_2d, Nf=(N_up, N_down), double_occupancy=True)
    hopping_left = [[-1.0, i, Tx[i]] for i in range(N_2d)] + [[-1.0, i, Ty[i]] for i in range(N_2d)]
    hopping_right = [[+1.0, i, Tx[i]] for i in range(N_2d)] + [[+1.0, i, Ty[i]] for i in range(N_2d)]
    potential = [[-mu, i] for i in range(N_2d)]
    interaction = [[U, i, i] for i in range(N_2d)]

    static = [
        ["+-|", hopping_left],
        ["-+|", hopping_right],
        ["|+-", hopping_left],
        ["|-+", hopping_right],
        ["n|", potential],
        ["|n", potential],
        ["n|n", interaction]
    ]

    H = hamiltonian(static, [], basis=basis_2d, dtype=np.float64).toarray()
    E = np.linalg.eigvalsh(H)
    E_min = E[0]


    epsilons = []
    for k_i in k_values:
        print(k_i)
        n_nodes = k_i * n_nodes_base
        x_k, w_k = truncated_hermgauss(n_nodes)
        t_k = np.sqrt(k_i*tau_base) * x_k
        z_vals = []
        O_vals = []
        for t in t_k:
            U_t = expm(-2j * t * H)
            z_vals.append(np.trace(U_t))
            O_vals.append(np.trace(U_t @ H))

        z_vals = np.array(z_vals)
        O_vals = np.array(O_vals)

        errors = []
        for E_j in E:
            num = 0.0 + 0.0j
            den = 0.0 + 0.0j
            for i in range(len(w_k)):
                if w_k[i] < weight_threshold:
                    continue
                w, t, z, O = w_k[i], t_k[i], z_vals[i], O_vals[i]
                phase = np.exp(2j * E_j * t)
                num += w * 2 * np.real(phase * O)
                den += w * 2 * np.real(phase * z)
            approx_val = np.real(num / den) if abs(den) > 1e-14 else np.nan
            errors.append((approx_val - E_j) ** 2)
        epsilons.append(np.sqrt(np.nanmean(errors)))

    results[Ly] = (tau_base*np.array(k_values), epsilons)

# Plotting
plt.figure(figsize=(8, 6))
for Ly in Ly_values:
    tau_vals, eps_vals = results[Ly]
    plt.plot(tau_vals, np.array(eps_vals), marker='o', label=f"$L_y={Ly}$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\epsilon_E(\tau)$")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("epsilon_E_vs_tau_for_various_Ly.pdf")
plt.show()
