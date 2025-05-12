
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, expm
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian
from scipy.optimize import curve_fit
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# Set fixed model parameters
Lx, Ly = 2, 3
N_2d = Lx * Ly
N_up = N_down = N_2d // 2
J, U, mu = 1.0, 2.0, 0.5

lambda_points = 200
k_values = [1, 2, 3,4]
n_nodes_base = 250
weight_threshold = 1e-12
tau_base=0.5
# color_list = ['blue', 'green', 'red','orange']

plt.figure(figsize=(7, 5))
full_alpha,full_e =[], []
for idx, k in enumerate(k_values):
    n_nodes = k * n_nodes_base
    # tau = (2 * n_nodes) / (J**2)  # |H|^2 ~ J^2
    x_base = (J ** 2 * tau_base) / (2 * n_nodes_base)
    # a_k = (x_base) ** (1 / k - 1) * (2 * k * n_nodes_base) / (J ** 2 * tau_base)/250
    # a_k = k*(x_base)**(1-1/k)
    tau = k * tau_base
    # print(a_k)
    max_M = int(n_nodes/3)

    # Construct Hamiltonian
    s = np.arange(N_2d)
    x = s % Lx
    y = s // Lx
    Tx = (x + 1) % Lx + Lx * y
    Ty = x + Lx * ((y + 1) % Ly)

    basis_2d = spinful_fermion_basis_general(
        N_2d, Nf=(N_up, N_down), double_occupancy=True
    )

    hopping_left = [[-J, i, Tx[i]] for i in range(N_2d)] + [[-J, i, Ty[i]] for i in range(N_2d)]
    hopping_right = [[+J, i, Tx[i]] for i in range(N_2d)] + [[+J, i, Ty[i]] for i in range(N_2d)]
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

    def truncated_hermgauss(N_total):
        i = np.arange(1, N_total)
        a = np.zeros(N_total)
        b = np.sqrt(i / 2)
        x_full, V = eigh_tridiagonal(a, b)
        w_full = (V[0, :] ** 2) * np.sqrt(np.pi)
        pos_mask = x_full > 0
        x_pos = x_full[pos_mask]
        w_pos = w_full[pos_mask]
        return x_pos, w_pos * 2

    x_k, w_k = truncated_hermgauss(n_nodes)
    t_k = np.sqrt(tau) * x_k

    if weight_threshold is not None:
        M_range = [i for i, w in enumerate(w_k) if w >= weight_threshold]
        max_M = max(M_range) + 1 if M_range else 1

    z_vals, O_vals = [], []
    for t in t_k:
        U_t = expm(-2j * t * H)
        z_vals.append(np.trace(U_t))
        O_vals.append(np.trace(U_t @ H))
    z_vals = np.array(z_vals)
    O_vals = np.array(O_vals)

    lam_values = np.linspace(-abs(E_min), abs(E_min), lambda_points)
    def exp_h(lam):
        w = np.exp(-tau * (E - lam) ** 2)
        return np.sum(E * w) / np.sum(w)
    exps = np.array([exp_h(lam) for lam in lam_values])

    alpha_vals, rmse_vals = [], []
    for M in range(1, max_M, 2):
        # alpha = tau / M
        alpha = M/k
        def approx(lam):
            num = 0.0 + 0.0j
            den = 0.0 + 0.0j
            for i in range(M):
                w, t, z, O = w_k[i], t_k[i], z_vals[i], O_vals[i]
                phase = np.exp(2j * lam * t)
                num += w * 2 * np.real(phase * O)
                den += w * 2 * np.real(phase * z)
            return np.real(num / den) if abs(den) > 1e-14 else np.nan

        val = np.array([approx(lam) for lam in lam_values])
        mask = ~np.isnan(val)
        err = np.sqrt(np.mean((val[mask] - exps[mask]) ** 2))**(1/k)
        alpha_vals.append(alpha)
        rmse_vals.append(err)
        if alpha >10:
            full_alpha.append(alpha)
            full_e.append(err)

    plt.plot(alpha_vals, rmse_vals, 'o-', label=f"$s=${k}")

# def exp_fit(x, c):
#     return np.exp(-c * x)
#
#
#
# popt, _ = curve_fit(exp_fit, full_alpha, full_e)
# c_fit = popt[0]
# plt.plot(full_alpha, exp_fit(np.array(full_alpha), c_fit), '--', color='gray')]
def exp_fit(x, a, c):
    return -a*x + c
    # return a * np.exp(-c * x)

# Fit with two parameters: scale (a), decay rate (c)
popt, _ = curve_fit(exp_fit, full_alpha, np.log(full_e), p0=[1.0, 0.1])  # Initial guess
a_fit, c_fit = popt

y=np.linspace(np.min(full_alpha),np.max(full_alpha),100)
# Plot the fitted curve
plt.plot(y, np.exp(exp_fit(y, a_fit, c_fit)), '--', color='gray', label=f'$ae^{{-cx}}$, c={c_fit:.3f}')
plt.xscale("linear")
plt.yscale("log")
plt.xlabel(r"$\frac{\bar{m}}{s}$")
plt.ylabel(r"$\epsilon^{1/s}$")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("scaled_error_collapse_vs_tau.png")
plt.show()
