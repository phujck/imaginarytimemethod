# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import eigh_tridiagonal, expm
# from quspin.basis import spinful_fermion_basis_general
# from quspin.operators import hamiltonian
# from tqdm import tqdm
#
# # Plotting setup
# plt.rc('font', size=12)
# plt.rc('axes', titlesize=12)
# plt.rc('axes', labelsize=14)
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)
# plt.rc('legend', fontsize=12)
# plt.rc('figure', titlesize=18)
#
# # System parameters
# Lx = 1
# Ly_values = [2,3,4,5,6]
# J, U, mu = 1.0, 2.0, 0.5
# tau_values = np.linspace(0.2, 15, 10)
# num_random_states = 100  # For estimating variance from different random initial states
#
# # Hamiltonian builder
# def build_Hamiltonian(Lx, Ly, J, U, mu):
#     N_2d = Lx * Ly
#     N_up = N_down = N_2d // 2
#     s = np.arange(N_2d)
#     x = s % Lx
#     y = s // Lx
#     Tx = (x + 1) % Lx + Lx * y
#     Ty = x + Lx * ((y + 1) % Ly)
#     basis = spinful_fermion_basis_general(N_2d, Nf=(N_up, N_down), double_occupancy=True)
#     hopping_left = [[-J, i, Tx[i]] for i in range(N_2d)] + [[-J, i, Ty[i]] for i in range(N_2d)]
#     hopping_right = [[+J, i, Tx[i]] for i in range(N_2d)] + [[+J, i, Ty[i]] for i in range(N_2d)]
#     potential = [[-mu, i] for i in range(N_2d)]
#     interaction = [[U, i, i] for i in range(N_2d)]
#     static = [
#         ["+-|", hopping_left], ["-+|", hopping_right],
#         ["|+-", hopping_left], ["|-+", hopping_right],
#         ["n|", potential], ["|n", potential], ["n|n", interaction]
#     ]
#     H = hamiltonian(static, [], basis=basis, dtype=np.complex128).toarray()
#     return H, basis
#
# # Run analysis
# results = {}
#
# for Ly in Ly_values:
#     H, basis = build_Hamiltonian(Lx, Ly, J, U, mu)
#     E = np.linalg.eigvalsh(H)
#     E_min = E[0]
#     D = H.shape[0]
#
#     mean_errors = []
#     std_errors = []
#
#     for tau in tqdm(tau_values, desc=f"Ly={Ly}"):
#         errors = []
#         for _ in range(num_random_states):
#             v = np.random.choice([1, -1], size=D) + 1j * np.random.choice([1, -1], size=D)
#             v /= np.linalg.norm(v)
#             psi = v
#             evolved = expm(-tau * (H - E_min * np.eye(D))**2) @ psi
#             evolved /= np.linalg.norm(evolved)
#             energy_est = np.real(np.vdot(evolved, H @ evolved))
#             errors.append(np.abs(energy_est - E_min)/np.abs(E_min))
#
#         mean_errors.append(np.mean(errors))
#         std_errors.append(np.std(errors))
#
#     results[Ly] = {
#         "tau": tau_values,
#         "mean": np.array(mean_errors),
#         "std": np.array(std_errors),
#     }
#
# # Plotting
# for Ly in Ly_values:
#     tau_vals = results[Ly]["tau"]
#     mean = results[Ly]["mean"]
#     std = results[Ly]["std"]
#     plt.plot(tau_vals, mean, marker='o', label=f"$L_y={Ly}$")
#     plt.fill_between(tau_vals, mean - std, mean + std, alpha=0.3)
#
# plt.xlabel(r"$\tau$")
# plt.ylabel("\epsilon_{E_0}")
# plt.yscale("log")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# plt.savefig("tau_vs_random_state_error_by_Ly.pdf")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal, expm
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian
from scipy.optimize import curve_fit

# Define RMSE calculation from random state expectations
def rmse_single_state(H, E_exact, taus, num_states=1):
    D = H.shape[0]
    errors = []
    for tau in taus:
        err_sum = 0.0
        for _ in range(num_states):
            v = np.random.choice([1, -1], size=D) + 1j * np.random.choice([1, -1], size=D)
            v /= np.linalg.norm(v)
            evolved = expm(-tau * (H - E_exact * np.eye(D)) ** 2) @ v
            evolved /= np.linalg.norm(evolved)
            energy_est = np.real(np.vdot(evolved, H @ evolved))
            errors.append(np.abs(energy_est - E_exact)/np.abs(E_exact))
        #
    return np.sqrt(np.mean(errors))

# Construct 1D Fermi-Hubbard model for various system sizes
def construct_H_1dFH(L, J=1.0, U=2.0, mu=0.5):
    N_up = N_down = L // 2
    basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down), double_occupancy=True)
    hop = [[-J, i, (i+1)%L] for i in range(L)]
    hop_rev = [[J, i, (i+1)%L] for i in range(L)]
    pot = [[-mu, i] for i in range(L)]
    inter = [[U, i, i] for i in range(L)]
    static = [
        ["+-|", hop], ["-+|", hop_rev],
        ["|+-", hop], ["|-+", hop_rev],
        ["n|", pot], ["|n", pot],
        ["n|n", inter]
    ]
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128).toarray()
    return H, np.linalg.eigvalsh(H)[0], basis.Ns

# Sweep over system sizes and number of random states
L_vals = [2,3,4,5]
random_state_counts = [1, 5, 10]
taus = [10]  # Fixed tau for simplicity

results = {}
for N in random_state_counts:
    rmse_list = []
    dim_list = []
    for L in L_vals:
        H, E0, D = construct_H_1dFH(L)
        err = rmse_single_state(H, E0, taus, num_states=N)
        rmse_list.append(err)
        dim_list.append(D)
    results[N] = (dim_list, rmse_list)

# Fit RMSE ~ c / sqrt(N * D)
fit_results = {}
for N, (dims, rmses) in results.items():
    def model(D, c): return c / np.sqrt(N * np.array(D))
    popt, _ = curve_fit(model, dims, rmses)
    fit_results[N] = popt[0]

# Plotting
plt.figure(figsize=(8, 6))
for N, (dims, rmses) in results.items():
    plt.plot(dims, rmses, 'o-', label=f'{N} random states')
    D_line = np.linspace(min(dims), max(dims), 100)
    plt.plot(D_line, fit_results[N] / np.sqrt(N * D_line), '--', label=f'fit: ~1/√({N}·D)')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Hilbert Space Dimension (D)")
plt.ylabel("RMSE")
plt.title("RMSE vs Hilbert Space Dimension for 1D FH")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()
