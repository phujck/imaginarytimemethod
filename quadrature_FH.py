#!/usr/bin/env python3
import os
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import linalg
from scipy.linalg import eigh_tridiagonal
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

def main():
    # -------------------- 1. Model and parameters --------------------
    Lx, Ly = 2, 2         # Lattice dimensions
    N_2d = Lx * Ly         # number of sites
    J = 1.0                # hopping
    U = 2.0                # on-site interaction
    mu = 0.5               # chemical potential

    # number of up/down fermions in the lattice
    N_up = N_2d // 2
    N_down = N_2d // 2

    # Imaginary time scale, number of Gauss-Hermite quadrature nodes
    tau = 5
    n_nodes = 1000
    trunc=200

    # Make directory to save data
    main_path = "2D_FH_testing_gauss/"
    os.makedirs(main_path, exist_ok=True)

    # Build a descriptive filename
    # e.g. "2DFH_gauss_Lx2_Ly2_NfUp2_NfDown2_J1.0_U2.0_mu0.5_tau5.0_n20.npz"
    out_file = (f"2DFH_gauss_Lx{Lx}_Ly{Ly}_NfUp{N_up}_NfDown{N_down}"
                f"_J{J}_U{U}_mu{mu}_tau{tau}_n{n_nodes}.npz")
    out_path = os.path.join(main_path, out_file)

    # -------------------- 2. Build the basis and Hamiltonian --------------------
    # Site index
    s = np.arange(N_2d)
    x = s % Lx
    y = s // Lx

    # Translations for the 2D lattice
    Tx = (x + 1) % Lx + Lx * y
    Ty = x + Lx * ((y + 1) % Ly)

    basis_2d = spinful_fermion_basis_general(
        N_2d, Nf=(N_up, N_down), double_occupancy=True
    )
    print(basis_2d)

    # Hamiltonian terms
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

    # Compute exact eigenspectrum for reference
    E = linalg.eigvalsh(H)
    print(E)
    print(f"Exact Hamiltonian spectrum: min={E[0]:.3f}, max={E[-1]:.3f}")

    # -------------------- 3. Gauss-Hermite nodes and weights --------------------
    # x_j, w_j = hermgauss(n_nodes)
    # pos_mask = x_j > 0
    # x_pos = x_j[pos_mask]
    # w_pos = w_j[pos_mask] * 2  # Double the weights to account for both +t and -t
    # def prune_gh_nodes(x, w, threshold=1e-10):
    #     """
    #     Returns truncated Gauss-Hermite nodes & weights.
    #     If w[k] < threshold, that node is dropped.
    #
    #     :param x: array of node locations
    #     :param w: array of corresponding weights
    #     :param threshold: minimal weight allowed
    #     :return: (x_pruned, w_pruned)
    #     """
    #     x_pruned, w_pruned = [], []
    #     for xi, wi in zip(x, w):
    #         if abs(wi) >= threshold:
    #             x_pruned.append(xi)
    #             w_pruned.append(wi)
    #     return np.array(x_pruned), np.array(w_pruned)
    #
    # x_k, w_k = prune_gh_nodes(x_pos, w_pos, threshold=1e-8)


    def truncated_hermgauss(N_total, N_keep):
        """
        Efficiently computes the smallest N_keep positive Gauss-Hermite nodes and weights
        from a total of N_total nodes using the Golub-Welsch algorithm.

        Parameters:
            N_total: total quadrature degree (large, e.g., 1000)
            N_keep: how many smallest positive nodes to return

        Returns:
            x: array of N_keep smallest positive nodes
            w: array of corresponding weights (doubled if even function assumed)
        """
        # Golub-Welsch: Build Jacobi matrix for Hermite polynomials
        i = np.arange(1, N_total)
        a = np.zeros(N_total)
        b = np.sqrt(i / 2)

        # Eigenvalues = nodes, eigenvectors give weights
        x_full, V = eigh_tridiagonal(a, b)

        # Weights are squares of first row of eigenvectors, times sqrt(pi)
        w_full = (V[0, :] ** 2) * np.sqrt(np.pi)

        # Select only positive x and corresponding weights
        pos_mask = x_full > 0
        x_pos = x_full[pos_mask]
        w_pos = w_full[pos_mask]

        # Keep only the smallest N_keep
        x_trim = x_pos[:N_keep]
        w_trim = w_pos[:N_keep] * 2  # double for symmetry

        return x_trim, w_trim

    x_k, w_k=truncated_hermgauss(n_nodes, trunc)
    print(f"After pruning: {len(x_k)} nodes remain out of {n_nodes}.")
    # We'll define times t_k = sqrt(tau)* x_k (adjust if your formula differs)
    t_k = np.sqrt(tau) * x_k
    print(w_k)

    # -------------------- 4. Build initial density matrix --------------------
    size = H.shape[0]
    rho = np.eye(size, dtype=complex)
    rho /= np.trace(rho)

    # -------------------- 5. Compute overlaps at Gauss-Hermite times --------------------
    # We'll store partition overlap (Tr[rho(t)]) and operator overlap (Tr[rho(t)*H])
    z_vals = []
    O_vals = []

    # Sort times in ascending order just for convenience
    # times = np.sort(t_k)
    ρ = np.eye(H.shape[0], dtype=complex)
    ρ /= np.trace(ρ)




    # ρs_tr.append(np.trace(temp))
    # ρHs_tr.append(np.trace(-temp.conj().T @ Ht))
    for t in t_k:
        U = linalg.expm(-2j * t * H)
        # rho_t=U.conj().T @ ρ @ U.conj().T
        # Evolve: U_f(t) = exp(-i * t * H)
        # U_f = linalg.expm(-2j * t * H)/size
        rho_t = U
        z_vals.append(np.trace(rho_t))
        O_vals.append(np.trace(rho_t@ H))

    z_vals = np.array(z_vals)
    O_vals = np.array(O_vals)

    # -------------------- 6. Save data --------------------
    np.savez(
        out_path,
        times=t_k,
        weights=w_k,
        z=z_vals,
        O=O_vals,
        spectrum=E,    # store the exact spectrum
        Lx=Lx,
        Ly=Ly,
        N_up=N_up,
        N_down=N_down,
        J=J,
        U=U,
        mu=mu,
        tau=tau,
        n_nodes=n_nodes
    )

    print(f"Data saved to {out_path}")
    print(f"times.shape={t_k.shape}, z_vals.shape={z_vals.shape}, O_vals.shape={O_vals.shape}")

if __name__ == '__main__':
    main()
