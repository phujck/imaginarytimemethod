#!/usr/bin/env python3
import os
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import linalg

# QuSpin import for 1D Ising
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def main():
    # -------------------- 1. Model and parameters --------------------
    L = 6  # System size (number of spins)
    J = 3.0  # Ising coupling for sigma^z_j sigma^z_{j+1}
    g = 1.0  # Transverse field strength (coefficient in front of sigma^x_j)
    bc = "periodic"  # boundary condition: "open" or "periodic"

    # Imaginary time scale, number of Gauss-Hermite quadrature nodes
    tau = 5.0
    n_nodes = 10

    # Make directory to save data
    main_path = "1D_Ising_gauss/"
    os.makedirs(main_path, exist_ok=True)

    # Build a descriptive filename
    # e.g. "1DIsing_gauss_L6_J1.0_g1.0_bcopen_tau10.0_n10.npz"
    out_file = (f"1DIsing_gauss_L{L}_J{J}_g{g}_bc{bc}_tau{tau}_n{n_nodes}.npz")
    out_path = os.path.join(main_path, out_file)

    # -------------------- 2. Build the basis and Hamiltonian --------------------
    # 2a. QuSpin basis for spin-1/2 chain
    basis = spin_basis_1d(L=L, pauli=True)  # using Pauli spin operators

    # 2b. Hamiltonian lists
    # TFI model: H = -J * sum_j (sigmaz_j * sigmaz_{j+1}) - g * sum_j (sigmax_j)
    # The negative sign is a convention; adapt as you like.
    # If bc="periodic", you’d also include the edge coupling (L-1,0).
    # We'll do open boundary in this example by default: sum_{j=0 to L-2}

    zz_interaction = []
    x_field = []

    # For open BC, connect j to j+1 for j in [0..L-2]
    for j in range(L - 1 if bc == "open" else L):
        jp1 = (j + 1) % L  # if bc=="periodic", this wraps around
        zz_interaction.append([+(-J), j, jp1])  # "zz" type with coefficient -J

    for j in range(L):
        x_field.append([+(-g), j])  # "x" type with coefficient -g

    static_list = [
        ["zz", zz_interaction],  # ising part
        ["x", x_field],  # transverse field
    ]

    # 2c. Build Hamiltonian in dense matrix form
    H = hamiltonian(static_list, [], basis=basis, dtype=np.float64).toarray()

    # -------------------- 3. Compute exact eigenspectrum for reference --------------------
    E = linalg.eigvalsh(H)
    print(f"Exact Hamiltonian spectrum: min={E[0]:.3f}, max={E[-1]:.3f}")

    # -------------------- 4. Gauss-Hermite nodes and weights --------------------
    x_k, w_k = hermgauss(n_nodes)
    # We'll define times t_k = sqrt(tau)* x_k (adjust if your formula differs)
    t_k = np.sqrt(tau) * x_k

    # -------------------- 5. Build initial 'density matrix' (here just an identity for example) --------------------
    size = H.shape[0]
    rho = np.eye(size, dtype=complex)
    rho /= np.trace(rho)

    # -------------------- 6. Compute overlaps at Gauss-Hermite times --------------------
    # We'll store partition overlap: Tr[rho(t)], and operator overlap: Tr[rho(t)*H]
    z_vals = []
    O_vals = []

    # Sort times in ascending order for convenience
    times = np.sort(t_k)

    # Imag-time-like evolution operator => use exp(-2i * t * H) if you want purely real part
    # or adapt for your ITQDE approach.
    for t in times:
        U_f = linalg.expm(-2j * t * H)
        # If you want an unnormalized operator:
        #   rho_t = U_f @ rho @ U_f.conj().T
        # For demonstration, just store the operator itself:
        rho_t = U_f

        z_vals.append(np.trace(rho_t))
        O_vals.append(np.trace(rho_t @ H))

    z_vals = np.array(z_vals)
    O_vals = np.array(O_vals)

    # -------------------- 7. Save data --------------------
    np.savez(
        out_path,
        times=times,
        weights=w_k,
        z=z_vals,
        O=O_vals,
        spectrum=E,
        L=L,
        J=J,
        g=g,
        bc=bc,
        tau=tau,
        n_nodes=n_nodes
    )

    print(f"Data saved to {out_path}")
    print(f"times.shape={times.shape}, z_vals.shape={z_vals.shape}, O_vals.shape={O_vals.shape}")


if __name__ == '__main__':
    main()
