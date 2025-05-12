import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit import Aer
from qiskit.opflow import X, Z, I, MatrixEvolution
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import Parameter
from qiskit import transpile, assemble
from scipy import linalg
from scipy.special import binom
import matplotlib.pyplot as plt
from qutip import *
import itertools as it
import copy
import stomp_functions as sf
from qiskit.quantum_info import random_clifford, DensityMatrix
from qiskit.providers.fake_provider import FakeBelem
import time

# Load data and calculate average
ovlps_file = "ris_ovlps_N=4_numsteps=1000_dtau=0.000_sim=aer.npz"
H_ovlps_file = "ris_H_ovlps_N=4_numsteps=1000_dtau=0.000_sim=aer.npz"

ovlps = np.load(ovlps_file, allow_pickle=True)['arr_0'].item()
H_ovlps = np.load(H_ovlps_file, allow_pickle=True)['arr_0'].item()

# Sum up arrays from the runs
total_ovlps = 0
total_H_ovlps = 0
for key in ovlps:
    total_ovlps += ovlps[key]
    total_H_ovlps += H_ovlps[key]

# Now get the average arrays
total_ovlps /= len(ovlps.keys())
total_H_ovlps /= len(H_ovlps.keys())

# Need to define beta for plotting to compare with other results
beta = 1
num_steps = 1000
betas, dbeta = np.linspace(0, beta, num_steps, retstep=True)

# Plot to see what things look like
plt.figure(1)
plt.plot(betas, ovlps['seed=88'].real, 'o-', label='Seed=87')
plt.plot(betas, total_ovlps.real, 'o-', label='Average')
plt.xlabel("$\\beta$")
plt.ylabel("Re(Overlaps)")
plt.legend()

plt.figure(2)
plt.plot(betas, ovlps['seed=88'].imag, 'o-', label='Seed=87')
plt.plot(betas, total_ovlps.imag, 'o-', label='Average')
plt.xlabel("$\\beta$")
plt.ylabel("Im(Overlaps)")
plt.legend()

plt.figure(3)
plt.plot(betas, H_ovlps['seed=88'].real, 'o-', label='Seed=87')
plt.plot(betas, total_H_ovlps.real, 'o-', label='Average')
plt.xlabel("$\\beta$")
plt.ylabel("Re(<E>)")
plt.legend()

plt.figure(4)
plt.plot(betas, H_ovlps['seed=88'].imag, 'o-', label='Seed=87')
plt.plot(betas, total_H_ovlps.imag, 'o-', label='Average')
plt.xlabel("$\\beta$")
plt.ylabel("Im(<E>)")
plt.legend()

# Need to construct H here
N = 2
g = 2
J = 1
# Construct operator lists and Hamiltonian
z_ops, x_ops = sf.construct_op_lists(N)
H = 0
for i in range(N - 1):
    H += -J * z_ops[i] @ z_ops[i + 1]
    for j in range(N):
        H += -J * g * x_ops[i]

# Get eigenvalues and vectors from H
E, V = linalg.eigh(H.to_matrix())

# Calculate partition stuff using one run
colors = plt.cm.jet(np.linspace(0, 1, len(E[::2])+2))
l = 0
plt.figure(5)
for e in E[::2]:
    if e < 0:
        f = -e
    else:
        f = -e

    if l == 0 or l == len(E[::2]) - 1:
        plt.plot(betas[2::2], sf.alt_partition_calc(ovlps['seed=88'], H_ovlps['seed=88'], num_steps,
                                                 f, dbeta)[1][1:] - f, color=colors[l + 1],
                 label='$\\lambda = E_' + str(l) + '$')
    plt.plot(betas[2::2], sf.alt_partition_calc(ovlps['seed=88'], H_ovlps['seed=88'], num_steps,
                                             f, dbeta)[1][1:] - f, color=colors[l])

    plt.plot(betas, -e * np.ones(len(betas)), '-.', color=colors[l + 1])
    l += 1
plt.xlabel("$\\beta$")
plt.ylabel("E")
plt.legend()
#plt.title("Qiskit - Seed=87")

# Now use the averaged values
l = 0
plt.figure(6)
for e in E[::2]:
    if e < 0:
        f = -e / 2
    else:
        f = -e / 2

    if l == 0 or l == len(E[::2]) - 1:
        plt.plot(betas[2::2], sf.partition_calc(total_ovlps, total_H_ovlps, num_steps,
                                                 f, dbeta)[1][1:] - f, color=colors[l + 1],
                 label='$\\lambda = E_' + str(l) + '$')
    plt.plot(betas[2::2], sf.partition_calc(total_ovlps, total_H_ovlps, num_steps,
                                             f, dbeta)[1][1:] - f, color=colors[l])

    plt.plot(betas, -e * np.ones(len(betas)), '-.', color=colors[l + 1])
    l += 1
plt.xlabel("$\\beta$")
plt.ylabel("E")
plt.legend()

# Now use the averaged values
l = 0
plt.figure(8)
for e in E[::2]:
    if e < 0:
        f = -e
    else:
        f = -e

    if l == 0 or l == len(E[::2]) - 1:
        plt.plot(betas[2::2], sf.alt_partition_calc(total_ovlps, total_H_ovlps, num_steps,
                                                 f, dbeta)[1][1:] - f, color=colors[l + 1],
                 label='$\\lambda = E_' + str(l) + '$')
    plt.plot(betas[2::2], sf.alt_partition_calc(total_ovlps, total_H_ovlps, num_steps,
                                             f, dbeta)[1][1:] - f, color=colors[l])

    plt.plot(betas, -e * np.ones(len(betas)), '-.', color=colors[l + 1])
    l += 1
plt.xlabel("$\\beta$")
plt.ylabel("E")
plt.legend()
#plt.title("Qiskit - Average (200 runs) - 200 steps")

plt.figure(7)
sf.plot_lambda_sweep(total_ovlps, total_H_ovlps, num_steps, dbeta, E)

plt.show()