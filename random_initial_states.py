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


def test_cliffords(num_qubits=2, num_seeds=100):
    """
    function for testing that running circuits with randomly chosen clifford gates does mimic the mixed state
    if enough are selected
    :param num_qubits:  number of qubits to use
    :param num_seeds:   number of randomly selected gaetes
    :return:
    """
    qr = QuantumRegister(num_qubits)

    result = []
    for n in range(num_seeds):
        # Make the circuit
        circ = QuantumCircuit(qr)

        # Make a clifford gate
        cliff = random_clifford(num_qubits, n).to_circuit().to_gate()

        # append to circ
        circ.append(cliff, qr)

        # Run statevector simulator
        result.append(DensityMatrix(circ))

    a = 0
    for _ in result:
        a += _.data

    plt.imshow(a.real)
    plt.colorbar()
    plt.show()

    return


if __name__ == "__main__":
    # Define useful parameters
    N = 4
    g = 2
    J = 1
    beta = 0.2
    num_steps = 1000
    betas, dbeta = np.linspace(0, beta, num_steps, retstep=True)

    # Construct operator lists and Hamiltonian
    z_ops, x_ops = sf.construct_op_lists(N)
    H = 0
    for i in range(N-1):
        H += -J * z_ops[i] @ z_ops[i+1]
        for j in range(N):
            H += -J * g * x_ops[i]

    # Get eigenvalues and vectors from H
    E, V = linalg.eigh(H.to_matrix())

    # Get Pauli decomposition of H
    pauli = sf.pauli_string_decomp(H, N)

    # Create registers for circuit
    qr = QuantumRegister(N)
    qar = AncillaRegister(1)
    cr = ClassicalRegister(1)

    # Define number of shots to use and backend
    num_shots = 8192
    #backend = FakeBelem()
    backend = Aer.get_backend('aer_simulator')

    # Set up dictionary for saving overlaps
    ovlps_dict = {}
    H_ovlps_dict = {}

    # Define parameters for timing things
    re_ovlp_times = []
    im_ovlp_times = []
    re_H_ovlp_times = []
    im_H_ovlp_times = []
    total_times = []
    # Repeat circuit in loop
    num_runs = 100
    for i in range(num_runs):
        total_start = time.time()
        # Create clifford gate for random initial state
        cliff = random_clifford(N, i)

        # Get hadamard estimation circuits for real and imaginary parts
        he_circs_re = sf.hadamard_estimation(qr, qar, cr, H, None, num_steps, dbeta, cliff)
        he_circs_im = sf.imag_hadamard_estimation(qr, qar, cr, H, None, num_steps, dbeta, cliff)

        # Calculate the overlaps
        start = time.time()
        ovlp_re = sf.get_ovlps(he_circs_re, backend, num_shots)
        re_ovlp_times.append(time.time() - start)

        start = time.time()
        ovlp_im = sf.get_ovlps(he_circs_im, backend, num_shots)
        im_ovlp_times.append(time.time() - start)

        ovlps = np.array([ovlp_re[i] + 1j * ovlp_im[i] for i in range(len(ovlp_re))])

        # Get hadamard estimation circuits for H with real and imaginary parts
        H_circs_re = {}
        H_circs_im = {}
        for key in pauli:
            if abs(pauli[key]) != 0:
                print(key)
                H_circs_re[key] = sf.hadamard_estimation(qr, qar, cr, H, Pauli(key), num_steps, dbeta,
                                                     cliff)

                H_circs_im[key] = sf.imag_hadamard_estimation(qr, qar, cr, H, Pauli(key), num_steps,
                                                          dbeta, cliff)

        # Get real part of H expectations
        H_ovlps_re = 0
        start = time.time()
        for key in H_circs_re:
            temp = sf.get_ovlps(H_circs_re[key], backend, num_shots)
            H_ovlps_re += pauli[key] * np.array(temp) / np.sqrt(2 ** (2 * N))
        re_H_ovlp_times.append(time.time() - start)

        # Get imaginary part of H expectations
        H_ovlps_im = 0
        start = time.time()
        for key in H_circs_im:
            temp = sf.get_ovlps(H_circs_im[key], backend, num_shots)
            H_ovlps_im += 1j * pauli[key] * np.array(temp) / np.sqrt(2 ** (2 * N))
        im_H_ovlp_times.append(time.time() - start)

        # Get total H expectation
        total_H_ovlp = H_ovlps_re + H_ovlps_im

        # Save both in dictionaries for later processing
        ovlps_dict["seed=" + str(i)] = ovlps
        H_ovlps_dict["seed=" + str(i)] = total_H_ovlp
        total_times.append(time.time() - total_start)
        print("Time for 1 run: ", total_times[-1])
        print("Run: ", i)

    print("Times for real ovlp: ", re_ovlp_times)
    print("Times for imag ovlp: ", im_ovlp_times)
    print("Times for real H ovlp: ", re_H_ovlp_times)
    print("Times for imag H ovlp: ", im_H_ovlp_times)
    print("Total times: ", total_times)

    np.savez("ris_ovlps_N=" + str(N) + "_numsteps=" + str(num_steps) + "_dtau=" + f'{dbeta:.3f}' + "_sim=aer.npz", ovlps_dict)
    np.savez("ris_H_ovlps_N=" + str(N) + "_numsteps=" + str(num_steps) + "_dtau=" + f'{dbeta:.3f}' + "_sim=aer.npz", H_ovlps_dict)