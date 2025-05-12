# === Refactored for Qiskit 1.0+ ===

import numpy as np
print(np.__version__)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, random_clifford
from qiskit.circuit import Parameter
from qiskit import transpile, assemble
from scipy import linalg
from scipy.special import binom
import matplotlib.pyplot as plt
from qutip import *
import itertools as it
from qiskit_ibm_runtime import Estimator, Sampler, Options, QiskitRuntimeService
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate

# Replacements for legacy opflow
X = Pauli("X")
Z = Pauli("Z")
I = Pauli("I")
# Use this command if you didn't save your credentials:
# service = QiskitRuntimeService(channel="ibm_quantum", token="63fe2dff151a1fa50f67b735e09f867d6a2b0380862c050ddac3bdb597586e4818b7a86afe0861d76d84be7f577561808c6a540f48dfb40f6406c056abf86e4c")


def pauli_string_decomp(obs, nq=1):
    """
    Function to obtain the Pauli string decomposition of an observable
    Note: obs should be a qiskit operator like object
    :param obs: the observable to be decomposed
    :param nq:  the number of qubits in the system
    :returns:   dictionary of the coefficients and corresponding pauli strings
    """
    # Create list of single element Pauli strings
    sing_Pauli = ['I', 'X', 'Y', 'Z']

    # Use itertools.product to get all combinations of length n of the Pauli elements
    temp = list(it.product(sing_Pauli, repeat=nq))

    # itertools.product returns a series of tuples, so we need to go through each element
    # of temp and turn the tuple into a string
    pauli_strings = []
    for _ in temp:
        s = ""
        for i in range(nq):
            s += _[i]
        pauli_strings.append(s)

    # Now we can get the coefficients for each Pauli string
    coeffs = [np.trace(obs.to_matrix() @ Pauli(_).to_matrix()) for _ in pauli_strings]

    return dict(zip(pauli_strings, coeffs))


def hadamard_estimation(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create mini circuit for applying desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Turn clifford operator into gate
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing circuits
    circuits = []
    for n in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)
        circ.append(clifford_gate, quant_reg)

        #[circ.h(_) for _ in quant_reg]

        # Construct unitaries
        U_tf = (-float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()
        U_tb = (float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()

        # Create mini circuit that applies U_f to qr
        forward = QuantumCircuit(quant_reg)
        forward.append(U_tf, quant_reg)
        con_f = forward.to_gate().control(1)

        # Create mini circuit that applies U_b to qr
        backward = QuantumCircuit(quant_reg)
        backward.append(U_tb, quant_reg)
        con_b = backward.to_gate().control(1)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply forward unitary to quantum register conditioned on ancilla
        circ.append(con_f, control_args)

        # Apply desired obs to quantum register conditioned on ancilla
        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply not to ancilla
        circ.x(ancilla_reg)

        # Apply backward unitary to quantum register conditioned on ancilla
        circ.append(con_b, control_args)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply measure ancilla
        circ.measure(ancilla_reg, class_reg)

        circuits.append(circ)

    return circuits


def had_est_barr(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation with barriers between each step
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create a mini circuit for appying the desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Now create the forward and backward evolution circuits
    U_tf = (-float(np.sqrt(dt / 2)) * H).exp_i().to_instruction()
    forward = QuantumCircuit(quant_reg)
    forward.append(U_tf, quant_reg)
    con_f = forward.to_gate().control(1)

    U_tb = (float(np.sqrt(dt / 2)) * H).exp_i().to_instruction()
    backward = QuantumCircuit(quant_reg)
    backward.append(U_tb, quant_reg)
    con_b = backward.to_gate().control(1)

    # Create clifford for initial state
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing circuits
    circuits = []
    for i in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # initialize state
        circ.append(clifford_gate, quant_reg)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Create a barrier
        circ.barrier()

        # Apply Uf i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_f, control_args)
            circ.barrier()

        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply X to ancilla
        circ.x(ancilla_reg)
        circ.barrier()

        # Apply Ub i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_b, control_args)
            circ.barrier()

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.barrier()

        # Measure ancilla
        circ.measure(ancilla_reg, class_reg)

        # Store finished circuit in list
        circuits.append(circ)

    return circuits


def had_est_barr_trot(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation with barriers between each step and explicit
    trotterization
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param Hz:              z part of hamiltonian - should be qiskit pauli sum
    :param Hx:              x part of hamiltonian
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create a mini circuit for appying the desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Now create the forward and backward evolution circuits
    U_f = PauliEvolutionGate(H, time=np.sqrt(dt) / 2, synthesis=LieTrotter(reps=1))
    f = QuantumCircuit(quant_reg)
    f.append(U_f, quant_reg)
    con_f = f.to_gate().control(1)

    U_b = PauliEvolutionGate(-H, time=np.sqrt(dt) / 2, synthesis=LieTrotter(reps=1))
    b = QuantumCircuit(quant_reg)
    b.append(U_b, quant_reg)
    con_b = b.to_gate().control(1)

    # Create clifford for initial state
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing circuits
    circuits = []
    for i in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # initialize state
        circ.append(clifford_gate, quant_reg)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Create a barrier
        circ.barrier()

        # Apply Uf i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_f, control_args)
            circ.barrier()

        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply X to ancilla
        circ.x(ancilla_reg)
        circ.barrier()

        # Apply Ub i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_b, control_args)
            circ.barrier()

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.barrier()

        # Measure ancilla
        circ.measure(ancilla_reg, class_reg)

        # Store finished circuit in list
        circuits.append(circ)

    return circuits


def imag_hadamard_estimation(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create mini circuit for applying desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Turn clifford operator into gate
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing overlaps
    circuits = []
    for n in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)
        circ.append(clifford_gate, quant_reg)
        #[circ.h(_) for _ in quant_reg]

        # Construct unitaries
        U_tf = (-float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()
        U_tb = (float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()

        # Create mini circuit that applies U_f to qr
        forward = QuantumCircuit(quant_reg)
        forward.append(U_tf, quant_reg)
        con_f = forward.to_gate().control(1)

        # Create mini circuit that applies U_b to qr
        backward = QuantumCircuit(quant_reg)
        backward.append(U_tb, quant_reg)
        con_b = backward.to_gate().control(1)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.s(ancilla_reg)

        # Apply forward unitary to quantum register conditioned on ancilla
        circ.append(con_f, control_args)

        # Apply desired obs to quantum register conditioned on ancilla
        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply not to ancilla
        circ.x(ancilla_reg)

        # Apply backward unitary to quantum register conditioned on ancilla
        circ.append(con_b, control_args)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply measure ancilla
        circ.measure(ancilla_reg, class_reg)

        # append to list
        circuits.append(circ)

    return circuits


def im_had_est_barr(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation with barriers between each step
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create a mini circuit for appying the desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Now create the forward and backward evolution circuits
    U_tf = (-float(np.sqrt(dt / 2)) * H).exp_i().to_instruction()
    forward = QuantumCircuit(quant_reg)
    forward.append(U_tf, quant_reg)
    con_f = forward.to_gate().control(1)

    U_tb = (float(np.sqrt(dt / 2)) * H).exp_i().to_instruction()
    backward = QuantumCircuit(quant_reg)
    backward.append(U_tb, quant_reg)
    con_b = backward.to_gate().control(1)

    # Create clifford for initial state
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing circuits
    circuits = []
    for i in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # initialize state
        circ.append(clifford_gate, quant_reg)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.barrier()

        # Apply S gate to ancilla
        circ.s(ancilla_reg)

        # Create a barrier
        circ.barrier()

        # Apply Uf i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_f, control_args)
            circ.barrier()

        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply X to ancilla
        circ.x(ancilla_reg)
        circ.barrier()

        # Apply Ub i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_b, control_args)
            circ.barrier()

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.barrier()

        # Measure ancilla
        circ.measure(ancilla_reg, class_reg)

        # Store finished circuit in list
        circuits.append(circ)

    return circuits


def imag_had_est_barr_trot(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt, clifford):
    """
    Function for calculating the overlaps using hadamard estimation with barriers between each step and explicit
    trotterization
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param Hz:              z part of hamiltonian - should be qiskit pauli sum
    :param Hx:              x part of hamiltonian
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create a mini circuit for appying the desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Now create the forward and backward evolution circuits
    U_f = PauliEvolutionGate(H, time=np.sqrt(dt) / 2, synthesis=LieTrotter(reps=1))
    f = QuantumCircuit(quant_reg)
    f.append(U_f, quant_reg)
    con_f = f.to_gate().control(1)

    U_b = PauliEvolutionGate(-H, time=np.sqrt(dt) / 2, synthesis=LieTrotter(reps=1))
    b = QuantumCircuit(quant_reg)
    b.append(U_b, quant_reg)
    con_b = b.to_gate().control(1)

    # Create clifford for initial state
    clifford_gate = clifford.to_circuit().to_gate()

    # Create list for storing circuits
    circuits = []
    for i in range(num_steps):
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # initialize state
        circ.append(clifford_gate, quant_reg)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Create a barrier
        circ.barrier()

        circ.s(ancilla_reg)
        circ.barrier()

        # Apply Uf i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_f, control_args)
            circ.barrier()

        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply X to ancilla
        circ.x(ancilla_reg)
        circ.barrier()

        # Apply Ub i times to circuit with a barrier after each step
        for j in range(i):
            circ.append(con_b, control_args)
            circ.barrier()

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.barrier()

        # Measure ancilla
        circ.measure(ancilla_reg, class_reg)

        # Store finished circuit in list
        circuits.append(circ)

    return circuits


def get_ovlps(circuit_list, backend, num_shots):
    """
    function to transpile and run all circuits in the list and calculate the overlap
    :param circuit_list:    list of circuits to run
    :param backend:         backend to use when running circuits
    :param num_shots:       number of shots to use in each circuit evaluation
    :return:                list of the overlaps
    """
    # First transpile the circuits
    transpiled_circuits = transpile(circuit_list, backend)

    # Assemble into quantum object
    #qobj = assemble(transpiled_circuits, backend)

    # Run the circuits
    job = backend.run(transpiled_circuits, shots=num_shots)

    # Get overlaps
    ovlps = []
    for c in range(len(circuit_list)):
        counts = job.result().get_counts(transpiled_circuits[c])
        if '0' in counts:
            p0 = counts['0'] / num_shots
        else:
            p0 = 0
        ovlps.append(2 * p0 - 1)

    return np.array(ovlps)


def classical_calc(initial_wf, H, obs, num_steps, dt):
    """
    function for performing the calculations numerically
    :param initial_wf:  initial value of the wavefunction
    :param H:           Hamiltonian
    :param obs:         observable to calculate expectation of
    :param num_steps:   number of iterations
    :param dt:          step size
    :return:            list of overlaps with and without observable
    """
    # Set up lists
    wf_root_list = [initial_wf]
    overlap_list = [(initial_wf.conj().T @ initial_wf)]
    obs_ovlp_list = [(initial_wf.conj().T @ H @ initial_wf)]

    # Get evolution unitaries
    U_f = linalg.expm(-1j * (np.sqrt(dt / 2)) * H)
    U_b = linalg.expm(1j * (np.sqrt(dt / 2)) * H)

    # Perform propagation
    for _ in range(num_steps - 1):
        # Propagate wf forward in time one step
        wf_root_list.append(U_f @ wf_root_list[-1])

        # Propagate wf backward in time one step
        wf_root_list.insert(0, U_b @ wf_root_list[0])

        # Calculate overlaps of most future and most past wfs and append to lists
        overlap_list.append(1 * (wf_root_list[0].conj().T @ wf_root_list[-1]))
        obs_ovlp_list.append(1 * (wf_root_list[0].conj().T @ obs @ wf_root_list[-1]))

    return np.array(overlap_list), np.array(obs_ovlp_list)


def partition_calc(ovlp_list, obs_ovlp_list, num_steps, lam, dt):
    """
    function for calculating the sums that give the expectation value as a function of imaginary time
    :param ovlp_list:       list of overlaps with no observable - used for the partition function
    :param obs_ovlp_list:   list of overlaps with the specified observable
    :param num_steps:       number of steps to include - binom grows quickly, so need to be careful
    :param lam:             shift parameter for the phase
    :param dt:              step size
    :return:                lists of the partition values and the expectation values
    """
    partition_list = [ovlp_list[1]]
    expect_list = [obs_ovlp_list[1]]

    for m in range(3, num_steps, 2):
        partition = 0
        E = 0

        for k in range(0, m, 2):
            phase = np.exp(2j * lam * np.sqrt(0.5 * dt) * (m - k))
            partition += binom(m, int(k / 2)) * 2 * (phase * ovlp_list[m - k]).real
            E += binom(m, int(k / 2)) * 2 * (phase * (obs_ovlp_list[m - k] + lam * ovlp_list[m - k])).real

        partition = np.abs(partition)

        partition_list.append(partition)
        expect_list.append(E / partition)

    return np.array(partition_list), np.array(expect_list)

def density_partition_calc(tr_ρs, tr_ρHs, num_steps, λ, dτ):
    Z = [tr_ρs[1]]
    O = [tr_ρHs[1]]
    for m in range(3, num_steps, 2):
        partition = 0
        expt_val = 0
        for k in range(0, m, 2):
            phase = np.exp(2j * λ * np.sqrt(0.5 * dτ) * (m - k))
            Z_k = tr_ρs[m - k]
            partition += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (phase * Z_k).real
            O_k = tr_ρHs[m - k] + λ * Z_k
            expt_val += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (phase * O_k).real

        partition = np.abs(partition)
        Z.append(partition)
        O.append(expt_val / partition)

    return np.array(Z), np.array(O)

def exact_density_partition_calc(tr_ρs, tr_ρHs, num_steps, λ, dτ):
    Z = [tr_ρs[1]]
    O = [tr_ρHs[1]]
    for m in range(3, num_steps, 2):
        partition = 0
        expt_val = 0
        for k in range(0, m, 2):
            phase = np.exp(2j * λ * np.sqrt(0.5 * dτ) * (m - k))
            Z_k = tr_ρs[m - k]
            partition += binom(m, int(k / 2)) * 2 * (phase * Z_k).real
            O_k = tr_ρHs[m - k] + λ * Z_k
            expt_val += binom(m, int(k / 2)) * 2 * (phase * O_k).real

        partition = np.abs(partition)
        Z.append(partition)
        O.append(expt_val / partition)

    return np.array(Z), np.array(O)

# def density_partition_calc(ovlp_list, obs_ovlp_list, num_steps, lam, dt):
#     """
#     function for calculating the sums that give the expectation value as a function of imaginary time
#     :param ovlp_list:       list of overlaps with no observable - used for the partition function
#     :param obs_ovlp_list:   list of overlaps with the specified observable
#     :param num_steps:       number of steps to include - binom grows quickly, so need to be careful
#     :param lam:             shift parameter for the phase
#     :param dt:              step size
#     :return:                lists of the partition values and the expectation values
#     """
#     partition_list = [ovlp_list[1]]
#     expect_list = [obs_ovlp_list[1]]
#
#     for m in range(3, num_steps, 2):
#         partition = 0
#         E = 0
#
#         for k in range(0, m, 2):
#             phase = np.exp(2j * lam * np.sqrt(0.5 * dt) * (m - k))
#             partition += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (phase * ovlp_list[m - k]).real
#             E += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (phase * (obs_ovlp_list[m - k] + lam * ovlp_list[m - k])).real
#
#         partition = np.abs(partition)
#
#         partition_list.append(partition)
#         expect_list.append(E / partition)
#
#     return np.array(partition_list), np.array(expect_list)
# def construct_op_lists(num_qubits):
#     """
#     function to create the lists of tensored operators we use
#     :param num_qubits: the number of qubits in the system
#     :returns:          lists of the full z and x operators
#     """
#     z_list = []
#     x_list = []
#     for i in range(num_qubits):
#         z_list.append((I ^ i) ^ Z ^ (I ^ (num_qubits - 1 - i)))
#         x_list.append((I ^ i) ^ X ^ (I ^ (num_qubits - 1 - i)))
#
#     return z_list, x_list

def construct_op_lists(num_qubits):
    z_list = []
    x_list = []
    for i in range(num_qubits):
        z_str = ["I"] * num_qubits
        x_str = ["I"] * num_qubits
        z_str[i] = "Z"
        x_str[i] = "X"
        # Qiskit uses little-endian convention → reverse the string
        z_list.append(SparsePauliOp("".join(reversed(z_str))))
        x_list.append(SparsePauliOp("".join(reversed(x_str))))
    return z_list, x_list

def alt_partition_calc(ovlp_list, obs_ovlp_list, num_steps, lam, dt):
    """
    function for calculating the sums that give the expectation value as a function of imaginary time
    :param ovlp_list:       list of overlaps with no observable - used for the partition function
    :param obs_ovlp_list:   list of overlaps with the specified observable
    :param num_steps:       number of steps to include - binom grows quickly, so need to be careful
    :param lam:             shift parameter for the phase
    :param dt:              step size
    :return:                lists of the partition values and the expectation values
    """
    partition_list = [ovlp_list[1]]
    expect_list = [obs_ovlp_list[1]]

    for m in range(3, num_steps, 2):
        partition = 0
        E = 0

        for k in range(0, m, 2):
            phase = np.exp(2j * lam * np.sqrt(0.5 *dt) * (m - k))
            partition += np.exp(-2*(int(k/2) - m /2) ** 2 / m) * 2 * (phase * ovlp_list[m - k]).real
            E += np.exp(-2*(int(k/2) - m /2) ** 2 / m) * 2 * (phase * (obs_ovlp_list[m - k] + lam * ovlp_list[m - k])).real

        partition = np.abs(partition)

        partition_list.append(partition)
        expect_list.append(E / partition)

    return np.array(partition_list), np.array(expect_list)

def extrapolated_partition_calc(ovlp_list, obs_ovlp_list, mbar, max_m, lam, dt):
    """
    Calculates expectation values as a function of imaginary time, using only the first mbar overlaps
    but extending the summation up to max_m steps.

    :param ovlp_list:     list of overlaps with no observable
    :param obs_ovlp_list: list of overlaps with the observable
    :param mbar:          number of known overlap steps to use (max index accessed = mbar)
    :param max_m:         maximum step to compute expectation for (defines max tau)
    :param lam:           shift parameter for phase
    :param dt:            time step size
    :return:              arrays of partition values and expectation values
    """
    partition_list = [ovlp_list[1]]
    expect_list = [obs_ovlp_list[1]]

    for m in range(3, max_m, 2):  # must remain odd for your tau indexing
        partition = 0
        E = 0

        for k in range(0, m, 2):
            index = m - k
            if index > mbar:
                continue  # Don't use overlaps we don't have
            weight = np.exp(-2 * ((k / 2 - m / 2) ** 2) / m)
            phase = np.exp(2j * lam * np.sqrt(0.5 * dt) * (m - k))
            psi = ovlp_list[index]
            Hpsi = obs_ovlp_list[index]

            partition += 2 * weight * (phase * psi).real
            E += 2 * weight * (phase * (Hpsi + lam * psi)).real

        partition = np.abs(partition)
        partition_list.append(partition)
        expect_list.append(E / partition)

    return np.array(partition_list), np.array(expect_list)

def extended_partition_calc(ovlp_list, obs_ovlp_list, num_steps, lam, dt, trunc_step):
    """
    function for calculating the sums for the expectation value beyond the number of steps we have data for
    :param ovlp_list:
    :param obs_ovlp_list:
    :param num_steps:
    :param lam:
    :param dt:
    :param trunc_step:
    :return:
    """
    m = num_steps - 0
    mbar = trunc_step - 0
    partition = 0
    E = 0

    for k in range((m - mbar + 0), m, 2):
        phase = np.exp(2j * lam * np.sqrt(2 * dt) * (mbar - k))
        partition += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (phase * ovlp_list[mbar - k]).real
        E += np.exp(-2 * (int(k / 2) - m / 2) ** 2 / m) * 2 * (
                    phase * (obs_ovlp_list[mbar - k] + lam * ovlp_list[mbar - k])).real

    partition = np.abs(partition)
    return partition, E / partition


def plot_energies_dif_lambdas(ovlp_list, obs_ovlp_list, num_steps, betas, dt, E_eigs):
    """
    function for generating plots of the energy as the system propagates in imaginary time using the energy
    eigenvalues as the lambda parameters
    :param ovlp_list:       the list of overlaps with no observable
    :param obs_ovlp_list:   the list of overlaps with an observable
    :param num_steps:       the number of steps used in the calculation
    :param dt:              the step size used in the calculations
    :param E_eigs:          the energy eigenvalues of H
    :return:
    """
    # Create colormap for the plots
    colors = plt.cm.jet(np.linspace(0, 1, len(E_eigs[::2]) + 2))
    l = 0
    for e in E_eigs[::2]:
        e = -e
        if l == 0 or l == len(E_eigs[::2]) - 1:
            plt.plot(betas[2::2], alt_partition_calc(ovlp_list, obs_ovlp_list, num_steps,
                                                         e, dt)[1][1:] - e, color=colors[l + 1],
                     label='$\\lambda = E_' + str(l) + '$')
        plt.plot(betas[2::2], alt_partition_calc(ovlp_list, obs_ovlp_list, num_steps,
                                                     e, dt)[1][1:] - e, color=colors[l])

        plt.plot(betas, e * np.ones(len(betas)), '-.', color=colors[l + 1])
        l += 1
    plt.xlabel("$\\beta$")
    plt.ylabel("E")
    plt.legend()
    plt.show()


def plot_lambda_sweep(ovlp_list, obs_ovlp_list, num_steps, dt, E_eigs):
    """
    function for plotting the results of sweeping over lambda
    :param ovlp_list:       the list of overlaps with no observable
    :param obs_ovlp_list:   the list of overlaps with an observable
    :param num_steps:       the number of steps propagated in imaginary time
    :param dt:              the step size
    :param E_eigs:          the eigenvalues of the observable
    :return:
    """
    # Create array of lambda values and use it to calculate the energy using the partition calculations
    lambdas = np.linspace(-1.1 * abs(E_eigs[0]), 1.1 * abs(E_eigs[0]), 1000)
    calculated_energies = [alt_partition_calc(ovlp_list, obs_ovlp_list, num_steps, _, dt)[1][-1]
                           for _ in lambdas]

    # Create colormap for plotting
    colors = plt.cm.jet(np.linspace(0, 1, len(E_eigs[::2]) + 2))
    h = 0
    for temp in E_eigs[::2]:
        plt.plot(lambdas / abs(E_eigs[0]), temp * np.ones(lambdas.shape[0]) + lambdas,
                 '-.', color=colors[h + 1])
        h += 1
        plt.plot(lambdas / abs(E_eigs[0]), calculated_energies, 'k')

    plt.ylabel("$\\langle H \\rangle + \\lambda (a.u.)$")
    plt.xlabel("$\\lambda/|E_0|$")
    plt.ylim([-10, 10])
    #plt.savefig("belem_N=2_lambda_sweep.png", format='png', dpi=300)
    plt.show()


def re_sampler_circs(qr, cr, H, obs, numsteps, dt, cliff):
    """
    Function for creating circuits to use in the Sampler primitive for Qiskit
    :param qr:          the quantum register - includes the ancilla
    :param cr:          the classical register
    :param H:           the hamiltonian
    :param obs:         the observable we want to measure
    :param numsteps:    the number of steps
    :param dt:          the time step size
    :param cliff:       the clifford for the initial setup
    :return:
    """
    circ_list = []
    if obs is not None:
        obs_circ = QuantumCircuit([qr[1], qr[2]])
        obs_circ.append(Operator(obs), [qr[1], qr[2]])
        obs_gate = obs_circ.to_gate().control(1)

    for n in range(numsteps):
        # Set up circuit
        circ = QuantumCircuit(qr, cr)

        # Apply clifford to main circuits and Hadamard to ancilla
        circ.append(cliff.to_circuit(), [1, 2])
        circ.h(0)

        # Create Uf
        U_f = (-float(n * np.sqrt(dt / 2)) * H).exp_i().to_circuit().to_gate().control(1)
        circ.append(U_f, [0, 1, 2])

        # Apply observable if it exits
        if obs is not None:
            circ.append(obs_gate, [0, 1, 2])

        # Apply x to ancilla
        circ.x(0)

        # Create Ub
        U_b = (float(n * np.sqrt(dt / 2)) * H).exp_i().to_circuit().to_gate().control(1)
        circ.append(U_b, [0, 1, 2])

        # Apply final Hadamard to ancilla
        circ.h(0)

        # Apply measurement to ancilla
        circ.measure([0], [0])

        # Append to list
        circ_list.append(circ)

    return circ_list


def im_sampler_circs(qr, cr, H, obs, numsteps, dt, cliff):
    """
    Function for creating circuits to use in the Sampler primitive for Qiskit
    :param qr:          the quantum register - includes the ancilla
    :param cr:          the classical register
    :param H:           the hamiltonian
    :param obs:         the observable we want to measure
    :param numsteps:    the number of steps
    :param dt:          the time step size
    :param cliff:       the clifford for the initial setup
    :return:
    """
    circ_list = []
    if obs is not None:
        obs_circ = QuantumCircuit([qr[1], qr[2]])
        obs_circ.append(Operator(obs), [qr[1], qr[2]])
        obs_gate = obs_circ.to_gate().control(1)

    for n in range(numsteps):
        # Set up circuit
        circ = QuantumCircuit(qr, cr)

        # Apply clifford to main circuits and Hadamard to ancilla
        circ.append(cliff.to_circuit(), [1, 2])
        circ.h(0)
        circ.s(0)

        # Create Uf
        U_f = (-float(n * np.sqrt(dt / 2)) * H).exp_i().to_circuit().to_gate().control(1)
        circ.append(U_f, [0, 1, 2])

        # Apply observable if it exits
        if obs is not None:
            circ.append(obs_gate, [0, 1, 2])

        # Apply x to ancilla
        circ.x(0)

        # Create Ub
        U_b = (float(n * np.sqrt(dt / 2)) * H).exp_i().to_circuit().to_gate().control(1)
        circ.append(U_b, [0, 1, 2])

        # Apply final Hadamard to ancilla
        circ.h(0)

        # Apply measurement to ancilla
        circ.measure([0], [0])

        # Append to list
        circ_list.append(circ)

    return circ_list


def get_p0s(results):
    """
    Function for getting the probability of measuring 0 in the ancilla for the sampler results
    :param results:     counts measured by the sampler
    :return:
    """
    p0s = []
    for _ in results:
        if '0' in _.keys():
            p0s.append(_['0'])
        else:
            p0s.append(0)

    return np.array(p0s)


def had_est_sampler(qr, cr, H, cliff, sampler_options, num_steps, dt, N):
    """
    function for performing hadamard estimation using the sampler primitive interface
    :param qr:                  the quantum register (note this includes the ancilla)
    :param cr:                  the classical register
    :param H:                   the Hamiltonian
    :param cliff:               the clifford setting the initial state
    :param sampler_options:     any extra options for the sampler
    :param num_steps:           the number of steps to iterate
    :param dt:                  the step size
    :param N:                   number of qubits
    :return:
    """
    backend = service.backend("ibmq_qasm_simulator")
    sampler = Sampler(backend=backend, options=sampler_options)

    # Real Z
    re_circ_list = re_sampler_circs(qr, cr, H, None, num_steps, dt, cliff)
    job = sampler.run(re_circ_list)
    re_results = [_.binary_probabilities() for _ in job.result().quasi_dists]
    re_ovlp = 2 * get_p0s(re_results) - 1

    # Imag Z
    im_circ_list = im_sampler_circs(qr, cr, H, None, num_steps, dt, cliff)
    job = sampler.run(im_circ_list)
    im_results = [_.binary_probabilities() for _ in job.result().quasi_dists]
    im_ovlp = 2 * get_p0s(im_results) - 1

    # Combine
    ovlps = re_ovlp + 1j * im_ovlp

    # Get pauli decomposition of H
    paulis = pauli_string_decomp(H, 2)

    # Get circuits for each key
    re_H_circs = {}
    im_H_circs = {}
    for key in paulis:
        if abs(paulis[key]) != 0:
            re_H_circs[key] = re_sampler_circs(qr, cr, H, Pauli(key), num_steps, dt, cliff)
            im_H_circs[key] = im_sampler_circs(qr, cr, H, Pauli(key), num_steps, dt, cliff)

    # Process real part
    re_H_ovlp = 0
    for key in re_H_circs:
        job = sampler.run(re_H_circs[key])
        results = [_.binary_probabilities() for _ in job.result().quasi_dists]
        temp = 2 * get_p0s(results) - 1
        re_H_ovlp += paulis[key] * temp / np.sqrt(2 ** (2 * N))

    # Process imaginary part
    im_H_ovlp = 0
    for key in im_H_circs:
        job = sampler.run(im_H_circs[key])
        results = [_.binary_probabilities() for _ in job.result().quasi_dists]
        temp = 2 * get_p0s(results) - 1
        im_H_ovlp += paulis[key] * temp / np.sqrt(2 ** (2 * N))

    # Combine
    H_ovlps = re_H_ovlp + 1j * im_H_ovlp

    return ovlps, H_ovlps


def get_coeffs_and_obs(A, N):
    paulis = pauli_string_decomp(A, N)
    coeffs = []
    obs_list = []
    for key in paulis:
        if np.abs(paulis[key]) != 0:
            coeffs.append(paulis[key])
            obs_list.append(Pauli(key))

    return np.array(coeffs), obs_list


def expt_estimator(qr, H, num_steps, cliff, estimator_options, dt, N):
    backend = service.backend("ibmq_qasm_simulator")
    estimator = Estimator(backend=backend, options=estimator_options)

    # Create initial circuit for appending into lists
    circ = QuantumCircuit(qr)
    circ.append(cliff, qr)

    # Create lists for storing expectation values
    U_sq_expt_vals = []
    Ud_sq_expt_vals = []
    HU_sq_expt_vals = []
    HUd_sq_expt_vals = []

    for n in range(num_steps):
        # Create U^2 and get coefficients and observables for pauli string decomposition
        U_sq = (-float(2 * n * np.sqrt(dt / 2)) * H).exp_i()
        U_sq_coeffs, U_sq_obs_list = get_coeffs_and_obs(U_sq, N)

        # Create list of circuits for U_sq and then run using estimator
        U_sq_circs = [circ for _ in U_sq_coeffs]
        U_sq_job = estimator.run(U_sq_circs, U_sq_obs_list)
        U_sq_vals = np.array(U_sq_job.result().values, dtype=complex)
        U_sq_expt_vals.append(np.sum(U_sq_vals * U_sq_coeffs) / np.sqrt(2 ** (2 * N)))

        # Now do the same for U_dag^2
        Ud_sq = (float(2 * n * np.sqrt(dt / 2)) * H).exp_i()
        Ud_sq_coeffs, Ud_sq_obs_list = get_coeffs_and_obs(Ud_sq, N)
        Ud_sq_circs = [circ for _ in Ud_sq_coeffs]
        Ud_sq_job = estimator.run(Ud_sq_circs, Ud_sq_obs_list)
        Ud_sq_vals = np.array(Ud_sq_job.result().values, dtype=complex)
        Ud_sq_expt_vals.append(np.sum(Ud_sq_vals * Ud_sq_coeffs) / np.sqrt(2 ** (2 * N)))

        # Now do the same for U^2 @ H
        HU_sq = (-float(2 * n * np.sqrt(dt / 2)) * H).exp_i() @ H
        HU_sq_coeffs, HU_sq_obs_list = get_coeffs_and_obs(HU_sq, N)
        HU_sq_circs = [circ for _ in HU_sq_coeffs]
        HU_sq_job = estimator.run(HU_sq_circs, HU_sq_obs_list)
        HU_sq_vals = np.array(HU_sq_job.result().values, dtype=complex)
        HU_sq_expt_vals.append(np.sum(HU_sq_vals * HU_sq_coeffs) / np.sqrt(2 ** (2 * N)))

        # Now do the same for U_dag^2 @ H
        HUd_sq = (float(2 * n * np.sqrt(dt / 2)) * H).exp_i() @ H
        HUd_sq_coeffs, HUd_sq_obs_list = get_coeffs_and_obs(HUd_sq, N)
        HUd_sq_circs = [circ for _ in HUd_sq_coeffs]
        HUd_sq_job = estimator.run(HUd_sq_circs, HUd_sq_obs_list)
        HUd_sq_vals = np.array(HUd_sq_job.result().values, dtype=complex)
        HUd_sq_expt_vals.append(np.sum(HUd_sq_vals * HUd_sq_coeffs) / np.sqrt(2 ** (2 * N)))

    # Calculate z
    U_sq_expt_vals = np.array(U_sq_expt_vals)
    Ud_sq_expt_vals = np.array(Ud_sq_expt_vals)
    z_re = (U_sq_expt_vals + Ud_sq_expt_vals) / 2
    z_im = (U_sq_expt_vals - Ud_sq_expt_vals) / 2
    z = z_re + 1j * z_im

    # Calculate <H(dt)>
    HU_sq_expt_vals = np.array(HU_sq_expt_vals)
    HUd_sq_expt_vals = np.array(HUd_sq_expt_vals)
    E_re = (HU_sq_expt_vals + HUd_sq_expt_vals) / 2
    E_im = (-HU_sq_expt_vals + HUd_sq_expt_vals) / 2
    E = E_re + E_im

    return z, E


def had_est_mult_cliffs(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt):
    """
    Function for calculating the overlaps using hadamard estimation
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create mini circuit for applying desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Create list for storing circuits
    circuits = []
    for n in range(num_steps):
        # Create initial circuit
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # Create new clifford gate and apply to circuit
        clifford_gate = random_clifford(len(quant_reg), n)
        circ.append(clifford_gate, quant_reg)

        # Construct unitaries
        U_tf = (-float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()
        U_tb = (float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()

        # Create mini circuit that applies U_f to qr
        forward = QuantumCircuit(quant_reg)
        forward.append(U_tf, quant_reg)
        con_f = forward.to_gate().control(1)

        # Create mini circuit that applies U_b to qr
        backward = QuantumCircuit(quant_reg)
        backward.append(U_tb, quant_reg)
        con_b = backward.to_gate().control(1)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply forward unitary to quantum register conditioned on ancilla
        circ.append(con_f, control_args)

        # Apply desired obs to quantum register conditioned on ancilla
        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply not to ancilla
        circ.x(ancilla_reg)

        # Apply backward unitary to quantum register conditioned on ancilla
        circ.append(con_b, control_args)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply measure ancilla
        circ.measure(ancilla_reg, class_reg)

        circuits.append(circ)

    return circuits


def im_had_est_mult_cliffs(quant_reg, ancilla_reg, class_reg, H, obs, num_steps, dt):
    """
    Function for calculating the overlaps using hadamard estimation
    :param quant_reg:       number of qubits in the register
    :param ancilla_reg:     number of ancilla qubits
    :param class_reg:       number of classical bits
    :param H:               hamiltonian - should be qiskit pauli sum
    :param obs:             observable to use for overlap calculation
    :param num_steps:       number of iterations to run
    :param dt:              step size
    :param init_wf:         value of the initial wavefunction
    :return:                list of overlap values and final circuit
    """
    # Create list for qubits to avoid issues with number when appending to circuits
    control_args = [q for q in quant_reg]
    control_args.insert(0, ancilla_reg)

    # Create mini circuit for applying desired observable as controlled unitary
    if obs is not None:
        obs_circ = QuantumCircuit(quant_reg)
        obs_circ.append(Operator(obs), quant_reg)
        obs_gate = obs_circ.to_gate().control(1)

    # Create list for storing circuits
    circuits = []
    for n in range(num_steps):
        # Create initial circuit
        circ = QuantumCircuit(quant_reg, ancilla_reg, class_reg)

        # Create new clifford gate and apply to circuit
        clifford_gate = random_clifford(len(quant_reg), n)
        circ.append(clifford_gate, quant_reg)

        # Construct unitaries
        U_tf = (-float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()
        U_tb = (float(np.sqrt(dt / 2) * n) * H).exp_i().to_instruction()

        # Create mini circuit that applies U_f to qr
        forward = QuantumCircuit(quant_reg)
        forward.append(U_tf, quant_reg)
        con_f = forward.to_gate().control(1)

        # Create mini circuit that applies U_b to qr
        backward = QuantumCircuit(quant_reg)
        backward.append(U_tb, quant_reg)
        con_b = backward.to_gate().control(1)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)
        circ.s(ancilla_reg)

        # Apply forward unitary to quantum register conditioned on ancilla
        circ.append(con_f, control_args)

        # Apply desired obs to quantum register conditioned on ancilla
        if obs is not None:
            circ.append(obs_gate, control_args)

        # Apply not to ancilla
        circ.x(ancilla_reg)

        # Apply backward unitary to quantum register conditioned on ancilla
        circ.append(con_b, control_args)

        # Apply hadamard to ancilla
        circ.h(ancilla_reg)

        # Apply measure ancilla
        circ.measure(ancilla_reg, class_reg)

        circuits.append(circ)

    return circuits
