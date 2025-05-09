import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryOverlap
from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


#############
# quantum circuit for fast fermionic Fourier transformation (ffft)
#############

def F2(dagger=False):
    qc = QuantumCircuit(2)
    q = list(reversed(range(2)))
    qc.cx(q[1], q[0])
    qc.ch(q[0], q[1])
    qc.cx(q[1], q[0])
    qc.cz(q[1], q[0])
    if dagger:
        return qc.inverse()
    else:
        return qc

def fswap():
    qc = QuantumCircuit(2)
    q = list(reversed(range(2)))
    qc.swap(q[0], q[1])
    qc.cz(q[0],q[1])
    return qc

def ffft(n, dagger=False):
    qc = QuantumCircuit(n)
    q = list(reversed(range(n)))
    if n==2:
        qc.compose(F2(), qubits=list(reversed([q[0],q[1]])), inplace=True)
        if dagger:
            return qc.inverse()
        return qc
    if n==4:
        qc.compose(fswap(), qubits=list(reversed([q[1],q[2]])), inplace=True)
        
        qc.compose(F2(), qubits=list(reversed([q[0],q[1]])), inplace=True)
        qc.compose(F2(), qubits=list(reversed([q[2],q[3]])), inplace=True)

        qc.compose(fswap(), qubits=list(reversed([q[1],q[2]])), inplace=True)

        qc.p(-2*np.pi/4, list(reversed([q[3]])))

        qc.compose(F2(), qubits=list(reversed([q[0],q[1]])), inplace=True)
        qc.compose(F2(), qubits=list(reversed([q[2],q[3]])), inplace=True)

        qc.compose(fswap(), qubits=list(reversed([q[1],q[2]])), inplace=True)

        if dagger:
            return qc.inverse()
        return qc
        
    if n==8:
        pass

def Bangle(J, h, k, n):
    num = J * np.sin(2*np.pi*k/n)
    den = -h + J * np.cos(2*np.pi*k/n)
    return np.arctan2(num, den)

def Bgate(angle):
    qc = QuantumCircuit(2)
    q = list(reversed(range(2)))
    qc.cx(q[0], q[1])
    qc.x(q[1])
    qc.crx(-angle, q[1], q[0])
    qc.x(q[1])
    qc.cx(q[0], q[1])
    return qc


#############
# quantum circuit for Bogoliubov transformation
#############

def bogoliubov_transform(J, h, n, dagger=False):
    qc = QuantumCircuit(n)
    q = list(reversed(range(n)))
    if n==2:
        return qc
    if n==4:   
        qc.compose(fswap(), qubits=list(reversed([q[1],q[2]])), inplace=True)
        for k in range(n//2):
            angle = Bangle(J, h, k, n)
            qc.compose(Bgate(angle), qubits=list(reversed([q[2*k],q[2*k+1]])), inplace=True)
        qc.compose(fswap(), qubits=list(reversed([q[1],q[2]])), inplace=True)
        if dagger:
            return qc.inverse()
        return qc


#############
# quantum circuits for comptuing average infidelity
#############
def basis_overlap_qcirc(approxU, exactU, basis=None):
    """
    computes basis overlap | <x|exactU^\dag approxU |x>|^2 where |x> represent basis.
    If basis is None, |x> is the all-zero state.
    """
    nqubits = exactU.num_qubits
    basis = basis or "0" * nqubits
    qc = QuantumCircuit(nqubits)
    qr = list(reversed(range(nqubits)))
    for i, bit in enumerate(basis):
        if bit == "1":
            qc.x(qr[i])

    overlap_qc = UnitaryOverlap(qc.compose(approxU), qc.compose(exactU))
    overlap_measured_qc = overlap_qc.measure_all(inplace=False)
    return overlap_measured_qc

def average_infidelity(approxU, exactU, backend=None, num_shots=10**4):
    """
    computes average infidility
    """
    nqubits = exactU.num_qubits
    sampler = Sampler(mode=backend, options={"default_shots":num_shots})
    basis_set = [format(i, '0' + str(nqubits) + 'b') for i in range(2**nqubits)]
    basis_infidelity_set = []
    for basis in basis_set:
        overlap_qc = basis_overlap_qcirc(approxU, exactU, basis)
        overlap_qc = transpile(overlap_qc, backend, optimization_level=2)
        result = sampler.run([overlap_qc], shots=num_shots).result()
        counts = result[0].data.meas.get_counts()
        allzero_state = '0'*nqubits
        overlap_allzero_state = counts.get(allzero_state, 0) / num_shots
        basis_infidelity = 1 - overlap_allzero_state
        basis_infidelity_set.append(basis_infidelity)
    return basis_infidelity_set