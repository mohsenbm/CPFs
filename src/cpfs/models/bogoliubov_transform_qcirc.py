import numpy as np
from qiskit import QuantumCircuit

#============================================================
# quantum circuit for Bogoliubov transformation
#============================================================
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

def fswap():
    qc = QuantumCircuit(2)
    q = list(reversed(range(2)))
    qc.swap(q[0], q[1])
    qc.cz(q[0],q[1])
    return qc

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