import numpy as np
from qiskit import QuantumCircuit
from src.ising_model import spectrum
from primitive_qcircs import ffft, bogoliubov_transform

def exact_evolution_diagonal_qcirc(J, h, n, t):
    qc = QuantumCircuit(n)  
    q = list(reversed(range(n)))
    for k in range(n):
        qc.rz(-2*spectrum(J,h,k,n)*t, q[k])
    return qc

def Udiag_2site_ising_qcirc(J,h):
    """
    Quantum circuit for a implementing a unitary U that diagonalizes 2-site Ising model H = J*XX + h*(ZI + IZ)
    :math:`U^{\dag} H U = D`.

    Args:
    J (float):  the nearest-neighbor interaction strength
    h (float): the transverse field strength

    Returns:
    A quantum circuit that implements U

    """
    qc = QuantumCircuit(2)
    q = list(reversed(range(2)))
    qc.cx(q[0], q[1])
    qc.ch(q[1], q[0])
    qc.cry(np.arctan2(J, 2*h), q[1], q[0], ctrl_state=0)
    qc.cx(q[0], q[1])
    return qc

def exact_evolution_2site_ising_qcirc(J, h, t):
    """
    Quantum circuits for exact implementation for e^{-iHt} with H = J*XX + h*(IZ+ZI), the 2-site ising model.
    We write e^{-iHt} = U e^{-iDt} Udag, where U is the unitary that diagonalized U.
    To implement e^{-iDt} we used the expression D = c0*ZI + c1*IZ where c_{0/1} = (lam0 +/- lam1)/2
    and where lam0 = sqrt(J^2+(2h)^2) and lam1=J are positive eginenvalues of H.
    Note that eignenvalues of H are \pm lam0 and \pm lam1.
    """
    U = Udiag_2site_ising_qcirc(J, h)
    Udag = U.inverse()

    def Udiag(J,h):
        qc = QuantumCircuit(2)
        q = list(reversed(range(2)))
        lam0 = np.sqrt(J**2 + (2*h)**2)
        lam1 = J
        c0 = (lam0 + lam1)/2
        c1 = (lam0 - lam1)/2
        qc.rz(2*c0*t, q[0])
        qc.rz(2*c1*t, q[1])
        return qc

    return Udag.compose(Udiag(J,h)).compose(U)

def exact_evolution_ising_qcirc(J, h, n, t):
    if n == 2:
        return exact_evolution_2site_ising_qcirc(J, h, t)
    F = ffft(n)
    Fdag = ffft(n, True)
    B = bogoliubov_transform(J, h, n)
    Bdag = bogoliubov_transform(J, h, n, True)
    diag_evolution = exact_evolution_diagonal_qcirc(J, h, n, t)
    return F.compose(B).compose(diag_evolution).compose(Bdag).compose(Fdag)



def Hxx_evolution_qcirc(Hxx, tau):
    n = Hxx.num_qubits
    qc = QuantumCircuit(n)
    q = list(reversed(range(n)))
    # evolution of Hxx_k = J Xk Xk+1 terms where J is the coeffient of each term.
    if n==2:
        qc.cx(q[1], q[0])
        qc.rx(2*tau*Hxx.coeffs.real[0], q[1])
        qc.cx(q[1], q[0])
        return qc
    for k in range(n-1):
        qc.cx(q[k+1], q[k])
        qc.rx(2*tau*Hxx.coeffs.real[k], q[k+1])
        qc.cx(q[k+1], q[k])
    # evolution of YZ...ZY term (can this be improved?)
    qc.rx(+np.pi/2, q[0])
    qc.rx(+np.pi/2, q[n-1])
    [qc.cx(q[k], q[k+1]) for k in range(n-1)]
    qc.rz(2*tau*Hxx.coeffs.real[k], q[n-1])
    [qc.cx(q[k-1], q[k]) for k in range(n-1,0,-1)]
    qc.rx(-np.pi/2, q[0])
    qc.rx(-np.pi/2, q[n-1])
    return qc


def Hz_evolution_qcirc(Hz, tau):
    n = Hz.num_qubits
    qc = QuantumCircuit(n)
    q = list(reversed(range(n)))
    [qc.rz(2*tau *Hz.coeffs.real[k], q[k]) for k in range(n)]
    return qc