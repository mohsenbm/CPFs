import numpy as np
from qiskit.quantum_info import SparsePauliOp
import openfermion as of



def ising_qubit_hamiltonian(n, J, h, periodic=True):
    """
    Returns the Hamiltonian for the transverse-field Ising model in qubit representation.
    The Hamiltonian is given by: :math:`H = Hxx + Hz`
    where :math:`Hxx = J \sum_{i} X_i X_{i+1}`
    and :math:`Hx = h \sum_i Z_i` for a line of qubits that are ordered from left to righ, i.e., `q_0 q_1 ... q_{n-1}`

    Args:
        n (int): number of qubits
        J (float):  the nearest-neighbor interaction strength
        h (float): the transverse field strength
        periodic (bool): if True, the term :math:`Y_0 Z_1 ... Z_{n-2} Y_{n-1}` is included.
        If False, then we have open boundaries.

    Returns:
        Operator: H, the Hamiltonian
        Operator: Hxx, the XX part
        Operator: Hz, the Z part of the H
    """
    # the qubit indices corresponding to the Pauli string (from left to right)
    q = list(reversed(range(n)))

    #list of Hamiltonian terms as tuples.
    XX_tuples = [("XX", [q[i], q[i + 1]], J) for i in range(n-1)]
    if periodic and n!=2:
        XX_tuples.append(("Y" + "Z"*(n-2) + "Y", range(n), J))

    Z_tuples = [("Z", [q[i]], h) for i in range(n)]

    Hxx = SparsePauliOp.from_sparse_list(XX_tuples, num_qubits=n)
    Hz = SparsePauliOp.from_sparse_list(Z_tuples, num_qubits=n)

    H = Hxx + Hz
    return H, Hxx, Hz


def ising_fermionic_hamiltonian(n, J, h):
    """
    Returns the Hamiltonian for the transverse-field Ising model in fermionic representation.
    """
    interaction_terms = [op + of.hermitian_conjugated(op)
          for op in (
        of.FermionOperator(((i, 1), ( (i + 1)%n , 0)))+
        of.FermionOperator(((i, 1), ( (i + 1)%n , 1)))
              for i in range(n))] 
    onsite_terms = [2*of.FermionOperator(((i, 1), (i , 0))) - of.FermionOperator(())
                for i in range(n)]
    ham =  J * sum(interaction_terms) - h * sum(onsite_terms)
    return ham


def ising_fermionic_hamiltonian_after_fft(n, J, h):
    ham_terms = []
    for k in range(n):
        coeff1 = 2*( -h + J * np.cos(2*np.pi*k/n) )
        coeff2 = 1j * J * np.sin(2*np.pi*k/n)

        if k==0:
            zero_term = coeff1 * of.FermionOperator(((k, 1), (k, 0))) +\
            coeff2 * (of.FermionOperator(((k, 1), (n//2, 1))) + of.FermionOperator(((k, 0), (n//2, 0))))
            ham_terms.append(zero_term)
        else:
            kth_term = coeff1 * of.FermionOperator(((k, 1), (k, 0))) +\
            coeff2 * (of.FermionOperator(((k, 1), (n-k, 1))) + of.FermionOperator(((k, 0), (n-k, 0))))
            ham_terms.append(kth_term)

    ham =  sum(ham_terms) + h * n * of.FermionOperator(())
    return ham

def ising_free_fermionic_hamiltoninan(J, h, n):
    diagH_list = [("Z", [n-1-k], -spectrum(J,h,k,n)) for k in range(n)] # note that qubits ordered from left to right.
    diagH = SparsePauliOp.from_sparse_list(diagH_list, num_qubits=n)
    return diagH

def spectrum(J, h, k, n):
    return np.sqrt((h + J * np.cos(2*np.pi*k/n))**2 + (J * np.sin(2*np.pi*k/n))**2)

