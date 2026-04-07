import numpy as np
from qiskit.quantum_info import SparsePauliOp
import openfermion as of



def ising_qubit_hamiltonian(n, J, h, periodic=True):
    """
    Returns the Hamiltonian for the transverse-field Ising model in qubit representation.
    The Hamiltonian is given by: :math:`H = Hxx + Hz`
    where :math:`Hxx = J \sum_{i} X_i X_{i+1}`
    and :math:`Hz = h \sum_i Z_i` for a line of qubits that are ordered from left to righ, i.e., `q_0 q_1 ... q_{n-1}`

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


def ising_qubit_hamiltonian_2d(rows, cols, J, h, periodic=False):
    """
    Constructs the 2D transverse-field Ising Hamiltonian with optional periodic boundary conditions.
    
    The Hamiltonian is given by: :math:`H = Hxx + Hz`
    where :math:`Hxx = J \sum_{<i,j>} X_i X_{j}` + periodic boundary terms: Y-Z-Z...Z-Y along each row and column (if periodic=True)
    and and :math:`Hz = h \sum_i Z_i` for a line of 2D gird of qubits that are ordered from left to righ and bottom to top.

    Args:
        rows (int): Number of rows in the 2D grid.
        cols (int): Number of columns in the 2D grid.
        J (float): Coupling strength for nearest-neighbor interactions and boundary terms.
        h (float): Strength of the transverse field (Z terms).
        periodic (bool): Enables periodic boundary conditions with Y-Z-Z...-Z-Y terms.

    Returns:
        H (SparsePauliOp): Full Hamiltonian.\n
        Hxx (SparsePauliOp): XX interaction part (including boundary terms).\n
        Hz (SparsePauliOp): Z field part.
    """

    num_qubits = rows * cols

    def qubit_index(i, j):
        # Reverse indexing to match Qiskit's ordering (right to left, top to bottom)
        return (rows - 1 - i) * cols + (cols - 1 - j)
    
    XX_terms = []
    Z_terms = []

    # Local XX interactions and Z fields
    for i in range(rows):
        for j in range(cols):
            q = qubit_index(i, j)

            # Local Z field
            Z_terms.append(("Z", [q], h))

            # Horizontal nearest neighbor (right)
            if j + 1 < cols:
                q_right = qubit_index(i, j + 1)
                XX_terms.append(("XX", [q, q_right], J))

            # Vertical nearest neighbor (down)
            if i + 1 < rows:
                q_down = qubit_index(i + 1, j)
                XX_terms.append(("XX", [q, q_down], J))

    # Periodic boundary terms (YZZ...ZY) for each row and column
    if periodic and num_qubits > 2:
        # Add one YZ-...ZY term for each row (horizontal periodicity)
        for i in range(rows):
            row_indices = [qubit_index(i, j) for j in range(cols)]
            if len(row_indices) > 1:
                pauli_str = "Y" + "Z" * (len(row_indices) - 2) + "Y"
                XX_terms.append((pauli_str, row_indices, J))

        # Add one YZ...ZY term for each column (vertical periodicity)
        for j in range(cols):
            col_indices = [qubit_index(i, j) for i in range(rows)]
            if len(col_indices) > 1:
                pauli_str = "Y" + "Z" * (len(col_indices) - 2) + "Y"
                XX_terms.append((pauli_str, col_indices, J))

    # Construct the SparsePauliOps
    Hxx = SparsePauliOp.from_sparse_list(XX_terms, num_qubits)
    Hz = SparsePauliOp.from_sparse_list(Z_terms, num_qubits)
    H = Hxx + Hz

    return H, Hxx, Hz

