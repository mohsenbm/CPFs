import numpy as np
import pandas as pd

def operator_norm(A):
    return np.linalg.norm(A, 2)


def mtx_commutator(A,B):
    """
    Commutator of two input matrices A and B
    """
    return A.dot(B) - B.dot(A)


def commutator(X, Y):
    """
    Commutator of two Pauli operators X and Y.
    """
    return (X @ Y - Y @ X).simplify(atol=0)

def show_mtx(mtx):
    """
    shows a matrix in a nice visual form using dataframe where zero is empty
    """
    x = pd.DataFrame(mtx)
    x[x.eq(0)] = ''
    return display(x)
