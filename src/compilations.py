import numpy as np
from scipy import linalg
from src.utils import operator_norm, mtx_commutator

#============================================================
# compilation for correctors

def X(a, b, A, B, lam):
    
    UA = linalg.expm(a*lam*A)
    UB = linalg.expm(b*lam*B)
    UnegA = linalg.expm(-a*lam*A)
    
    return UA.dot(UB).dot(UnegA)
                          
def Y(a, b, A, B, lam):
    return X(lam, a, b, A, B).dot(X(lam, -a, -b, A, B))

def Z(a, b, c, A, B, lam):
    UcB = linalg.expm(c*lam*B)
    return UcB.dot(Y(a, b, A, B, lam)).dot(UcB)


def corrector(c1, c2, c3, A, B, lam):
    return c1*lam*B + c2*(lam**2)*mtx_commutator(A,B) + c3*(lam**3)*mtx_commutator(B, mtx_commutator(A,B))

#============================================================

def commutator_compilation(c, A, B, lam):
    """
    Compilation for W(c) = exp(c*lam^2 [A,B]) = exp(-c*tau^2 [A,B]) where lam = -1j*tau
    """
    a1 = (np.sqrt(5)-1)/2
    b1 = -(np.sqrt(5)+1)/2
    a2 = (3-np.sqrt(5))/2

    Ua1cA  = linalg.expm(a1*c*lam*A)
    Ua1B   = linalg.expm(a1*lam*B)
    UnegcA = linalg.expm(-c*lam*A)
    Ub1B   = linalg.expm(b1*lam*B)
    Ua2cA  = linalg.expm(a2*c*lam*A)
    UB     = linalg.expm(lam*B)

    return Ua1cA.dot(Ua1B).dot(UnegcA).dot(Ub1B).dot(Ua2cA).dot(UB) 

def linear_plus_commutator_compilation(b, c, A, B, lam):
    """
    Compilation for exp(C) with C = b*lam*B + c*lam^2*[A,B] where lam = -1j*tau.
    This provides compilation for exp(lam*b*B/2)W(a)exp(lam*b*B/2),
    where W(a) is the commutator compilcation.
    """
    a1 = (np.sqrt(5)-1)/2
    b1 = -(np.sqrt(5)+1)/2
    a2 = (3-np.sqrt(5))/2

    UbB2 = linalg.expm(b*lam/2*B)
    Ua1cA = linalg.expm(a1*c*lam*A)
    Ua1B  = linalg.expm(a1*lam*B)
    UnegcA = linalg.expm(-c*lam*A)
    Ub1B  = linalg.expm(b1*lam*B)
    Ua2cA = linalg.expm(a2*c*lam*A)
    U1plusbB2 = linalg.expm((1+b/2)*lam*B)
    return UbB2.dot(Ua1cA).dot(Ua1B).dot(UnegcA).dot(Ub1B).dot(Ua2cA).dot(U1plusbB2)


def compile_UC(c1, c2, c3, A, B, lam):
    """"
    Provides compilation for exp(C)
    where C = c1*lam*B + c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau.
    """
    
    # compilation for c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau
    if c1 == 0 and c2 != 0 != c3:
        a = c2**2/(4*c3)
        b = 2*(c3/c2)
        return Y(a, b, A, B, lam)
    
    # compilation for c2*lam^2[A,B] with lam = -1j*tau
    if c2 != 0 and c1 == 0 == c3:
        return commutator_compilation(c2, A, B, lam)
        # below is an alternative approach
        #a, b = c2/2, 1
        #return linalg.expm(-lam*B/2).dot(Y(a, b, A, B, lam)).dot(linalg.expm(lam*B/2))
    
    # compilation for c1*B + c2*lam^2[A,B]
    if c3 == 0 and c1 != 0 != c2:
        return linear_plus_commutator_compilation(c1, c2, A, B, lam)
        # below is an alternative approach
        #a, b = c2/4, 1
        #return Z(a, b, c1/4, A, B, lam).dot(Z(-a, -b, c1/4, A, B, lam))
    
    # compilation for c3*lam^3[B,A,B]
    if c3 !=0 and c1 == 0 == c2:
        a, b = c3/2, 1
        return Y(a, b, A, B, lam).dot(Y(a, -b, A, B, lam))
    

def compile_UnegC(c1, c2, c3, A, B, lam):
    """
    Compilation for exp(-C)
    """
    return compile_UC(-c1, -c2, -c3, A, B, lam)

def compilation_error(UC, compiledUC):
    return operator_norm(UC - compiledUC)