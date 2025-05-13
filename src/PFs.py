
import numpy as np
from scipy import linalg
from src.utils import operator_norm

#============================================================
# Standard product formulas
def PF1(A, B, lam):
    UA = linalg.expm(A*lam)
    UB = linalg.expm(B*lam)
    return UA.dot(UB)

def PF2(A, B, lam):
    UA2 = linalg.expm(A*lam/2)
    UB = linalg.expm(B*lam)
    return UA2.dot(UB).dot(UA2)

def PF4(A, B, lam):
    p2 = 1/(4-4**(1/(4-1)))
    outer = PF2(A, B, p2*lam).dot(PF2(A, B, p2*lam))
    inner = PF2(A, B, (1-4*p2)*lam)
    return outer.dot(inner).dot(outer)

def PF(order, A, B, lam):
    if order == 1:
        return PF1(A, B, lam)
    elif order == 2:
        return PF2(A, B, lam)
    else:
        p = 1/(4-4**(1/(order-1)))
        outer = PF(order-2, A, B, p*lam).dot(PF(order-2, A, B, p*lam))
        inner = PF(order-2, A, B, (1-4*p)*lam)
        return outer.dot(inner).dot(outer)

def PF_error(order, H, A, B, t, r):
    tau = t/r
    lam = -1j*tau

    U = linalg.expm(-1j*t*H)
    Utau = linalg.expm(lam*H)
    
    appxUtau = PF(order, A, B, lam)
    appxU = np.linalg.matrix_power(appxUtau, r)
    
    PF_step_error= operator_norm(Utau - appxUtau)
    PF_error = operator_norm(U - appxU)
    
    return PF_step_error, PF_error