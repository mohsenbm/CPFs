import numpy as np
from scipy import linalg
from tqdm.notebook import tqdm
from scipy.special import bernoulli
from src.utils import operator_norm, mtx_commutator



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

#============================================================
# types of correctors

def symplectic_corrector(U, C): # works for matrix inputs
    
    UC= linalg.expm(C)
    UnegC= linalg.expm(-1*C)
    
    return UC.dot(U).dot(UnegC)

def symmetric_corrector(U, C): # works for matrix inputs
    
    UC= linalg.expm(C)
    
    return UC.dot(U).dot(UC)

def composite_corrector(U, Csym, Csymp): # works for matrix inputs
    
    UCsym= linalg.expm(Csym)
    UC= linalg.expm(Csymp)
    UnegC= linalg.expm(-1*Csymp)
    
    return UC.dot(UCsym).dot(U).dot(UCsym).dot(UnegC)

#============================================================
# error for corrected product formulas (CPFs)

def CPF1_error(H, A, B, t, r):
    
    tau = t/r
    lam = -1j*tau
    
    # correctors
    Csymp = (lam/2)*B + (lam**2/12)*mtx_commutator(A,B)
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PF1tau = PF(1, A, B, lam)
    PF1tauCsymp = symplectic_corrector(PF1tau, Csymp)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PF1 = np.linalg.matrix_power(PF1tau, r)
    PF1Csymp = symplectic_corrector(PF1, Csymp)
    
    # one-step errors
    PF1_step_error = operator_norm(Utau - PF1tau)
    PF1Csymp_step_error = operator_norm(Utau - PF1tauCsymp)
    
    # total errors
    PF1_error = operator_norm(U - PF1)
    PF1Csymp_error = operator_norm(U - PF1Csymp)
    
    
    # composite corrector
    Csym = (-lam**2 /4)*mtx_commutator(A,B) + (lam**3 /12)*mtx_commutator(B,mtx_commutator(A,B))
    
    PF1tauCsym = symmetric_corrector(PF1tau, Csym)
    
    Csymp2 = +(lam**2 /12)*mtx_commutator(A,B)
    PF1tauCcom = composite_corrector(PF1tau, Csym, Csymp2)
    
    # one-step error for composite corrector
    PF1Ccom_step_error = operator_norm(Utau - PF1tauCcom)
    
    # total error for composite corrector
    PF1Csym = np.linalg.matrix_power(PF1tauCsym, r)
    PF1Ccom = symplectic_corrector(PF1Csym, Csymp2)
    
    PF1Ccom_error = operator_norm(U - PF1Ccom)
    

    return PF1_step_error, PF1Csymp_step_error, PF1Ccom_step_error, PF1_error, PF1Csymp_error, PF1Ccom_error

def CPF2_error(H, A, B, t, r):
    
    tau = t/r
    lam = -1j*tau
    
    # correctors
    Csymp = (-lam**2 /24)*mtx_commutator(A,B)
    Csym  = (lam**3 /48)*mtx_commutator(B,mtx_commutator(A,B))
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PF2tau = PF(2, A, B, lam)
    PF2tauCsymp = symplectic_corrector(PF2tau, Csymp)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PF2   = np.linalg.matrix_power(PF2tau, r)
    PF2Csymp = symplectic_corrector(PF2, Csymp)
    
    # composite corrector
    PF2tauCsym = symmetric_corrector(PF2tau, Csym)
    PF2tauCcom = composite_corrector(PF2tau, Csym, Csymp)
    
    PF2Csym  = np.linalg.matrix_power(PF2tauCsym, r)
    PF2Ccom = symplectic_corrector(PF2Csym, Csymp)
    
    # one-step errors
    PF2_step_error = operator_norm(Utau - PF2tau)
    PF2Csymp_step_error = operator_norm(Utau - PF2tauCsymp)
    PF2Ccom_step_error = operator_norm(Utau - PF2tauCcom)
    
    # total errors
    PF2_error = operator_norm(U - PF2)
    PF2Csymp_error = operator_norm(U - PF2Csymp)
    PF2Ccom_error = operator_norm(U - PF2Ccom)
    
    return PF2_step_error, PF2Csymp_step_error, PF2Ccom_step_error, PF2_error, PF2Csymp_error, PF2Ccom_error

def CPF2_symp_error(mtxH, mtxA, mtxB, t, r):
    """
    Computer timestep and full errors for PF2 and CPF2 with only the symplectic corrector.
    """
    
    tau = t/r
    lam = -1j*tau
    
    
    # symp corrector
    Csymp = (-lam**2 /24)*mtx_commutator(mtxA,mtxB)

    
    # ideal one-step evolution
    Utau = linalg.expm(mtxH*lam)
    
    # approximate one-step evolutions
    PF2tau = PF(2, mtxA, mtxB, lam)
    PF2tauCsymp = symplectic_corrector(PF2tau, Csymp)

    
    # ideal evolution
    U = linalg.expm(-1j*mtxH*t)
    
    # approximate ideal evolutions
    PF2   = np.linalg.matrix_power(PF2tau, r)
    PF2Csymp = symplectic_corrector(PF2, Csymp)

    
    # one-step errors
    PF2_step_error = operator_norm(Utau - PF2tau)
    PF2Csymp_step_error = operator_norm(Utau - PF2tauCsymp)
    
    # total errors
    PF2_error = operator_norm(U - PF2)
    PF2Csymp_error = operator_norm(U - PF2Csymp)
    
    return PF2_step_error, PF2Csymp_step_error, PF2_error, PF2Csymp_error

#============================================================
# error data for CPFs
def data_CPF1_error(mtxH, mtxA, mtxB, time_ticks, num_steps): 

    error_data = [CPF1_error(mtxH, mtxA, mtxB, t, num_steps) for t in tqdm(time_ticks)]

    return error_data

def data_CPF2_error(mtxH, mtxA, mtxB, time_ticks, num_steps): 

    error_data = [CPF2_error(mtxH, mtxA, mtxB, t, num_steps) for t in tqdm(time_ticks)]

    return error_data

def data_CPF2_symp_error(mtxH, mtxA, mtxB, time_ticks, num_steps): 
    """
    collect the timestep and full errors for PF2 and CPF2 with simplectic corrector at various times t. 
    """

    error_data = [CPF2_symp_error(mtxH, mtxA, mtxB, t, num_steps) for t in tqdm(time_ticks)]

    return error_data

#============================================================
# High-order CPFs for non-perturbed systems

def CPF2comp(A, B, lam):
    
    Csymp = -(lam**2 /24)*mtx_commutator(A,B)
    Csym = (lam**3 /48)*mtx_commutator(B,mtx_commutator(A,B))
    
    PF2lam = PF2(A, B, lam)
    
    compPF2lam = composite_corrector(PF2lam, Csym, Csymp)
    
    return compPF2lam

def CPFcomp(order, A, B, lam): # order := 2k
    
    if order == 2:
        return CPF2comp(A, B, lam)

    else:
        ak = 1/(4-4**(1/(order+1)))
        outer = CPFcomp(order-2, A, B, ak*lam).dot(CPFcomp(order-2, A, B, ak*lam))
        inner = CPFcomp(order-2, A, B, (1-4*ak)*lam) 
    
        return outer.dot(inner).dot(outer)

def CPFcomp_error(order, H, A, B, t, r):
    
    tau = t/r
    lam = -1j*tau
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order,A,B,lam)
    CPFtau = CPFcomp(order,A, B, lam)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt = np.linalg.matrix_power(CPFtau, r)
    
    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_step_error = operator_norm(Utau - CPFtau)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_error = operator_norm(U - CPFt)
    
    return PF_step_error, CPF_step_error, PF_error, CPF_error

def data_CPFcomp_error(order, H, A, B, time_ticks, num_steps): 

    error_data = [CPFcomp_error(order, H, A, B, t, num_steps) for t in tqdm(time_ticks)]

    return error_data

#============================================================
# High-order CPFs for non-perturbed systems

def bern_poly(n): # Bernoulli polynomial B_n(x) evaluated at x=1/2
    
    return (2**(1-n)-1)*bernoulli(n)[n]

def b(n):
    return bern_poly(n)/np.math.factorial(n)

def C(n,b,lam, A, B):
    
    ad = B
    for j in range(n-1):
        ad = mtx_commutator(A,ad)
    return b*(lam**n)*ad

def PF2symp_corrector(order, A, B, lam): #order is an even number
    Csymp = 0
    for j in range(1,int(order/2)+1):
        Csymp += C(2*j,b(2*j),lam,A,B)
    return Csymp

def CPF2symp(order, A, B, lam): #order is an even number
    
    Csymp = PF2symp_corrector(order, A, B, lam)
    
    PF2lam = PF2(A, B, lam)
    
    CPF2lam = symplectic_corrector(PF2lam, Csymp)
    
    return CPF2lam

def CPF_perturbed_sys(order, PF_order, A, B, lam): # PF_order is used for CPF2k with k>=2
    
    if order == 2:
        return CPF2symp(order, A, B, lam)
    
    if order ==4:
        pk = 1/(4-4**(1/(order-1)))
        outer = CPF2symp(PF_order, A, B, pk*lam).dot(CPF2symp(PF_order, A, B, pk*lam))
        inner = CPF2symp(PF_order, A, B, (1-4*pk)*lam) 
        return outer.dot(inner).dot(outer)
    
    else:
        pk = 1/(4-4**(1/(order-1)))
    
        outer = CPF_perturbed_sys(order-2, PF_order, A, B, pk*lam).dot(CPF_perturbed_sys(order-2, PF_order, A, B, pk*lam))
        inner = CPF_perturbed_sys(order-2, PF_order, A, B, (1-4*pk)*lam) 
    
        return outer.dot(inner).dot(outer)
    
def CPF_perturbed_sys_error(order, PF_order, H, A, B, t, r):
    tau = t/r
    lam = -1j*tau
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau = CPF_perturbed_sys(order, PF_order, A, B, lam)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt = np.linalg.matrix_power(CPFtau, r)
    
    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_step_error = operator_norm(Utau - CPFtau)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_error = operator_norm(U - CPFt)
    
    return PF_step_error, CPF_step_error, PF_error, CPF_error


def data_CPF_perturbed_sys_error(order, PF_order, H, A, B, time_ticks, num_steps):
    
    error_data = [CPF_perturbed_sys_error(order, PF_order, H, A, B, t, num_steps) for t in tqdm(time_ticks)]
    
    return error_data

#============================================================
# compilation for correctors

def X(lam,a,b,A,B):
    
    UA = linalg.expm(a*lam*A)
    UB = linalg.expm(b*lam*B)
    UnegA = linalg.expm(-a*lam*A)
    
    return UA.dot(UB).dot(UnegA)
                          
def Y(lam,a,b,A,B):
    return X(lam,a,b,A,B).dot(X(lam,-a,-b,A,B))

def Z(lam,a,b,c,A,B):
    UcB = linalg.expm(c*lam*B)
    return UcB.dot(Y(lam,a,b,A,B)).dot(UcB)


def corrector(lam,A,B,c1,c2,c3):
    return c1*lam*B + c2*(lam**2)*mtx_commutator(A,B) + c3*(lam**3)*mtx_commutator(B,mtx_commutator(A,B))

def compile_UC(lam,A,B,c1,c2,c3):
    
    if c1 == 0 and c2 != 0 != c3:       # compilation for c2*lam^2[A,B] + c3*lam^3[B,A,B]
        a = c2**2/(4*c3)
        b = 2*(c3/c2)
        compiledUC = Y(lam,a,b,A,B)
    
    if c2 != 0 and c1 == 0 == c3: # compilation for c2*lam^2[A,B]
        a = c2/2
        b = 1
        compiledUC = linalg.expm(-lam*B/2).dot(Y(lam,a,b,A,B)).dot(linalg.expm(lam*B/2))
    
    if c3 == 0 and c1 !=0 != c2:       # compilation for c1*B + c2*lam^2[A,B]
        a = c2/4
        b = 1
        compiledUC = Z(lam,a,b,c1/4,A,B).dot(Z(lam,-a,-b,c1/4,A,B))
    
    if c3 !=0 and c1 == 0 == c2: # compilation for c3*lam^3[B,A,B]
        a = c3/2
        b = 1
        compiledUC = Y(lam,a,b,A,B).dot(Y(lam,a,-b,A,B))
    
    return compiledUC

def compilation_error(UC, compiledUC):
    return operator_norm(UC - compiledUC)


def symplectic_corrector(U, C): # works for matrix inputs
    
    UC= linalg.expm(C)
    UnegC= linalg.expm(-1*C)
    
    return UC.dot(U).dot(UnegC)

def CPF1_symp_error(H, A, B, t, r):
    
    tau = t/r
    lam = -1j*tau
    
    # correctors
    #Csymp = (lam/2)*B + (lam**2/12)*mtx_commutator(A,B)
    
    UC = compile_UC(lam,A,B,1/2,1/12,0)
    UnegC= compile_UC(lam,A,B,-1/2,-1/12,0)
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PF1tau = PF(1, A, B, lam)
    PF1tauCsymp = UC.dot(PF1tau).dot(UnegC)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PF1 = np.linalg.matrix_power(PF1tau, r)
    PF1Csymp = UC.dot(PF1).dot(UnegC)
    
    # one-step errors
    PF1_step_error = operator_norm(Utau - PF1tau)
    PF1Csymp_step_error = operator_norm(Utau - PF1tauCsymp)
    
    # total errors
    PF1_error = operator_norm(U - PF1)
    PF1Csymp_error = operator_norm(U - PF1Csymp)
    

    return PF1_step_error, PF1Csymp_step_error, PF1_error, PF1Csymp_error

def CPF2_symp_error(mtxH, mtxA, mtxB, t, r):
    """
    Computer timestep and full errors for PF2 and CPF2 with only the symplectic corrector.
    """
    
    tau = t/r
    lam = -1j*tau
    
    UC = compile_UC(lam,mtxA,mtxB,0,-1/24,0)
    UnegC= compile_UC(lam,mtxA,mtxB,0,+1/24,0)

    
    # ideal one-step evolution
    Utau = linalg.expm(mtxH*lam)
    
    # approximate one-step evolutions
    PF2tau = PF(2, mtxA, mtxB, lam)
    PF2tauCsymp = UC.dot(PF2tau).dot(UnegC)

    
    # ideal evolution
    U = linalg.expm(-1j*mtxH*t)
    
    # approximate ideal evolutions
    PF2   = np.linalg.matrix_power(PF2tau, r)
    PF2Csymp = UC.dot(PF2).dot(UnegC)

    
    # one-step errors
    PF2_step_error = operator_norm(Utau - PF2tau)
    PF2Csymp_step_error = operator_norm(Utau - PF2tauCsymp)
    
    # total errors
    PF2_error = operator_norm(U - PF2)
    PF2Csymp_error = operator_norm(U - PF2Csymp)
    
    return PF2_step_error, PF2Csymp_step_error, PF2_error, PF2Csymp_error

def data_CPF1_symp_error(mtxH, mtxA, mtxB, time_ticks, num_steps): 

    error_data = [CPF1_symp_error(mtxH, mtxA, mtxB, t, num_steps) for t in tqdm(time_ticks)]

    return error_data

def data_CPF2_symp_error(mtxH, mtxA, mtxB, time_ticks, num_steps): 

    error_data = [CPF2_symp_error(mtxH, mtxA, mtxB, t, num_steps) for t in tqdm(time_ticks)]

    return error_data
