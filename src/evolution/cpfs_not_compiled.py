import numpy as np
from scipy import linalg
from tqdm.notebook import tqdm
from scipy.special import bernoulli
from src.utils import operator_norm, mtx_commutator
from src.PFs import *

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

# error for CPFs
def CPF_error(order, H, A, B, t, r):
    """
    Computer total simulation errors for PFs and CPFs of order 1 and 2.
    """
    tau = t/r
    lam = -1j*tau
    
    if order == 1:
        # symplectic corrector for PF1
        Csymp = (lam/2)*B + (lam**2/12)*mtx_commutator(A,B)
        # symmetric corrector for PF1
        Csym = (-lam**2 /4)*mtx_commutator(A,B) + (lam**3 /12)*mtx_commutator(B,mtx_commutator(A,B))
        # symplectic corrector used in composite corrector for PF1
        Csymp_used_in_com = +(lam**2 /12)*mtx_commutator(A,B)

    if order == 2:
        # symplectic corrector for PF2
        Csymp = (-lam**2 /24)*mtx_commutator(A,B)
        Csym  = (lam**3 /48)*mtx_commutator(B,mtx_commutator(A,B))

    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau_symp = symplectic_corrector(PFtau, Csymp)
    CPFtau_sym = symmetric_corrector(PFtau, Csym)
    if order == 1:
        CPFtau_com = composite_corrector(PFtau, Csym, Csymp_used_in_com)
    if order == 2:
        CPFtau_com = composite_corrector(PFtau, Csym, Csymp)

    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPF_symp = symplectic_corrector(PFt, Csymp)
    CPF_sym = np.linalg.matrix_power(CPFtau_sym, r)
    if order == 1:
        CPF_com = symplectic_corrector(CPF_sym, Csymp_used_in_com)
    if order == 2:
        CPF_com = symplectic_corrector(CPF_sym, Csymp)

    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_symp_step_error = operator_norm(Utau - CPFtau_symp)
    CPF_com_step_error = operator_norm(Utau - CPFtau_com)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_symp_error = operator_norm(U - CPF_symp)    
    CPF_com_error = operator_norm(U - CPF_com)

    return PF_step_error, CPF_symp_step_error, CPF_com_step_error, PF_error, CPF_symp_error, CPF_com_error

def CPF2_symp_error(H, A, B, t, r):
    """
    Computes total simulation errors for PF2 and CPF2 with only symplectic corrector.
    """
    tau = t/r
    lam = -1j*tau
    
    # symp corrector
    Csymp = (-lam**2 /24)*mtx_commutator(A, B)

    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PF2tau = PF(2, A, B, lam)
    CPF2tau_symp = symplectic_corrector(PF2tau, Csymp)

    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PF2   = np.linalg.matrix_power(PF2tau, r)
    CPF2_symp = symplectic_corrector(PF2, Csymp)

    # one-step errors
    PF2_step_error = operator_norm(Utau - PF2tau)
    CPF2_symp_step_error = operator_norm(Utau - CPF2tau_symp)
    
    # total errors
    PF2_error = operator_norm(U - PF2)
    CPF2_symp_error = operator_norm(U - CPF2_symp)
    
    return PF2_step_error, CPF2_symp_step_error, PF2_error, CPF2_symp_error
#============================================================
# error data for CPFs

def data_CPF_error(order, H, A, B, time_ticks, nsteps): 
    error_data = [CPF_error(order, H, A, B, t, nsteps) for t in tqdm(time_ticks)]
    return error_data

def data_CPF2_symp_error(H, A, B, time_ticks, nsteps): 
    error_data = [CPF2_symp_error(H, A, B, t, nsteps) for t in tqdm(time_ticks)]
    return error_data

#============================================================
# High-order CPFs for non-perturbed systems

def CPF2_com(A, B, lam):
    Csymp = -(lam**2 /24)*mtx_commutator(A,B)
    Csym = (lam**3 /48)*mtx_commutator(B,mtx_commutator(A,B))

    PF2lam = PF2(A, B, lam)
    CPF2lam_com = composite_corrector(PF2lam, Csym, Csymp)
    return CPF2lam_com

def CPF_com(order, A, B, lam): # order := 2k
    if order == 2:
        return CPF2_com(A, B, lam)
    else:
        ak = 1/(4-4**(1/(order+1)))
        outer = CPF_com(order-2, A, B, ak*lam).dot(CPF_com(order-2, A, B, ak*lam))
        inner = CPF_com(order-2, A, B, (1-4*ak)*lam) 
        return outer.dot(inner).dot(outer)

def CPF_com_error(order, H, A, B, t, r):
    
    tau = t/r
    lam = -1j*tau
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau = CPF_com(order, A, B, lam)
    
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

def data_CPF_com_error(order, H, A, B, time_ticks, nsteps): 
    error_data = [CPF_com_error(order, H, A, B, t, nsteps) for t in tqdm(time_ticks)]
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
        ad = mtx_commutator(A, ad)
    return b*(lam**n)*ad

def PF2_symp_corrector(order, A, B, lam): #order is an even number
    Csymp = 0
    for j in range(1,int(order/2)+1):
        Csymp += C(2*j,b(2*j),lam,A,B)
    return Csymp

def CPF2_symp(order, A, B, lam): #order is an even number
    Csymp = PF2_symp_corrector(order, A, B, lam)
    PF2lam = PF2(A, B, lam)
    CPF2lam = symplectic_corrector(PF2lam, Csymp)
    return CPF2lam

def CPF_perturbed_sys(order, PF_order, A, B, lam): # PF_order is used for CPF2k with k>=2
    
    if order == 2:
        return CPF2_symp(order, A, B, lam)
    if order ==4:
        pk = 1/(4-4**(1/(order-1)))
        outer = CPF2_symp(PF_order, A, B, pk*lam).dot(CPF2_symp(PF_order, A, B, pk*lam))
        inner = CPF2_symp(PF_order, A, B, (1-4*pk)*lam) 
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


def data_CPF_perturbed_sys_error(order, PF_order, H, A, B, time_ticks, nsteps):
    error_data = [CPF_perturbed_sys_error(order, PF_order, H, A, B, t, nsteps) for t in tqdm(time_ticks)]
    return error_data