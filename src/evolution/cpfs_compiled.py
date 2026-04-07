import numpy as np
from scipy import linalg

from tqdm import tqdm

from scipy.special import bernoulli
from src.utils import operator_norm, mtx_commutator
from src.evolution.compilations import *
from src.evolution.pfs import *

#============================================================

# error for CPFs
def CPF_error(order, H, A, B, t, r):
    """
    Computes simulation errors for PFs and CPFs of order 1 and 2.\n
    Both symplectic and composite correctors are used.\n
    System can be perturbed or non-perturbed (pertubred = False).\n
    Returns:\n
    0: step error of PF\n
    1: step error of CPF with symp corrector\n
    2: step error of CPF with comp corrector\n
    3: total error of PF\n
    4: total error of CPF with symp corrector\n
    5: total error of CPF with comp corrector
    """
    tau = t/r
    lam = -1j*tau
    
    if order == 1:
        #Symp corrector: Csymp = (lam/2)*B + (lam**2/12)*[A,B]
        UC_symp = compile_UC(1/2, 1/12, 0, A, B, lam)
        UnegC_symp = compile_UC(-1/2, -1/12, 0, A, B, lam)

        #Sym corrector: Csym = -(lam**2/4)*[A,B] + (lam**3/12)[B,A,B]
        UC_sym = compile_UC(0, -1/4, 1/12, A, B, lam)

        #Symp corrector used in composite corrector: Csym = (lam**2/12)[A,B]
        UC_symp_used_in_Ccom = compile_UC(0, 1/12, 0, A, B, lam)
        UnegC_symp_used_in_Ccom = compile_UC(0, -1/12, 0, A, B, lam)

    if order == 2:
        #Sym corrector: Csym = (lam^3/48)[B,A,B]        
        UC_sym = compile_UC(0, 0, 1/48, A, B, lam)
        #Symp corrector: Csymp = -(lam^2/24)*[A,B]
        UC_symp = compile_UC(0, -1/24, 0, A, B, lam)
        UnegC_symp = compile_UC(0, +1/24, 0, A, B, lam)

    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau_symp = UC_symp.dot(PFtau).dot(UnegC_symp)
    CPFtau_sym = UC_sym.dot(PFtau).dot(UC_sym)

    if order == 1:
        CPFtau_com = UC_symp_used_in_Ccom.dot(CPFtau_sym).dot(UnegC_symp_used_in_Ccom)
    if order == 2:
        CPFtau_com = UC_symp.dot(CPFtau_sym).dot(UnegC_symp)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt_symp = UC_symp.dot(PFt).dot(UnegC_symp)
    CPFt_sym = np.linalg.matrix_power(CPFtau_sym, r)

    if order == 1:
        CPFt_com = UC_symp_used_in_Ccom.dot(CPFt_sym).dot(UnegC_symp_used_in_Ccom)
    if order == 2:
        CPFt_com = UC_symp.dot(CPFt_sym).dot(UnegC_symp)

    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_symp_step_error = operator_norm(Utau - CPFtau_symp)
    CPF_com_step_error = operator_norm(Utau - CPFtau_com)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_symp_error = operator_norm(U - CPFt_symp)    
    CPF_com_error = operator_norm(U - CPFt_com)

    return PF_step_error, CPF_symp_step_error, CPF_com_step_error, PF_error, CPF_symp_error, CPF_com_error

#============================================================
# error data for CPFs
def data_CPF_error(order, H, A, B, time_ticks, nsteps):
    """
    returns error data list for PFs and CPFs of orders 1 and 2:\n
    list 0: step error of PF\n
    list 1: step error of CPF with symp corrector\n
    list 2: step error of CPF with comp corrector\n
    list 3: total error of PF\n
    list 4: total error of CPF with symp corrector\n
    list 5: total error of CPF with comp corrector
    """
    error_data = [CPF_error(order, H, A, B, t, nsteps) for t in tqdm(time_ticks)]
    return error_data


#============================================================
# High-order CPFs with symmetric corrector for non-perturbed systems

def CPF2_sym(A, B, lam):
    #Sym corrector: Csym = (lam^3/48)[A+2B,A,B] 
    UC_sym = compile_UC_CPF2_sym(1/48, A, B, lam)
    CPF2_sym = UC_sym.dot(PF2(A, B, lam)).dot(UC_sym)
    return CPF2_sym


def CPF_sym(order, A, B, lam): # order := 2k
    if order == 2:
        return CPF2_sym(A, B, lam)
    else:
        ak = 1/(4-4**(1/(order+1)))
        outer = CPF_sym(order-2, A, B, ak*lam).dot(CPF_sym(order-2, A, B, ak*lam))
        inner = CPF_sym(order-2, A, B, (1-4*ak)*lam) 
        return outer.dot(inner).dot(outer)

def CPF_sym_error(order, H, A, B, t, r):
    tau = t/r
    lam = -1j*tau

    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau = CPF_sym(order, A, B, lam)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt = np.linalg.matrix_power(CPFtau, r)
    
    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_sym_step_error = operator_norm(Utau - CPFtau)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_sym_error = operator_norm(U - CPFt)
    
    return PF_step_error, CPF_sym_step_error, PF_error, CPF_sym_error

#============================================================
# error data for high-order CPFs with symmetric corrector
def data_CPF_sym_error(order, H, A, B, ticks, nsteps): 
    error_data = [CPF_sym_error(order, H, A, B, t, nsteps) for t in tqdm(ticks)]
    return error_data



#============================================================
# Error for fixed step size (fixed tau)

def CPF_error_fixed_step(order, H, A, B, tau, r):
    """
    Computes simulation errors for PFs and CPFs of order 1 and 2.\n
    Both symplectic and composite correctors are used.\n
    System can be perturbed or non-perturbed (pertubred = False).\n
    Returns:\n
    0: step error of PF\n
    1: step error of CPF with symp corrector\n
    2: step error of CPF with comp corrector\n
    3: total error of PF\n
    4: total error of CPF with symp corrector\n
    5: total error of CPF with comp corrector
    """

    lam = -1j*tau
    
    if order == 1:
        #Symp corrector: Csymp = (lam/2)*B + (lam**2/12)*[A,B]
        UC_symp = compile_UC(1/2, 1/12, 0, A, B, lam)
        UnegC_symp = compile_UC(-1/2, -1/12, 0, A, B, lam)

        #Sym corrector: Csym = -(lam**2/4)*[A,B] + (lam**3/12)[B,A,B]
        UC_sym = compile_UC(0, -1/4, 1/12, A, B, lam)

        #Symp corrector used in composite corrector: Csym = (lam**2/12)[A,B]
        UC_symp_used_in_Ccom = compile_UC(0, 1/12, 0, A, B, lam)
        UnegC_symp_used_in_Ccom = compile_UC(0, -1/12, 0, A, B, lam)

    if order == 2:
        #Sym corrector: Csym = (lam^3/48)[B,A,B]        
        UC_sym = compile_UC(0, 0, 1/48, A, B, lam)
        #Symp corrector: Csymp = -(lam^2/24)*[A,B]
        UC_symp = compile_UC(0, -1/24, 0, A, B, lam)
        UnegC_symp = compile_UC(0, +1/24, 0, A, B, lam)

    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau_symp = UC_symp.dot(PFtau).dot(UnegC_symp)
    CPFtau_sym = UC_sym.dot(PFtau).dot(UC_sym)

    if order == 1:
        CPFtau_com = UC_symp_used_in_Ccom.dot(CPFtau_sym).dot(UnegC_symp_used_in_Ccom)
    if order == 2:
        CPFtau_com = UC_symp.dot(CPFtau_sym).dot(UnegC_symp)
    
    # ideal evolution
    t = r*tau
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt_symp = UC_symp.dot(PFt).dot(UnegC_symp)
    CPFt_sym = np.linalg.matrix_power(CPFtau_sym, r)

    if order == 1:
        CPFt_com = UC_symp_used_in_Ccom.dot(CPFt_sym).dot(UnegC_symp_used_in_Ccom)
    if order == 2:
        CPFt_com = UC_symp.dot(CPFt_sym).dot(UnegC_symp)

    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_symp_step_error = operator_norm(Utau - CPFtau_symp)
    CPF_com_step_error = operator_norm(Utau - CPFtau_com)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_symp_error = operator_norm(U - CPFt_symp)    
    CPF_com_error = operator_norm(U - CPFt_com)

    return PF_step_error, CPF_symp_step_error, CPF_com_step_error, PF_error, CPF_symp_error, CPF_com_error

#============================================================
# error data for CPFs
def data_CPF_error_fixed_step(order, H, A, B, tau, steps_list):
    """
    returns error data list for PFs and CPFs of orders 1 and 2:\n
    list 0: step error of PF\n
    list 1: step error of CPF with symp corrector\n
    list 2: step error of CPF with comp corrector\n
    list 3: total error of PF\n
    list 4: total error of CPF with symp corrector\n
    list 5: total error of CPF with comp corrector
    """
    error_data = [CPF_error_fixed_step(order, H, A, B, tau, r) for r in tqdm(steps_list)]
    return error_data