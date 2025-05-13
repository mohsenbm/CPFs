import numpy as np
from scipy import linalg
from tqdm.notebook import tqdm
from scipy.special import bernoulli
from src.utils import operator_norm, mtx_commutator
from src.compilations import *
from src.PFs import *

#============================================================

def CPF_symp_error(order, H, A, B, t, r):
    """
    Computes total simulation errors for PF2 and for CPF2 with symplectic corrector.
    Only for order 1 and 2.
    """
    tau = t/r
    lam = -1j*tau
    
    if order == 1:
        #Symp corrector: Csymp = (lam/2)*B + (lam**2/12)*mtx_commutator(A,B)
        UC = compile_UC(1/2, 1/12, 0, A, B, lam)
        UnegC= compile_UC(-1/2, -1/12, 0, A, B, lam)
    if order == 2:
        #Symp corrector: Csymp = (-1/24)*(lam**2)*mtx_commutator(A,B)
        UC = compile_UC(0, -1/24, 0, A, B, lam)
        UnegC= compile_UC(0, +1/24, 0, A, B, lam)
    
    # ideal one-step evolution
    Utau = linalg.expm(H*lam)
    
    # approximate one-step evolutions
    PFtau = PF(order, A, B, lam)
    CPFtau_symp = UC.dot(PFtau).dot(UnegC)
    
    # ideal evolution
    U = linalg.expm(-1j*H*t)
    
    # approximate ideal evolutions
    PFt = np.linalg.matrix_power(PFtau, r)
    CPFt_symp = UC.dot(PFt).dot(UnegC)
    
    # one-step errors
    PF_step_error = operator_norm(Utau - PFtau)
    CPF_symp_step_error = operator_norm(Utau - CPFtau_symp)
    
    # total errors
    PF_error = operator_norm(U - PFt)
    CPF_symp_error = operator_norm(U - CPFt_symp)
    
    return PF_step_error, CPF_symp_step_error, PF_error, CPF_symp_error

def data_CPF_symp_error(order, H, A, B, time_ticks, nsteps): 
    error_data = [CPF_symp_error(order, H, A, B, t, nsteps) for t in tqdm(time_ticks)]
    return error_data