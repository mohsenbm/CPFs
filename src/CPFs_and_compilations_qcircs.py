import numpy as np
from qiskit.quantum_info import Operator
from src.utils import operator_norm
from ising_model_qcircs import *


def PF1_qcirc(A, B, tau, ppart): # ppart here specifies the perturbation part of the Hamiltonian
    if ppart == 'Hxx':
        UA = Hz_evolution_qcirc(A, tau)
        UB = Hxx_evolution_qcirc(B, tau)
    if ppart == 'Hz':
        UA = Hxx_evolution_qcirc(A, tau)
        UB = Hz_evolution_qcirc(B, tau)
    return UB.compose(UA) # this is equal to UA UB in the matrix form

def PF2_qcirc(A, B, tau, ppart):
    if ppart == 'Hxx':
        UA2 = Hz_evolution_qcirc(A, tau/2)
        UB = Hxx_evolution_qcirc(B, tau)
    if ppart == 'Hz':
        UA2 = Hxx_evolution_qcirc(A, tau/2)
        UB = Hz_evolution_qcirc(B, tau)
    return UA2.compose(UB).compose(UA2)

def PF_qcirc(order, A, B, tau, ppart, reps=1):

    PF1 = PF1_qcirc(A, B, tau, ppart)
    UPF1 = PF1

    if order == 1:
        for _ in range(reps-1):
            UPF1 = UPF1.compose(PF1)
        return UPF1
    
    if order == 2:
        if reps == 1:
            return PF2_qcirc(A, B, tau, ppart)
        
        for _ in range(reps-2):
            UPF1 = UPF1.compose(PF1)
        if ppart == 'Hxx':
            UA2 = Hz_evolution_qcirc(A, tau/2)
            UB = Hxx_evolution_qcirc(B, tau)
        if ppart == 'Hz':
            UA2 = Hxx_evolution_qcirc(A, tau/2)
            UB = Hz_evolution_qcirc(B, tau)

        UPF2 = UA2.compose(UPF1).compose(UB).compose(UA2)
        return UPF2
    
def X_qcirc(a, b, A, B, tau, ppart):
    """
    Generates quantum circuit for implementing X = exp(a*lam*A) exp(b*lam*B) exp(-a*lam*A),
    where lam = -1j*tau. The evolutin circuits below implement exp(-1j*H*t),
    where H is the first argument and t is the second argument.
    So to implement e.g. exp(a*lam*A) we pass A as first argument and a*tau as the second argument.
    """
    if ppart == 'Hxx':
        UA = Hz_evolution_qcirc(A, a*tau)
        UB = Hxx_evolution_qcirc(B, b*tau)
        UnegA = Hz_evolution_qcirc(A, -a*tau)
    if ppart == 'Hz':
        UA = Hxx_evolution_qcirc(A, a*tau)
        UB = Hz_evolution_qcirc(B, b*tau)
        UnegA = Hxx_evolution_qcirc(A, -a*tau)
    return UnegA.compose(UB).compose(UA)

def Y_qcirc(a, b, A, B, tau, ppart):
    """"
    Generates quantum circuit for implementing
    Y = exp(a*lam*A) exp(b*lam*B) exp(-2*a*lam*A) exp(-b*lam*B) exp(a*lam*A),
    where lam = -1j*tau.
    """
    if ppart == 'Hxx':
        UA = Hz_evolution_qcirc(A, a*tau)
        Uneg2A = Hz_evolution_qcirc(A, -2*a*tau)
        UB = Hxx_evolution_qcirc(B, b*tau)
        UnegB = Hxx_evolution_qcirc(B, -b*tau)
        return UA.compose(UnegB).compose(Uneg2A).compose(UB).compose(UA)

def Z_qcirc(a, b, c, A, B, tau, ppart):
    if ppart == 'Hxx':
        UcB = Hxx_evolution_qcirc(B, c*tau)
    if ppart == 'Hz':
        UcB = Hz_evolution_qcirc(B, c*tau)
    return UcB.compose(Y_qcirc(a, b, A, B, tau, ppart)).compose(UcB)

def commutator_compilation_qcirc(c, tau, A, B, ppart):
    """
    Generates quantum circuit for implementing W(c) = exp(c*lam^2 [A,B]) = exp(-c*tau^2 [A,B]) where lam = -1j*tau
    The evolutin circuits below implement exp(-1j*H*t), where H is the first argument and t is the second argument.
    So to implement e.g. exp(a*lam*A) = exp(-1j*a*tau*A) we pass A as first argument and a*tau as the second argument.
    """
    a1 = (np.sqrt(5)-1)/2
    b1 = -(np.sqrt(5)+1)/2
    a2 = (3-np.sqrt(5))/2
    if ppart == 'Hxx':
        Ua1cA = Hz_evolution_qcirc(A, a1*c*tau)
        Ua1B = Hxx_evolution_qcirc(B, a1*tau)
        UnegcA = Hz_evolution_qcirc(A, -c*tau)
        Ub1B = Hxx_evolution_qcirc(B, b1*tau)
        Ua2cA = Hz_evolution_qcirc(A, a2*c*tau)
        UB = Hxx_evolution_qcirc(B, tau)
        return UB.compose(Ua2cA).compose(Ub1B).compose(UnegcA).compose(Ua1B).compose(Ua1cA)

def linear_plus_commutator_compilation_qcirc(b, c, tau, A, B, ppart):
    """
    Generates quantum circuit for implementing exp(C) with C = b*lam*B + c*lam^2*[A,B] where lam = -1j*tau.
    This implements exp(lam*b*B/2)W(a)exp(lam*b*B/2), where W(a) is the commutator compilcation.
    The evolutin circuits below implement exp(-1j*H*t), where H is the first argument and t is the second argument.
    So to implement e.g. exp(a*lam*A) = exp(-1j*a*tau*A) we pass A as first argument and a*tau as the second argument.
    """
    a1 = (np.sqrt(5)-1)/2
    b1 = -(np.sqrt(5)+1)/2
    a2 = (3-np.sqrt(5))/2
    if ppart == 'Hxx':
        UbB2 = Hxx_evolution_qcirc(B, b*tau/2)
        Ua1cA = Hz_evolution_qcirc(A, a1*c*tau)
        Ua1B = Hxx_evolution_qcirc(B, a1*tau)
        UnegcA = Hz_evolution_qcirc(A, -c*tau)
        Ub1B = Hxx_evolution_qcirc(B, b1*tau)
        Ua2cA = Hz_evolution_qcirc(A, a2*c*tau)
        U1plusbB2 = Hxx_evolution_qcirc(B, (1+b/2)*tau)
        return U1plusbB2.compose(Ua2cA).compose(Ub1B).compose(UnegcA).compose(Ua1B).compose(Ua1cA).compose(UbB2)

def UC_qcirc(c1, c2, c3, A, B, tau, ppart):
    """"
    Generates quantum circuit for implementing exp(C)
    with C = c1*lam*B + c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau.
    """
    
    # compilation for c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau
    if c1 == 0 and c2 != 0 != c3:
        a = c2**2/(4*c3)
        b = 2*(c3/c2)
        return Y_qcirc(a, b, A, B, tau, ppart)
    
    # compilation for c2*lam^2[A,B] with lam = -1j*tau
    if c2 != 0 and c1 == 0 == c3:
        return commutator_compilation_qcirc(c2, tau, A, B, ppart)
        # below is an alternative approach
        #a, b = c2/2, 1
        #if ppart == 'Hxx':
        #    return Hxx_evolution_qcirc(B, tau/2).compose(Y_qcirc(a, b, A, B, tau, ppart)).compose(Hxx_evolution_qcirc(B, -tau/2))
        #if ppart == 'Hz':
        #    return Hz_evolution_qcirc(B, tau/2).compose(Y_qcirc(a, b, A, B, tau, ppart)).compose(Hz_evolution_qcirc(B, -tau/2))
    
    # compilation for c1*B + c2*lam^2[A,B]
    if c3 == 0 and c1 != 0 != c2:
        return linear_plus_commutator_compilation_qcirc(c1, c2, tau, A, B, ppart)
        # below is an alternative approach
        #a, b = c2/4, 1
        #return Z_qcirc(-a, -b, c1/4, A, B, tau, ppart).compose(Z_qcirc(a, b, c1/4, A, B, tau, ppart))
    
    # compilation for c3*lam^3[B,A,B]
    if c3 !=0 and c1 == 0 == c2:
        a, b = c3/2, 1
        return Y_qcirc(a, -b, A, B, tau, ppart).compose(Y_qcirc(a, b, A, B, tau, ppart))
    
def UC_qcirc0(c1, c2, c3, A, B, tau, ppart):
    """"
    Generates quantum circuit for implementing exp(C)
    with C = c1*lam*B + c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau.
    """
    
    # compilation for c2*lam^2 [A,B] + c3*lam^3 [B,A,B] with lam = -1j*tau
    if c1 == 0 and c2 != 0 != c3:
        a = c2**2/(4*c3)
        b = 2*(c3/c2)
        return Y_qcirc(a, b, A, B, tau, ppart)
    
    # compilation for c2*lam^2[A,B] with lam = -1j*tau
    if c2 != 0 and c1 == 0 == c3:
        a, b = c2/2, 1
        if ppart == 'Hxx':
            return Hxx_evolution_qcirc(B, tau/2).compose(Y_qcirc(a, b, A, B, tau, ppart)).compose(Hxx_evolution_qcirc(B, -tau/2))
        if ppart == 'Hz':
            return Hz_evolution_qcirc(B, tau/2).compose(Y_qcirc(a, b, A, B, tau, ppart)).compose(Hz_evolution_qcirc(B, -tau/2))
    
    # compilation for c1*B + c2*lam^2[A,B]
    if c3 == 0 and c1 != 0 != c2:
        a, b = c2/4, 1
        return Z_qcirc(-a, -b, c1/4, A, B, tau, ppart).compose(Z_qcirc(a, b, c1/4, A, B, tau, ppart))
    
    # compilation for c3*lam^3[B,A,B]
    if c3 !=0 and c1 == 0 == c2:
        a, b = c3/2, 1
        return Y_qcirc(a, -b, A, B, tau, ppart).compose(Y_qcirc(a, b, A, B, tau, ppart))

def UnegC_qcirc(c1, c2, c3, A, B, tau, ppart):
    """
    Generates quantum circuit for implementing exp(-C)
    """
    return UC_qcirc(-c1, -c2, -c3, A, B, tau, ppart)

def CPF_symp_qcirc(order, A, B, tau, ppart, reps=1):
    """
    Generates quantum circuits for implementing exp(C) PF exp(-C)
    where PF product formula of order 1 or 2 and C is the associated symplectic corrector.
    """
    #tau = t/reps
    if order == 1:
        # Symp corrector: Csymp = (lam/2)*B + (lam**2/12)[A,B] with lam = -1j*tau
        # tau is taken for quantum circuits becuase they implement Hamiltonin evolution
        UC = UC_qcirc(1/2, 1/12, 0, A, B, tau, ppart)
        UnegC = UC_qcirc(-1/2, -1/12, 0, A, B, tau, ppart)
    if order == 2:
        # Symp corrector: Csymp = (-lam**2 /24) [A,B] = (tau**2 /24) [A,B]; note here c2 = -1/24.
        UC = UC_qcirc(0, -1/24, 0, A, B, tau, ppart) # generates exp(Csymp)
        UnegC= UC_qcirc(0, 1/24, 0, A, B, tau, ppart) # generates exp(-Csymp)
    
    PF = PF_qcirc(order, A, B, tau, ppart, reps)
    CPF = UnegC.compose(PF).compose(UC)
    return CPF

def CPF_symp_qcirc_error(order, A, B, t, ppart, reps):
    tau=t/reps
    PF = PF_qcirc(order, A, B, tau, ppart, reps)
    CPF = CPF_symp_qcirc(order, A, B, t, ppart, reps)

    if ppart == 'Hxx':
        J = B.coeffs[0].real
        h = A.coeffs[0].real
    if ppart == 'Hz':
        h = B.coeffs[0].real
        J = A.coeffs[0].real
    n = A.num_qubits
    Uexact = exact_evolution_ising_qcirc(J,h,n,t)

    mtxPF = Operator(PF).data
    mtxCPF = Operator(CPF).data
    mtxUexact = Operator(Uexact).data

    PF_error = operator_norm(mtxUexact - mtxPF)
    CPF_error = operator_norm(mtxUexact - mtxCPF)
    return PF_error, CPF_error

def data_CPF_symp_qcirc_error(order, A, B, time_ticks, ppart, num_steps):

    error_data = [CPF_symp_qcirc_error(order, A, B, t, ppart, num_steps) for t in time_ticks]
    return error_data