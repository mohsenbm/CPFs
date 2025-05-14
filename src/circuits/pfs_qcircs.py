from src.circuits.ising_model_qcircs import *


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