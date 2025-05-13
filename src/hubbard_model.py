import openfermion as of

def hubbard_model_terms(size, t_hop, U_int):
    one_body_terms = [
    op + of.hermitian_conjugated(op) for op in (
        of.FermionOperator(((j, 1), ((j + 1) % size, 0)), coefficient=-t_hop) for j in range(size)
    )]

    one_body = sum(one_body_terms)
    
    # Two-body (charge-charge) terms.
    two_body_terms = [
    of.FermionOperator(((j, 1), (j, 0), ((j + 1)% size, 1), ((j + 1)% size, 0)), coefficient=U_int)
    for j in range(size)]
    
    two_body = sum(two_body_terms)
    return one_body, two_body

def weak_coupling_hubbard_model(size, alpha):
    """
    Hubbard model with with n = size fermions where t_hop = 1 and U_int = alpha
    """
    hopping_term, interaction_term  = hubbard_model_terms(size=8, t_hop=1, U_int=alpha)
    
    A = of.get_sparse_operator(hopping_term).todense()
    B = of.get_sparse_operator(interaction_term).todense()
    H = A + B
    return H, A, B

def intermediate_coupling_hubbard_model(size):
    """
    Hubbard model with with n = size fermions where  U_int = 2 * t_hop = 2
    """
    hopping_term, interaction_term  = hubbard_model_terms(size=8, t_hop=1, U_int=2)
    A = of.get_sparse_operator(interaction_term).todense()
    B = of.get_sparse_operator(hopping_term).todense()
    H = A + B
    return H, A, B

def weak_hopping_hubbard_model(size, alpha):
    """
    Hubbard model with with n = size fermions where t_hop = alpha , U_int = 1
    """
    hopping_term, interaction_term  = hubbard_model_terms(size=8, t_hop=alpha, U_int=1)
    A = of.get_sparse_operator(interaction_term).todense()
    B = of.get_sparse_operator(hopping_term).todense()
    H = A + B
    return H, A, B