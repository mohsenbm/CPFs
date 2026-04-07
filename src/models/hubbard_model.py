import openfermion as of

def hubbard_model_terms(size, t_hop, U_int):
    """
    Terms of 1D Hubbard model with periodic boundaries and with n = size fermions.
    """
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
    1D Hubbard model with periodic boundaries with n = size fermions where t_hop = 1 and U_int = alpha
    """
    hopping_term, interaction_term  = hubbard_model_terms(size, t_hop=1, U_int=alpha)
    
    A = of.get_sparse_operator(hopping_term).todense()
    B = of.get_sparse_operator(interaction_term).todense()
    H = A + B
    return H, A, B

def weak_hopping_hubbard_model(size, alpha):
    """
    1D Hubbard model with periodic boundaries with n = size fermions where t_hop = alpha , U_int = 1
    """
    hopping_term, interaction_term  = hubbard_model_terms(size, t_hop=alpha, U_int=1)
    A = of.get_sparse_operator(interaction_term).todense()
    B = of.get_sparse_operator(hopping_term).todense()
    H = A + B
    return H, A, B

def intermediate_coupling_hubbard_model(size):
    """
    1D Hubbard model with periodic boundaries with n = size fermions where  U_int = 2 * t_hop = 2
    """
    hopping_term, interaction_term  = hubbard_model_terms(size, t_hop=1, U_int=2)
    A = of.get_sparse_operator(interaction_term).todense()
    B = of.get_sparse_operator(hopping_term).todense()
    H = A + B
    return H, A, B

# 2D Hubbard model
def site_index(i, j, cols):
    """Returns linear index for site (i, j). Sites are ordered from left to right and top to bottom."""
    return i * cols + j

def hubbard_model_terms_2d(rows, cols, t_hop, U_int, periodic=False):
    """
    Terms of 2D Hubbard model
    """
    one_body_terms = []
    two_body_terms = []

    for i in range(rows):
        for j in range(cols):
            q = site_index(i, j, cols)

            # Right neighbor
            if j + 1 < cols or periodic:
                j_next = (j + 1) % cols
                q_right = site_index(i, j_next, cols)

                # Hopping: -t (c†_i c_j + c†_j c_i)
                hop_term = of.FermionOperator(((q, 1), (q_right, 0)), coefficient=-t_hop)
                one_body_terms.append(hop_term + of.hermitian_conjugated(hop_term))

                # Interaction: U n_i n_j
                n_i_n_j = of.FermionOperator(((q, 1), (q, 0), (q_right, 1), (q_right, 0)), coefficient=U_int)
                two_body_terms.append(n_i_n_j)

            # Down neighbor
            if i + 1 < rows or periodic:
                i_next = (i + 1) % rows
                q_down = site_index(i_next, j, cols)

                # Hopping
                hop_term = of.FermionOperator(((q, 1), (q_down, 0)), coefficient=-t_hop)
                one_body_terms.append(hop_term + of.hermitian_conjugated(hop_term))

                # Interaction
                n_i_n_j = of.FermionOperator(((q, 1), (q, 0), (q_down, 1), (q_down, 0)), coefficient=U_int)
                two_body_terms.append(n_i_n_j)

    one_body = sum(one_body_terms, start=of.FermionOperator())
    two_body = sum(two_body_terms, start=of.FermionOperator())
    return one_body, two_body

def weak_coupling_hubbard_model_2d(rows, cols, alpha, periodic=False):
    """
    2D Hubbard model with t_hop = 1.0 and U_int = alpha.
    """
    one_body, two_body = hubbard_model_terms_2d(rows, cols, t_hop=1.0, U_int=alpha, periodic=periodic)
    
    A = of.get_sparse_operator(one_body).todense()
    B = of.get_sparse_operator(two_body).todense()
    H = A + B
    return H, A, B

def weak_hopping_hubbard_model_2d(rows, cols, alpha, periodic=False):
    """
    2D Hubbard model with t_hop = alpha , U_int = 1
    """
    one_body, two_body = hubbard_model_terms_2d(rows, cols, t_hop=alpha, U_int=1, periodic=periodic)
    
    A = of.get_sparse_operator(one_body).todense()
    B = of.get_sparse_operator(two_body).todense()
    H = A + B
    return H, A, B

def intermediate_coupling_hubbard_model_2d(rows, cols, periodic=False):
    """
    2D Hubbard model with U_int = 2 * t_hop = 2
    """
    one_body, two_body = hubbard_model_terms_2d(rows, cols, t_hop=1, U_int=2, periodic=periodic)

    A = of.get_sparse_operator(one_body).todense()
    B = of.get_sparse_operator(two_body).todense()
    H = A + B
    return H, A, B