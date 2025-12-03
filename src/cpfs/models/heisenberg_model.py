import cirq

def heisenberg_model(n):
    """
    Heisenberg model over n sites. We assume n is even number.
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(n - 1):
        ham += cirq.X(qs[i]) * cirq.X(qs[i + 1])
        ham += cirq.Y(qs[i]) * cirq.Y(qs[i + 1])
        ham += cirq.Z(qs[i]) * cirq.Z(qs[i + 1])
    ham += cirq.X(qs[n-1]) * cirq.X(qs[0])
    ham += cirq.Y(qs[n-1]) * cirq.Y(qs[0])
    ham += cirq.Z(qs[n-1]) * cirq.Z(qs[0])
    return ham

def heisenberg_model_even(n):
    """
    Heisenberg model over even-index sites.
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(int(n/2)):
        ham += cirq.X(qs[2*i]) * cirq.X(qs[2*i + 1])
        ham += cirq.Y(qs[2*i]) * cirq.Y(qs[2*i + 1])
        ham += cirq.Z(qs[2*i]) * cirq.Z(qs[2*i + 1])
    return ham

def heisenberg_model_odd(n):
    """
    Heisenberg model over odd-index sites.
    """
    qs = [cirq.GridQubit(0, i) for i in range(n)]
    ham = cirq.PauliSum()
    for i in range(int(n/2)-1):
        ham += cirq.X(qs[2*i+1]) * cirq.X(qs[2*i + 2])
        ham += cirq.Y(qs[2*i+1]) * cirq.Y(qs[2*i + 2])
        ham += cirq.Z(qs[2*i+1]) * cirq.Z(qs[2*i + 2])
    ham += cirq.X(qs[n-1]) * cirq.X(qs[0])
    ham += cirq.Y(qs[n-1]) * cirq.Y(qs[0])
    ham += cirq.Z(qs[n-1]) * cirq.Z(qs[0])
    return ham