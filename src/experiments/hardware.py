import os
from dotenv import load_dotenv

import numpy as np

from src.models.ising_model import *
from src.utils import *
# from src.corrector import *

from src.circuits.pfs_qcircs import *
from src.circuits.cpfs_qcircs import *
from src.circuits.ising_model_qcircs import *
from src.circuits.average_infidelity_qcircs import *

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from qiskit_aer import AerSimulator

import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings(action='once')

token = os.getenv("IBMQ_TOKEN")

service = QiskitRuntimeService(
    channel= os.getenv("IBMQ_CHANNEL"), #add channel here; channel used is: "ibm_quantum",
    instance= os.getenv("IBMQ_INSTANCE"), #add instance here; instance used is: 'pinq-quebec-hub/univ-toronto/matterlab',
    token=token)

#backend = service.least_busy(operational=True, simulator=False)
backend = service.backend(name=os.getenv("IBMQ_BACKEND")) # add your backend here; backend used is: "ibm_quebec"


pparam = 0.1
n, J, h = 2, pparam, 1 #r is number of qubits (nqubits)
H, Hxx, Hz = ising_qubit_hamiltonian(n, J, h)
A, B = Hz, Hxx

num_shots = 10**5

time_ticks = np.linspace(0.1,1,10)

list_avg_infid_PF1 = []
list_std_infid_PF1 = []
list_avg_infid_PF2 = []
list_std_infid_PF2 = []

list_avg_infid_CPF1 = []
list_std_infid_CPF1 = []
list_avg_infid_CPF2 = []
list_std_infid_CPF2 = []

for t in time_ticks:
    exactU = exact_evolution_ising_qcirc(J, h, n, t)
    r = 10 #r is number of stepts (nsteps)
    tau = t/r

    UPF1 = PF_qcirc(1, A, B, tau, ppart='Hxx', reps=r)
    UPF2 = PF_qcirc(2, A, B, tau, ppart='Hxx', reps=r)

    UCPF1 = CPF_symp_qcirc(1, A, B, tau, ppart='Hxx', reps=r)
    UCPF2 = CPF_symp_qcirc(2, A, B, tau, ppart='Hxx', reps=r)

    avg_infid_set_PF1 = average_infidelity(UPF1, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_PF2 = average_infidelity(UPF2, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_CPF1 = average_infidelity(UCPF1, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_CPF2 = average_infidelity(UCPF2, exactU, backend=backend, num_shots=num_shots)


    list_avg_infid_PF1.append(np.mean(avg_infid_set_PF1))
    list_std_infid_PF1.append(np.std(avg_infid_set_PF1))

    list_avg_infid_PF2.append(np.mean(avg_infid_set_PF2))
    list_std_infid_PF2.append(np.std(avg_infid_set_PF2))

    list_avg_infid_CPF1.append(np.mean(avg_infid_set_CPF1))
    list_std_infid_CPF1.append(np.std(avg_infid_set_CPF1))

    list_avg_infid_CPF2.append(np.mean(avg_infid_set_CPF2))
    list_std_infid_CPF2.append(np.std(avg_infid_set_CPF2))

print(f"""
Depth comparison
-----------------
Depth of Uexact: {exactU.decompose().depth()}
Depth of UPF1: {UPF1.decompose().depth()}
Depth of UPF2: {UPF2.decompose().depth()}
Depth of UCPF1: {UCPF1.decompose().depth()}
Depth of UCPF2: {UCPF2.decompose().depth()}
""")

print(exactU.draw(fold=-1))
print(exactU.decompose().draw(fold=-1))

# Save results as a dataframe.
data = {'description': ['nqubits=2','nsteps=10','nshots=10^5','pparam=0.1','ticks=np.linspace(0.1,1,10)',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'avg_infid_PF1': list_avg_infid_PF1,
        'std_infid_PF1': list_std_infid_PF1,
        'avg_infid_PF2': list_avg_infid_PF2,
        'std_infid_PF2': list_std_infid_PF2,
        'avg_infid_CPF1': list_avg_infid_CPF1,
        'std_infid_CPF1': list_std_infid_CPF1,
        'avg_infid_CPF2': list_avg_infid_CPF2,
        'std_infid_CPF2': list_std_infid_CPF2
        }

# Create DataFrame
df = pd.DataFrame(data)
# we put a descriptive title for csv file with parameters used
df.to_csv('ibm_implementation_ising_2sites_10steps_1e5shots_0.1pparam_10ticks.csv', index = False)

# Display the csv file of the hardwaree implementation results
file_path = '../hardware experiments/ibm_implementation_ising_2sites_10steps_1e5shots_0.1pparam_20ticks.csv'
df = pd.read_csv(file_path)

#backend = FakeQuebec()
backend = AerSimulator(method='statevector')

pparam = 0.1
n, J, h = 2, pparam, 1
H, Hxx, Hz = ising_qubit_hamiltonian(n, J, h)
A, B = Hz, Hxx

num_shots = 10**6

time_ticks = np.linspace(0.1,1,20)

list_avg_infid_PF1 = []
list_std_infid_PF1 = []
list_avg_infid_PF2 = []
list_std_infid_PF2 = []

list_avg_infid_CPF1 = []
list_std_infid_CPF1 = []
list_avg_infid_CPF2 = []
list_std_infid_CPF2 = []

for t in time_ticks:
    exactU = exact_evolution_ising_qcirc(J, h, n, t)
    r = 1 #r is number of stepts (nsteps)
    tau = t/r

    UPF1 = PF_qcirc(1, A, B, tau, ppart='Hxx', reps=r)
    UPF2 = PF_qcirc(2, A, B, tau, ppart='Hxx', reps=r)

    UCPF1 = CPF_symp_qcirc(1, A, B, tau, ppart='Hxx', reps=r)
    UCPF2 = CPF_symp_qcirc(2, A, B, tau, ppart='Hxx', reps=r)

    avg_infid_set_PF1 = average_infidelity(UPF1, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_PF2 = average_infidelity(UPF2, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_CPF1 = average_infidelity(UCPF1, exactU, backend=backend, num_shots=num_shots)
    avg_infid_set_CPF2 = average_infidelity(UCPF2, exactU, backend=backend, num_shots=num_shots)


    list_avg_infid_PF1.append(np.mean(avg_infid_set_PF1))
    list_std_infid_PF1.append(np.std(avg_infid_set_PF1))

    list_avg_infid_PF2.append(np.mean(avg_infid_set_PF2))
    list_std_infid_PF2.append(np.std(avg_infid_set_PF2))

    list_avg_infid_CPF1.append(np.mean(avg_infid_set_CPF1))
    list_std_infid_CPF1.append(np.std(avg_infid_set_CPF1))

    list_avg_infid_CPF2.append(np.mean(avg_infid_set_CPF2))
    list_std_infid_CPF2.append(np.std(avg_infid_set_CPF2))


# Save results as a dataframe.
data = {'description': ['nqubits=2','nsteps=1','nshots=10^6', 'pparam=0.1', 'ticks=np.linspace(0.1,1,20)',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'avg_infid_PF1': list_avg_infid_PF1,
        'std_infid_PF1': list_std_infid_PF1,
        'avg_infid_PF2': list_avg_infid_PF2,
        'std_infid_PF2': list_std_infid_PF2,
        'avg_infid_CPF1': list_avg_infid_CPF1,
        'std_infid_CPF1': list_std_infid_CPF1,
        'avg_infid_CPF2': list_avg_infid_CPF2,
        'std_infid_CPF2': list_std_infid_CPF2
        }

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('noiseless_simulation_ising_2sites_1step_1e6shots_0.1pparam_20ticks.csv', index = False)
print(df)