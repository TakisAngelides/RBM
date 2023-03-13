# RBM NNQS for Takis' Schwinger model

# This is based on the netket example https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html

from scipy.sparse.linalg import eigsh
import scipy

# Import netket library
import netket as nk

# Import Json, this will be needed to load log files
import json

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time

from netket.operator.spin import sigmax, sigmay, sigmaz

# Pauli matrices etc.
def sigx(i):
    return nk.operator.spin.sigmax(hi, i-1, dtype=np.complex128)

def sigy(i):
    return nk.operator.spin.sigmay(hi, i-1, dtype=np.complex128)

def sigz(i):
    return nk.operator.spin.sigmaz(hi, i-1, dtype=np.complex128)

def sigp(i):
    return 0.5*(sigx(i) + 1j*sigy(i))

def sigm(i):
    return 0.5*(sigx(i) - 1j*sigy(i))

def Qcharge(i):
    return 0.5*(sigz(2*i-1) + sigz(2*i))


# Define a 1d chain
L = 6
g = nk.graph.Hypercube(length=2*L, n_dim=1, pbc=False)

# Define the Hilbert space based on this graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

# Now we define some parameters for the Hamiltonian
l_0 = 0.0
mg = 0.1
volume = 10
x = (L/volume)**2
lambd = 10

# Now the Hamiltonian with Wilson parameter = 1 from Takis' paper

# setup local Hamiltonian
ha = nk.operator.LocalOperator(hi, dtype=np.complex128)

for n in range(1, L): # kinetic term
  ha += (2j*x)*(sigp(2*n-1)*sigz(2*n)*sigz(2*n+1)*sigm(2*n+2))
  ha += -(2j*x)*(sigm(2*n-1)*sigz(2*n)*sigz(2*n+1)*sigp(2*n+2))

for n in range(1, L+1): # mass term
  ha += (2j)*(mg*(x)**(0.5) + x)*(sigm(2*n-1)*sigp(2*n))
  ha += -(2j)*(mg*(x)**(0.5) + x)*(sigp(2*n-1)*sigm(2*n))

for n in range(1, 2*L-1): # single Z term
  ha += l_0*(L-np.ceil(n/2))*sigz(n)

for n in range(1, 2*L+1): # long range ZZ interaction from Coulomb interaction
  for k in range(n+1, 2*L+1):
    ha += 0.5*(L-np.ceil(k/2)+lambd)*sigz(n)*sigz(k)

ha += (l_0**2)*(L-1) + 0.25*L*(L-1) + lambd*L/2 # constants

print("Hamiltonian is Hermitian: {}".format(ha.is_hermitian))
print("Number of local operators: {:d}".format(len(ha.operators)))
print("Hamiltonian size: {}".format(ha.to_dense().shape))

# Exact compututation for the ground-state energy
evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
exact_gs_energy = evals[0]
print('The exact ground-state energy is E0=',exact_gs_energy)

# Extra test for exact diagonalization agrees with above results from nk.exact.lanczos_ed and with my results from qiskit definition of Hamiltonian
# evals, evecs = eigsh(scipy.sparse.csr_array(ha.to_dense()), which = 'SR', k = 6, ncv = 40)
# print(evals)

# TODO: Understand the hyper-parameters and how best to initialize the weights
# as the convergence of the RBM is sensitive to both

# RBM ansatz
ma = nk.models.RBM()

# Build the sampler
sa = nk.sampler.MetropolisExchange(hilbert = hi, graph = g)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(diag_shift=0.1)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=100)

# The ground-state optimization loop
gs = nk.VMC(hamiltonian=ha, optimizer=op, preconditioner=sr, variational_state=vs)

start = time.time()
gs.run(out='RBM', n_iter=1000)
end = time.time()

print('### RBM calculation')
print('Has',vs.n_parameters,'parameters')
print('The RBM calculation took',end-start,'seconds')

# Condensate

#corr = (1/(2*L))*sum([((-1)**(i+1))*sigz(i) for i in range(L)])
#vs.n_samples=40000
#vs.expect(corr)
#print(vs.expect(corr))

## import the data from log file
data=json.load(open("RBM.log"))

# Extract the relevant information
iters_RBM = data["Energy"]["iters"]
energy_RBM = data["Energy"]["Mean"]["real"]

plt.plot(iters_RBM, energy_RBM, '-o', color='red', label='Energy (RBM)')
plt.hlines(y=exact_gs_energy, xmin=0, xmax=iters_RBM[-1], linewidth=2, color='k', label='Exact')
plt.legend()
plt.savefig('energy_vs_iteration.png', bbox_inches = 'tight')
plt.show()