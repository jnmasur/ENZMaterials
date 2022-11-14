from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from exact_methods import *
from matplotlib import pyplot as plt


t_init = time()

"""Hubbard model Parameters"""
L = 10  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
U = 1. * t0  # interaction strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10.  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms
cycles = 10  # time in cycles of field frequency

"""ENZ Parameter"""
kappa = 0.125

"""instantiate parameters with proper unit scaling"""
lat = Parameters(L, U, t0, a, cycles, field, F0, pbc)

"""System Evolution Time"""
n_steps = 2000
start = 0
stop = 2 * np.pi * cycles / lat.field
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""
parameters = f'-nsteps{n_steps}-nsites{L}-U{U/t0}-c{kappa}-F{F0}-a{a}'

"""create basis"""
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down), sblock=1, kblock=1)

"""Create static part of hamiltonian - the interaction b/w electrons"""
int_list = [[1.0, i, i] for i in range(L)]
static_Hamiltonian_list = [
    ["n|n", int_list]  # onsite interaction
]
# n_j,up n_j,down
onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

"""Create dynamic part of hamiltonian - composed of a left and a right hopping parts"""
hop = [[1.0, i, i+1] for i in range(L-1)]
if lat.pbc:
    hop.append([1.0, L-1, 0])
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# c^dag_j,sigma c_j+1,sigma
hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=basis, **no_checks)
# c^dag_j+1,sigma c_j,sigma
hop_right = hop_left.getH()

"""Create complete Hamiltonian"""
H = -lat.t0 * (hop_left + hop_right) + lat.u * onsite

psi_0_parameters = f"-nsites{L}-U{U/t0}-F{F0}-a{a}"
try:
    psi_0 = np.load("./Data/Exact/ENZ/psi0" + psi_0_parameters + ".npy")
    print("Loaded initial psi")
except:
    """get ground state as the eigenstate corresponding to the lowest eigenergy"""
    print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    print("ground state calculated, energy is {:.2f}".format(E[0]))

    print('evolving system')
    ti = time()
    """evolving system, using our own solver for derivatives"""
    f_params = (onsite, hop_left, hop_right, lat, phi_tl)
    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=f_params)
    psi_t = np.squeeze(psi_t)
    print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
    psi_0 = psi_t[:, -1]
    np.save("./Data/Exact/ENZ/psi0" + psi_0_parameters + ".npy", psi_0)

expec = np.vdot(psi_0, hop_left.static.dot(psi_0))
r, theta = np.abs(expec), np.angle(expec)
scale = r * np.sin(theta)
print('evolving system')
ti = time()
"""evolving system, using our own solver for derivatives"""
f_params = (onsite, hop_left, hop_right, lat, phi_enz_exact, dict(kappa=kappa, scale=scale))
psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=f_params)
psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))

"""Calculate Expectation Values"""
ti = time()
phis = get_enz_phis(lat, hop_left, psi_t, kappa, scale, n_steps)
expectations = {'H': H_expec(psi_t, times, onsite, hop_left, hop_right, lat, phis),
                'J': J_expec(psi_t, times, hop_left, hop_right, lat, phis)}

print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

plt.plot(times, expectations["J"])
plt.plot(times, phis * kappa + expectations["J"][0], label="$\\frac{\\Phi}{\\mathfrak{L}} + J(0)$")
plt.legend()
plt.show()

np.save('./Data/Exact/ENZ/energies'+parameters+'.npy', expectations['H'])
np.save('./Data/Exact/ENZ/currents'+parameters+'.npy', expectations['J'])
np.save('./Data/Exact/ENZ/phis'+parameters+'.npy', phis)
np.save("./Data/Exact/ENZ/times" + f'-nsteps{n_steps}' + ".npy", times)

print('All finished. Total time was {:.2f} seconds'.format((time() - t_init)))

def run(uot, a, kappa):
    """Hubbard model Parameters"""
    L = 10  # system size
    N_up = L // 2 + L % 2  # number of fermions with spin up
    N_down = L // 2  # number of fermions with spin down
    N = N_up + N_down  # number of particles
    t0 = 0.52  # hopping strength
    U = uot * t0  # interaction strength
    pbc = True

    """Laser pulse parameters"""
    field = 32.9  # field angular frequency THz
    F0 = 10.  # Field amplitude MV/cm
    cycles = 10  # time in cycles of field frequency

    """instantiate parameters with proper unit scaling"""
    lat = Parameters(L, U, t0, a, cycles, field, F0, pbc)

    """System Evolution Time"""
    n_steps = 2000
    start = 0
    stop = 2 * np.pi * cycles / lat.field
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    """set up parameters for saving expectations later"""
    parameters = f'-nsteps{n_steps}-nsites{L}-U{U/t0}-c{kappa}-F{F0}-a{a}'

    """create basis"""
    basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down), sblock=1, kblock=1)

    """Create static part of hamiltonian - the interaction b/w electrons"""
    int_list = [[1.0, i, i] for i in range(L)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

    """Create dynamic part of hamiltonian - composed of a left and a right hopping parts"""
    hop = [[1.0, i, i+1] for i in range(L-1)]
    if lat.pbc:
        hop.append([1.0, L-1, 0])
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    """Create complete Hamiltonian"""
    H = -lat.t0 * (hop_left + hop_right) + lat.u * onsite

    psi_0_parameters = f"-nsites{L}-U{U/t0}-F{F0}-a{a}"
    try:
        psi_0 = np.load("./Data/Exact/ENZ/psi0" + psi_0_parameters + ".npy")
        print("Loaded initial psi")
    except:
        """get ground state as the eigenstate corresponding to the lowest eigenergy"""
        E, psi_0 = H.eigsh(k=1, which='SA')
        psi_0 = np.squeeze(psi_0)
        psi_0 = psi_0 / np.linalg.norm(psi_0)

        print('evolving system')
        ti = time()
        """evolving system, using our own solver for derivatives"""
        f_params = (onsite, hop_left, hop_right, lat, phi_tl)
        psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=f_params)
        psi_t = np.squeeze(psi_t)
        print("Pump pulse evolution done " + parameters)
        psi_0 = psi_t[:, -1]
        np.save("./Data/Exact/ENZ/psi0" + psi_0_parameters + ".npy", psi_0)

    expec = np.vdot(psi_0, hop_left.static.dot(psi_0))
    r, theta = np.abs(expec), np.angle(expec)
    scale = r * np.sin(theta)
    print('evolving system')
    ti = time()
    """evolving system, using our own solver for derivatives"""
    f_params = (onsite, hop_left, hop_right, lat, phi_enz_exact, dict(kappa=kappa, scale=scale))
    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=f_params)
    psi_t = np.squeeze(psi_t)
    print("ENZ evolution done " + parameters)

    """Calculate Expectation Values"""
    ti = time()
    phis = get_enz_phis(lat, hop_left, psi_t, kappa, scale, n_steps)
    expectations = {'H': H_expec(psi_t, times, onsite, hop_left, hop_right, lat, phis),
                    'J': J_expec(psi_t, times, hop_left, hop_right, lat, phis)}

    np.save('./Data/Exact/ENZ/energies'+parameters+'.npy', expectations['H'])
    np.save('./Data/Exact/ENZ/currents'+parameters+'.npy', expectations['J'])
    np.save('./Data/Exact/ENZ/phis'+parameters+'.npy', phis)
    np.save("./Data/Exact/ENZ/times" + f'-nsteps{n_steps}' + ".npy", times)
