"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from evolve import TimeDependentExpMPOEvolution as WPO1
from tenpy.algorithms.truncation import TruncationError
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time
import sys

"""
Command line arguments should be used as follows:
--U [float] : U/t_0 ratio
--dim [int multiple of 200] : maximum dimension
--nx [int] : number of unit cells in x direction
--ny [int] : number of unit cells in the y direction
--nsteps [int] : number of evolution steps
--F [float] : field strength of pump pulse (in MV/cm)
"""
maxdim = None
nx = None
ny = None
nsteps = None
iF0 = None
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == "--dim":
        maxdim = int(sys.argv[i + 1])
    elif sys.argv[i] == "--nx":
        nx = int(sys.argv[i + 1])
    elif sys.argv[i] == "--ny":
        ny = int(sys.argv[i + 1])
    elif sys.argv[i] == "--nsteps":
        nsteps = int(sys.argv[i + 1])
    elif sys.argv[i] == "--F":
        iF0 = float(sys.argv[i + 1])
    else:
        print("Unrecognized argument: {}".format(sys.argv[i]))

##########################
"""IMPORTANT PARAMETERS"""
##########################
# maximum bond dimension, used for both DMRG and TEBD, multiple of 200
maxdim = 1000 if maxdim is None else maxdim
# number of unit cells in the x and y direction
nx = 2 if nx is None else nx
ny = 2 if ny is None else ny
# number of sites (unit cell has 2 sites)
N = nx * ny * 2
# the number of steps if not apdative
nsteps = 4000 if nsteps is None else nsteps
iF0 = 10. if iF0 is None else iF0

kwargs = {}
enz = True
if enz:
    kappa = 1.
    kwargs["kappa"] = kappa
    kwargs["scale"] = None
else:
    kwargs["time"] = 0.


"""We will hold these parameters constant"""
maxerr = 1e-12  # used for DMRG
# hopping parameter, in units eV
it = 2.7
# lattice spacing, in angstroms
ia = 2.5
# pulse parameters
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10
pbc = False  # periodic boundary conditions

# this works for normal evolution and for enz
phi_func = phi_tl

out = """Evolving with
maximum dimension = {}
number of unit cells in the x direction = {}
number of unit cells in the y direction = {}
number of steps = {}
pump field strength = {} MV/cm
""".format(maxdim, nx, ny, nsteps, iF0)
print(out)

p = Parameters(N, 0, it, ia, cycles, iomega0, iF0, pbc, nx=nx, ny=ny)

start_time = time.time()

gps = dict(p=p, phi_func=phi_func)
model = GrapheneHamiltonian(gps, 0.)
# current = GrapheneCurrent(p, 0.)
sites = model.lat.mps_sites()
state = ["up", "down"] * (N // 2)
psi0_i = MPS.from_product_state(sites, state)

# the max bond dimension
chi_list = {0:20, 1:40, 2:100, 4:200}
chi = 200
iter = 4
while chi < maxdim:
    iter += 2
    chi += 200
    chi_list[iter] = chi
dmrg_dict = {"chi_list":chi_list, "max_E_err":maxerr, "max_sweeps":(maxdim / 100) + 4, "mixer":True, "combine":False}
dmrg_params = Config(dmrg_dict, "DMRG-maxerr{}".format(maxerr))
dmrg = DMRG(psi0_i, model, dmrg_params)
E, psi0 = dmrg.run()
print("Ground state energy:", E)

if enz:
    kwargs["nnop"] = GrapheneNearestNeighbor(gps).H_MPO
    kwargs["psi"] = psi0

psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
times, delta = np.linspace(ti, tf, num=nsteps, endpoint=True, retstep=True)
# we pass in nsteps - 1 because we would like to evauluate the system at
# nsteps time points, including the ground state calculations
tebd_dict = {"dt":delta, "order":1, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":maxdim}, "N_steps":nsteps-1, "F0":iF0, "compression_method":"SVD"}
tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
evolver = WPO1(psi, model, tebd_params, phi_func, material="Graphene", kwargs=kwargs)
times, phis, psis = evolver.run(enz)

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

ti = time.time()
print("Calculating expectations")
currents = []
energies = []
for phi, psi in zip(phis, psis):
    currents.append(GrapheneCurrent(gps, phi).H_MPO.expectation_value(psi))
    energies.append(GrapheneHamiltonian(gps, phi).H_MPO.expectation_value(psi))

print("Expectations calculated, total time:", time.time() - ti)

savedir = "./Data/Graphene/"
if enz:
    savedir += "ENZ/"
else:
    savedir += "Basic/"
ecps = "-nsteps{}-nx{}-ny{}-U{}".format(nsteps, p.nx, p.ny, p.u)
if enz:
    ecps += "-c{}-F{}".format(kappa, iF0)
ecps += "-maxdim{}".format(maxdim)
np.save(savedir + "energies" + ecps + ".npy", energies)
np.save(savedir + "currents" + ecps + ".npy", currents)
np.save(savedir + "phis" + ecps + ".npy", phis)

plt.plot(times, currents)
plt.plot(times, phis, ls="dashed", label="$\\Phi(t)$")
plt.legend()
plt.show()
