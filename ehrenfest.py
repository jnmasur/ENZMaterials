"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from tebd import Engine as TEBD
from tenpy.algorithms.truncation import TruncationError
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time
import sys

it = .52
##########################
"""IMPORTANT PARAMETERS"""
##########################
# maximum bond dimension, used for both DMRG and TEBD, multiple of 200
maxdim = 1000
N = 10
iU = 0.5 * it
# the number of steps
nsteps = 4000
iF0 = 10.
# c = 1. # constant to modify the amplitude of the current

"""We will hold these parameters constant"""
maxerr = 1e-12  # used for DMRG
# lattice spacing, in angstroms
ia = 4
# pulse parameters
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10
pbc = False  # periodic boundary conditions

# load pulse info
loaddir = "./Data/Tenpy/Basic/"
phi_times = np.load(loaddir + "times-nsteps{}.npy".format(nsteps))
phi_vals = np.load(loaddir + "phis-nsteps{}-a{}-f{}-w{}-cycles{}.npy".format(nsteps, ia, int(iF0), iomega0, cycles))
phi_func = dict(times=phi_times, phis=phi_vals)


out = """Evolving with
U/t0 = {:.1f}
maximum dimension = {}
number of sites = {}
number of steps = {}
pump field strength = {} MV/cm
""".format(iU/it, maxdim, N, nsteps, iF0)
print(out)

p = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

model = FHHamiltonian(p, 0)

# set up expectation values to track
nnop = FHNearestNeighborModel(p)
nncommop = NearestNeighborCommutatorModel(p)
nnintcommop = InteractionNNCommutatorModel(p)
expecops = [nnop, nncommop, nnintcommop]

# get the start time
start_time = time.time()
model = FHHamiltonian(p, 0)
current = FHCurrent(p, 0)
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

psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
times, delta = np.linspace(ti, tf, num=nsteps, endpoint=True, retstep=True)
# we pass in nsteps - 1 because we would like to evauluate the system at
# nsteps time points, including the ground state calculations
tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":maxdim}, "N_steps":nsteps-1, "F0":iF0}
tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
tebd = TEBD(psi, model, p, phi_func, None, None, None, tebd_params, expecops)
times, energies, currents, phis = tebd.run()

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

expectation_vals = tebd.expectation_vals
phigrad = np.gradient(phis, delta)
leftside = np.gradient(currents, delta)
rightside = []

for i in range(nsteps):
    r = np.abs(expectation_vals[0][i])
    theta = np.angle(expectation_vals[0][i])
    q = expectation_vals[1][i].real
    pr = np.abs(expectation_vals[2][i])
    lamb = np.angle(expectation_vals[2][i])
    rightside.append(2 * np.e * p.a * p.t * (-p.t * q + pr * np.cos(phis[i] - lamb) - phigrad[i] * r * np.cos(phi[i] - theta)))

savedir = "./Data/Tenpy/Ehrenfest/"
ps = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, p.nsites, p.u, maxdim)
np.save(savedir + "LHS" + ps + ".npy", leftside)
np.save(savedir + "RHS" + ps + ".npy", rightside)
np.save(savedir + "currents" + ps + ".npy", currents)
np.save(savedir + "phis" + ps + ".npy", phis)
np.save(savedir + "NearestNeighborExpec" + ps + ".npy", expectation_vals[0])
np.save(savedir + "NearestNeighborCommutatorExpec" + ps + ".npy", expectation_vals[1])
np.save(savedir + "InteractionNearestNeighborCommutatorExpec" + ps + ".npy", expectation_vals[2])

plt.plot(times, leftside, label="$\\frac{dJ}{dt}$")
plt.plot(times, rightside, ls="dashed")
plt.legend()
plt.show()
