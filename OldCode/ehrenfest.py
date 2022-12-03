"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from evolve import TEBD
from tenpy.algorithms.truncation import TruncationError
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time
import datetime
import sys

enz = True

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
if enz:
    c = 1. # constant to modify the amplitude of the current
else:
    c = None

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
kappa = {}
""".format(iU/it, maxdim, N, nsteps, iF0, c)
print(out)

p = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

model = FHHamiltonian(p, 0)

# get the start time
start_time = time.time()
model = FHHamiltonian(p, 0)
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
tebd = TEBD(psi, model, p, phi_func, None, c, tebd_params)
times, phis, psis = tebd.run()

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

# set up expectation values to track
nnop = FHNearestNeighborModel(p)
nncommop = NearestNeighborCommutatorModel(p)
nnintcommop = InteractionNNCommutatorModel(p)

start_time = time.time()
print("Calculating expectations")
phigrads = np.gradient(phis, delta)
currents = []
rightside = []
nnopexpecs = []
nncommexpecs = []
nnintcommexpecs = []
for i, (phi, phigrad, psi) in enumerate(zip(phis, phigrads, psis)):
    nnopexpec = nnop.H_MPO.expectation_value(psi)
    nnopexpecs.append(nnopexpec)
    r, theta = np.abs(nnopexpec), np.angle(nnopexpec)
    q = nncommop.H_MPO.expectation_value(psi).real
    nncommexpecs.append(q)
    nnintexpec = p.u * nnintcommop.H_MPO.expectation_value(psi)
    nnintcommexpecs.append(nnintexpec)
    pr, lamb = np.abs(nnintexpec), np.angle(nnintexpec)
    rightside.append(2 * p.a * p.t0 * (p.t0 * q + pr * np.cos(phi - lamb) - phigrad * r * np.cos(phi - theta)))

    currents.append(FHCurrent(p, phi).H_MPO.expectation_value(psi))

    seconds = ((i + 1) / nsteps) * (time.time() - start_time) * (nsteps - i - 1)
    days = int(seconds // (3600 * 24))
    seconds = seconds % (3600 * 24)
    hrs = int(seconds // 3600)
    seconds = seconds % 3600
    mins = int(seconds // 60)
    seconds = int(seconds % 60)
    status = "Status: {:.2f}% -- ".format((i + 1) / nsteps * 100)
    status += "Estimated time remaining: {} days, {}".format(days, datetime.time(hrs, mins, seconds))
    print(status, end="\r")

leftside = np.gradient(currents, delta)

print("\nDone calculating expectations, this took", time.time() - start_time, "seconds")

savedir = "./Data/Tenpy/Ehrenfest/"
ps = "-nsteps{}-nsites{}-U{}".format(nsteps, p.nsites, p.u)
if enz:
    ps += "-c{}-F{}".format(c, iF0)
ps += "-maxdim{}".format(nsteps, p.nsites, p.u, maxdim)
np.save(savedir + "LHS" + ps + ".npy", leftside)
np.save(savedir + "RHS" + ps + ".npy", rightside)
np.save(savedir + "currents" + ps + ".npy", currents)
np.save(savedir + "phis" + ps + ".npy", phis)
np.save(savedir + "NearestNeighborExpec" + ps + ".npy", nnopexpecs)
np.save(savedir + "NearestNeighborCommutatorExpec" + ps + ".npy", nncommexpecs)
np.save(savedir + "InteractionNearestNeighborCommutatorExpec" + ps + ".npy", nnintcommexpecs)

# leftside = np.load(savedir + "LHS" + ps + ".npy")
# rightside = np.load(savedir + "RHS" + ps + ".npy")
# currents = np.load(savedir + "currents" + ps + ".npy")
# phis = np.load(savedir + "phis" + ps + ".npy")
# nnopexpecs = np.load(savedir + "NearestNeighborExpec" + ps + ".npy")
# nncommexpecs = np.load(savedir + "NearestNeighborCommutatorExpec" + ps + ".npy")
# nnintcommexpecs = np.load(savedir + "InteractionNearestNeighborCommutatorExpec" + ps + ".npy")

# phigrads = np.gradient(phis, delta)
# currentgrads = np.gradient(currents, delta)

# print(nnopexpecs[0])
# print(phigrads[0])
# print(2 * p.a * nnintcommexpecs[0])
# print(np.cos(phis[0] - np.angle(nnintcommexpecs[0])))

# rhs =  2 * p.a * p.t0 * (p.t0 * nncommexpecs.real + np.abs(nnintcommexpecs) * np.cos(phis - np.angle(nnintcommexpecs)) - phigrads * np.abs(nnopexpecs) * np.cos(phis - np.angle(nnopexpecs)))
# rhs =  2 * p.a * p.t0 * (p.t0 * nncommexpecs.real - phigrads * np.abs(nnopexpecs) * np.cos(phis - np.angle(nnopexpecs)))
# print(nncommexpecs[0])
# plt.plot(times, leftside)
# plt.plot(times, 2 * p.a * p.t0 * (-p.t0 * nncommexpecs + p.u * np.abs(nnintcommexpecs) * np.cos(phis - np.angle(nnintcommexpecs)) - phigrads * np.abs(nnopexpecs) * np.cos(phis - np.angle(nnopexpecs))))
# plt.plot(times, rhs - rhs[0], ls="dashed")

# plt.plot(times, rhs, label="rhs")
# plt.plot(times, currentgrads, ls="dashed", label="$\\frac{dJ}{dt}$")
# plt.legend()
# plt.show()

# plt.plot(times, np.abs(nnopexpecs))
# plt.plot(times, np.abs(nncommexpecs))

# plt.show()

plt.plot(times, leftside, label="$\\frac{dJ}{dt}$")
plt.plot(times, rightside, ls="dashed")
plt.legend()
plt.show()
