
"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from tebd import Engine as TEBD
from tenpy.algorithms.truncation import TruncationError
from tenpy.tools import hdf5_io
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import sys

"""
Command line arguments should be used as follows:
--U [float] : U/t_0 ratio
--dim [int multiple of 200] : maximum dimension
--N [even int] : number of sites
--nsteps [int] : number of evolution steps
--c [float] : amount to scale current by in enz
--F [float] : field strength of pump pulse (in MV/cm)
"""
# hopping parameter, in units eV
it = .52
maxdim = None
N = None
iU = None
nsteps = None
c = None # only changed if we are performing an enz simulation
iF0 = None
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == "--U":
        iU = float(sys.argv[i + 1]) * it
    elif sys.argv[i] == "--dim":
        maxdim = int(sys.argv[i + 1])
    elif sys.argv[i] == "--N":
        N = int(sys.argv[i + 1])
    elif sys.argv[i] == "--nsteps":
        nsteps = int(sys.argv[i + 1])
    elif sys.argv[i] == "--c":
        c = float(sys.argv[i + 1])
    elif sys.argv[i] == "--F":
        iF0 = float(sys.argv[i + 1])
    else:
        print("Unrecognized argument: {}".format(sys.argv[i]))

##########################
"""IMPORTANT PARAMETERS"""
##########################
# maximum bond dimension, used for both DMRG and TEBD, multiple of 200
maxdim = 1000 if maxdim is None else maxdim
N = 10 if N is None else N
iU = 0.5 * it if iU is None else iU
# the number of steps if not apdative
nsteps = 4000 if nsteps is None else nsteps
iF0 = 10. if iF0 is None else iF0
c = 1. if c is None else c # constant to modify the amplitude of the current

"""We will hold these parameters constant"""
maxerr = 1e-12  # used for DMRG
# lattice spacing, in angstroms
ia = 4
# pulse parameters
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10
pbc = False  # periodic boundary conditions

# an enz simulation must first be evolved by a tl or specified pulse
phi_func = phi_tl
# loaddir = "./Data/Tenpy/Tracking/"
# suot = 0.
# tuot = 1.
# tps = "-nsteps{}".format(nsteps)
# phips = "-nsteps{}-nsites{}-sU{}-tU{}-maxdim{}".format(nsteps, N, suot, tuot, maxdim)
# phi_times = np.load(loaddir + "times" + tps + ".npy")
# phi_vals = np.load(loaddir + "phis" + phips + ".npy")
# phi_func = dict(times=phi_times, phis=phi_vals)
# cps = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, tuot, maxdim)
# comp_current = np.load("./Data/Tenpy/Basic/currents" + cps + ".npy")


out = """Evolving with
U/t0 = {:.1f}
maximum dimension = {}
number of sites = {}
number of steps = {}
scaling factor = {}
pump field strength = {} MV/cm
""".format(iU/it, maxdim, N, nsteps, c, iF0)
print(out)

p = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

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
# E, psi0 = dmrg.run()
#
# psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
times, delta = np.linspace(ti, tf, num=nsteps, endpoint=True, retstep=True)
# we pass in nsteps - 1 because we would like to evauluate the system at
# nsteps time points, including the ground state calculations
# tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":maxdim}, "N_steps":nsteps-1, "F0":iF0}
# tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
# tebd = TEBD(psi, model, p, phi_func, None, None, c, tebd_params)
# times, energies, currents, phis = tebd.run()

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

fps = "-nsites{}-U{}-F{}-maxdim{}".format(p.nsites, p.u, iF0, maxdim)

nnop = FHNearestNeighbor(p)

with h5py.File("./Data/Tenpy/ENZ/psi0" + fps + ".h5", 'r') as f:
    psi = hdf5_io.load_from_hdf5(f)

expec = nnop.H_MPO.expectation_value(psi)
r = np.abs(expec)
theta = np.angle(expec)

scale = 2 * p.a * p.t0 * r * np.sin(theta)

savedir = "./Data/Tenpy/ENZ/INITIALPHITEST-"
ecps = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, p.nsites, p.u, c, iF0, maxdim)
# np.save(savedir + "energies" + ecps + ".npy", energies)
# np.save(savedir + "currents" + ecps + ".npy", currents)
# np.save(savedir + "phis" + ecps + ".npy", phis)
currents = np.load(savedir + "currents" + ecps + ".npy")
phis = np.load(savedir + "phis" + ecps + ".npy")

plt.plot(phis, label="test $\\Phi$")
plt.plot(currents, ls="dashed", label="test current")
compphi = np.load("./Data/Tenpy/ENZ/phis" + ecps + ".npy")
# plt.plot(compphi, ls="dashed", label="comparison $\\Phi$")
plt.legend()
plt.show()

# write metadata to file (evolution time and error)
# np.save(savedir + "metadata" + ecps + ".npy", tot_time)

# plt.plot(currents)
# plt.plot(phis, label="$\\Phi(t)$", ls="dashed")
# plt.legend()
# plt.show()