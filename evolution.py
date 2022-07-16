
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

"""
Command line arguments should be used as follows:
--U [float] : U/t_0 ratio
--dim [int multiple of 200] : maximum dimension
--N [even int] : number of sites
--nsteps [int] : number of evolution steps
--c [float] : amount to scale current by in enz
--eps [float] : the acceptable change in states for adaptive time step
"""
# hopping parameter, in units eV
it = .52
maxdim = None
N = None
iU = None
nsteps = None
c = None # only changed if we are performing an enz simulation
epsilon = None  # only changed for adaptive method
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
    elif sys.argv[i] == "--eps":
        epsilon = float(sys.argv[i + 1])
    else:
        print("Unrecognized argument: {}".format(sys.argv[i]))

##########################
"""IMPORTANT PARAMETERS"""
##########################
# maximum bond dimension, used for both DMRG and TEBD, multiple of 200
maxdim = 1000 if maxdim is None else maxdim
N = 10 if N is None else N
iU = 0. * it if iU is None else iU
# the number of steps if not apdative, determines the initial dt if adaptive
nsteps = 4000 if nsteps is None else nsteps

"""We will hold these parameters constant"""
maxerr = 1e-12  # used for DMRG
# lattice spacing, in angstroms
ia = 4
# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10
pbc = False  # periodic boundary conditions

########################
"""TYPE OF SIMULATION"""
########################
adaptive = False if epsilon is None else True
tracking = False
enz = False
assert not (tracking and enz)  # tracking and enz are mutually exclusive
if adaptive:
    epsilon = 1e-5 if epsilon is None else epsilon
if not tracking:
    tracking_info = None
    # a basic simulation evolves by a tl pulse or one specified by loading
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
    if enz:
        c = 1 if c is None else c # constant to modify the amplitude of the pulse
else:
    # TRACKING PARAMETERS
    tuot = 1.  # tracking U/t_0
    dir = "./Data/Tenpy/Basic/"
    tps = "-nsteps{}".format(nsteps)
    ps = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, tuot, maxdim)
    tracking_time = np.load(dir + "times" + tps + ".npy")
    tracking_current = np.load(dir + "currents" + ps + ".npy")

    # tracking info is a dictionary w/ keys "times" and "currents"
    tracking_info = dict(times=tracking_time, currents=tracking_current)
    phi_func = phi_tracking

out = """Evolving with
U/t0 = {:.1f}
adaptive = {}
epsilon = {}
maximum dimension = {}
number of sites = {}
number of steps = {}
""".format(iU/it, adaptive, epsilon, maxdim, N, nsteps)
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
E, psi0 = dmrg.run()

psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
times, delta = np.linspace(ti, tf, num=nsteps, endpoint=True, retstep=True)
# we pass in nsteps - 1 because we would like to evauluate the system at
# nsteps time points, including the ground state calculations
tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":maxdim}, "N_steps":nsteps-1}
tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
tebd = TEBD(psi, model, p, phi_func, epsilon, tracking_info, c, tebd_params)
times, energies, currents, phis = tebd.run(adaptive=adaptive)

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

savedir = "./Data/"
allps = ""
ecps = "-nsites{}".format(p.nsites)
if adaptive:
    savedir += "AdaptiveTimeStep/"
    ecps += "-epsilon{}".format(epsilon)
else:
    savedir += "Tenpy/"
    allps += "-nsteps{}".format(nsteps)
if tracking:
    savedir += "Tracking/"
    ecps += "-sU{}-tU{}".format(p.u, tuot)
elif enz:
    savedir += "ENZ/"
    ecps += "-U{}-c{}".format(p.u, c)
else:
    savedir += "Basic/"
    ecps += "-U{}".format(p.u)
ecps += "-maxdim{}".format(maxdim)
# if phi function is a predefined pulse, save those parameters too
if not callable(phi_func):
    ecps += "--phips" + phips
if adaptive:
    np.save(savedir + "times" + allps + ecps + ".npy", times)
else:
    np.save(savedir + "times" + allps + ".npy", times)
np.save(savedir + "energies" + allps + ecps + ".npy", energies)
np.save(savedir + "currents" + allps + ecps + ".npy", currents)
if tracking or enz:
    np.save(savedir + "phis" + allps + ecps + ".npy", phis)
else:
    # no need to save phi if it was loaded
    if callable(phi_func):
        np.save(savedir + "phis" + allps + "-a{}-f{}-w{}-cycles{}.npy".format(ia, iF0, iomega0, cycles), phis)

# write metadata to file (evolution time and error)
np.save(savedir + "metadata" + allps + ecps + ".npy", tot_time)

# if tracking:
#     plt.plot(times, currents)
#     plt.plot(times, tracking_current, label="Tracked Current", ls="dashed")
#     plt.legend()
#     plt.show()
# elif enz:
#     plt.plot(currents)
#     plt.plot(phis, label="$\\Phi(t)$", ls="dashed")
#     plt.legend()
#     plt.show()
# else:
#     plt.plot(times, currents, label="U/t=0")
#     plt.plot(times, comp_current, label="U/t=1")
#     plt.show()
