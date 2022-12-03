import os
os.environ["OMP_NUM_THREADS"] = "4"

"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from mps_evolve import *
from tenpy.algorithms.truncation import TruncationError
from tenpy.tools import hdf5_io
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import sys
import datetime

def run(nsites, nsteps, uot, ia, ind, maxdim, lock=None):
    # default parameters
    if nsites is None:
        nsites = 10
    if nsteps is None:
        nsteps = 2000
    if uot is None:
        uot = 1.
    if ia is None:
        ia = 4
    if ind is None:
        ind = 1.
    if maxdim is None:
        maxdim = 2000

    """Hubbard model Parameters"""
    it = 0.52  # hopping strength
    iU = uot * it  # interaction strength

    """We will hold these parameters constant"""
    maxerr = 1e-12  # used for DMRG
    # pulse parameters
    iF0 = 10.  # Field amplitude MV/cm
    iomega0 = 32.9  # driving (angular) frequency, in THz
    cycles = 10
    pbc = False  # periodic boundary conditions

    # an enz simulation must first be evolved by a tl pulse
    phi_func = phi_tl

    p = Parameters(nsites, iU, it, ia, cycles, iomega0, iF0, pbc)

    out = """Evolving with
    U/t0 = {:.1f}
    maximum dimension = {}
    number of sites = {}
    number of steps = {}
    inductance = {}
    pump field strength = {} MV/cm
    """.format(iU/it, maxdim, nsites, nsteps, ind, iF0)
    print(out)

    # get the start time
    start_time = time.time()
    model = FHHamiltonian(p, 0)
    current = FHCurrent(p, 0)
    sites = model.lat.mps_sites()
    state = ["up", "down"] * (nsites // 2)
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
    tebd = TEBD(psi, model, p, phi_func, ind, tebd_params)
    times, phis, currents, energies = tebd.run_parallel(lock)

    tot_time = time.time() - start_time

    print("Evolution complete, total time:", tot_time)

    # savedir = "./Data/Tenpy/ENZ/"
    # ecps = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, p.nsites, p.u, c, iF0, maxdim)
    # np.save(savedir + "energies" + ecps + ".npy", energies)
    # np.save(savedir + "currents" + ecps + ".npy", currents)
    # np.save(savedir + "phis" + ecps + ".npy", phis)

    plt.plot(currents)
    plt.plot(np.array(phis) / ind + currents[0], label="$\\Phi(t)/\\mathfrak{L} + J(0)$", ls="dashed")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    UOT = N = NSTEPS = IND = A = MAXDIM = None
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == "--U":
            UOT = float(sys.argv[i + 1])
        elif sys.argv[i] == "--N":
            N = int(sys.argv[i + 1])
        elif sys.argv[i] == "--nsteps":
            NSTEPS = int(sys.argv[i + 1])
        elif sys.argv[i] == "--L":
            IND = float(sys.argv[i + 1])
        elif sys.argv[i] == "--a":
            A = int(sys.argv[i + 1])
        elif sys.argv[i] == "--dim":
            MAXDIM = int(sys.argv[i + 1])
        else:
            print("Unrecognized argument: {}".format(sys.argv[i]))

    run(N, NSTEPS, UOT, A, IND, MAXDIM)
