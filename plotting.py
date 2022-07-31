import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error, phi_tl, Parameters
from scipy.stats import linregress

"""CODE FOR PLOTTING ENZ"""
# nsteps = 4000
# nsites = 10
# uot = 0.5
# maxdim = 800
# c = 0.25 # scaling factor
# eps = None
#
# dir = "./Data/"
# params = ""
# if nsteps is not None:
#     dir += "Tenpy/ENZ/"
#     params += "-nsteps{}".format(nsteps)
# params += "-nsites{}".format(nsites)
# if eps is not None:
#     dir += "AdaptiveTimeStep/ENZ/"
#     params += "-epsilon{}".format(eps)
# params += "-U{}-c{}-maxdim{}".format(uot, c, maxdim)
#
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
# plt.plot(currents, color="blue")
# plt.plot(phis, ls="dashed", color="orange", label="$\\Phi(t)$")
# plt.xlabel("Time Step")
# plt.ylabel("Current")
# plt.legend()
# plt.show()
# print(relative_error(phis * c, currents))

"""CODE FOR PLOTTING TRACKING"""
# nsteps = 2000
# nsites = 20
# tuot = 1.  # tracked u/t
# suot = 0.  # system u/t
# maxdim = 800
# eps = None
#
# # directories must be changed manually here
# # data for the tracked current
# tdir = "./Data/Tenpy/Basic/"
# tparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, nsites, tuot, maxdim)
# tcurrents = np.load(tdir + "currents" + tparams + ".npy")
#
# # data for the tracking current
# dir = "./Data/Tenpy/Tracking/"
# params = "-nsteps{}-nsites{}-sU{}-tU{}-maxdim{}".format(nsteps, nsites, suot, tuot, maxdim)
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
#
# # does not have to have the same number of time steps if adaptive step is used
# # but for now, use the same size
# times = np.load(tdir + "times-nsteps{}.npy".format(nsteps))
#
# # calculate transform limited pulse for plotting
# it = 0.52
# ia = 4
# iF0 = 10
# iomega0 = 32.9
# cycles = 10
# p = Parameters(nsites, tuot * it, it, ia, cycles, iomega0, iF0, False)
# tphis = [phi_tl(t, p) for t in times]
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
# ax1.plot(times, currents, color="blue")
# ax1.plot(times, tcurrents, ls="dashed", color="orange", label="Tracked Current")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Current")
# ax1.legend()
# ax2.plot(times, tphis, label="Tracked $\\Phi$", color="orange")
# ax2.plot(times, phis, label="$\\Phi_T$", color="blue")
# ax2.legend()
# ax2.set_xlabel("Time")
# ax2.set_ylabel("$\\Phi(t)$")
# plt.show()
# print(relative_error(tcurrents, currents))

"""CODE FOR PLOTTING MPS VS EXACT"""
# nsteps = 4000
# nsites = 12
# uot = 0.
# maxdim = 1800
# eps = None
#
# # adaptive time step for mps
# if eps is not None:
#     mpsdir = "./Data/AdaptiveTimeStep/Basic/"
#     mpsparams = "-nsites{}-epsilon{}-U{}-maxdim{}".format(nsites, eps, uot, maxdim)
#     mpstparams = "-nsites{}-epsilon{}-U{}-maxdim{}".format(nsites, eps, uot, maxdim)
# else:
#     mpsdir = "./Data/Tenpy/Basic/"
#     mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, nsites, uot, maxdim)
#     mpstparams = "-nsteps{}".format(nsteps)
#
# exactdir = "./Data/Exact/"
# exactparams = "-U{}-nsites{}-nsteps{}".format(uot, nsites, nsteps)
# exacttparams = "-nsteps{}".format(nsteps)
#
# currents = np.load(mpsdir + "currents" + mpsparams + ".npy")
# times = np.load(mpsdir + "times" + mpstparams + ".npy")
# ecurrents = np.load(exactdir + "current" + exactparams + ".npy")
# etimes = np.load(exactdir + "times" + exacttparams + ".npy")
#
# plt.plot(times, currents, color="blue")
# plt.plot(etimes, ecurrents, ls="dashed", color="orange", label="Exact Current")
# plt.title("$\\frac{U}{t_0} = %.2f, N = %d, maxdim = %d, steps = %d, eps = %s$" % (uot, nsites, maxdim, nsteps, str(eps)))
# plt.xlabel("Time")
# plt.ylabel("Current")
# plt.legend()
# plt.show()


"""PLOTTING CURRENT SCALING WITH SYSTEM SIZE"""
# Ns = list(range(6, 27, 2))
# currents = []
# evolve_times = []
# for N in Ns:
#     mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, uot, maxdim)
#     current = np.load(mpsdir + "currents" + mpsparams + ".npy") / N
#     with open(mpsdir + "metadata" + mpsparams + ".txt") as f:
#         evolve_times.append(float(f.readlines()[0]))
#     currents.append(current)
#
# print(evolve_times)
# errors = []
# for i, N in enumerate(Ns):
#     if N != 26:
#         errors.append(relative_error(currents[i+1], currents[i]))
#     print("\\hline\n{} to {} & {:.2f} \\\\".format(N, Ns[i+1], errors[i]))
#
# plt.plot(Ns, evolve_times)
# plt.xlabel("System Size")
# plt.xticks(Ns)
# plt.ylabel("Time (seconds)")
# plt.show()

# energies = np.load(mpsdir + "energies" + mpsparams + ".npy")
#
#
# etimes = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
# ecurrents = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}-pbc.npy".format(uot, 10, nsteps))
# eenergies = np.load("./Data/Exact/energy-U{}-nsites{}-nsteps{}.npy".format(uot, 10, nsteps))
#
#
# plt.plot(times, currents, label="MPS")
# plt.plot(etimes, ecurrents, ls="dashed", label="Exact")
# plt.legend()
# plt.show()


# Ns = list(range(4, 101, 4))
# times = np.load("./Data/Tenpy/GroundState/times-4Nto100N.npy")
#
# res = linregress(Ns, times)
# xs = np.linspace(4, 100, num=1000)
#
# plt.plot(Ns, times, ".")
# plt.plot(xs, res.slope * xs + res.intercept, label="r value: {:.3f}".format(res.rvalue))
# plt.xlabel("System Size")
# plt.ylabel("Time")
# plt.legend()
# plt.show()
