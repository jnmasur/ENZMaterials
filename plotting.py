import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error, phi_tl, Parameters
from scipy.stats import linregress

nsteps = 2000
nsites = 20
uot = 0. # u over t0 ratio
maxdim = 800
pbc = False

mpsdir = "./Data/Tenpy/Basic/"
mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, nsites, uot, maxdim)

times = np.load(mpsdir + "times-nsteps{}.npy".format(nsteps))

trackdir = "./Data/Tenpy/Tracking/"
trackparams = "-nsteps{}-nsites{}-sU{}-tU{}-maxdim{}".format(nsteps, nsites, 0., uot, maxdim)

exactdir = "./Data/Exact/"
exactparams = "-U{}-nsites{}-nsteps{}".format(uot, nsites, nsteps)

"""CODE FOR PLOTTING TRACKING"""
# tcurrents = np.load(mpsdir + "currents" + mpsparams + ".npy")
# currents = np.load(trackdir + "currents" + trackparams + ".npy")
# phis = np.load(trackdir + "phis" + trackparams + ".npy")
# # make this True to calculate and plot the regular pulse
# if True:
#     it = 0.52
#     ia = 4
#     iF0 = 10
#     iomega0 = 32.9
#     cycles = 10
#     p = Parameters(nsites, uot * it, it, ia, cycles, iomega0, iF0, False)
#     tphis = [phi_tl(t, p) for t in times]
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

"""PLOTTING CURRENT SCALING WITH SYSTEM SIZE"""
Ns = list(range(6, 27, 2))
currents = []
evolve_times = []
for N in Ns:
    mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, uot, maxdim)
    current = np.load(mpsdir + "currents" + mpsparams + ".npy") / N
    with open(mpsdir + "metadata" + mpsparams + ".txt") as f:
        evolve_times.append(float(f.readlines()[0]))
    currents.append(current)

print(evolve_times)
errors = []
for i, N in enumerate(Ns):
    if N != 26:
        errors.append(relative_error(currents[i+1], currents[i]))
    print("\\hline\n{} to {} & {:.2f} \\\\".format(N, Ns[i+1], errors[i]))

plt.plot(Ns, evolve_times)
plt.xlabel("System Size")
plt.xticks(Ns)
plt.ylabel("Time (seconds)")
plt.show()

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
