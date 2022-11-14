import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error, phi_tl, Parameters, relative_error_interp, spectrum
from scipy.stats import linregress

nsteps = 2000
nsites = 10
uot = 1.
maxdim = 2000
c = 4. # scaling factor
F0 = 10.

"""CODE FOR PLOTTING ENZ"""
# p = Parameters(nsites, uot * .52, .52, 4, 10, 32.9, 10., False)
# ypsi = 2 * p.a * p.t0 * (nsites - 1) / c
# print("Y(ψ) ≤ %.4f" % ypsi)
#
# dir = "./Data/Tenpy/ENZ/"
# params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
# print(relative_error(phis * c + currents[0], currents))
# plt.plot(currents, color="blue")
# plt.plot(c * phis + currents[0], ls="dashed", color="orange", label="$ \\frac{\\Phi(t)}{\\mathfrak{L}} + J(0)$")
# plt.xlabel("Time Step")
# plt.ylabel("Current")
# plt.legend()
# plt.savefig("./Data/Images/ENZ/" + params[1:] + ".png")
# plt.show()

"""PLOT ENZ EHRENFEST"""
# dir = "./Data/Tenpy/Ehrenfest/"
# params = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, nsites, uot, maxdim)
#
# currents = np.load(dir + "currents" + params + ".npy")
# rhs = np.load(dir + "RHS" + params + ".npy")
# plt.plot(currents, color="blue", label="$\\frac{dJ}{dt}$")
# plt.plot(rhs, ls="dashed", color="orange")
# plt.xlabel("Time Step")
# # plt.ylabel("Current")
# plt.legend()
# plt.show()

"""PLOT ENZ PHIS AGAINST EACH OTHER"""
# dir = "./Data/Tenpy/ENZ/"
# fig, axs = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
#
# for i, uot in enumerate([.5, 1., 2.]):
#     for c in [.25, .5, 1., 2., 4.]:
#         params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#         phis = np.load(dir + "phis" + params + ".npy")
#         axs[i].plot(phis, label="c={}".format(c))
#
#     axs[i].set_title("$U/t_0$={}".format(uot))
#     axs[i].legend()
#
# axs[1].set_xlabel("Time Step")
# axs[0].set_ylabel("$\\Phi$")
#
# fig.subplots_adjust(left=.07, bottom=.15, right=.98, top=.92, wspace=.2, hspace=None)
#
# plt.savefig("./Data/Images/PhiVsKappa-F{}.png".format(F0))
# plt.show()

"""PLOT CHANGE IN CURRENT^2 VS CHANGE IN ENERGY (MPS)"""
# p = Parameters(nsites, uot * .52, .52, 4, 10, 32.9, 10., False)
#
# dir = "./Data/Tenpy/ENZ/"
# params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
# times = np.load("./Data/Tenpy/Basic/times-nsteps{}.npy".format(nsteps))
# energies = np.load(dir + "energies" + params + ".npy")
# currents = np.load(dir + "currents" + params + ".npy")
# deltaE = energies[1:] - energies[0]
# deltaJ2 = currents[1:]**2 - currents[0]**2
#
# xs = np.linspace(min(deltaJ2), max(deltaJ2), num=1000)
# comparison = -0.5 * ((nsites - 1) / c) * np.sqrt(p.a) * xs + max(deltaE)
# plt.scatter(deltaJ2, deltaE, c=np.arange(len(deltaE)))
# plt.plot(xs, comparison, ls="dashed")
# plt.xlabel("$J^2 (t) - J^2 (0)$")
# plt.ylabel("$E(t) - E(0)$")
# plt.show()

"""PLOT CHANGE IN CURRENT^2 VS CHANGE IN ENERGY (Exact)"""
p = Parameters(nsites, uot * .52, .52, 4, 10, 32.9, 10., True)

dir = "./Data/Exact/ENZ/"
params = "-nsteps{}-nsites{}-U{}-c{}-F{}".format(nsteps, nsites, uot, c, F0)
times = np.load("./Data/Exact/ENZ/times-nsteps{}.npy".format(nsteps))
energies = np.load(dir + "energies" + params + ".npy")
currents = np.load(dir + "currents" + params + ".npy")
deltaE = energies[1:] - energies[0]
deltaJ2 = currents[1:]**2 - currents[0]**2

xs = np.linspace(min(deltaJ2), max(deltaJ2), num=1000)
comparison = -0.5 * (nsites / c) * p.a**(1/4) * xs + max(deltaE)
plt.scatter(deltaJ2, deltaE, c=np.arange(len(deltaE)))
plt.plot(xs, comparison, ls="dashed")
plt.xlabel("$J^2 (t) - J^2 (0)$")
plt.ylabel("$E(t) - E(0)$")
plt.show()


"""PLOT ENZ ENERGIES AGAINST EACH OTHER"""
# nsteps = 4000
# nsites = 10
# uot = 2.
# maxdim = 1000
# c = 1. # scaling factor
# F0 = 10.
#
# dir = "./Data/Tenpy/ENZ/"
#
# fig, axs = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
#
# for i, uot in enumerate([.5, 1., 2.]):
#     for c in [.25, .5, 1., 2., 4.]:
#         params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#         energies = np.load(dir + "energies" + params + ".npy")
#         axs[i].plot(energies[-1] - energies, label="c={}".format(c))
#
#     axs[i].set_title("$U/t_0$={}".format(uot))
#     axs[i].legend()
#
# axs[1].set_xlabel("Time Step")
# axs[0].set_ylabel("Energy")
#
# fig.subplots_adjust(left=.09, bottom=.15, right=.98, top=.92, wspace=.2, hspace=None)
#
# plt.savefig("./Data/Images/EnergyVsKappa-F{}.png".format(F0))
#
# plt.show()

"""PLOT INDUCTOR ENERGIES"""
# nsteps = 4000
# nsites = 10
# uot = 2.
# maxdim = 1000
# c = 1. # scaling factor
# F0 = 10.
#
# dir = "./Data/Tenpy/ENZ/"
#
# fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
#
# for i, uot in enumerate([.5, 1., 2.]):
#     for c in [.25, .5, 1., 2., 4.]:
#         params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#         energies = np.load(dir + "energies" + params + ".npy")
#         currents = np.load(dir + "currents" + params + ".npy")
#         axs[0, i].plot((energies - energies[0]) - ((nsites / c) * (currents**2 - currents[0]**2)), label="c={}".format(c))
#         axs[1, i].plot((nsites / c) * (currents**2 - currents[0]**2), label="c={}".format(c))
#
#     axs[0, i].set_title("$U/t_0$={}".format(uot))
#     axs[0, i].legend()
#
# axs[1, 1].set_xlabel("Time Step")
# axs[0, 0].set_ylabel("Energy")
# axs[1, 0].set_ylabel("Inductor Energy")
#
# fig.subplots_adjust(left=.09, bottom=.15, right=.98, top=.92, wspace=.2, hspace=None)
#
# # plt.savefig("./Data/Images/EnergyVsKappa-F{}.png".format(F0))
#
# plt.show()

"""PLOT A SINGLE ENZ SPECTRUM AGAINST THE TL SPECTRUM"""
# nsteps = 4000
# nsites = 10
# uot = 0.5
# maxdim = 1000
# c = 1. # scaling factor
# F0 = 10.  # ONLY WORKS WITH 10 F0 RN
#
# # getting frequency in AU
# t = .52
# freq = 32.9 * 0.0001519828442 / (2 * np.pi * t * 0.036749323)
#
# dir = "./Data/Tenpy/ENZ/"
# params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
# phis = np.load(dir + "phis" + params + ".npy")
#
# phitl = np.load("./Data/Tenpy/Basic/phis-nsteps4000-a4-f10-w32.9-cycles10.npy")
# times = np.load("./Data/Tenpy/Basic/times-nsteps4000.npy")
# delta = times[1] - times[0]
#
# efreqs, especs = spectrum(phis, delta)
# tlfreqs, tlspecs = spectrum(phitl, delta)
#
# efreqs /= freq
# tlfreqs /= freq
#
# plt.semilogy(efreqs, especs, label="ENZ")
# plt.semilogy(tlfreqs, tlspecs, label="TL")
# plt.legend()
# plt.show()


"""PLOT ENZ SPECTRA AGAINST EACH OTHER"""
# nsteps = 4000
# nsites = 10
# uot = 0.5
# maxdim = 1000
# c = 1. # scaling factor
# F0 = 10.  # ONLY WORKS WITH 10 F0 RN
#
# # getting frequency in AU
# t = .52
# freq = 32.9 * 0.0001519828442 / (2 * np.pi * t * 0.036749323)
#
# dir = "./Data/Tenpy/ENZ/"
# # params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
# #
# # times = np.load(dir + "times-nsteps{}.npy".format(nsteps))
# # delta = times[1] - times[0]
# #
# # current2 = np.load(dir + "currents" + params + ".npy").real
# # current1 = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, nsites, uot, maxdim)).real
# # currents = np.append(current1, current2)
# #
# # freq1, spec1 = spectrum(current1, delta)
# # freqs, specs = spectrum(currents, delta)
# # plt.semilogy(freq1, spec1)
# # plt.semilogy(freqs, specs, label="ENZ")
# # plt.legend()
# # plt.show()
#
# times = np.load(dir + "times-nsteps{}.npy".format(nsteps))
# current1 = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, nsites, uot, maxdim)).real
# delta = times[1] - times[0]
# freq1, spec1 = spectrum(current1, delta)
# freq1 /= freq
# plt.semilogy(freq1, spec1)
# current1 = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, nsites, uot, maxdim)).real
# for c in [.25, .5, 1., 2., 4.]:
#     params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#
#     current2 = np.load(dir + "currents" + params + ".npy").real
#     zindx = np.where(abs(current2) < 1e-5)[0][-1]
#     current2 = current2[:zindx]
#     # currents = np.append(current1, current2)
#     # freqs, specs = spectrum(currents, delta)
#     freqs, specs = spectrum(current2, delta)
#     freqs /= freq
#
#     plt.semilogy(freqs, specs, label="c={}".format(c))
#
# plt.legend()
# plt.xlim((0, None))
# plt.xlabel("Harmonic Order")
# plt.ylabel("Spectrum")
# plt.title("$U/t_0 = {}$".format(uot))
# plt.show()


"""PLOT ENZ INDUCED BY PRELOADED FIELD"""
# savedir = "./Data/Tenpy/ENZ/"
# ecps = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(4000, 10, 0.5, 1., 10., 1000)
# currents = np.load(savedir + "TEST" + "currents" + ecps + ".npy")
# phis = np.load(savedir + "TEST" + "phis" + ecps + ".npy")
#
# plt.plot(currents, label="J(t)")
# plt.plot(phis, label="phi(t)", ls="dashed")
# plt.legend()
# plt.show()

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
# nsites = 10
# uot = 0.
# maxdim = 1000
# eps = 1e-5
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

# print(relative_error_interp(ecurrents, etimes, currents, times))
#
# plt.plot(times, currents, color="blue")
# plt.plot(etimes, ecurrents, ls="dashed", color="orange", label="Exact Current")
# plt.title("$\\frac{U}{t_0} = %.2f, N = %d, maxdim = %d, steps = %d, eps = %s$" % (uot, nsites, maxdim, nsteps, str(eps)))
# plt.xlabel("Time")
# plt.ylabel("Current")
# plt.legend()
# plt.savefig("./Data/Images/Comparison-exact" + exactparams + "-mps" + mpsparams + ".png")
# plt.show()
# plt.plot(np.diff(times))
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
