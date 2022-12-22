import numpy as np
from matplotlib import pyplot as plt
from tools import Parameters
from scipy.stats import linregress

nsteps = 2000
nsites = 10
uot = 0.5
maxdim = 2000
ind = 4. # inductance
F0 = 10.
a = 4

"""CODE FOR PLOTTING ENZ"""
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# ypsi = 2 * p.a * p.t0 * nsites * ind
# # print("Y(ψ) ≤ %.4f" % ypsi)
#
# dir = "./Data/Exact/ENZ/"
# params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
#
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
# plt.plot(currents, color="blue")
# plt.plot(phis / ind + currents[0], ls="dashed", color="orange", label="$ \\frac{\\Phi(t)}{\\mathfrak{L}} + J(0)$")
# plt.xlabel("Time Step")
# plt.ylabel("Current")
# plt.legend()
# plt.savefig("./Data/Images/ENZ/ENZplot" + params + ".pdf")
# plt.show()

"""PLOT ENERGY AGAINST J^2 VARYING INDUCTANCE EXACT"""
# dir = "./Data/Exact/ENZ/"
# fig, ax = plt.subplots()
# fig.subplots_adjust(left=.13, right=.95, bottom=.1, top=.95)
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# for ind in [-2., -1., -.5, .5, 1., 2.]:
#     params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
#     currents = np.load(dir + "currents" + params + ".npy")
#     energies = np.load(dir + "energies" + params + ".npy")
#
#     deltaE = energies - energies[0]
#     deltaJ2 = currents**2 - currents[0]**2
#
#     ax.scatter(deltaJ2, deltaE)
#     ax.plot(deltaJ2, -.5 * ind * deltaJ2 / p.a, label="$\\mathfrak{L} = %.1f$" % ind)
#
# params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-F{F0}-a{a}'
# ax.set_xlabel("$J^2(t) - J^2(0)$")
# ax.set_ylabel("$\\mathcal{E}(t) - \\mathcal{E}(0)$")
# ax.legend()
# plt.savefig("./Data/Images/ENZ/deltaEvsDeltaJ2" + params + ".pdf")
# plt.show()

"""PLOT ENERGY AGAINST J^2 VARYING INDUCTANCE MPS"""
dir = "./Data/Tenpy/ENZ/"
fig, ax = plt.subplots()
fig.subplots_adjust(left=.13, right=.95, bottom=.1, top=.95)
p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
for ind in [-2., -1., -.5, .5, 1., 2.]:
    params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
    currents = np.load(dir + "currents" + params + ".npy")
    energies = np.load(dir + "energies" + params + ".npy")

    deltaE = energies - energies[0]
    deltaJ2 = currents**2 - currents[0]**2

    ax.scatter(deltaJ2, deltaE)
    ax.plot(deltaJ2, -.5 * ind * deltaJ2 / p.a, label="$\\mathfrak{L} = %.1f$" % ind)

params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-F{F0}-a{a}'
ax.set_xlabel("$J^2(t) - J^2(0)$")
ax.set_ylabel("$\\mathcal{E}(t) - \\mathcal{E}(0)$")
ax.legend()
plt.savefig("./Data/Images/ENZ/deltaEvsDeltaJ2" + params + ".pdf")
plt.show()

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
