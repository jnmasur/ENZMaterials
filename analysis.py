import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tools import Parameters, phi_tl
from scipy.stats import linregress

nsteps = 4000
nsites = 10
uot = 1.
maxdim = 2000
ind = 1. # inductance
F0 = 10.
a = 4.
kp = 100
exact = False

##########################
"""DELTA E VS DELTA J^2"""
##########################

# if exact:
#     dir = "./Data/Exact/ENZ/"
#     params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
# else:
#     dir = "./Data/Tenpy/ENZ/"
#     params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-c{1/ind}-F{10.0}-maxdim{maxdim}'
# energies = np.load(dir + "energies" + params + ".npy")
# currents = np.load(dir + "currents" + params + ".npy")
#
# deltaE = energies[0] - energies[1:]
# deltaJ2 = currents[1:]**2 - currents[0]**2
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# res = linregress(deltaJ2, deltaE)
# print(f"R-val = {res.rvalue}")
# print("slope * 2 * a / \mathfrak{L} =", res.slope * p.a * 2 / ind)
#
# plt.scatter(deltaJ2, deltaE)
# plt.plot(deltaJ2, .5 * ind * deltaJ2 / p.a, color="orange")
# plt.show()

################################################
"""DELTA E VS DELTA J^2 for non ENZ evolution"""
################################################

# dir = "./Data/Exact/Basic/"
# params = f'-U{uot}-nsites{nsites}-nsteps{nsteps}'
# p = Parameters(nsites, uot * .52, .52, 4, 10, 32.9, 10., True)
# energies = np.load(dir + "energy" + params + ".npy")
# currents = np.load(dir + "current" + params + ".npy")
# times = np.load(dir + f"times-nsteps{nsteps}.npy")
# phis = phi_tl(p, times)
# dx = times[1]
# phigrad = np.gradient(phis, dx)
#
# expec_deltaE = [np.trapz(phigrad[:i] * currents[:i], dx=dx) / p.a for i in range(2, nsteps)]
# deltaE = energies[0] - energies[2:]
#
# plt.plot(deltaE)
# plt.plot(expec_deltaE, ls="dashed")
# plt.show()

###################################
"""DEPENDENCE OF FEEDBACK ON IND"""
###################################
dir = "./Data/Exact/FeedbackENZ/"

params = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
times = np.load(dir + "times" + params + ".npy")
plt.plot(times, [1] * len(times), label="expected")
for ind in [-2., -1., -.5, .5, 1., 2.]:

    params = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
    p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
    currents = np.load(dir + "currents" + params + ".npy")
    phis = np.load(dir + "phis" + params + ".npy")
    params = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
    times = np.load(dir + "times" + params + ".npy")
    plt.plot(times, (currents / phis) / (-p.a * ind), label="$\\mathfrak{L}=%.2f$" % ind)

plt.xlabel("Time Step")
plt.ylabel("$J / \\Phi$")
plt.legend()
# plt.savefig("./Data/Images/ENZ/ENZplot" + params + ".pdf")
plt.show()

#########################################
"""DEPENDENCE OF a ON CHANGE IN ENERGY"""
#########################################
# dir = "./Data/Exact/ENZ/"
#
# avals = np.array(list(range(1,11)))
# slopes = []
# for a in avals:
#     params = f"-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}"
#     energies = np.load(dir + "energies" + params + ".npy")
#     currents = np.load(dir + "currents" + params + ".npy")
#
#     deltaE = energies[0] - energies[1:]
#     deltaJ2 = currents[1:]**2 - currents[0]**2
#
#     res = linregress(deltaJ2, deltaE)
#     print(f"R-val = {res.rvalue}, a = {a}")
#     slopes.append(res.slope)
#
# slopes = np.array(slopes)
# resofres = linregress(np.log(slopes), np.log(avals))
# print("assuming the slope m = Ca^p")
# print("p = {} and C = {}".format(resofres.slope, np.exp(resofres.intercept)))
#
# plt.plot(avals, slopes)
# plt.show()


#########################################
"""DEPENDENCE OF inductance ON CHANGE IN ENERG"""
#########################################
# dir = "./Data/Exact/ENZ/"
#
# inds = [.125, .25, .5, 1., 2., 4., 8.]
# slopes = []
# for ind in inds:
#     params = f"-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}"
#     energies = np.load(dir + "energies" + params + ".npy")
#     currents = np.load(dir + "currents" + params + ".npy")
#
#     deltaE = energies[0] - energies[1:]
#     deltaJ2 = currents[1:]**2 - currents[0]**2
#
#     res = linregress(deltaJ2, deltaE)
#     print(f"R-val = {res.rvalue}, L = {ind}")
#     slopes.append(res.slope)
#
# slopes = np.array(slopes)
# resofres = linregress(np.log(slopes), np.log(inds))
# print("assuming the slope m = C\mathfrak{L}^p")
# print("p = {} and C = {}".format(resofres.slope, np.exp(resofres.intercept)))
#
# plt.xlabel("$\\mathfrak{L}$")
# plt.plot(inds, slopes)
# plt.show()

############################################
"""DEPENDENCE OF U/t0 ON CHANGE IN ENERGY"""
############################################
# dir = "./Data/Exact/ENZ/"
#
# uots = [.25, .5, 1., 2., 4.]
# slopes = []
# for uot in uots:
#     params = f"-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}"
#     energies = np.load(dir + "energies" + params + ".npy")
#     currents = np.load(dir + "currents" + params + ".npy")
#
#     deltaE = energies[0] - energies[1:]
#     deltaJ2 = currents[1:]**2 - currents[0]**2
#
#     res = linregress(deltaJ2, deltaE)
#     print(f"R-val = {res.rvalue}, U/t0 = {uot}")
#     slopes.append(res.slope)
#
# slopes = np.array(slopes)
# uots = np.array(uots)
# resofres = linregress(np.log(slopes), np.log(uots))
# print("assuming the slope m = C(U/t0)^p")
# print("p = {} and C = {}".format(resofres.slope, np.exp(resofres.intercept)))
#
#
# plt.xlabel("U/t0")
# plt.plot(uots, slopes)
# plt.show()


############################################
"""DEPENDENCE OF sites ON CHANGE IN ENERGY"""
############################################
# dir = "./Data/Exact/ENZ/"
#
# nsitesvals = [4, 6, 8, 10]
# slopes = []
# for nsites in nsitesvals:
#     params = f"-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}"
#     energies = np.load(dir + "energies" + params + ".npy")
#     currents = np.load(dir + "currents" + params + ".npy")
#
#     deltaE = energies[0] - energies[1:]
#     deltaJ2 = currents[1:]**2 - currents[0]**2
#
#     res = linregress(deltaJ2, deltaE)
#     print(f"R-val = {res.rvalue}, N = {nsites}")
#     slopes.append(res.slope)
#
# slopes = np.array(slopes)
# nsitesvals = np.array(nsitesvals)
# resofres = linregress(np.log(slopes), np.log(nsitesvals))
# print("assuming the slope m = CN^p")
# print("p = {} and C = {}".format(resofres.slope, np.exp(resofres.intercept)))
#
# plt.xlabel("N")
# plt.plot(nsitesvals, slopes)
# plt.show()
