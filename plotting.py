import numpy as np
from matplotlib import pyplot as plt
from tools import Parameters, phi_tl
from scipy.stats import linregress

nsteps = 2000
nsites = 10
uot = 1.
maxdim = 2000
ind = 1. # inductance
F0 = 10.
a = 4
kp = 10000  # amplification

"""PLOT TRACKING AND FEEDBACK CONTROL ENZ SIDE BY SIDE"""
p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
trackingdir = "./Data/Exact/ENZ/"
trackingparams = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'

trackingcurrents = np.load(trackingdir + "currents" + trackingparams + ".npy")
trackingphis = np.load(trackingdir + "phis" + trackingparams + ".npy")
trackingtimes = np.load(trackingdir + f"times-nsteps{nsteps}.npy")

feedbackdir = "./Data/Exact/FeedbackENZ/"
feedbackparams = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'

feedbackcurrents = np.load(feedbackdir + "currents" + feedbackparams + ".npy")
feedbackphis = np.load(feedbackdir + "phis" + feedbackparams + ".npy")
feedbacktimes = np.load(feedbackdir + "times" + feedbackparams + ".npy")

fig, axs = plt.subplots(2)
fig.subplots_adjust(left=.13, right=.95, bottom=.1, top=.95)
axs[0].tick_params(axis="both", labelsize=12)
axs[1].tick_params(axis="both", labelsize=12)

axs[0].plot(trackingtimes, trackingcurrents, color="blue", label="$J(t)$")
axs[0].plot(trackingtimes, -trackingphis / (p.a * ind) + trackingcurrents[0],
ls="dashed", color="orange")
axs[0].legend(loc="lower left")
axs[0].text(-7, 0.02, "(a)", fontsize="x-large")
axs[0].set_yticks([-.02, 0., .02])

axs[1].plot(feedbacktimes, feedbackcurrents, color="blue")
axs[1].plot(feedbacktimes, -feedbackphis / (p.a * ind) + feedbackcurrents[0],
ls="dashed", color="orange", label="$ -\\frac{\\Phi(t)}{a\\mathfrak{L}} + J(0)$")
axs[1].legend(loc="lower left")
axs[1].text(-7, 0.02, "(b)", fontsize="x-large")
axs[1].set_yticks([-.02, 0., .02])

params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
plt.xlabel("Time", fontsize=14)
fig.supylabel("Current", fontsize=14)
plt.savefig("./Data/Images/ENZ/TrackingAndFeedbackENZplot" + params + ".pdf")
plt.show()

"""PLOT FEEDBACK AND TRACKING ON THE SAME AXES"""
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# trackingdir = "./Data/Exact/ENZ/"
# trackingparams = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
#
# trackingcurrents = np.load(trackingdir + "currents" + trackingparams + ".npy")
# trackingphis = np.load(trackingdir + "phis" + trackingparams + ".npy")
# trackingtimes = np.load(trackingdir + f"times-nsteps{nsteps}.npy")
#
# feedbackdir = "./Data/Exact/FeedbackENZ/"
# feedbackparams = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
#
# feedbackcurrents = np.load(feedbackdir + "currents" + feedbackparams + ".npy")
# feedbackphis = np.load(feedbackdir + "phis" + feedbackparams + ".npy")
# feedbacktimes = np.load(feedbackdir + "times" + feedbackparams + ".npy")
#
# # plt.plot(trackingtimes, trackingcurrents, color="blue", label="$J(t)$")
# plt.plot(trackingtimes, -trackingphis / (p.a * ind) + trackingcurrents[0],
# color="blue", label="tracking $\\Phi$")
#
# # plt.plot(feedbacktimes, feedbackcurrents, color="blue", label="$J(t)$")
# plt.plot(feedbacktimes, -feedbackphis / (p.a * ind) + feedbackcurrents[0],
# ls="dashed", color="orange", label="feedback$\\Phi$")
#
# # params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
# plt.xlabel("Time")
# plt.ylabel("Current")
# # plt.savefig("./Data/Images/ENZ/TrackingAndFeedbackENZplot" + params + ".pdf")
# plt.show()


"""CODE FOR PLOTTING ENZ"""
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# ypsi = 2 * p.a**2 * p.t0 * nsites * ind
# print("Y(ψ) ≤ %.4f" % ypsi)
#
# dir = "./Data/Exact/ENZ/"
# params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
#
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
# plt.plot(currents, color="blue")
# plt.plot(-phis / (p.a * ind) + currents[0], ls="dashed", color="orange", label="$ -\\frac{\\Phi(t)}{a\\mathfrak{L}} + J(0)$")
# plt.xlabel("Time Step")
# plt.ylabel("Current")
# plt.legend()
# plt.savefig("./Data/Images/ENZ/ENZplot" + params + ".pdf")
# plt.show()

"""PLOT ENZ INDUCED BY FEEDBACK CONTROL"""
# dir = "./Data/Exact/FeedbackENZ/"
# params = f'-nsites{nsites}-U{uot}-ind{ind}-kp{kp}-F{F0}-a{a}'
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# start = 0
# stop = 2 * np.pi * p.cycles / p.field
#
# currents = np.load(dir + "currents" + params + ".npy")
# phis = np.load(dir + "phis" + params + ".npy")
# plt.plot(currents * -p.a * ind, color="blue", label="$J(t)$")
# plt.plot(phis, ls="dashed", color="orange", label="$ -\\frac{\\Phi(t)}{a\\mathfrak{L}}$")
# # plt.plot(phi_tl(p, np.linspace(start, stop, num=len(currents))), label="$\\Phi_{\\rm tl}(t)$")
# plt.xlabel("Time Step")
# plt.ylabel("Current")
# plt.legend()
# # plt.savefig("./Data/Images/ENZ/ENZplot" + params + ".pdf")
# plt.show()

"""PLOT ENERGY AGAINST J^2 VARYING INDUCTANCE EXACT"""
# dir = "./Data/Exact/ENZ/"
# fig, ax = plt.subplots()
# fig.subplots_adjust(left=.18, right=.95, bottom=.13, top=.95)
# ax.tick_params(axis="both", labelsize=12)
#
# p = Parameters(nsites, uot * .52, .52, a, 10, 32.9, F0, True)
# for ind in [-2., -1., -.5, .5, 1., 2.]:
#     params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-ind{ind}-F{F0}-a{a}'
#     currents = np.load(dir + "currents" + params + ".npy")
#     energies = np.load(dir + "energies" + params + ".npy")
#
#     deltaE = energies - energies[0]
#     deltaJ2 = currents**2 - currents[0]**2
#
#     ax.scatter(deltaJ2, deltaE, marker=".")
#     ax.plot(deltaJ2, .5 * ind * deltaJ2, label="$\\mathfrak{L} = %.1f$" % ind)
#
# params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-F{F0}-a{a}'
# ax.set_xlabel("$J^2(t) - J^2(0)$", fontsize=14)
# ax.set_ylabel("$\\mathcal{E}(t) - \\mathcal{E}(0)$", fontsize=14)
# ax.legend(loc="lower left")
# plt.savefig("./Data/Images/ENZ/deltaEvsDeltaJ2" + params + ".pdf")
# plt.show()

"""PLOT ENERGY AGAINST J^2 VARYING INDUCTANCE MPS"""
# dir = "./Data/Tenpy/ENZ/"
# fig, ax = plt.subplots()
# fig.subplots_adjust(left=.12, right=.95, bottom=.1, top=.95)
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
#     ax.plot(deltaJ2, .5 * ind * deltaJ2, label="$\\mathfrak{L} = %.1f$" % ind)
#
# params = f'-nsteps{nsteps}-nsites{nsites}-U{uot}-F{F0}-a{a}'
# ax.set_xlabel("$J^2(t) - J^2(0)$")
# ax.set_ylabel("$\\mathcal{E}(t) - \\mathcal{E}(0)$")
# ax.legend()
# plt.savefig("./Data/Images/ENZ/deltaEvsDeltaJ2" + params + ".pdf")
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
