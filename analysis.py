import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tools import relative_error, relative_error_interp

###########################################
"""COMPARING ENERGY TO CURRENT AND KAPPA"""
###########################################
# nsteps = 4000
# nsites = 10
# uot = 0.5
# c = 2.
# maxdim = 1000
# F0 = 10.
#
# dir = "./Data/Tenpy/ENZ/"
# params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
# times = np.load("./Data/Tenpy/ENZ/times-nsteps{}.npy".format(nsteps))
# delta = times[1] - times[0]
#
# # energies = np.load(dir + "energies" + params + ".npy")
# # currents = np.load(dir + "currents" + params + ".npy")
# # phis = np.load(dir + "phis" + params + ".npy")
# energies = np.load("Data/Tenpy/ENZ/INITIALPHITEST-energies-nsteps4000-nsites10-U0.5-c1.0-F10.0-maxdim1000.npy")
# currents = np.load("Data/Tenpy/ENZ/INITIALPHITEST-currents-nsteps4000-nsites10-U0.5-c1.0-F10.0-maxdim1000.npy")
# phis = np.load("Data/Tenpy/ENZ/INITIALPHITEST-phis-nsteps4000-nsites10-U0.5-c1.0-F10.0-maxdim1000.npy")
# # efield_energy = .5 * np.gradient(phis, delta)**2
#
# plt.plot(energies - .5 * (nsites / c) * currents**2)
# plt.show()


######################################
"""COMPARING THE PHIS TO EACH OTHER"""
######################################
# nsteps = 4000
# nsites = 10
# uot = 0.5
# maxdim = 1000
# c = 4. # scaling factor
# F0 = 10.
#
# dir = "./Data/Tenpy/ENZ/"
# fig, axs = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
#
# enzphis = []
# for c in [.25, .5, 1., 2., 4.]:
#     params = "-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}".format(nsteps, nsites, uot, c, F0, maxdim)
#     phis = np.load(dir + "phis" + params + ".npy")
#     enzphis.append(phis / np.max(phis))
#
# for i in range(5):
#     for j in range(5):
#         print("{:.2f}".format(relative_error(enzphis[i], enzphis[j])), end=" ")
#     print()


#########################################################
"""2D Current Heat Map of Scaling Factor or F0 vs time"""
#########################################################
# nsteps = 4000
# uot = 2.0
# maxdim = 1000
# N = 10
# grad = False  # whether to plot the gradient of the current
#
# times = np.load("./Data/Tenpy/ENZ/times-nsteps{}.npy".format(nsteps))
# delta = times[1] - times[0]
#
# ylabel = "Scaling factor"
# # ylabel = "Pump Field Strength (MV/cm)"
#
# if ylabel == "Scaling factor":
#     cs = yvals = [.25, .5, 1., 2., 4.]
#     F0 = 10.
#     data = []
#     for c in cs:
#         curr = np.load("./Data/Tenpy/ENZ/currents-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}.npy".format(nsteps, N, uot, c, F0, maxdim))
#         curr = curr.real
#         curr /= np.linalg.norm(curr)
#         if grad:
#             data.append(np.gradient(curr, delta))
#         else:
#             data.append(curr)
# else:
#     c = 1.
#     F0s = yvals = [5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
#     data = []
#     for F0 in F0s:
#         curr = np.load("./Data/Tenpy/ENZ/currents-nsteps{}-nsites{}-U{}-c{}-F{}-maxdim{}.npy".format(nsteps, N, uot, c, F0, maxdim))
#         curr = curr.real
#         curr /= np.linalg.norm(curr)
#         if grad:
#             data.append(np.gradient(curr, delta))
#         else:
#             data.append(curr)
# data = np.array(data)
# lval = data.min()
# hval = data.max()
#
# norm = mpl.colors.Normalize(lval, hval)
# cmap = mpl.cm.viridis
# mp = mpl.cm.ScalarMappable(norm, cmap)
#
# colors = mp.to_rgba(data)
#
# ymin = 0
# ymax = 1
# ticks = []
# for i in range(len(yvals)):
#     plt.vlines(times, ymin, ymax, colors=colors[i])
#     ticks.append((ymin + ymax) / 2)
#     ymin += 1.5
#     ymax += 1.5
# plt.xlim((0, times[-1]))
# plt.ylim((0, ymax - 1.5))
# plt.xlabel("Time")
# plt.ylabel(ylabel)
# plt.title("$U/t_0 = {}$".format(uot))
# plt.colorbar(mp, label="Normalized Current")
# plt.yticks(ticks, yvals)
# savef = "./Data/Images/HeatMap"
# if ylabel == "Scaling factor":
#     savef += "OverC-F{}".format(F0)
# else:
#     savef += "OverF0-c{}".format(c)
# savef += "-nsteps{}-nsites{}-U{}-maxdim{}.png".format(nsteps, N, uot, maxdim)
# plt.savefig(savef)
# plt.show()


###################################
"""Bond Dimension Error Analysis"""
###################################
# nsteps = 4000
# nsites = [6, 8, 10, 12]
# Us = [0., .5, 1., 2.]
# maxdims = [600 + 200 * i for i in range(8)]
#
# fig = plt.figure()
# axs = fig.subplots(2, 2, sharex=True)
#
# for i in range(4):
#     ax = axs[i // 2, i % 2]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         exact = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(U, N, nsteps))
#         errors = []
#         for md in maxdims:
#             curr = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, N, U, md))
#             errors.append(relative_error(exact, curr))
#         ax.plot(maxdims, errors, label="$U/t_0 = {}$".format(U))
#
# handles, labels = axs[1,1].get_legend_handles_labels()
# axs[1,1].legend(handles, labels, loc='upper right')
#
# fig.subplots_adjust(left=.1, bottom=.1, right=.98, top=.9, wspace=None, hspace=None)
#
# fig.supxlabel("Maximum Bond Dimension")
# fig.supylabel("Percent Error")
#
# plt.show()

#####################################
"""Bond Dimension Runtime Analysis"""
#####################################
# nsteps = 4000
# nsites = [6, 8, 10, 12]
# Us = [0., .5, 1., 2.]
# maxdims = [600 + 200 * i for i in range(8)]
#
# fig = plt.figure()
# axs = fig.subplots(2, 2, sharex=True)
#
# for i in range(4):
#     ax = axs[i // 2, i % 2]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         times = []
#         for md in maxdims:
#             t = np.load("./Data/Tenpy/Basic/metadata-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, N, U, md))
#             times.append(float(t) / 3600)
#         ax.plot(maxdims, times, label="$U/t_0 = {}$".format(U))
#     # ax.legend()
#
# handles, labels = axs[1,1].get_legend_handles_labels()
# axs[1,1].legend(handles, labels, loc='lower right')
#
# fig.subplots_adjust(left=.15, bottom=.1, right=.98, top=.9, wspace=None, hspace=None)
#
# fig.supxlabel("Maximum Bond Dimension")
# fig.supylabel("Runtime (hours)")
#
# plt.show()

######################################
"""Dynamic Time Step Error Analysis"""
######################################
# nsteps = 4000
# nsites = [6, 8, 10, 12]
# epsilons = [.01, .001, .0001, 1e-5]
# Us = [0., .5, 1., 2.]
# maxdim = 1000
#
# fig = plt.figure()
# axs = fig.subplots(2, 2, sharex=True)
#
# for i in range(4):
#     ax = axs[i // 2, i % 2]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         exact = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(U, N, nsteps))
#         exactts = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
#         errors = []
#         for eps in epsilons:
#             curr = np.load("./Data/AdaptiveTimeStep/Basic/currents-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(N, eps, U, maxdim))
#             ts = np.load("./Data/AdaptiveTimeStep/Basic/times-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(N, eps, U, maxdim))
#             errors.append(relative_error_interp(exact, exactts, curr, ts))
#         ax.plot(-np.log10(epsilons), errors, label="$U/t_0 = {}$".format(U))
#
# handles, labels = axs[1,0].get_legend_handles_labels()
# axs[1,0].legend(handles, labels, loc='upper right')
#
# fig.supxlabel("$-\\log_{10}\\epsilon}$")
# fig.supylabel("Percent Error")
#
# plt.show()


########################################
"""Dynamic Time Step Runtime Analysis"""
########################################
# nsites = [6, 8, 10, 12]
# epsilons = [.01, .001, .0001, 1e-5]
# Us = [0., .5, 1., 2.]
# maxdim = 1000
#
# fig = plt.figure()
# axs = fig.subplots(2, 2, sharex=True)
#
# for i in range(4):
#     ax = axs[i // 2, i % 2]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         times = []
#         for eps in epsilons:
#             t = np.load("./Data/AdaptiveTimeStep/Basic/metadata-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(N, eps, U, maxdim))
#             times.append(float(t) / 3600)
#         ax.plot(-np.log10(epsilons), times, label="$U/t_0 = {}$".format(U))
#     ax.legend()
#
# fig.supxlabel("$-\\log_{10}\\epsilon}$")
# fig.supylabel("Time (hours)")
#
# plt.show()
