import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error, relative_error_interp

###################################
"""Bond Dimension Error Analysis"""
###################################
# nsteps = 4000
# nsites = [6, 8, 10, 12]
# Us = [0., .5, 1., 2.]
# maxdims = [600 + 200 * i for i in range(8)]
#
# fig = plt.figure()
# axs = fig.subplots(3, sharex=True)
#
# for i in range(3):
#     ax = axs[i]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         exact = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(U, N, nsteps))
#         errors = []
#         for md in maxdims:
#             curr = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, N, U, md))
#             errors.append(relative_error(exact, curr))
#         ax.plot(maxdims, errors, label="$U/t_0 = {}$".format(U))
#     ax.legend()
#
# axs[2].set_xlabel("Maximum Bond Dimension")
# axs[1].set_ylabel("Percent Error")
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
# axs = fig.subplots(3, sharex=True)
#
# for i in range(3):
#     ax = axs[i]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         times = []
#         for md in maxdims:
#             t = np.load("./Data/Tenpy/Basic/metadata-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, N, U, md))
#             times.append(t)
#         ax.plot(maxdims, times, label="$U/t_0 = {}$".format(U))
#     ax.legend()
#
# axs[2].set_xlabel("Maximum Bond Dimension")
# axs[1].set_ylabel("Runtime")
#
# plt.show()

######################################
"""Dynamic Time Step Error Analysis"""
######################################
# nsteps = 4000
# nsites = [6, 8, 10]
# epsilons = [.01, .001, .0001, 1e-5]
# Us = [0., .5, 1., 2.]
# maxdim = 1000
#
# fig = plt.figure()
# axs = fig.subplots(2, sharex=True)
#
# for i in range(2):
#     ax = axs[i]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         exact = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(U, N, nsteps))
#         exactts = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
#         errors = []
#         for eps in epsilons:
#             curr = np.load("./Data/AdaptiveTimeStep/Basic/currents-nsteps{}-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(nsteps, N, eps, U, maxdim))
#             ts = np.load("./Data/AdaptiveTimeStep/Basic/times-nsteps{}-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(nsteps, N, eps, U, maxdim))
#             errors.append(relative_error_interp(exact, exactts, curr, ts))
#         ax.semilogx(epsilons, errors, label="$U/t_0 = {}$".format(U))
#     ax.legend()
#
# axs[1].set_xlabel("Epsilon")
# axs[1].set_ylabel("Percent Error")
#
# plt.show()


########################################
"""Dynamic Time Step Runtime Analysis"""
########################################
# nsites = [6, 8, 10]
# epsilons = [.01, .001, .0001, 1e-5]
# Us = [0., .5, 1., 2.]
# maxdim = 1000
#
# fig = plt.figure()
# axs = fig.subplots(3, sharex=True)
#
# for i in range(3):
#     ax = axs[i]
#     N = nsites[i]
#     ax.set_title("N = {}".format(N))
#     for U in Us:
#         times = []
#         for eps in epsilons:
#             t = np.load("./Data/AdaptiveTimeStep/Basic/metadata-nsites{}-epsilon{}-U{}-maxdim{}.npy".format(N, eps, U, maxdim))
#             times.append(t)
#         ax.plot(-np.log10(epsilons), times, label="$U/t_0 = {}$".format(U))
#     ax.legend()
#
# axs[1].set_xlabel("Epsilon")
# axs[1].set_ylabel("Time")
#
# plt.show()
