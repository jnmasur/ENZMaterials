import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error

#############################
"""Bond Dimension Analysis"""
#############################

nsteps = 4000
nsites = [6, 8, 10, 12]
Us = [0., .5, 1., 2.]
maxdims = [600 + 200 * i for i in range(8)]

fig = plt.figure()
axs = fig.subplots(4, sharex=True)

for i in range(4):
    ax = axs[i]
    N = nsites[i]
    ax.set_title("N = {}".format(N))
    for U in Us:
        exact = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(U, N, nsteps))
        errors = []
        for md in maxdims:
            curr = np.load("./Data/Tenpy/Basic/currents-nsteps{}-nsites{}-U{}-maxdim{}.npy".format(nsteps, N, U, md))
            errors.append(relative_error(exact, curr))
        ax.plot(maxdims, errors, label="$U/t_0 = {}$".format(U))
    ax.legend()

axs[2].set_xlabel("Maximum Bond Dimension")
axs[1].set_ylabel("Percent Error")

plt.show()
