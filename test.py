import numpy as np
from matplotlib import pyplot as plt
from tenpy.tools import hdf5_io
import h5py

# U = 2.
#
# phis = np.load("Data/Tenpy/ENZ/phis-nsteps4000-nsites10-U{}-c1-maxdim800.npy".format(U))
# currents = np.load("Data/Tenpy/ENZ/currents-nsteps4000-nsites10-U{}-c1-maxdim800.npy".format(U))
#
# plt.plot(currents)
# plt.plot(phis, label="$\\Phi(t)$", ls="dashed")
# plt.legend()
# plt.show()

with h5py.File("Data/Tenpy/ENZ/psi0-nsites10-U0.5-maxdim800.h5", 'r') as f:
    a = hdf5_io.load_from_hdf5(f)

print(np.abs(a.overlap(a)))
