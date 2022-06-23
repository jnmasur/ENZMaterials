import numpy as np
from matplotlib import pyplot as plt

phis = np.load("Data/Tenpy/ENZ/phis-nsteps4000-nsites10-U0.5-maxdim800.npy")
currents = np.load("Data/Tenpy/ENZ/currents-nsteps4000-nsites10-U0.5-maxdim800.npy")

plt.plot(currents)
plt.plot(phis, label="$\\Phi(t)$", ls="dashed")
plt.legend()
plt.show()
