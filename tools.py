import numpy as np
from scipy.optimize import root_scalar

class Parameters:
    """
    Scales parameters to atomic units in terms of t_0.
    input units: eV (t, U)
    """

    def __init__(self, nsites, u, t, a, cycles, field, strength, pbc, nx=None, ny=None, ub=None, un=None):
        self.nsites = nsites
        self.nx = nx
        self.ny = ny
        self.nup = nsites // 2 + nsites % 2
        self.ndown = nsites // 2

        self.u = u / t
        if ub is not None:
            self.ub = ub / t
        if un is not None:
            self.un = un / t
        self.t0 = 1.

        self.cycles = cycles

        # CONVERTING TO ATOMIC UNITS, w/ energy normalized to t_0
        factor = 1 / (t * 0.036749323)
        self.field = field * factor * 0.0001519828442
        self.a = a * 1.889726125/factor
        self.strength = strength * 1.944689151e-4 * (factor**2)

        self.pbc = pbc #periodic boundary conditions

def phi_tl(p, time):
    """
    Calculate transform limited phi at time
    Params:
        p - an instance of Parameters
        time - current time
    """
    return (p.a * p.strength / p.field) * (np.sin(p.field * time / (2*p.cycles))**2) * np.sin(p.field * time)

def efield_tl(p, time):
    r1 = -p.strength * np.sin(p.field * time / (2 * p.cycles))**2
    r2 = -p.strength / p.cycles  * np.sin(p.field * time) * np.cos(p.field * time / (2 * p.cycles)) * np.sin(p.field * time / (2 * p.cycles))
    return r1 + r2
