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

def phi_enz_exact(p, nnop, psi, kappa, scale):
    expec = np.vdot(psi, nnop.static.dot(psi))
    r = np.abs(expec)
    theta = np.angle(expec)
    scale = scale / r
    y = kappa / (2 * p.a* p.t0 *r)
    # when this function is 0, induced current = phi
    f = lambda phi: np.sin(phi - theta) + y * phi + scale
    res = root_scalar(f, bracket=[-1/y - scale/y, 1/y - scale/y])
    if not res.converged:
        raise Exception("Could not find zero")
    return res.root

def get_enz_phis(p, nnop, psi_t, kappa, scale, nsteps):
    phis = []
    for i in range(nsteps):
        psi = psi_t[:, i]
        expec = np.vdot(psi, nnop.static.dot(psi))
        r = np.abs(expec)
        theta = np.angle(expec)
        new_scale = scale / r
        y = kappa / (2 * p.a* p.t0 *r)
        # when this function is 0, induced current = phi
        f = lambda phi: np.sin(phi - theta) + y * phi + new_scale
        res = root_scalar(f, bracket=[-1/y - new_scale/y, 1/y - new_scale/y])
        if not res.converged:
            raise Exception("Could not find zero")
        phis.append(res.root)
    return np.array(phis)

def evolve_psi(current_time, psi, onsite, hop_left, hop_right, lat, phi_func, kwargs={}):
    """
    Evolves psi
    :param current_time: time in evolution
    :param psi: the current wavefunction
    :param phi_func: the function used to calculate phi
    :return: -i * H|psi>
    """
    if phi_func.__name__ == "phi_tl":
        phi = phi_func(lat, current_time)
    elif phi_func.__name__ == "phi_enz_exact":
        phi = phi_func(lat, hop_left, psi, kwargs["kappa"], kwargs["scale"])

    a = -1j * (-lat.t0 * (np.exp(-1j*phi)*hop_left.static.dot(psi) + np.exp(1j*phi)*hop_right.static.dot(psi))
               + lat.u * onsite.static.dot(psi))

    return a


def H_expec(psis, times, onsite, hop_left, hop_right, lat, phis):
    """
    Calculates expectation of the hamiltonian
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a Hamiltonian
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phis[i]
        # H|psi>
        Hpsi = -lat.t0 * (np.exp(-1j*phi) * hop_left.dot(psi) + np.exp(1j*phi) * hop_right.dot(psi)) + \
            lat.u * onsite.dot(psi)
        # <psi|H|psi>
        expec.append((np.vdot(psi, Hpsi)).real)
    return np.array(expec)

def J_expec(psis, times, hop_left, hop_right, lat, phis):
    """
    Calculates expectation of the current density
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a density
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phis[i]
        # J|psi>
        Jpsi = -1j*lat.a*lat.t0* (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <psi|J|psi>
        expec.append((np.vdot(psi, Jpsi)).real)
    return np.array(expec)
