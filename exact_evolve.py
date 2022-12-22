import numpy as np
from tools import *

def phi_enz_exact(p, nnop, psi, ind, scale):
    expec = np.vdot(psi, nnop.static.dot(psi))
    r = np.abs(expec)
    theta = np.angle(expec)
    g = scale / r
    y = 2 * p.a**2 * p.t0 * r * ind
    # when this function is 0, induced current = phi
    f = lambda phi: np.sin(phi - theta) - (phi / y) + g
    res = root_scalar(f, bracket=[-abs(y) + g*y, abs(y) + g*y])
    if not res.converged:
        raise Exception("Could not find zero")
    return res.root

def get_enz_phis(p, nnop, psi_t, ind, scale, nsteps):
    phis = []
    for i in range(nsteps):
        psi = psi_t[:, i]
        expec = np.vdot(psi, nnop.static.dot(psi))
        r = np.abs(expec)
        theta = np.angle(expec)
        g = scale / r
        y = 2 * p.a * p.t0 * r * ind
        # when this function is 0, induced current = phi
        f = lambda phi: np.sin(phi - theta) + (phi / y) + g
        res = root_scalar(f, bracket=[-abs(y) - g*y, abs(y) - g*y])
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
        phi = phi_func(lat, hop_left, psi, kwargs["ind"], kwargs["scale"])

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
