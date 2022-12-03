import numpy as np
from scipy.optimize import root_scalar
from scipy import signal
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel, CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.models.lattice import Chain, Honeycomb
from tenpy.networks.site import SpinHalfFermionSite

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

class FHHamiltonian(FermiHubbardChain):
    def __init__(self, p, phi):
        t0 = p.t0 * np.exp(-1j * phi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", "explicit_plus_hc":True,
        "L":p.nsites, "mu":0, "V":0, "U":p.u, "t":t0}
        self.options = model_params = Config(model_dict, "FHHam-U{}".format(p.u))
        FermiHubbardChain.__init__(self, model_params)

class FHCurrentModel(CouplingMPOModel):
    def __init__(self, p, phi):
        t0 = p.t0 * np.exp(-1j * phi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites, "t":t0, "a":p.a}
        model_params = Config(model_dict, "FHCurrent-U{}".format(p.u))
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t = model_params.get('t', 1.)
        a = model_params.get('a', 4 * t * 1.889726125 * 0.036749323)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # the -dx is necessary for hermitian conjugation, see documentation
            self.add_coupling(-1j * a * t, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdu', u1, 'Cu', -dx)
            self.add_coupling(-1j * a * t, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdd', u1, 'Cd', -dx)

class FHCurrent(FHCurrentModel, NearestNeighborModel):
    default_lattice = Chain
    force_default_lattice = True

class FHNearestNeighborModel(CouplingMPOModel):
    def __init__(self, p):
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites}
        self.options = model_params = Config(model_dict, "FHNearestNeighbors")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # the -dx is necessary for hermitian conjugation, see documentation
            self.add_coupling(1, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(1, u1, 'Cdd', u2, 'Cd', dx)

class FHNearestNeighbor(FHNearestNeighborModel, NearestNeighborModel):
    default_lattice = Chain
    force_default_lattice = True

class FHHamiltonian(FermiHubbardChain):
    def __init__(self, p, phi):
        t0 = p.t0 * np.exp(-1j * phi)
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", "explicit_plus_hc":True,
        "L":p.nsites, "mu":0, "V":0, "U":p.u, "t":t0}
        self.options = model_params = Config(model_dict, "FHHam-U{}".format(p.u))
        FermiHubbardChain.__init__(self, model_params)

# class FHCurrentModel(CouplingMPOModel):
#     def __init__(self, p, phi):
#         t0 = p.t0 * np.exp(-1j * phi)
#         model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
#         "L":p.nsites, "t":t0, "a":p.a}
#         model_params = Config(model_dict, "FHCurrent-U{}".format(p.u))
#         CouplingMPOModel.__init__(self, model_params)
#
#     def init_sites(self, model_params):
#         cons_N = model_params.get('cons_N', 'N')
#         cons_Sz = model_params.get('cons_Sz', 'Sz')
#         site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
#         return site
#
#     def init_terms(self, model_params):
#         # 0) Read out/set default parameters.
#         t = model_params.get('t', 1.)
#         a = model_params.get('a', 4 * t * 1.889726125 * 0.036749323)
#
#         for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
#             # the -dx is necessary for hermitian conjugation, see documentation
#             self.add_coupling(-1j * a * t, u1, 'Cdu', u2, 'Cu', dx)
#             self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdu', u1, 'Cu', -dx)
#             self.add_coupling(-1j * a * t, u1, 'Cdd', u2, 'Cd', dx)
#             self.add_coupling(1j * a * np.conjugate(t), u2, 'Cdd', u1, 'Cd', -dx)
#
# class FHCurrent(FHCurrentModel, NearestNeighborModel):
#     default_lattice = Chain
#     force_default_lattice = True
#
# class FHNearestNeighborModel(CouplingMPOModel):
#     def __init__(self, p):
#         model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
#         "L":p.nsites}
#         self.options = model_params = Config(model_dict, "FHNearestNeighbors")
#         CouplingMPOModel.__init__(self, model_params)
#
#     def init_sites(self, model_params):
#         cons_N = model_params.get('cons_N', 'N')
#         cons_Sz = model_params.get('cons_Sz', 'Sz')
#         site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
#         return site
#
#     def init_terms(self, model_params):
#         for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
#             # the -dx is necessary for hermitian conjugation, see documentation
#             self.add_coupling(1, u1, 'Cdu', u2, 'Cu', dx)
#             self.add_coupling(1, u1, 'Cdd', u2, 'Cd', dx)
#
# class FHNearestNeighbor(FHNearestNeighborModel, NearestNeighborModel):
#     default_lattice = Chain
#     force_default_lattice = True

class GrapheneHamiltonian(CouplingModel, MPOModel):
    def __init__(self, model_params, phi):
        self.options = model_params

        model_params["phi"] = phi

        p = model_params["p"]

        t0 = p.t0
        site = SpinHalfFermionSite()
        lat = Honeycomb(p.nx, p.ny, site)

        CouplingModel.__init__(self, lat)

        # we assume a pulse polarized in the x direction
        # nearest neighbor in the i direction
        s1 = -t0 * np.exp(-1j * phi)
        self.add_coupling(s1, 1, "Cdu", 0, "Cu", [1, 0], plus_hc=True)
        self.add_coupling(s1, 1, "Cdd", 0, "Cd", [1, 0], plus_hc=True)

        # nearest neighbor in the cos(pi/3)i + sin(pi/3)j direction
        s2 = -t0 * np.exp(-1j * phi * .5)
        self.add_coupling(s2, 0, "Cdu", 1, "Cu", [0, 0], plus_hc=True)
        self.add_coupling(s2, 0, "Cdd", 1, "Cd", [0, 0], plus_hc=True)

        # nearest neighbor in the cos(2pi/3)i + sin(2pi/3)j direction
        s3 = -t0 * np.exp(1j * phi * .5)
        self.add_coupling(s3, 1, "Cdu", 0, "Cu", [0, 1], plus_hc=True)
        self.add_coupling(s3, 1, "Cdd", 0, "Cd", [0, 1], plus_hc=True)

        MPOModel.__init__(self, lat, self.calc_H_MPO())


class GrapheneNearestNeighbor(CouplingModel, MPOModel):
    def __init__(self, model_params):
        self.options = model_params

        p = model_params["p"]

        t0 = p.t0
        a = p.a

        site = SpinHalfFermionSite()
        lat = Honeycomb(p.nx, p.ny, site)

        CouplingModel.__init__(self, lat)

        # we assume a pulse polarized in the x direction
        # nearest neighbor in the i direction
        self.add_coupling(1, 1, "Cdu", 0, "Cu", [1, 0], plus_hc=False)
        self.add_coupling(1, 1, "Cdd", 0, "Cd", [1, 0], plus_hc=False)

        # nearest neighbor in the cos(pi/3)i + sin(pi/3)j direction
        self.add_coupling(1, 0, "Cdu", 1, "Cu", [0, 0], plus_hc=False)
        self.add_coupling(1, 0, "Cdd", 1, "Cd", [0, 0], plus_hc=False)

        # nearest neighbor in the cos(2pi/3)i + sin(2pi/3)j direction
        self.add_coupling(1, 1, "Cdu", 0, "Cu", [0, 1], plus_hc=False)
        self.add_coupling(1, 1, "Cdd", 0, "Cd", [0, 1], plus_hc=False)

        MPOModel.__init__(self, lat, self.calc_H_MPO())


class GrapheneCurrent(CouplingModel, MPOModel):
    def __init__(self, model_params, phi):
        self.options = model_params

        model_params["phi"] = phi

        p = model_params["p"]

        t0 = p.t0
        a = p.a

        site = SpinHalfFermionSite()
        lat = Honeycomb(p.nx, p.ny, site)

        CouplingModel.__init__(self, lat)

        # we assume a pulse polarized in the x direction
        # nearest neighbor in the i direction
        s1 = a * t0 * np.exp(-1j * phi)
        self.add_coupling(-1j * s1, 1, "Cdu", 0, "Cu", [1, 0], plus_hc=False)
        self.add_coupling(-1j * s1, 1, "Cdd", 0, "Cd", [1, 0], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s1), 0, "Cdu", 1, "Cu", [-1, 0], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s1), 0, "Cdd", 1, "Cd", [-1, 0], plus_hc=False)

        # nearest neighbor in the cos(pi/3)i + sin(pi/3)j direction
        s2 = a * t0 * np.exp(-1j * phi * .5)
        self.add_coupling(-1j * s2, 0, "Cdu", 1, "Cu", [0, 0], plus_hc=False)
        self.add_coupling(-1j * s2, 0, "Cdd", 1, "Cd", [0, 0], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s2), 1, "Cdu", 0, "Cu", [0, 0], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s2), 1, "Cdd", 0, "Cd", [0, 0], plus_hc=False)

        # nearest neighbor in the cos(2pi/3)i + sin(2pi/3)j direction
        s3 = a * t0 * np.exp(1j * phi * .5)
        self.add_coupling(-1j * s3, 1, "Cdu", 0, "Cu", [0, 1], plus_hc=False)
        self.add_coupling(-1j * s3, 1, "Cdd", 0, "Cd", [0, 1], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s3), 0, "Cdu", 1, "Cu", [0, -1], plus_hc=False)
        self.add_coupling(1j * np.conjugate(s3), 0, "Cdd", 1, "Cd", [0, -1], plus_hc=False)

        MPOModel.__init__(self, lat, self.calc_H_MPO())


class HBNHamiltonian(CouplingModel, MPOModel):
    def __init__(self, model_params, phi):
        self.options = model_params

        model_params["phi"] = phi

        p = model_params["p"]

        t0 = p.t0
        ub = p.ub
        un = p.un
        site = SpinHalfFermionSite()
        lat = Honeycomb(p.nx, p.ny, site)

        CouplingModel.__init__(self, lat)

        # we assume a pulse polarized in the x direction
        # nearest neighbor in the i direction
        s1 = -t0 * np.exp(-1j * phi)
        self.add_coupling(s1, 1, "Cdu", 0, "Cu", [1, 0], plus_hc=True)
        self.add_coupling(s1, 1, "Cdd", 0, "Cd", [1, 0], plus_hc=True)

        # nearest neighbor in the cos(pi/3)i + sin(pi/3)j direction
        s2 = -t0 * np.exp(-1j * phi * .5)
        self.add_coupling(s2, 0, "Cdu", 1, "Cu", [0, 0], plus_hc=True)
        self.add_coupling(s2, 0, "Cdd", 1, "Cd", [0, 0], plus_hc=True)

        # nearest neighbor in the cos(2pi/3)i + sin(2pi/3)j direction
        s3 = -t0 * np.exp(1j * phi * .5)
        self.add_coupling(s3, 1, "Cdu", 0, "Cu", [0, 1], plus_hc=True)
        self.add_coupling(s3, 1, "Cdd", 0, "Cd", [0, 1], plus_hc=True)

        # add onsite interaction
        self.add_onsite(ub, 0, "NuNd")
        self.add_onsite(un, 1, "NuNd")

        MPOModel.__init__(self, lat, self.calc_H_MPO())


HBNNearestNeighbor = GrapheneNearestNeighbor
HBNCurrent = GrapheneCurrent


class NearestNeighborCommutatorModel(CouplingMPOModel):
    def __init__(self, p):
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites}
        model_params = Config(model_dict, "NNComm")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        L = model_params.get('L', 10)

        self.add_onsite_term(1., 0, 'Ntot')
        self.add_onsite_term(-1., L-1, 'Ntot')

class InteractionNNCommutatorModel(CouplingMPOModel):
    def __init__(self, p):
        model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", 'explicit_plus_hc':False,
        "L":p.nsites, "a":p.a}
        model_params = Config(model_dict, "NNComm")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-1., u1, 'Cdd', u2, 'Nu Cd', dx)
            self.add_coupling(1., u1, 'Nu Cdd', u2, 'Cd', dx)
            self.add_coupling(-1., u1, 'Cdu', u2, 'Nd Cu', dx)
            self.add_coupling(1., u1, 'Nd Cdu', u2, 'Cu', dx)

def get_phi(fh, model_params, evals):
    """
    Returns phi based on the time given in model_params and expectation values
    Params:
        fh - boolean indicating whether the system is FH or graphene
        model_params - dictionary containing a lot of important info
        evals - dictonary containing the expectation values of observables
    """
    phi_func = model_params["phi_func"]

    # if the phi has been predefined by a dictionary
    if type(phi_func) is dict:
        t = model_params["time"]
        phi_vals = self.phi_func["phis"]
        phi_times = self.phi_func["times"]
        if t in phi_func["times"]:
            return phi_vals[phi_times.index(t)]
        else:
            # interpolation to find the value of phi
            if t >= phi_times[-1]:
                phi = phi_vals[-1]
            else:
                indx = np.where(phi_times > t)[0][0]
                phi = phi_vals[indx-1] + (phi_vals[indx] - phi_vals[indx-1]) / \
                      (phi_times[indx] - phi_times[indx-1]) * (t - phi_times[indx-1])

    elif phi_func.__name__ == "phi_tl":
        args = [model_params["time"]]

    elif phi_func.__name__ == "phi_tracking":
        ttimes = model_params["tracking_times"]
        tcurrents = model_params["tracking_currents"]
        t = model_params["time"]
        args = [evals["FHNearestNeighbor"][-1] if fh else evals["GrapheneNearestNeighbor"][-1]]
        if t in ttimes:
            args.append(tcurrents[ttimes.index(t)])
        else:
            # interpolation to find the current
            if t >= ttimes[-1]:
                phi = tcurrents[-1]
            else:
                indx = np.where(ttimes > t)[0][0]
                phi = tcurrents[indx-1] + (tcurrents[indx] - tcurrents[indx-1]) / \
                      (ttimes[indx] - ttimes[indx-1]) * (t - ttimes[indx-1])

    elif phi_func.__name__ == "phi_enz":
        args = [evals["FHNearestNeighbor"][-1] if fh else evals["GrapheneNearestNeighbor"][-1]]
        args.extend([model_params["kappa"], model_params["scale"]])

    else:
        raise Exception("Invalid phi function: {}".format(phi_func.__name__))

    return phi_func(model_params["p"], *args)

def phi_tl(p, time):
    """
    Calculate transform limited phi at time
    Params:
        p - an instance of Parameters
        time - current time
    """
    return (p.a * p.strength / p.field) * (np.sin(p.field * time / (2*p.cycles))**2) * np.sin(p.field * time)

def phi_tracking(p, nnop, psi, target_current):
    """
    Calculates phi(time) for some current expectation we would like to track
    Params:
        p - an instance of Parameters
        expec - the nearest neighbor expectation
        target_current - the current we would like to track
    """
    expec = nnop.expectation_value(psi)
    r = np.abs(expec)
    theta = np.angle(expec)
    return np.arcsin(-target_current / (2 * p.a * p.t0 * r)) + theta

def phi_enz(p, nnop, psi, kappa, scale):
    """
    Calculates phi for an enz material
    Params:
        p - an instance of Parameters
        expec - the nearest neighbor expectation
        kappa - multiplicative scaling factor
        scale - addititve scaling factor
    """
    expec = nnop.expectation_value(psi)
    r = np.abs(expec)
    theta = np.angle(expec)
    scale = scale / r
    y = kappa / (2 * p.a* p.t0 *r)
    # when this function is 0, induced current = phi
    f = lambda phi: np.sin(phi - theta) + y * phi + scale
    res = root_scalar(f, bracket=[-1/y - scale/y, 1/y - scale/y])
    if not res.converged:
        raise Exception("Could not find zero at time {}".format(time))
    return res.root

def relative_error(exact, mps):
    return 100 * np.linalg.norm(exact - mps) / np.linalg.norm(exact)

def relative_error_interp(exact, exactts, mps, mpsts):
    assert len(exact) == len(exactts)
    assert len(mps) == len(mpsts)
    if len(exact) > len(mps):
        bigarr = exact
        bigarrt = exactts
        litarr = mps
        litarrt = mpsts
    elif len(exact) < len(mps):
        bigarr = mps
        bigarrt = mpsts
        litarr = exact
        litarrt = exactts
    else:
        return relative_error(exact, mps)

    diff = np.zeros(len(litarr))
    for i in range(len(litarr)):
        # extrapolation
        if litarrt[i] > bigarrt[-1]:
            intcurr = bigarr[-1] + (bigarr[-1] - bigarr[-2]) / (bigarrt[-1] - bigarrt[-2]) * \
                                        (litarrt[i] - bigarrt[-1])
        else:
            indx = np.where(bigarrt > litarrt[i])[0][0]
            intcurr = bigarr[indx-1] + (bigarr[indx] - bigarr[indx-1]) / (bigarrt[indx] - bigarrt[indx-1]) * \
                                        (litarrt[i] - bigarrt[indx-1])
        diff[i] = litarr[i] - intcurr
    return 100 * np.linalg.norm(diff) / np.linalg.norm(exact)



# calculate difference between two MPS
def difference(a, b):
    return 1 - abs(a.overlap(b))

def spectrum(current, delta):
    """
    Gets power spectrum of the current
    :param current: the induced current in the lattice
    :param delta: time step between current points
    :return: the power spectrum of the current
    """
    at = np.gradient(current, delta)
    return spectrum_welch(at, delta)

def spectrum_welch(at, delta):
    return signal.welch(at, 1. / delta, nperseg=len(at), scaling='spectrum')
