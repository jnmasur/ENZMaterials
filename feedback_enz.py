from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from evolve import *
from matplotlib import pyplot as plt
import sys
from scipy.integrate import DOP853

def run(nsites, uot, a, ind, kp):
    # default parameters
    if nsites is None:
        nsites = 10
    if uot is None:
        uot = 2.
    if a is None:
        a = 4
    if ind is None:
        ind = 0.5
    if kp is None:
        kp = 10000

    # system setup
    N_up = nsites // 2 + nsites % 2  # number of fermions with spin up
    N_down = nsites // 2  # number of fermions with spin down
    N = N_up + N_down  # number of particles
    t0 = 0.52  # hopping strength
    U = uot * t0  # interaction strength
    pbc = True

    field = 32.9  # field angular frequency THz
    F0 = 10.  # Field amplitude MV/cm
    cycles = 10  # time in cycles of field frequency for tl pulse

    lat = Parameters(nsites, U, t0, a, cycles, field, F0, pbc)

    # start and stop time
    start = 0
    stop = 2 * np.pi * cycles / lat.field

    parameters = f'-nsites{nsites}-U{U/t0}-ind{ind}-kp{kp}-F{F0}-a{a}'
    # create basis
    basis = spinful_fermion_basis_1d(nsites, Nf=(N_up, N_down), sblock=1, kblock=1)

    # static part of the hamiltonian
    int_list = [[1.0, i, i] for i in range(nsites)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

    # dynamic part of the hamiltonian (hopping terms)
    hop = [[1.0, i, i+1] for i in range(nsites-1)]
    if lat.pbc:
        hop.append([1.0, nsites-1, 0])
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    # complete hamiltonian
    H = -lat.t0 * (hop_left + hop_right) + lat.u * onsite

    # get the ground state of the systems
    # E, psi_0 = H.eigsh(k=1, which='SA')
    # psi_0 = np.squeeze(psi_0)
    # psi_0 = psi_0 / np.linalg.norm(psi_0)

    # load state excited by pump pulse
    psi_0 = np.load("./Data/Exact/ENZ/psi0-nsites{}-U{}-F{}-a{}.npy".format(nsites, uot, F0, a))

    # current and energy expectation values
    def current_expec(phi, psi):
        Jpsi = -1j*lat.a*lat.t0* (np.exp(-1j*phi) * hop_left.static.dot(psi) - np.exp(1j*phi) * hop_right.static.dot(psi))
        return np.vdot(psi, Jpsi).real

    def energy_expec(phi, psi):
        Hpsi = -lat.t0 * (np.exp(-1j*phi) * hop_left.static.dot(psi) + np.exp(1j*phi) * hop_right.static.dot(psi)) + \
            lat.u * onsite.static.dot(psi)
        return np.vdot(psi, Hpsi).real

    # for keeping track of observables
    tsteps = [0.]
    phis = [0.]
    currents = [current_expec(0., psi_0)]
    energies = [energy_expec(0., psi_0)]

    # returns dpsi/dt in atomic units
    def evolve_feedback(t, psi):
        phi = feedback_phi(t)
        Hpsi = -lat.t0 * (np.exp(-1j*phi) * hop_left.static.dot(psi) + np.exp(1j*phi) * hop_right.static.dot(psi)) + \
            lat.u * onsite.static.dot(psi)
        return -1j * Hpsi

    # calculates value of phi
    def feedback_phi(t):
        phitl = phi_tl(lat, t)
        if t in tsteps:
            J = currents[tsteps.index(t)]
        elif t >= tsteps[-1]:
            J = currents[-1]
        # interpolate to find current
        else:
            indx = np.where(tsteps > t)[0][0]
            J = currents[indx-1] + (currents[indx] - currents[indx-1]) / \
                (tsteps[indx] - tsteps[indx-1]) * (t - tsteps[indx-1])

        return phitl + ((kp / (1 + kp)) * ((- lat.a * ind * (J - currents[0])) - phitl))

    # solve the schrodinger equation
    solver = DOP853(evolve_feedback, start, psi_0, stop)
    while solver.status == "running":
        solver.step()
        if solver.status == "failed":
            print("FAILURE IN DOP853")
        input_phi = feedback_phi(solver.t)
        tsteps.append(solver.t)
        phis.append(input_phi)
        currents.append(current_expec(input_phi, solver.y))
        energies.append(energy_expec(input_phi, solver.y))

    plt.plot(tsteps, np.array(currents), color="blue", label="$J(t)$")
    plt.plot(tsteps, np.array(phis) / (-lat.a * ind) + currents[0], ls="dashed", color="orange", label="$ -\\frac{\\Phi(t)}{a\\mathfrak{L}} + J(0)$")
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.legend()
    plt.savefig("./Data/Images/ENZ/FeedbackENZplot" + parameters + ".pdf")
    plt.show()

    np.save('./Data/Exact/FeedbackENZ/energies'+parameters+'.npy', energies)
    np.save('./Data/Exact/FeedbackENZ/currents'+parameters+'.npy', currents)
    np.save('./Data/Exact/FeedbackENZ/phis'+parameters+'.npy', phis)
    np.save("./Data/Exact/FeedbackENZ/times" + parameters + ".npy", tsteps)


if __name__ == "__main__":
    UOT = N = KP = IND = A = None
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == "--U":
            UOT = float(sys.argv[i + 1])
        elif sys.argv[i] == "--N":
            N = int(sys.argv[i + 1])
        elif sys.argv[i] == "--L":
            IND = float(sys.argv[i + 1])
        elif sys.argv[i] == "--a":
            A = int(sys.argv[i + 1])
        elif sys.argv[i] == "--kp":
            KP = float(sys.argv[i + 1])
        else:
            print("Unrecognized argument: {}".format(sys.argv[i]))

    run(N, UOT, A, IND, KP)
