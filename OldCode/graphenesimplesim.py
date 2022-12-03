from __future__ import print_function, division
import os
import sys
"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 16
os.environ['NUMEXPR_MAX_THREADS']='{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS']='{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_general  # Hilbert space basis
from quspin.basis import spinful_fermion_basis_1d # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
from quspin.tools.evolution import evolve # ODE evolve tool
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt  # plotting library
from tqdm import tqdm
sys.path.append('../')
# from tools import parameter_instantiate_2D as hhg  # Used for scaling units.
import psutil
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
import networkx as nx # networkx package, see https://networkx.github.io/documentation/stable/

class hhg:
    """
    Scales parameters to atomic units in terms of t_0.
    input units: eV (t, U)
    """

    def __init__(self, nsites, u, t, a, cycles, field, strength, pbc, nx=None, ny=None):
        self.nsites = nsites
        self.nx = nx
        self.ny = ny
        self.nup = nsites // 2 + nsites % 2
        self.ndown = nsites // 2

        self.u = u / t
        self.t0 = 1.

        self.cycles = cycles

        # CONVERTING TO ATOMIC UNITS, w/ energy normalized to t_0
        factor = 1 / (t * 0.036749323)
        self.field = field * factor * 0.0001519828442
        self.a = a * 1.889726125/factor
        self.strength = strength * 1.944689151e-4 * (factor**2)

        self.pbc = pbc #periodic boundary conditions

"""system params"""


# """hopping in e.v."""
t0=2.7
"""lattice constant in angstrom"""
a=2.5
"""values for hBN in e.v.- graphene has no onsite potential."""
U_A=3.3
# U_A=-1.4
U_B=-1.4
# U_A=10
# # U_A=-1.4
# U_B=10
"""laser pulse params for kick"""
field = 32.9  # field angular frequency THz
F0 = 10 # Field amplitude MV/cm
field_angle=0
cycles = 1  # time in cycles of field frequency
n_steps = 5000

# """misha paper settings"""
# wl=3*10**(-6)
# light=2.99792458*10**8
# freq=light/wl
# field=2*np.pi*freq*10**(-12)
# # print(cycles)
# F0=40
# # field_angle=np.pi/4
# """hopping in e.v."""
# t0=2.92
# """lattice constant in angstrom"""
# a=1.5
# """values for hBN in e.v.- graphene has no onsite potential."""
# U_A=2.81
# U_B=-2.81
# # U_A=10
# # # U_A=-1.4
# # U_B=10
#
#
# """System Evolution Time"""
# cycles = 10  # time in cycles of field frequency
# n_steps = 5000
# field_angle=np.pi/2



###### create honeycomb lattice
# lattice graph parameters
m =3
# number of rows of hexagons in the lattice
n = 2  # number of columns of hexagons in the lattice
# isPBC = False # if True, use periodic boundary conditions
isPBC = True # if True, use periodic boundary conditions


"""build graph using networkx"""
hex_graph = nx.generators.lattice.hexagonal_lattice_graph(m, n, periodic=isPBC)
# label graph nodes by consecutive integers
hex_graph = nx.convert_node_labels_to_integers(hex_graph)
# set number of lattice sites
N = hex_graph.number_of_nodes()
G=hex_graph
print('constructed hexagonal lattice with {0:d} sites.\n'.format(N))

"""use this to parametrise rest of QuSpin system"""
L = N# system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
pbc=False

"""scale units to e.v'"""
lat = hhg(N, 0., t0, a, cycles, field, F0, pbc)
# lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U_A=U_A,U_B=U_B, t=t0, F0=F0, a=a, pbc=pbc)
"""Define e^i*phi for later dynamics. Important point here is that for later implementations of tracking, we
will pass phi as a global variable that will be modified as appropriate during evolution"""

start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)



"""This is used for setting up Hamiltonian in Quspin."""
dynamic_args = []




def phi(current_time):
    np.random.seed(current_time)
    phi = (lat.a * lat.F0 / lat.field) * (
            np.sin(lat.field * current_time / (2. * cycles)) ** 2. * np.sin(
        lat.field * current_time))
    return phi


"""set up parameters for saving expectations later"""
outfile_graphene = './Data/graphene-expectations:{}sites-{}up-{}down-{}cycles-{}steps-{}field_amplitude-{}field_angle.npz'.format(L, N_up, N_down,
                                                                                                cycles,
                                                                                               n_steps,F0,field_angle)

outfile_hbn = './Data/hbn-expectations:{}sites-{}up-{}down-{}cycles-{}steps-{}field_amplitude-{}field_angle.npz'.format(L, N_up, N_down,
                                                                                                cycles,
                                                                                               n_steps,F0,field_angle)


"""define Phi for each direction"""

def phi_x(current_time):
    phi = np.cos(field_angle)*(lat.a * lat.F0 / lat.field) * (
            np.sin(lat.field * current_time / (2. * cycles)) ** 2. * np.sin(
        lat.field * current_time))
    return phi
# def phi_x(current_time):
#     phi = np.cos(field_angle)*(lat.a * lat.F0 / lat.field) *np.exp(-0.01*(current_time-0.3*cycles/lat.freq)**2)
#     return phi

def phi_y(current_time):
    phi = np.sin(field_angle)*(lat.a * lat.F0 / lat.field) * (
            np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return phi
"""while it's natural to ultimately express the current and field in cartesian coordinates,
they should appear in the Hamiltonian in terms of the three nearest neighbour lattice directions."""


def phi_1(current_time):
    phi=phi_x(current_time)
    return phi

def phi_2(current_time):
    phi=0.5*phi_x(current_time)+0.5*np.sqrt(3)*phi_y(current_time)
    return phi

def phi_3(current_time):
    phi=0.5*phi_x(current_time)-0.5*np.sqrt(3)*phi_y(current_time)
    return phi


def expiphi_1(current_time):

    return np.exp(1j * phi_1(current_time))


def expiphiconj_1(current_time):

    return np.exp(-1j * phi_1(current_time))


def expiphi_2(current_time):

    return np.exp(1j * phi_2(current_time))


def expiphiconj_2(current_time):

    return np.exp(-1j * phi_2(current_time))


def expiphi_3(current_time):

    return np.exp(1j * phi_3(current_time))


def expiphiconj_3(current_time):

    return np.exp(-1j * phi_3(current_time))


"""Needlessly elaborate manner for assigning directions for field on lattice"""

for j in range(N):
    print(G.adj[j])

for e in G.edges():
        G[e[0]][e[1]]['color'] = 'black'

# for j in range(N):
#     redcount = 0
#     bluecount = 0
#     greencount = 0
#     for e in G.edges(j):
#         print(e[0])
#         print(e[1])
#         # print(G[e[0]][e[1]]['color'])
#         # G[e[0]][e[1]]['color'] = 'red'
#         # y= or G[e[1]][e[0]]['color']
#         if G[e[0]][e[1]]['color'] == 'red':
#             redcount+=1
#         if G[e[0]][e[1]]['color'] == 'blue':
#             bluecount+=1
#         if G[e[0]][e[1]]['color'] == 'green':
#             greencount+=1
#     for e in G.edges(j):
#         if redcount==0:
#             if G[e[0]][e[1]]['color'] !='green' and G[e[0]][e[1]]['color'] !='blue':
#                 G[e[0]][e[1]]['color'] = 'red'
#                 print(G[e[0]][e[1]]['color'])
#                 print(G[e[1]][e[0]]['color'])
#                 redcount+=1
#         elif bluecount==0:
#             if G[e[0]][e[1]]['color'] !='green' and G[e[0]][e[1]]['color'] !='red':
#                 bluecount+=1
#                 G[e[0]][e[1]]['color'] = 'blue'
#         elif greencount==0:
#             if G[e[0]][e[1]]['color'] !='red' and G[e[0]][e[1]]['color'] !='green':
#                 greencount+=1
#                 G[e[0]][e[1]]['color'] = 'green'

"""Ugly fix to ensure the periodic structure has the right field in the right direction"""
if m==2 and n==2:
    G[5][6]['color'] = 'green'
    G[1][2]['color'] = 'green'
    G[1][5]['color'] = 'blue'
    G[2][6]['color'] = 'blue'
#
# if m==3 and n==2:
#     G[0][6]['color'] = 'green'
#     G[5][11]['color'] = 'green'
#     G[0][5]['color'] = 'blue'
#     G[11][6]['color'] = 'blue'
edge_color_list = [ G[e[0]][e[1]]['color'] for e in G.edges() ]

hop_left_1=[]
hop_right_1=[]
hop_left_2=[]
hop_right_2=[]
hop_left_3=[]
hop_right_3=[]
onsite_A=[]
onsite_B=[]
finished_sites=[]
done_list_A=[]
done_list_B=[]


"""constructing site list from edge colours. The extra if statements for the hopping operators are to account for the
sites with the periodic boundary."""

"""this setup has been specifically constructed for m,n=2. The graph doesn't contain enough information to automate the
 proper classification of edges into one of the field directions. This isn't true in the case of non-pbc, but we're not
 doing that, so fuck it."""
# if m==2 and n==2:
#     for e in G.edges():
#         print(e[0])
#         print(e[1])
#         if m==2 and n==2:
#             if G[e[0]][e[1]]['color'] =='green':
#                 if e[0]==1 or e[1]==7:
#                     hop_right_2.append([-lat.t, e[0], e[1]])
#                     hop_left_2.append([lat.t, e[0], e[1]])
#                 else:
#                     hop_left_2.append([-lat.t,e[0],e[1]])
#                     hop_right_2.append([lat.t,e[0],e[1]])
#             if G[e[0]][e[1]]['color'] =='blue':
#                 if e[0] == 1 or e[1] == 7:
#                     hop_right_1.append([-lat.t, e[1], e[0]])
#                     hop_left_1.append([lat.t, e[1], e[0]])
#                 else:
#                     hop_right_1.append([-lat.t, e[0], e[1]])
#                     hop_left_1.append([lat.t, e[0], e[1]])
#             if G[e[0]][e[1]]['color'] =='red':
#                 if e[0] == 1 or e[1] == 7:
#                     hop_right_3.append([-lat.t, e[0], e[1]])
#                     hop_left_3.append([lat.t, e[0], e[1]])
#                 else:
#                     hop_left_3.append([-lat.t, e[0], e[1]])
#                     hop_right_3.append([lat.t, e[0], e[1]])
#         if e[0] in done_list_A:
#             if e[1] not in done_list_B:
#                 onsite_B.append([lat.U_B, e[1], e[1]])
#                 done_list_B.append(e[1])
#         elif e[0] in done_list_B:
#             if e[1] not in done_list_A:
#                 onsite_A.append([lat.U_A, e[1], e[1]])
#                 done_list_A.append(e[1])
#         else:
#             onsite_A.append([lat.U_A, e[0], e[0]])
#             onsite_B.append([lat.U_B, e[1], e[1]])
#             done_list_A.append(e[0])
#             done_list_B.append(e[1])

for e in G.edges():
    print(e[0])
    print(e[1])
    if e[0] in done_list_A:
        if e[1] not in done_list_B:
            onsite_B.append([lat.U_B, e[1], e[1]])
            done_list_B.append(e[1])
    elif e[0] in done_list_B:
        if e[1] not in done_list_A:
            onsite_A.append([lat.U_A, e[1], e[1]])
            done_list_A.append(e[1])
    else:
        onsite_A.append([lat.U_A, e[0], e[0]])
        onsite_B.append([lat.U_B, e[1], e[1]])
        done_list_A.append(e[0])
        done_list_B.append(e[1])

if m==2 and n==2:
    hop_right_1.append([-lat.t,7,3])
    hop_left_1.append([lat.t, 7, 3])
    hop_right_2.append([-lat.t,6,7])
    hop_left_2.append([lat.t,6,7])
    hop_right_3.append([-lat.t,4,7])
    hop_left_3.append([lat.t, 4, 7])

    hop_right_1.append([-lat.t, 0, 4])
    hop_left_1.append([lat.t, 0, 4])
    hop_right_2.append([-lat.t, 1, 0])
    hop_left_2.append([lat.t, 1, 0])
    hop_right_3.append([-lat.t, 3, 0])
    hop_left_3.append([lat.t, 3, 0])

    hop_right_2.append([-lat.t, 4, 5])
    hop_left_2.append([lat.t, 4, 5])

    hop_right_1.append([-lat.t, 5, 1])
    hop_left_1.append([lat.t, 5, 1])
    hop_right_3.append([-lat.t, 6, 5])
    hop_left_3.append([lat.t, 6, 5])

    hop_right_1.append([-lat.t, 2, 6])
    hop_left_1.append([lat.t, 2, 6])

    hop_right_2.append([-lat.t, 3, 2])
    hop_left_2.append([lat.t, 3, 2])
    hop_right_3.append([-lat.t, 1, 2])
    hop_left_3.append([lat.t, 1, 2])

    if m == 3 and n == 2:
        hop_right_1.append([-lat.t, 11, 10])
        hop_left_1.append([lat.t, 11, 10])
        hop_right_1.append([-lat.t, 4, 5])
        hop_left_1.append([lat.t, 4, 5])
        hop_right_1.append([-lat.t, 0, 6])
        hop_left_1.append([lat.t, 0, 6])
        hop_right_1.append([-lat.t, 7, 8])
        hop_left_1.append([lat.t, 7, 8])
        hop_right_1.append([-lat.t, 2, 1])
        hop_left_1.append([lat.t, 2, 1])
        hop_right_1.append([-lat.t, 9, 3])
        hop_left_1.append([lat.t, 9, 3])

        hop_right_2.append([-lat.t, 5, 11])
        hop_left_2.append([lat.t, 5, 11])
        hop_right_2.append([-lat.t, 1, 0])
        hop_left_2.append([lat.t, 1, 0])
        hop_right_2.append([-lat.t, 6, 7])
        hop_left_2.append([lat.t, 6, 7])
        hop_right_2.append([-lat.t, 8, 2])
        hop_left_2.append([lat.t, 8, 2])
        hop_right_2.append([-lat.t, 10, 9])
        hop_left_2.append([lat.t, 10, 9])
        hop_right_2.append([-lat.t, 3, 4])
        hop_left_2.append([lat.t, 3, 4])

        hop_right_3.append([-lat.t, 10, 4])
        hop_left_3.append([lat.t, 10, 4])
        hop_right_3.append([-lat.t, 6, 11])
        hop_left_3.append([lat.t, 6, 11])
        hop_right_3.append([-lat.t, 5, 0])
        hop_left_3.append([lat.t, 5, 0])
        hop_right_3.append([-lat.t, 1, 7])
        hop_left_3.append([lat.t, 1, 7])
        hop_right_3.append([-lat.t, 8, 9])
        hop_left_3.append([lat.t, 8, 9])
        hop_right_3.append([-lat.t, 3, 2])
        hop_left_3.append([lat.t, 3, 2])

"""for m=3, it's much easier to just code up the site list manually, using networkx's graph as a reference."""
###ADD THIS LATER.


print('left_hoperators first direction')
print(hop_left_1)
print('right_hoperators first direction')
print(hop_right_1)
print('left_hoperators second direction')
print(hop_left_2)
print('right_hoperators second direction')
print(hop_right_2)
print('left_hoperators third direction')
print(hop_left_3)
print('right_hoperators third direction')
print(hop_right_3)
print(onsite_A)
print(onsite_B)

"""graphing the lattice structure. Don't bother unless you're trying to work out the proper coupling list."""
# edge_color_list = ['black']
pos = nx.spring_layout(G, seed=42, iterations=100)
# plt.subplot(211)
nx.draw(G,edge_color=edge_color_list, pos=pos, with_labels=True)
# plt.subplot(212)
# nx.draw(hex_graphpbc,edge_color=edge_color_list, pos=pos, with_labels=True)
plt.show()


"""finally able to setup QuSpin!"""
# basis = spinful_fermion_basis_general(N, Nf=(N_up, N_down))
basis=spinful_fermion_basis_1d(L=L,Nf=(N_up,N_down),a=2,sblock=1,kblock=0)
print('Hilbert space size: {0:d}.\n'.format(basis.Ns))
int_list=onsite_A+onsite_B
print(int_list)

# create static lists
# Note that the pipe determines the spinfulness of the operator. | on the left corresponds to down spin, | on the right
# is for up spin. For the onsite interaction here, we have:

"""This term only required for graphene!"""
static_Hamiltonian_list = [
    ["n|n", int_list],  # onsite interaction
]

dynamic_args = []
# After creating the site lists, we attach an operator and a time-dependent function to them
"""three directions and two spin species makes this a bloody long list!"""
dynamic_Hamiltonian_list = [
    ["+-|", hop_left_1, expiphiconj_1, dynamic_args],  # up hop left
    ["-+|", hop_right_1, expiphi_1, dynamic_args],  # up hop right
    ["|+-", hop_left_1, expiphiconj_1, dynamic_args],  # down hop left
    ["|-+", hop_right_1, expiphi_1, dynamic_args],  # down hop right
    ["+-|", hop_left_2, expiphiconj_2, dynamic_args],  # up hop left
    ["-+|", hop_right_2, expiphi_2, dynamic_args],  # up hop right
    ["|+-", hop_left_2, expiphiconj_2, dynamic_args],  # down hop left
    ["|-+", hop_right_2, expiphi_2, dynamic_args],  # down hop right
    ["+-|", hop_left_3, expiphiconj_3, dynamic_args],  # up hop left
    ["-+|", hop_right_3, expiphi_3, dynamic_args],  # up hop right
    ["|+-", hop_left_3, expiphiconj_3, dynamic_args],  # down hop left
    ["|-+", hop_right_3, expiphi_3, dynamic_args],  # down hop right
]



ham_graphene = hamiltonian([], dynamic_Hamiltonian_list, basis=basis)
# ham_graphene = hamiltonian(static_Hamiltonian_list, dynamic_Hamiltonian_list, basis=basis)
# ham_hbn = hamiltonian(static_Hamiltonian_list, dynamic_Hamiltonian_list, basis=basis,check_symm=False)
"""build up the other operator expectations here as a dictionary"""

"""note that graphene and hbn probably _don't_ need separate operator dictionaries, but I don't see the harm in
keeping a cordon sanitaire between them. """
operator_dict_graphene = dict(H=ham_graphene)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# hopping operators for building current. Note that the easiest way to build an operator is just to cast it as an
# instance of the Hamiltonian class. Note in this instance the hops up and down have the e^iphi factor attached directly
operator_dict_graphene['neighbour_1']= hamiltonian([["+-|", hop_left_1],["|+-", hop_left_1]],[], basis=basis, **no_checks)
operator_dict_graphene['neighbour_2']= hamiltonian([["+-|", hop_left_2],["|+-", hop_left_2]],[], basis=basis, **no_checks)
operator_dict_graphene['neighbour_3']= hamiltonian([["+-|", hop_left_3],["|+-", hop_left_3]],[], basis=basis, **no_checks)
operator_dict_graphene["lhopup_1"] = hamiltonian([], [["+-|", hop_left_1, expiphiconj_1, dynamic_args]], basis=basis, **no_checks)
operator_dict_graphene["lhopup_2"] = hamiltonian([], [["+-|", hop_left_2, expiphiconj_2, dynamic_args]], basis=basis, **no_checks)
operator_dict_graphene["lhopup_3"] = hamiltonian([], [["+-|", hop_left_3, expiphiconj_3, dynamic_args]], basis=basis, **no_checks)
operator_dict_graphene["lhopdown_1"] = hamiltonian([], [["|+-", hop_left_1, expiphiconj_1, dynamic_args]], basis=basis, **no_checks)
operator_dict_graphene["lhopdown_2"] = hamiltonian([], [["|+-", hop_left_2, expiphiconj_2, dynamic_args]], basis=basis, **no_checks)
operator_dict_graphene["lhopdown_3"] = hamiltonian([], [["|+-", hop_left_3, expiphiconj_3, dynamic_args]], basis=basis, **no_checks)
for j in range(L):
    # spin up densities for each site
    operator_dict_graphene["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    operator_dict_graphene["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # doublon densities
    operator_dict_graphene["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)


operator_dict_hbn = dict(H=ham_hbn)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# hopping operators for building current. Note that the easiest way to build an operator is just to cast it as an
# instance of the Hamiltonian class. Note in this instance the hops up and down have the e^iphi factor attached directly
operator_dict_hbn['neighbour_1']= hamiltonian([["+-|", hop_left_1],["|+-", hop_left_1]],[], basis=basis, **no_checks)
operator_dict_hbn['neighbour_2']= hamiltonian([["+-|", hop_left_2],["|+-", hop_left_2]],[], basis=basis, **no_checks)
operator_dict_hbn['neighbour_3']= hamiltonian([["+-|", hop_left_3],["|+-", hop_left_3]],[], basis=basis, **no_checks)
operator_dict_hbn["lhopup_1"] = hamiltonian([], [["+-|", hop_left_1, expiphiconj_1, dynamic_args]], basis=basis, **no_checks)
operator_dict_hbn["lhopup_2"] = hamiltonian([], [["+-|", hop_left_2, expiphiconj_2, dynamic_args]], basis=basis, **no_checks)
operator_dict_hbn["lhopup_3"] = hamiltonian([], [["+-|", hop_left_3, expiphiconj_3, dynamic_args]], basis=basis, **no_checks)
operator_dict_hbn["lhopdown_1"] = hamiltonian([], [["|+-", hop_left_1, expiphiconj_1, dynamic_args]], basis=basis, **no_checks)
operator_dict_hbn["lhopdown_2"] = hamiltonian([], [["|+-", hop_left_2, expiphiconj_2, dynamic_args]], basis=basis, **no_checks)
operator_dict_hbn["lhopdown_3"] = hamiltonian([], [["|+-", hop_left_3, expiphiconj_3, dynamic_args]], basis=basis, **no_checks)
for j in range(L):
    # spin up densities for each site
    operator_dict_hbn["nup" + str(j)] = hamiltonian([["n|", [[1.0, j]]]], [], basis=basis, **no_checks)
    # spin down
    operator_dict_hbn["ndown" + str(j)] = hamiltonian([["|n", [[1.0, j]]]], [], basis=basis, **no_checks)
    # doublon densities
    operator_dict_hbn["D" + str(j)] = hamiltonian([["n|n", [[1.0, j, j]]]], [], basis=basis, **no_checks)

"""build ground state for graphene"""
print("calculating ground state for graphene")
E_graphene, psi_0_graphene = ham_graphene.eigsh(k=1, which='SA')
E_graphene=E_graphene[0]
print(E_graphene)
# E_gs, psi = ham_graphene.eigsh(which='SA')
# E_graphene=E_gs[-1]
# psi_0_graphene=psi[:,-1]
print(psi_0_graphene.shape)
print('normalisation')
# psi_0=psi_0[:,2]
print(np.linalg.norm(psi_0_graphene))
psi_0=psi_0_graphene/np.linalg.norm(psi_0_graphene)
print("ground state for graphene calculated, energy is {:.2f}".format(E_graphene))
# psi_0.reshape((-1,))
# psi_0=psi_0.flatten
print('evolving system')
ti = time()
"""evolving system. In this simple case we'll just use the built in solver"""
# this version returns the generator for psi
# psi_t=ham.evolve(psi_0,0.0,times,iterate=True)

# this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
# obs_vs_time to work properly
psi_t = ham_graphene.evolve(psi_0, 0.0, times,verbose=True)
psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
# calculate the expectations for every bastard in the operator dictionary
ti = time()
# note that here the expectations
expectations_graphene = obs_vs_time(psi_t, times, operator_dict_graphene)
print(type(expectations_graphene))
current_partial_1 = (expectations_graphene['lhopup_1'] + expectations_graphene['lhopdown_1'])
current_partial_2 = (expectations_graphene['lhopup_2'] + expectations_graphene['lhopdown_2'])
current_partial_3 = (expectations_graphene['lhopup_3'] + expectations_graphene['lhopdown_3'])

current_1 = -1j * lat.a * (current_partial_1 - current_partial_1.conjugate())
current_2 = -1j * lat.a * (current_partial_2 - current_partial_2.conjugate())
current_3 = -1j * lat.a * (current_partial_3 - current_partial_3.conjugate())
expectations_graphene['current_1'] = current_1
expectations_graphene['current_2'] = current_2
expectations_graphene['current_3'] = current_3
expectations_graphene['phi_1']=phi_1(times)
expectations_graphene['phi_2']=phi_2(times)
expectations_graphene['phi_3']=phi_3(times)
print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

print("Saving Expectations. We have {} of them".format(len(expectations_graphene)))
np.savez(outfile_graphene, **expectations_graphene)

print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))



# print("calculating ground state for hbn")
# E_hbn, psi_0_hbn = ham_hbn.eigsh(k=1, which='SA')
# E_hbn=E_hbn[0]
# print(E_hbn)
# print(psi_0_hbn.shape)
# print('normalisation')
# # psi_0=psi_0[:,2]
# print(np.linalg.norm(psi_0_hbn))
# psi_0=psi_0_hbn/np.linalg.norm(psi_0_hbn)
# print("ground state for hbn calculated, energy is {:.2f}".format(E_hbn))
# # psi_0.reshape((-1,))
# # psi_0=psi_0.flatten
# print('evolving system')
# ti = time()
# """evolving system. In this simple case we'll just use the built in solver"""
# # this version returns the generator for psi
# # psi_t=ham.evolve(psi_0,0.0,times,iterate=True)
#
# # this version returns psi directly, last dimension contains time dynamics. The squeeze is necessary for the
# # obs_vs_time to work properly
# psi_t = ham_hbn.evolve(psi_0, 0.0, times,verbose=True)
# psi_t = np.squeeze(psi_t)
# print("Evolution done! This one took {:.2f} seconds".format(time() - ti))
# # calculate the expectations for every bastard in the operator dictionary
# ti = time()
# # note that here the expectations
# expectations_hbn = obs_vs_time(psi_t, times, operator_dict_hbn)
# print(type(expectations_hbn))
# current_partial_1 = (expectations_hbn['lhopup_1'] + expectations_hbn['lhopdown_1'])
# current_partial_2 = (expectations_hbn['lhopup_2'] + expectations_hbn['lhopdown_2'])
# current_partial_3 = (expectations_hbn['lhopup_3'] + expectations_hbn['lhopdown_3'])
#
# current_1 = -1j * lat.a * (current_partial_1 - current_partial_1.conjugate())
# current_2 = -1j * lat.a * (current_partial_2 - current_partial_2.conjugate())
# current_3 = -1j * lat.a * (current_partial_3 - current_partial_3.conjugate())
# expectations_hbn['current_1'] = current_1
# expectations_hbn['current_2'] = current_2
# expectations_hbn['current_3'] = current_3
# expectations_hbn['phi_1']=phi_1(times)
# expectations_hbn['phi_2']=phi_2(times)
# expectations_hbn['phi_3']=phi_3(times)
# print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))
#
# print("Saving Expectations. We have {} of them".format(len(expectations_hbn)))
# np.savez(outfile_hbn, **expectations_hbn)
#
# print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))
