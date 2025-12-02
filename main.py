import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

from generators import *
from plot import *
from src.utils import *
from src import diffusion_hamiltonian, reaction_hamiltonian, reduced_hilbert

M = 5
AC = 3# the total number of AC molecules that wll be conserved through evolution
num_of_chemicals = 5  # mandatory for cAMP reaction model
num_of_voxels = 1
voxel_x = 1  # from 1 to num_of_chemicals, determine for the plots
labels = ['Gi', 'Gs', 'AC', 'AC-Gi', 'AC-Gs'] #labels = ['Gi', 'Gs','AC', 'AC-Gi', 'AC-Gs']
k_l = [.05, .05, .1, .1, .01, .01, .1, .1]
reduced = True
bath = False

if __name__ == "__main__":

    # Initialize cAMP production system with y_A_i and y_bar_A_i
    if bath:
        cAMP_y, cAMP_y_bar = CRNStructureGenerator.cAMP_bath()
        reduced = False
    else:
        cAMP_y, cAMP_y_bar = CRNStructureGenerator.cAMP()

    H_cAMP = reaction_hamiltonian.ReactionHamiltonian.generate_general_reaction_hamiltonian(
        M, k_l, cAMP_y, cAMP_y_bar, num_of_voxels, num_of_chemicals, proj = True
    )
    H_diff = diffusion_hamiltonian.DiffusionHamiltonian.generate_diffusion_hamiltonian_by_threading(M, num_of_voxels, num_of_chemicals)
    print(H_cAMP.shape)

    #WARNING : the reduction can occur only with finite number of AC, not bath
    if reduced and not bath:
        H_reduced = reduced_hilbert.ReducedHilbertSpace.generate_reduced_matrix(H_cAMP + H_diff, AC, M, num_of_chemicals, num_of_voxels)
    else:
        H_reduced = H_cAMP
    print(H_reduced.shape)

    # In case the matrix is small enough that we can find the exact GS
    E, vecs = eig(H_reduced.toarray())
    print(np.sort(E)[:7].real)
    E_list = []
    for e in E:
        E_list.append(e.real)
    E_list = sorted(E_list)
    plt.plot(E_list)
    dim = dim = H_reduced.shape[0] #(M +1)**num_of_chemicals
    P_bra = np.ones((M + 1)**(num_of_chemicals*num_of_voxels))
    P_bra = P_bra[:dim]

    vec = []

    for i in range(len(E)):
        if P_bra @ (ide(dim) @ vecs[:, i]) != 0:
            vec2 = vecs[:, i] /(P_bra @ (ide(dim) @ vecs[:, i]))
        vec.append((vec2.real, E[i].real))
    vec = sorted(vec, key = lambda x: x[1])

    N = []
    for i in range(num_of_chemicals):
        n = reaction_hamiltonian.ReactionHamiltonian.number_operator(M, i, num_of_chemicals)
        n = sum(
                sparse_kron([ide((M + 1)**(voxel * num_of_chemicals)), n,
                            ide((M + 1)**((num_of_voxels - voxel - 1) * num_of_chemicals))])
                for voxel in range(num_of_voxels))
        if reduced:
            n = reduced_hilbert.ReducedHilbertSpace.generate_reduced_matrix(n, AC, M, num_of_chemicals, num_of_voxels)
        N.append(n)

    v0 = vec[0][0]

    E0 = P_bra @ (H_reduced @ v0)
    print('E0 = ', E0)
    nAC_v0 = 0
    for k in range(num_of_chemicals):
        if k == 2 or k==3 or k==4:
            nAC_v0 += P_bra @ (N[k] @ v0)
        print(labels[k], '=',P_bra @ (N[k] @ v0))
    print('Total number of AC: ',nAC_v0)

    omegas= np.linspace(0.001*np.pi, np.pi, 200)
    timesteps_number = 10000

    voxel_x = 1 # from 1 to num_of_chemicals, determine for the plots
    Floquet_voxels = [0] #first site
    labels = ['Gi', 'Gs','AC', 'AC-Gi', 'AC-Gs']
    W =min(omegas)#0.1#omegas[8]#*30
    A = 0.99
    phase =  - np.pi/2
    print('Frequency = ', W)

    # v0 = ReducedHilbertSpace.generate_reduced_psi(sparse_kron([state(M, 1), state(M, 1), state(M, 3), state(M,1), state(M, 1)]),
    #                                               AC, M, num_of_chemicals, num_of_voxels)

    v0 = np.asarray(v0).reshape(-1)

    delta_t = 1

    print(timesteps_number)
    before = time.time()

    diff = Plot.plt('one', M, H_reduced, v0, delta_t, num_of_voxels, num_of_chemicals, labels, AC,
        timesteps_number, voxel_x, k_l, Floquet_voxels, W, A, phase, RK = True, model = 'cAMP', reduced = reduced, proj = True)
    plt.savefig('poster_zero.svg')
    print('Computation time ', time.time() - before)