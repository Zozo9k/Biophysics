from collections import deque
from functools import reduce

import numpy as np
from scipy import sparse
from scipy.linalg import eig

sigma_p = sparse.csr_matrix([[0, 0], [1, 0]])
sigma_m = sparse.csr_matrix([[0, 1], [0, 0]])
ide = sparse.identity


def lower_op(M: int) -> sparse.csr_matrix:
    return sparse.diags([[0,] * M, [0,] * (M + 1), [i + 1 for i in range(M)]], [-1, 0, 1])


def raise_op(M: int) -> sparse.csr_matrix:
    return sparse.diags([[1,] * M, [0,] * (M + 1), [0,] * M], [-1, 0, 1])


def state(M: int, occupied_num: int) -> sparse.csr_matrix:
    """Returns a state vector with specified number of occupating chemicals.

    Parameters
    ----------
        M (int):
            Maximum occupation number of each chemical in each voxel.
        occupied_num (int):
            Number of occupying chemicals.

    Returns
    -------
        state (sparse.csr_matrix):
            State vector with specified number of occupating chemicals.

    Examples
    --------
        >>> state(4, 1).toarray()
        array([[0.],
               [1.],
               [0.],
               [0.]])
    """
    return sparse.csr_matrix(([1], ([occupied_num], [0])), shape=(M + 1, 1))


def sparse_kron(tensor) -> sparse.csr_matrix:
    """returns a tensor product of all the term in the right order
     e.g tensor=[a,b,c] returns kron(a, kron(b,c)) """
    return reduce(sparse.kron, tensor)


def NN_ground_states_exact_diag(H: sparse.csr_matrix, tolerance=1e-8) -> list[float]:
    """Returns the list of ground states with nonnegative coefficients values,
      or with some negative coefs if there is no nonneg."""
    H = H.toarray()
    GS_list = []
    # H = np.round(H, decimals = int(-np.log10(tolerance)))
    vals, vecs = eig(H)
    vals = np.sort(vals)
    for i in range(len(vals)):
        E = vals[i].real
        if np.round(E, decimals=int(-np.log10(tolerance))) == 0:
            GS_list.append([np.round(vecs[:, i].real, decimals=int(-np.log10(tolerance))) + 0.0, E])

    indices = set()
    for k in range(len(GS_list)):
        neg_count = 0
        zeros = 0
        for i in range(len(GS_list[k][0])):
            GS_list[k][0][i] += 0.0
            if GS_list[k][0][i] == 0.0:
                zeros += 1
            elif GS_list[k][0][i] < 0.0:
                neg_count += 1
                indices.add(k)
            if neg_count + zeros == len(GS_list[k][0]):
                # if only negative elements, vec is positive up to a phase
                GS_list[k][0] = [x * (-1) + 0.0 for x in GS_list[k][0]]

    NN_GS_list = deque(GS_list)
    NN_GS_list = list(deque([value for index, value in enumerate(GS_list) if index not in indices]))
    if NN_GS_list:
        GS_list = NN_GS_list
        print('We found some non negative coefs GS ! :)')
    elif GS_list:
        print('We found some negative coefs GS ! :( ')

    return GS_list


def power_method(
    lambd: float, u0: sparse.csr_matrix, H: sparse.csr_matrix, tolerance=1e-8, iter=100000
) -> sparse.csr_matrix:
    """Returns one ground state of H iff the initial vector u0 is non-orthogonal to the GS"""
    dim = H.diagonal().shape[0]
    up = (lambd * ide(dim) - H) @ u0
    up = up / np.sum(up)
    for _ in range(iter):
        if (_ % 10000) == 0:
            print('Downloading...', int(_/iter * 100),'%')
        new_up = (lambd * ide(dim) - H) @ up
        if np.allclose(np.asarray(new_up - up).reshape(-1), [0] * dim, atol = tolerance):
            print('Stopped before final number of iterations')
            break
        else:
            up = new_up
            up = up / np.sum(up)
    if np.allclose(np.asarray(H @ up).reshape(-1), [0] * dim, atol = tolerance):
        print('This is a ground state')
    else:
        print('This doesn\'t reach the ground state')

    return up.real