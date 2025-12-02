from threading import Thread

import numpy as np
from scipy import sparse

from src.utils import sparse_kron, ide, raise_op, lower_op


class DiffusionHamiltonian:
    r"""Create hamiltonian in the form of a matrix. This corresponds to a diffusion term.
    $$
        H_d =
            - \sum_{x, y \in \expval{x}; i}
            D_{x \to y; i}} \left[
                \sigma^+_i(y)\sigma^-_i(x)
                - \sigma^+_i(x)\sigma^-_i(x) (1 - \sigma^+_i(y)\sigma^-_i(y))
            \right]
    %%

    Hilbert space is spanned on the basis of the tensor product of the chemicals in each voxel.
    """
    DEBUG_ = False

    @staticmethod
    def generate_D(num_of_chemicals: int) -> list:
        """
        Parameters
        ----------
            num_of_chemicals (int, optional):
                Number of chemicals in each voxel.
                e.g.1) 1-chemical, i = 0.
                e.g.2) 2-chemical, i = 0, 1.
        Returns
        -------
            D (list):
                Diffusion coefficient matrix.
                D[i] contains the left and right (respectively) diffusion coefficients of chemical i.
        """
        D = sparse.lil_array((num_of_chemicals, 2))  # = (DL,DR)

        for i in range(num_of_chemicals):
            np.random.seed(10)
            random_value = 0#np.random.rand()
            print('Diffusion :', random_value)
            D[i, 0], D[i, 1] = random_value, 1 - random_value

        return D.tocsr()

    @staticmethod
    def operator(
        sigma: sparse.csr_matrix, M: int, x: int, i: int, num_of_voxels: int, num_of_chemicals: int
    ) -> sparse.csr_matrix:
        """Generate a single operator acting on the specified voxel and chemical.

        Parameters
        ----------
            sigma (sparse.csr_matrix):
                Operator to be applied.
            M (int):
                Maximum occupation number of each chemical in each voxel.
            x (int):
                Voxel index.
            i (int):
                Chemical index.
            num_of_voxels (int):
                Number of voxels.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
        """
        return sparse_kron([ide((M + 1)**(num_of_chemicals * x + i)), sigma,
                            ide((M + 1)**(num_of_chemicals * (num_of_voxels - x) - i - 1))])

    @classmethod
    def generate_single_diffusion_term(
        cls, M: int, x: int, i: int, num_of_voxels: int, num_of_chemicals: int
    ) -> tuple[sparse.csr_matrix]:
        r"""Generate a single diffusion term corresponding to the following term in the Hamiltonian.
        $$
            \sigma^+_{i,y}\sigma^-_{i,x} - (1 - \ket{M}_{i,y}\bra{M}_{i,y}) n_{i,x}
        $$

        Returns
        -------
            DL_term, DR_term (tuple(sparse.csr_matrix)):
                A single diffusion term without coefficient.
        """

        if cls.DEBUG_:
            assert x >= 0 and x < num_of_voxels and type(x) is int, \
                "x must be a non-negative integer less than num_of_voxels."
            assert i >= 0 and i < num_of_chemicals and type(i) is int, \
                "i must be a non-negative integer less than num_of_chemicals."
            assert num_of_voxels > 0 and type(num_of_voxels) is int, \
                "num_of_voxels must be a positive integer."
            assert num_of_chemicals > 0 and type(num_of_chemicals) is int, \
                "num_of_chemicals must be a positive integer."

        y = x + 1  # OBCs

        raise_x = cls.operator(raise_op(M), M, x, i, num_of_voxels, num_of_chemicals)
        lower_x = cls.operator(lower_op(M), M, x, i, num_of_voxels, num_of_chemicals)
        raise_y = cls.operator(raise_op(M), M, y, i, num_of_voxels, num_of_chemicals)
        lower_y = cls.operator(lower_op(M), M, y, i, num_of_voxels, num_of_chemicals)

        projM = sparse.csr_matrix((M + 1,) * 2)
        projM[M, M] = 1
        projector_i = ide(M + 1) - projM

        DL_term = raise_y @ lower_x - raise_x @ lower_x \
            @ cls.operator(projector_i, M, y, i, num_of_voxels, num_of_chemicals)

        DR_term = raise_x @ lower_y - raise_y @ lower_y \
            @ cls.operator(projector_i, M, x, i, num_of_voxels, num_of_chemicals)

        return DL_term, DR_term

    @classmethod
    def worker_for_threading(
            cls, M: int, x: int, i: int, num_of_voxels: int, num_of_chemicals: int, D: list, terms: list
    ) -> None:
        DL_term, DR_term = cls.generate_single_diffusion_term(
            M, x, i, num_of_voxels, num_of_chemicals)
        term = - D[i, 0] * DL_term - D[i, 1] * DR_term
        terms.append(term)

    @classmethod
    def generate_diffusion_hamiltonian_by_threading(
        cls, M: int, num_of_voxels: int, num_of_chemicals: int
    ) -> sparse.csr_matrix:
        D = cls.generate_D(num_of_chemicals)
        H_d = sparse.csr_matrix(((M + 1)**(num_of_voxels * num_of_chemicals),) * 2)

        terms = []
        threads = []
        for x in range(num_of_voxels):
            # NOTE: to avoid double counting when num_of_voxels is 2
            # TODO: For num_of_voxels > 2, if we want to involve PBCs, we need to modify this part.
            if x == num_of_voxels - 1:
                break
            for i in range(num_of_chemicals):
                thread = Thread(
                    target=cls.worker_for_threading,
                    args=(M, x, i, num_of_voxels, num_of_chemicals, D, terms),
                )
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        for term in terms:
            H_d += term

        return H_d
