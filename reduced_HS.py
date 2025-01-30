from scipy import sparse

from reaction_hamiltonian import ReactionHamiltonian
from utils import sparse_kron, ide


class ReducedHilbertSpace:

    @staticmethod
    def AC_number(M: int, num_of_chemicals: int, num_of_voxels: int) -> sparse.csr_matrix:
        """Returns the number operator obtaining the total number of AC molecules in all voxels

        Parameters
        ----------
            M (int):
                Maximum occupation number of each chemical in each voxel.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
            num_of_voxels (int):
                Number of voxels in the system.
        """
        total_AC = sum(ReactionHamiltonian.number_operator(M, i, num_of_chemicals) for i in [2, 3, 4])
        op = sum(
            sparse_kron([ide((M + 1)**(voxel * num_of_chemicals)), total_AC,
                         ide((M + 1)**((num_of_voxels - voxel - 1) * num_of_chemicals))])
            for voxel in range(num_of_voxels))
        return op

    @classmethod
    def reduced_HS_proj(cls, AC: int, M: int, num_of_chemicals: int, num_of_voxels: int) -> sparse.csr_matrix:
        """Returns the projector on the reduced Hilbert space invariant
        under conservation of a given number of AC molecules

        Parameters
        ----------
            AC (int):
                Conserved number of AC molecules.
            M (int):
                Maximum occupation number of each chemical in each voxel.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
            num_of_voxels (int):
                Number of voxels in the system.

        Returns
        -------
            red_proj (sparse.csr_matrix):
                Projector to the reduced Hilbert space corresponding to the given AC number.
                This projector is a diagonal matrix with 1 on the diagonal corresponding to the
                AC-conserving states and 0 otherwise.
        """
        N_ac = cls.AC_number(M, num_of_chemicals, num_of_voxels).diagonal()
        red_proj = sparse.lil_matrix(((M + 1)**(num_of_chemicals * num_of_voxels), ) * 2)

        for n in range(len(N_ac)):
            if N_ac[n] == AC:
                red_proj[n, n] = 1
        return red_proj.tocsr()

    @classmethod
    def reduced_HS_basis(
        cls, matrix: sparse.csr_matrix, AC: int, M: int, num_of_chemicals: int, num_of_voxels: int
    ) -> sparse.csr_matrix:
        """Returns the matrix in the reduced Hilbert space targeted by the squared projector.
        Hilbert space order of P is reordered to have the AC-conserving states in the first diagonal block.
        P_T is the transpose of the reordered P.

        Parameters
        ----------
            matrix (sparse.csr_matrix):
                Matrix to be reduced.
            AC (int):
                Conserved number of AC molecules.
            M (int):
                Maximum occupation number of each chemical in each voxel.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
            num_of_voxels (int):
                Number of voxels in the system.

        Returns
        -------
            new_matrix (sparse.csr_matrix):
                Matrix in the reduced Hilbert space.
            P_T (sparse.csr_matrix):
                Transpose of the projector to the reduced Hilbert space.
        """
        P = cls.reduced_HS_proj(AC, M, num_of_chemicals, num_of_voxels)
        # P_T will be the transposed matrix in the reduced basis
        P_T = sparse.lil_matrix(P.shape)
        indices = P.indices.tolist()

        dim = len(indices)

        P = P.tolil()
        for i in range(dim):
            if P[i, i] == 1:
                P_T[i, i] = 1
            else:
                j = indices.pop()
                # Here we permute column i with column j.
                # note that this is just a convention and we could
                # also permute the rows, but has to be consistent
                P[j, j], P[j, i] = P[j, i], P[j, j]

                P_T[i, j] = 1

        new_matrix = P_T @ matrix @ P
        return new_matrix[:dim, :dim], P_T

    @classmethod
    def generate_reduced_matrix(
        cls, matrix: sparse.csr_matrix, AC: int, M: int, num_of_chemicals: int, num_of_voxels: int
    ) -> sparse.csr_matrix:
        """Returns the matrix in the reduced Hilbert space.

        Parameters
        ----------
            matrix (sparse.csr_matrix):
                Matrix to be reduced.
            AC (int):
                Conserved number of AC molecules.
            M (int):
                Maximum occupation number of each chemical in each voxel.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
            num_of_voxels (int):
                Number of voxels in the system.

        Returns
        -------
            new_matrix (sparse.csr_matrix):
                Matrix in the reduced Hilbert space.
        """
        return cls.reduced_HS_basis(matrix, AC, M, num_of_chemicals, num_of_voxels)[0]

    @classmethod
    def generate_reduced_psi(
            cls, psi_init: sparse.csr_matrix, AC: int, M: int, num_of_chemicals: int, num_of_voxels: int
    ) -> sparse.csr_matrix:
        """Returns the reduced psi.

        Parameters
        ----------
            psi_init (sparse.csr_matrix):
                Initial state.
            AC (int):
                Conserved number of AC molecules.
            M (int):
                Maximum occupation number of each chemical in each voxel.
            num_of_chemicals (int):
                Number of chemicals in each voxel.
            num_of_voxels (int):
                Number of voxels in the system.

        Returns
        -------
            reduced_psi (sparse.csr_matrix):
                Reduced psi.
        """
        P = cls.reduced_HS_proj(AC, M, num_of_chemicals, num_of_voxels)
        dim = len(P.indices)
        P_T = cls.reduced_HS_basis(ide(psi_init.shape[0]), AC, M, num_of_chemicals, num_of_voxels)[1]
        reduced_psi = (P_T @ psi_init.todense())[:dim]
        return reduced_psi
