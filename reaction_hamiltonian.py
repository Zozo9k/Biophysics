from scipy import sparse
from math import factorial

from utils import sparse_kron, sigma_p, sigma_m, ide, lower_op, raise_op


class ReactionHamiltonian:

    @staticmethod
    def operator(M: int, op_func: sparse.csr_matrix, i: int, num_of_chemicals: int) -> sparse.csr_matrix:
        """Operator acting on chemical i.

        Parameters
        ----------
            M (int):
                Maximum occupation number of each chemical in each voxel.
            op_func (sparse.csr_matrix):
                Operator to be applied.
            i (int):
                Chemical index where the operator is applied.
            num_of_chemicals (int):
                Number of chemicals in each voxel.

        Returns
        -------
            (sparse.csr_matrix):
                Operator of (M + 1)**num_of_chemicals x (M + 1)**num_of_chemicals size sparse matrix.
        """
        return sparse_kron([ide((M + 1)**i), op_func(M), ide((M + 1)**(num_of_chemicals - i - 1))])

    @staticmethod
    def projector_allowed_iA(M: int, y_bar_Ai: int, y_Ai: int, i: int, num_of_chemicals: int) -> sparse.csr_matrix:
        """Projector acting on chemical i, which projects on allowed states with y_bar > y.

        Parameters
        ----------
            M (int):
                Maximum occupation number of each chemical in each voxel.
            y_bar_Ai (int):
                y_bar_iA.
            y_Ai (int):
                y_iA.
            i (int):
                Chemical index where the operator is applied.
            num_of_chemicals (int):
                Number of chemicals in each voxel.

        Returns
        -------
            (sparse.csr_matrix):
                Operator of (M + 1)**num_of_chemicals x (M + 1)**num_of_chemicals size sparse matrix.
        """
        proj_prohibited = sparse.lil_matrix((M + 1,) * 2)
        for k in range(y_bar_Ai - y_Ai):
            proj_prohibited[M - k, M - k] = 1
        proj_allowed = ide(M + 1) - proj_prohibited
        return sparse_kron([ide((M + 1)**i), proj_allowed, ide((M + 1)**(num_of_chemicals - i - 1))])

    @classmethod
    def number_operator(cls, M: int, i: int, num_of_chemicals: int) -> sparse.csr_matrix:
        """Number operator counting the number of chemical i inside single voxel.

        Parameters
        ----------
            M (int):
                Maximum occupation number of each chemical in each voxel.
            i (int):
                Chemical index where the operator is applied.
            num_of_chemicals (int):
                Number of chemicals in each voxel.

        Returns
        -------
            (sparse.csr_matrix):
                Operator of (M + 1)**num_of_chemicals x (M + 1)**num_of_chemicals size sparse matrix.
        """
        return cls.operator(M, raise_op, i, num_of_chemicals) @ cls.operator(M, lower_op, i, num_of_chemicals)

    @staticmethod
    def cAMP_operator(op: sparse.csr_matrix, i: int, num_of_chemicals: int) -> sparse.csr_matrix:
        """Operator acting on chemical i for cAMP reaction model.
        NOTE: This is deprecated. Use operator instead. This method is only applicable for M = 1.
        """
        return sparse_kron([ide(2**i), op, ide(2**(num_of_chemicals - i - 1))])

    @classmethod
    def _inflow_term(
            cls, M: int, i_bar: int, y_bar: int, num_of_chemicals: int, proj: bool = True
    ) -> sparse.csr_matrix:
        if proj:
            matrix = cls.projector_allowed_iA(M, y_bar, 0, i_bar, num_of_chemicals)
        else:
            matrix = ide((M + 1)**num_of_chemicals)

        return cls.operator(M, raise_op, i_bar, num_of_chemicals)**y_bar - matrix

    @classmethod
    def _outflow_term(
        cls, M: int, i: int, y: int, num_of_chemicals: int
    ) -> sparse.csr_matrix:
        return (ide((M + 1)**num_of_chemicals) \
            - cls.operator(M, raise_op, i, num_of_chemicals)**y) @ cls.operator(M, lower_op, i, num_of_chemicals)**y

    @classmethod
    def _general_reaction_term_x(
            cls, M: int, k_l: list,
            y_values: sparse.csr_matrix, y_bar_values: sparse.csr_matrix, num_of_chemicals: int, proj: bool = True
    ) -> sparse.csr_matrix:
        """

        Parameters
        ----------
            M (int):
                Number of states in each voxel.
            k_l (list):
                List of rate constants for each reaction.
            y_values (sparse.csr_matrix):
                y_iA for all chemical i and reaction A.
            y_bar_values (sparse.csr_matrix):
                y_bar_iA for all chemical i and reaction A.
            num_of_chemicals (int):
                Number of chemicals.
            proj (bool):
                Whether to project on allowed states or not.

        Returns
        -------
            H_r_x (sparse.csr_matrix):
                Reaction hamiltonian for voxel x in the form of sparse matrix
                with dim (((M + 1)**num_of_chemicals, (M + 1)**num_of_chemicals)).
        """

        inflow_term = sparse.csr_matrix(((M + 1)**num_of_chemicals,) * 2)
        outflow_term = sparse.csr_matrix(((M + 1)**num_of_chemicals,) * 2)
        reaction_term = sparse.csr_matrix(((M + 1)**num_of_chemicals,) * 2)

        # TODO: parallelize
        for reac in range(len(k_l)):
            # now we want all nonzero elements of y_iA and y_bar_iA and their indices
            # for a specific reaction A
            y_col_data = y_values.getcol(reac).tocoo()
            y_iA = y_col_data.data
            i_indices = y_col_data.row
            y_bar_col_data = y_bar_values.getcol(reac).tocoo()
            y_bar_iA = y_bar_col_data.data
            i_bar_indices = y_bar_col_data.row
            both_i = set(i_indices).intersection(set(i_bar_indices))
            y_factorial_prod = 1

            # inflow term
            if not y_iA.any():
                for y_bar, i_bar in zip(y_bar_iA, i_bar_indices):
                    inflow_term -= k_l[reac] *  cls._inflow_term(M, int(i_bar), int(y_bar), num_of_chemicals, proj)

            # outflow term
            if not y_bar_iA.any():
                for y, i in zip(y_iA, i_indices):
                    outflow_term -= (k_l[reac] / factorial(int(y))) \
                        * cls._outflow_term(M, int(i), int(y), num_of_chemicals)

            # reaction term
            if y_iA.any() and y_bar_iA.any():
                first_term = ide((M + 1)**num_of_chemicals)
                second_term = ide((M + 1)**num_of_chemicals)

                for y_bar, i_bar in zip(y_bar_iA, i_bar_indices):
                    y_bar = int(y_bar)
                    i_bar = int(i_bar)
                    first_term = first_term @ cls.operator(M, raise_op, i_bar, num_of_chemicals)**y_bar
                    if i_bar in both_i:
                        S_Ai = y_bar - int(y_iA[i_bar])
                        if S_Ai > 0:
                            projector_Ai = cls.projector_allowed_iA(M, y_bar, int(
                                y_iA[i_bar]), i_bar, num_of_chemicals)
                    else:  # means only y_bar_i is nonzero
                        S_Ai = y_bar
                        if S_Ai > 0:
                            projector_Ai = cls.projector_allowed_iA(M, y_bar, 0, i_bar, num_of_chemicals)

                    if S_Ai > 0 and proj:
                        second_term = second_term @ projector_Ai

                for y, i in zip(y_iA, i_indices):
                    y = int(y)
                    i = int(i)
                    y_factorial_prod *= factorial(y)
                    first_term = first_term @ cls.operator(M, lower_op, i, num_of_chemicals)**y
                    second_term = second_term \
                        @ cls.operator(M, raise_op, i, num_of_chemicals)**y \
                        @ cls.operator(M, lower_op, i, num_of_chemicals)**y

                reaction_term -= (k_l[reac] / y_factorial_prod) * (first_term - second_term)

        return inflow_term + reaction_term + outflow_term

    @classmethod
    def _cAMP_reaction_term_x(cls, k_l: list, ASEP: bool) -> sparse.csr_matrix:
        """Create reaction hamiltonian in the form of a matrix.
        This is the reaction term for cAMP reaction model.

        NOTE: This is deprecated. Use _general_reaction_term_x instead. This method is only applicable for M = 1.

        Parameters
        ----------
            k_l (list):
                k_1 : Floquet rate constant for reaction 1. In general, it is a function of time.
                k_2 : Floquet rate constant for reaction 2. In general, it is a function of time.
                k_3, ..., k_8 : Rate constants for reactions 3, ..., 8.

        Returns
        -------
            H_r_x (sparse.csr_matrix)):
                Reaction hamiltonian for voxel x in the form of matrix
                with dim (((M + 1)**num_of_chemicals, (M + 1)**num_of_chemicals)).
        """

        sigma_1_p = cls.cAMP_operator(sigma_p, 0, 5)
        sigma_2_p = cls.cAMP_operator(sigma_p, 1, 5)
        sigma_3_p = cls.cAMP_operator(sigma_p, 2, 5)
        sigma_4_p = cls.cAMP_operator(sigma_p, 3, 5)
        sigma_5_p = cls.cAMP_operator(sigma_p, 4, 5)
        sigma_1_m = cls.cAMP_operator(sigma_m, 0, 5)
        sigma_2_m = cls.cAMP_operator(sigma_m, 1, 5)
        sigma_3_m = cls.cAMP_operator(sigma_m, 2, 5)
        sigma_4_m = cls.cAMP_operator(sigma_m, 3, 5)
        sigma_5_m = cls.cAMP_operator(sigma_m, 4, 5)

        H_r_x = - k_l[0] * (sigma_1_p - ide(2**5))

        H_r_x -= k_l[1] * (sigma_2_p - ide(2**5))

        H_r_x -= k_l[2] * (sigma_4_p - sigma_1_p @ sigma_3_p) @ sigma_1_m @ sigma_3_m
        if ASEP:
            H_r_x -= k_l[2] * \
                sigma_1_p @ sigma_1_m @ sigma_3_p @ sigma_3_m @ sigma_4_p @ sigma_4_m

        H_r_x -= k_l[3] * (sigma_5_p - sigma_2_p @ sigma_3_p) @ sigma_2_m @ sigma_3_m
        if ASEP:
            H_r_x -= k_l[3] * \
                sigma_2_p @ sigma_2_m @ sigma_3_p @ sigma_3_m @ sigma_5_p @ sigma_5_m

        H_r_x -= k_l[4] * (sigma_1_p @ sigma_3_p - sigma_4_p) @ sigma_4_m
        if ASEP:
            H_r_x -= k_l[4] * sigma_4_p @ sigma_4_m \
                @ (sigma_1_p @ sigma_1_m + sigma_3_p @ sigma_3_m
                   - sigma_1_p @ sigma_1_m @ sigma_3_p @ sigma_3_m)

        H_r_x -= k_l[5] * (sigma_2_p @ sigma_3_p - sigma_5_p) @ sigma_5_m
        if ASEP:
            H_r_x -= k_l[5] * sigma_5_p @ sigma_5_m \
                @ (sigma_2_p @ sigma_2_m + sigma_3_p @ sigma_3_m
                   - sigma_2_p @ sigma_2_m @ sigma_3_p @ sigma_3_m)

        H_r_x -= k_l[6] * (ide(2**5) - sigma_1_p) @ sigma_1_m

        H_r_x -= k_l[7] * (ide(2**5) - sigma_2_p) @ sigma_2_m

        return H_r_x

    @classmethod
    def generate_cAMP_reaction_hamiltonian(
            cls, k_l: list, num_of_voxels: int, num_of_chemicals: int, ASEP: bool = True
    ) -> sparse.csr_matrix:
        """Create reaction hamiltonian in the form of sparse matrix.

        Parameters
        ----------
            k_l (list):
                k_1 : Floquet rate constant for reaction 1. In general, it is a function of time.
                k_2 : Floquet rate constant for reaction 2. In general, it is a function of time.
                k_3, ..., k_8 : Rate constants for reactions 3, ..., 8.
            num_of_voxels (int):
                Number of voxels.
            num_of_chemicals (int):
                Number of chemicals.
            ASEP (bool):
                Whether to use ASEP or not.

        Returns
        -------
            H_r (sparse.csr_matrix)):
                Reaction hamiltonian in the form of sparse matrix
                with dim (((M + 1)**(num_of_voxels * num_of_chemicals), (M + 1)**(num_of_voxels * num_of_chemicals))).
        """
        h_r = cls._cAMP_reaction_term_x(k_l, ASEP)
        H_r = sum(sparse_kron([
            ide(2**(i * num_of_chemicals)), h_r, ide(2**((num_of_voxels - i - 1) * num_of_chemicals))
        ]) for i in range(num_of_voxels))
        return H_r

    @classmethod
    def generate_general_reaction_hamiltonian(
            cls, M: int, k_l: list, y_values: sparse.csr_matrix, y_bar_values: sparse.csr_matrix,
            num_of_voxels: int, num_of_chemicals: int, proj: bool
    ) -> sparse.csr_matrix:
        """
        Parameters
        ----------
            M (int):
                Number of states in each voxel.
            k_l (list):
                List of rate constants for each reaction.
            y_values (sparse.csc_matrix):
                y_iA for all chemical i and reaction A.
            y_bar_values (sparse.csr_matrix):
                y_bar_iA for all chemical i and reaction A.
            num_of_voxels (int):
                Number of voxels.
            num_of_chemicals (int):
                Number of chemicals.
            proj (bool):
                Whether to project on allowed states or not.

        Returns
        -------
            H_r (sparse.csr_matrix)):
                Reaction hamiltonian in the form of sparse matrix
                with dim (((M + 1)**(num_of_voxels * num_of_chemicals), (M + 1)**(num_of_voxels * num_of_chemicals))).
        """
        h_r = cls._general_reaction_term_x(M, k_l, y_values, y_bar_values, num_of_chemicals, proj)
        H_r = sum(sparse_kron([
            ide((M + 1)**(i * num_of_chemicals)), h_r, ide((M + 1)**((num_of_voxels - i - 1) * num_of_chemicals))
        ]) for i in range(num_of_voxels))
        return H_r
