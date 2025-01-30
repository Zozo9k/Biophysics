from scipy import sparse


class CRNStructureGenerator:
    """Generates y_iA and y_bar_iA for different CRN structures.
    """
    @staticmethod
    def cAMP() -> tuple[sparse.csr_matrix]:
        """Returns y_A_i and y_bar_A_i for cAMP reaction model.
        Returns
        -------
            cAMP_y (sparse.csr_matrix):
                y_A_i for cAMP reaction model.
            cAMP_y_bar (sparse.csr_matrix):
                y_bar_A_i for cAMP reaction model.
        """
        _NUM_OF_CHEMICALS = 5
        _NUM_OF_REACTIONS = 8
        cAMP_y = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y_bar = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y[0, 2] = 1
        cAMP_y[2, 2] = 1
        cAMP_y[1, 3] = 1
        cAMP_y[2, 3] = 1
        cAMP_y[3, 4] = 1
        cAMP_y[4, 5] = 1
        cAMP_y[0, 6] = 1
        cAMP_y[1, 7] = 1
        cAMP_y_bar[0, 0] = 1
        cAMP_y_bar[1, 1] = 1
        cAMP_y_bar[3, 2] = 1
        cAMP_y_bar[4, 3] = 1
        cAMP_y_bar[0, 4] = 1
        cAMP_y_bar[2, 4] = 1
        cAMP_y_bar[1, 5] = 1
        cAMP_y_bar[2, 5] = 1
        return cAMP_y.tocsr(), cAMP_y_bar.tocsr()

    @staticmethod
    def closed_cAMP() -> tuple[sparse.csr_matrix]:
        """Returns y_A_i and y_bar_A_i for cAMP reaction model in a closed system.
        Returns
        -------
            cAMP_y (sparse.csr_matrix):
                y_A_i for cAMP reaction model.
            cAMP_y_bar (sparse.csr_matrix):
                y_bar_A_i for cAMP reaction model.
        """
        _NUM_OF_CHEMICALS = 5
        _NUM_OF_REACTIONS = 8
        cAMP_y = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y_bar = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y[0, 0] = 1
        cAMP_y[2, 0] = 1
        cAMP_y[1, 1] = 1
        cAMP_y[2, 1] = 1
        cAMP_y[3, 2] = 1
        cAMP_y[4, 3] = 1
        cAMP_y_bar[3, 0] = 1
        cAMP_y_bar[4, 1] = 1
        cAMP_y_bar[0, 2] = 1
        cAMP_y_bar[2, 2] = 1
        cAMP_y_bar[1, 3] = 1
        cAMP_y_bar[2, 3] = 1
        return cAMP_y.tocsr(), cAMP_y_bar.tocsr()

    @staticmethod
    def cAMP_bath() -> tuple[sparse.csr_matrix]:
        """Returns y_A_i and y_bar_A_i for cAMP reaction model when there is a bath of ACGi and ACGs molecules.
        Returns
        -------
            cAMP_y (sparse.csr_matrix):
                y_A_i for cAMP reaction model.
            cAMP_y_bar (sparse.csr_matrix):
                y_bar_A_i for cAMP reaction model.
        """
        _NUM_OF_CHEMICALS = 5
        _NUM_OF_REACTIONS = 12
        cAMP_y = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y_bar = sparse.lil_matrix((_NUM_OF_CHEMICALS, _NUM_OF_REACTIONS))
        cAMP_y[0, 2] = 1
        cAMP_y[1, 3] = 1
        cAMP_y[2, 4] = 1
        cAMP_y[3, 5] = 1
        cAMP_y[0, 6] = 1
        cAMP_y[1, 7] = 1
        cAMP_y[3, 8] = 1
        cAMP_y[4, 9] = 1
        cAMP_y_bar[0, 0] = 1
        cAMP_y_bar[1, 1] = 1
        cAMP_y_bar[2, 2] = 1
        cAMP_y_bar[3, 3] = 1
        cAMP_y_bar[0, 4] = 1
        cAMP_y_bar[1, 5] = 1
        cAMP_y_bar[3, 10] = 1
        cAMP_y_bar[4, 11] = 1
        return cAMP_y.tocsr(), cAMP_y_bar.tocsr()


    @staticmethod
    def Schloegel(num_of_chemicals: int, num_of_reactions: int) -> tuple[sparse.csr_matrix]:
        """Returns y_iA and y_bar_iA for Schloegl model.
        Returns
        -------
            Schl_y (sparse.csr_matrix):
                y_iA for Schloegl model.
            Schl_y_bar (sparse.csr_matrix):
                y_bar_iA for Schloegl model.
        """
        Schl_y = sparse.lil_matrix((num_of_chemicals, num_of_reactions))
        Schl_y_bar = sparse.lil_matrix((num_of_chemicals, num_of_reactions))
        Schl_y[2, 0] = 2
        Schl_y[0, 0] = 1
        Schl_y[2, 1] = 3
        Schl_y[1, 2] = 1
        Schl_y[2, 3] = 1
        Schl_y_bar[2, 0] = 3
        Schl_y_bar[2, 1] = 2
        Schl_y_bar[0, 1] = 1
        Schl_y_bar[2, 2] = 1
        Schl_y_bar[1, 3] = 1
        return Schl_y.tocsr(), Schl_y_bar.tocsr()
