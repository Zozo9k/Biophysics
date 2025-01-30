from typing import Union
#from numba import jit

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm
from scipy.sparse import (
    bsr_matrix, coo_matrix, csr_array, csr_matrix,
)

from reaction_hamiltonian import ReactionHamiltonian
from diffusion_hamiltonian import DiffusionHamiltonian
from reduced_HS import ReducedHilbertSpace
from utils import sparse_kron, ide


class Plot:

    @staticmethod
    #@jit
    def periodic_time_evol(
        t: int, y: list, model: str, hamiltonian: csr_matrix, k: list,
        inflow_Gi: csr_matrix, inflow_Gs: csr_matrix, omega: float, amplitude: float, phase: float = 0.0
    ) -> csr_matrix:
        """Update coefficients of the reaction Hamiltonian at each time step

        Parameters
        ----------
            t (int):
                Time.
            y (list):
                List of coefficients of the wavefunction.
            model (str):
                Name of the reaction model.
            hamiltonian (csr_matrix):
                Hamiltonian without inflow terms for the reaction part
            inflow_Gi (csr_matrix):
                Matrix to update reaction Hamiltonian with cosinus
            inflow_Gs (csr_matrix):
                Matrix to update reaction Hamiltonian with sinus
            omega (float):
                Angular frequency of the oscillating reaction rates.
            amplitude (float):
                Amplitude of the oscillating reaction rates.
            phase (float):
                Phase of the oscillating reaction rates.
        """
        if model == 'cAMP':
            H = hamiltonian - inflow_Gi * k[0] *(1 + amplitude*np.cos(omega*t))\
                - inflow_Gs * k[1] *(1 + amplitude*np.cos(omega*t + phase)) #(k[1] + (amplitude/2) * (1 - np.cos(omega * t + phase)))

        return - H @ y

    @staticmethod
    #@jit
    def number_ops(
        M: int, num_of_chemicals: int, num_of_voxels: int, AC: int, reduced: bool = False
    ) -> dict[dict[csr_matrix]]:
        """Returns a mapping of number operators for each chemical in each voxel.

        Returns
        -------
            num_op_map (dict[dict[csr_matrix]]):
                num_op_map[i][x - 1] returns the number operator for chemical i in voxel x.
        """
        num_op_map: dict[dict[csr_matrix]] = {}  # contains all chemicals and all voxels operators

        for chem in range(num_of_chemicals):
            num_op_chem_map = {}

            for voxel in range(num_of_voxels):
                num_op_chem = ReactionHamiltonian.number_operator(M, chem, num_of_chemicals)
                num_op_chem_voxel = sparse_kron(
                    [ide((M + 1)**(voxel * num_of_chemicals)), num_op_chem,
                     ide((M + 1)**((num_of_voxels - voxel - 1) * num_of_chemicals))])
                if reduced:
                    num_op_chem_voxel = ReducedHilbertSpace.generate_reduced_matrix(
                        num_op_chem_voxel, AC, M, num_of_chemicals, num_of_voxels)

                num_op_chem_map[voxel] = num_op_chem_voxel

            num_op_map[chem] = num_op_chem_map
        return num_op_map

    @staticmethod
    def _psi_init_for_RK(
        psi_init: Union[csr_matrix, coo_matrix, bsr_matrix, np.matrix, np.ndarray]
    ) -> np.ndarray:
        if isinstance(psi_init, (csr_matrix, coo_matrix, bsr_matrix)):
            psi_init_for_RK = psi_init.toarray().flatten()
        elif isinstance(psi_init, (np.matrix, np.ndarray)):
            psi_init_for_RK = np.ravel(psi_init)
        else:
            raise TypeError('psi_init must be a sparse or numpy array')
        return psi_init_for_RK

    @staticmethod
    #@jit
    def plot_init(num_of_chemicals: int) -> tuple[dict[list], dict[list]]:
        plot_all_chem_voxel_x = dict()
        plot_all_chem_all_voxels = dict()
        for chem in range(num_of_chemicals):
            plot_all_chem_voxel_x[chem] = []
            plot_all_chem_all_voxels[chem] = []
        return plot_all_chem_voxel_x, plot_all_chem_all_voxels

    @staticmethod
    def P_bra(AC: int, M: int, num_of_chemicals: int, num_of_voxels: int, reduced: bool = False):
        P_bra = np.ones((M + 1)**(num_of_voxels * num_of_chemicals))
        if reduced:
            proj = ReducedHilbertSpace.reduced_HS_proj(AC, M, num_of_chemicals, num_of_voxels).copy()
            dim = len(proj.indices)
            P_bra = P_bra[:dim]
        return P_bra

    @staticmethod
    def figure(
        time: np.array, labels: dict[int, str], num_of_chemicals: int,
        plot_all_chem_all_voxels: dict[int, list],
        plot_all_chem_voxel_x: dict[int, list],
        voxel_x: int, k_values: list,
    ):
        fig, axs = plt.subplots(2, 1)
        axs[0].set(
            ylabel=r'$\langle n \rangle$ for all voxels',
            title=r'$\langle n \rangle$ for reaction rates  k = {}'.format(k_values),
        )
        axs[1].set(
            xlabel='Time',
            ylabel=r'$\langle n \rangle$ for voxel {}'.format(voxel_x),
        )

        for chem in range(num_of_chemicals):
            axs[0].plot(time, plot_all_chem_all_voxels[chem], label=labels[chem])
            axs[1].plot(time, plot_all_chem_voxel_x[chem], label=labels[chem])

        for i in range(2):
            axs[i].legend()
            axs[i].grid(which='minor', alpha=0.2)
            axs[i].grid(which='major', alpha=0.5)
            axs[i].minorticks_on()

    @staticmethod
    def single_figure(
        time: np.array, labels: dict[int, str], num_of_chemicals: int,
        target: dict[int, list], voxel_x: int, k_values: list, omega: float, phase: float
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel(r'Time [s]', fontsize=24)
        ax.set_ylabel(r'Density [a.u.]', fontsize=24)#r'$\langle n \rangle$ for voxel {}'.format(voxel_x),
            #title=#'k = {}, $\omega$ = {} and $\phi$ = 0'.format(k_values, omega),#r'$\langle n \rangle$ for reaction rates  k = {} and omega = {}'.format(k_values, omega),

        for chem in range(num_of_chemicals):
            if chem==3 or chem==4:
                ax.plot(time, target[chem], label=labels[chem])
            else:
                ax.plot(time, target[chem], label=labels[chem], alpha=0.7)


        ax.legend(
           bbox_to_anchor=(0.99, 0.01), borderaxespad=0., loc= 'lower right', fontsize=20
        )
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.minorticks_on()
        fig.tight_layout()


    @staticmethod
    def evol_update_matrices(
        Floquet_voxels: list, M: int, AC: int, num_of_chemicals: int, num_of_voxels: int,
        reduced: bool = True, proj: bool = True
    ):
        Gi = ReactionHamiltonian._inflow_term(M, 0, 1, num_of_chemicals, proj)
        Gs = ReactionHamiltonian._inflow_term(M, 1, 1, num_of_chemicals, proj)
        Gi_matrix = sum(sparse_kron([
            ide((M + 1)**(i * num_of_chemicals)), Gi, ide((M + 1)**((num_of_voxels - i - 1) * num_of_chemicals))
        ]) for i in Floquet_voxels)
        Gs_matrix = sum(sparse_kron([
            ide((M + 1)**(i * num_of_chemicals)), Gs, ide((M + 1)**((num_of_voxels - i - 1) * num_of_chemicals))
        ]) for i in Floquet_voxels)
        if reduced:
            Gi_matrix = ReducedHilbertSpace.generate_reduced_matrix(Gi_matrix, AC, M, num_of_chemicals, num_of_voxels)
            Gs_matrix = ReducedHilbertSpace.generate_reduced_matrix(Gs_matrix, AC, M, num_of_chemicals, num_of_voxels)

        return Gi_matrix, Gs_matrix

    @classmethod
    def pump_current(cls, average_N_all_chem: dict[int, list], chemical: int, deltat: float) -> float:
        average_N = average_N_all_chem[chemical]
        J = np.gradient(average_N, deltat)
        return J

    @classmethod
    def plt(
        cls,
        plot: str,
        M: int, matrix: csr_matrix, psi_init: csr_array, delta_t: int, num_of_voxels: int,
        num_of_chemicals: int, labels: list, AC: int, timesteps_number: int, voxel_x: int, k_values: list,
        Floquet_voxels: list, omega: float, amplitude: float, phase: float = np.pi / 2,
        RK: bool = False, model: str = 'cAMP', reduced=False, proj: bool = True
    ):
        """
        Parameters
        ----------
            plot (str):
                'both' plots both all voxels and chosen voxel x,
                'one' plots voxel x, 'all' plots all voxels only
            RK (bool):
                whether Runge-Kutta method is used or not (by default is the exact solution)
            reduced (bool):
                if the Hamiltonian is determined in the reduced basis
            proj (bool):
                whether we use projectors in the definition of the annihilation and creation operators or not
        """


        psi = psi_init
        time = np.arange(timesteps_number) * delta_t

        P_bra = cls.P_bra(AC, M, num_of_chemicals, num_of_voxels, reduced)
        num_chem_all_voxels = cls.number_ops(M, num_of_chemicals, num_of_voxels, AC, reduced)

        # we compute the matrices for Floquet evolution once outside the loop to be more efficient
        Gi_matrix, Gs_matrix = cls.evol_update_matrices(
            Floquet_voxels, M, AC, num_of_chemicals, num_of_voxels, reduced, proj)

        plot_all_chem_voxel_x, plot_all_chem_all_voxels = cls.plot_init(num_of_chemicals)

        if RK:
            H_wo_k1_k2 = matrix + k_values[0] * Gi_matrix + k_values[1] * Gs_matrix
            y0 = cls._psi_init_for_RK(psi_init)
            RK_solver = solve_ivp(
                fun=cls.periodic_time_evol, t_span=[0, timesteps_number * delta_t], y0=y0,
                t_eval=time, method='RK45',
                args=([str(model), H_wo_k1_k2, k_values, Gi_matrix, Gs_matrix,
                       omega, amplitude, phase]))
        else:
            H_red = matrix

        for iteration in range(timesteps_number):
            if not RK and iteration % 100 == 0:
                print(iteration)
            for chem in range(num_of_chemicals):
                op = num_chem_all_voxels[chem][voxel_x - 1]
                num_of_chem_voxel_x = P_bra @ (op @ psi)
                plot_all_chem_voxel_x[chem].append(num_of_chem_voxel_x.item(0))

                op_chem_all_voxels = sum(num_chem_all_voxels[chem].values())
                num_of_chem_all_voxels = P_bra @ (op_chem_all_voxels @ psi)
                plot_all_chem_all_voxels[chem].append(num_of_chem_all_voxels.item(0))

            if RK:
                psi = RK_solver.y[:, iteration]
                psi = psi / np.sum(psi)
            else:
                if not reduced:
                    U = matrix
                else:  # This is the exact but more time consuming solution
                    # WARNING : reduction of HS only available for cAMP model
                    #WARNING :  This is with ancient parametrizations of cos (cf latex document)
                    if iteration == 0:
                        H_red = H_red \
                            - Gi_matrix * amplitude * np.cos(omega * delta_t) \
                            - Gs_matrix * amplitude * np.sin(omega * delta_t)
                    elif iteration > 0:
                        H_red = \
                            H_red \
                            - Gi_matrix * amplitude * (
                                np.cos(omega * (iteration + 1) * delta_t) - np.cos(omega * iteration * delta_t)) \
                            - Gs_matrix * amplitude * (
                                np.sin(omega * (iteration + 1) * delta_t) - np.cos(omega * iteration * delta_t + phase))
                    U = expm(- delta_t * H_red)
                psi = U @ psi
                psi = psi / np.sum(psi)

        if plot == 'both':
            cls.figure(time, labels, num_of_chemicals, plot_all_chem_all_voxels,
                       plot_all_chem_voxel_x, voxel_x, k_values)
        elif plot == 'one':
            cls.single_figure(time, labels, num_of_chemicals,
                              plot_all_chem_voxel_x, voxel_x, k_values, omega, phase)
        elif plot == 'all':
            cls.single_figure(time, labels, num_of_chemicals,
                              plot_all_chem_all_voxels, voxel_x, k_values, omega)

        #diff_ACGs_minus_ACGi = np.array([ACGs - ACGi for ACGi, ACGs in zip(plot_all_chem_all_voxels[3], plot_all_chem_all_voxels[4])])
        ACGi_current = np.gradient(plot_all_chem_all_voxels[3], delta_t)
        #J = cls.pump_current(plot_all_chem_voxel_x, 3, delta_t)
        #with open('.\data\current_th_model.pkl', 'wb') as file:
         #   pickle.dump(J, file)
        return ACGi_current #diff_ACGs_minus_ACGi
