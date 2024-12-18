import numpy as np
from modules.boundaryconditions import apply_boundary_conditions
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)


class Monitor():
    def __init__(self, ConvectiveFluxDiscretizationScheme, TimeIntegrationMethod, itermax = 50000, eps = 10**(-8)):
        self.scheme = ConvectiveFluxDiscretizationScheme
        self.time_integration = TimeIntegrationMethod

        self.itermax = itermax
        self.n_iter  = 0
        self.eps     = eps

        self.res_rho  = []
        self.res_rhou = []
        self.res_rhov = []
        self.res_rhoE = []

    def iterate(self, mesh, W, M_inf, AOA):
        self.scheme.getResiduals(W, mesh)

        # Initialization of residuals
        self.res_rho.append(np.max(self.scheme.res_rho[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
        self.res_rhou.append(np.max(self.scheme.res_rhou[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
        self.res_rhov.append(np.max(self.scheme.res_rhov[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
        self.res_rhoE.append(np.max(self.scheme.res_rhoE[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))

        while (self.n_iter<=self.itermax):
            self.scheme.getResiduals(W, mesh)
            self.time_integration.step(self.scheme, W, mesh, M_inf, AOA)
            W = apply_boundary_conditions(W, mesh, M_inf, AOA)

            self.n_iter +=1
            self.res_rho.append(np.max(self.scheme.res_rho[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
            self.res_rhou.append(np.max(self.scheme.res_rhou[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
            self.res_rhov.append(np.max(self.scheme.res_rhov[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
            self.res_rhoE.append(np.max(self.scheme.res_rhoE[mesh.n_ghosts:-(mesh.n_ghosts + 1), mesh.n_ghosts:-(mesh.n_ghosts + 1)]))
            # Data to display
            data = [
                ["Residual rho",  self.res_rho[-1]/self.res_rho[0]],
                ["Residual rhou", self.res_rhou[-1]/self.res_rhou[0]],
                ["Residual rhov", self.res_rhov[-1]/self.res_rhov[0]],
                ["Residual rhoE", self.res_rhoE[-1]/self.res_rhoE[0]],
            ]

            # Calculate column widths
            col1_width = max(len(row[0]) for row in data) + 2  # Add padding
            col2_width = 20  # Fixed width for values for alignment

            # Print the table header
            print("+" + "-" * col1_width + "+" + "-" * col2_width + "+")
            print(f"| {'Quantity'.ljust(col1_width - 1)} | {'Value'.ljust(col2_width - 1)} |")
            print("+" + "-" * col1_width + "+" + "-" * col2_width + "+")

            # Print the table rows
            for row in data:
                quantity, value = row
                print(f"| {quantity.ljust(col1_width - 1)} | {str(value).ljust(col2_width - 1)} |")

            # Print the bottom border
            print("+" + "-" * col1_width + "+" + "-" * col2_width + "+")

            if (self.res_rho[-1]/self.res_rho[0]<=self.eps):
                print("Computation has converged !\n")
                break

    def plot_residuals(self, output_path_fig, output_path_txt):
        iterations = np.arange(len(self.res_rho))

        # Compute normalized residuals
        res_rho_norm = np.array(self.res_rho) / self.res_rho[0]
        res_rhou_norm = np.array(self.res_rhou) / self.res_rhou[0]
        res_rhov_norm = np.array(self.res_rhov) / self.res_rhov[0]
        res_rhoE_norm = np.array(self.res_rhoE) / self.res_rhoE[0]

        # Save residuals to a text file
        residuals_data = np.column_stack((iterations, res_rho_norm, res_rhou_norm, res_rhov_norm, res_rhoE_norm))
        np.savetxt(output_path_txt, residuals_data, header="Iteration\tRes_rho\tRes_rhou\tRes_rhov\tRes_rhoE", fmt="%.6e", delimiter="\t")

        # Plot the residuals
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, res_rho_norm, label=r"Residual $\rho$", color="tab:orange")
        plt.semilogy(iterations, res_rhou_norm, label=r"Residual $\rho u$", color="tab:green")
        plt.semilogy(iterations, res_rhov_norm, label=r"Residual $\rho v$", color="tab:purple")
        plt.semilogy(iterations, res_rhoE_norm, label=r"Residual $\rho E$", color="tab:blue")

        # Add labels, legend, and grid
        plt.xlabel("Iteration", fontsize=15)
        plt.ylabel("Normalized Residual (log scale)", fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_path_fig, dpi=300, bbox_inches='tight')





