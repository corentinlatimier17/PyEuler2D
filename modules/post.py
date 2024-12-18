import numpy as np
import matplotlib.pyplot as plt
from modules.thermodynamics import GAMMA

def plot_save_CP(W, mesh, M_inf, AOA, output_path_Cp_fig, output_path_txt):
    # Compute the pressure
    P = (GAMMA - 1) * (W.rhoE - 0.5 * (W.rhou**2 + W.rhov**2) / W.rho)

    # Interpolate pressure values to vertices (adjusting for ghost cells)
    P_vertex = 0.5 * (P[mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts-1] + P[mesh.n_ghosts, mesh.n_ghosts+1:-mesh.n_ghosts])

    # Slice the mesh to match the correct number of vertices
    x_vertex = mesh.x_coordinates[0, mesh.n_ghosts:-mesh.n_ghosts]  # Adjusting for ghost cells

    # Extract wing profile (x, y coordinates at i = 0)
    x_profile = mesh.x_coordinates[0, :]  # All x-coordinates
    y_profile = mesh.y_coordinates[0, :]  # Corresponding y-coordinates

    # Save x and -C_p to a text file
    x_cp_data = np.column_stack((x_vertex, -P_vertex[1:-1]))
    np.savetxt(output_path_txt, x_cp_data, header="x/c\t-C_p", fmt="%.6f", delimiter="\t")
    
    # Create a figure and axis for both plots
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plotting Cp vs x/c
    ax.scatter(0, -1, marker="s", s=15, color="tab:blue")
    ax.scatter(x_vertex, -P_vertex[1:-1], marker="s", s=15, color="tab:blue", 
            label=r"$M_{\infty}$ =" + str(M_inf) + r", $\alpha$ = " + str(AOA) + " deg, mesh : " + str(mesh.ni) + r"$\times$" + str(mesh.nj))

    # Labels and grid for Cp plot
    ax.set_xlabel(r"$x/c$", fontsize=15)
    ax.set_ylabel(r"$-C_p$", fontsize=15)
    ax.grid()
    ax.legend(fontsize=12)

    # Plotting the wing profile (shifted y by -1)
    ax.plot(x_profile, y_profile - 1, color="tab:orange", linewidth=2)

    # Show the plot
    plt.tight_layout()  # Ensure no label overlap
    plt.savefig(output_path_Cp_fig, dpi=300, bbox_inches='tight')


def compute_save_CL_CD(M_inf, AOA, mesh, W, output_path_txt):
    P_0 = 1                     # Stagnation pressure of free stream flow
    rho_0 = 1                   # Stagnation density of free stream flow
    c_0 = np.sqrt(P_0 / rho_0)  # Velocity of sound 

    # Isentropic relations for pressure and density
    pressure_ratio = (1 + 0.5 * (GAMMA - 1) * M_inf**2)**(-GAMMA / (GAMMA - 1))
    rho_ratio = pressure_ratio**(1 / GAMMA)
                                
    P = pressure_ratio * P_0
    rho = rho_ratio * rho_0
    c = np.sqrt(GAMMA * P / rho) / c_0
    u = c * M_inf * np.cos(AOA * (np.pi) / 180)
    v = c * M_inf * np.sin(AOA * (np.pi) / 180)
    normV = np.sqrt(u**2 + v**2)

    # Calculate wall pressure
    P_wall = (GAMMA - 1) * 0.5 * (
        (W.rhoE[mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts] -
         0.5 * (W.rhou[mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]**2 + 
                W.rhov[mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]**2) /
         W.rho[mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]) +
        (W.rhoE[mesh.n_ghosts - 1, mesh.n_ghosts:-mesh.n_ghosts] -
         0.5 * (W.rhou[mesh.n_ghosts - 1, mesh.n_ghosts:-mesh.n_ghosts]**2 + 
                W.rhov[mesh.n_ghosts - 1, mesh.n_ghosts:-mesh.n_ghosts]**2) /
         W.rho[mesh.n_ghosts - 1, mesh.n_ghosts:-mesh.n_ghosts])
    )

    # Assuming mesh.x_ext and mesh.y_ext are the vertex coordinates
    x_mean = np.zeros_like(mesh.x_ext[2, mesh.n_ghosts:-mesh.n_ghosts])
    y_mean = np.zeros_like(mesh.y_ext[2, mesh.n_ghosts:-mesh.n_ghosts])

    # Loop through the cells and compute mean coordinates
    for j in range(mesh.n_ghosts, mesh.n_ghosts + mesh.nj):
        # Handle periodic boundary
        j_next = (j + 1) % mesh.nj
        
        # Compute mean vertex coordinates
        x_mean[j - mesh.n_ghosts] = (mesh.x_ext[2, j] + mesh.x_ext[2, j_next]) / 2
        y_mean[j - mesh.n_ghosts] = (mesh.y_ext[2, j] + mesh.y_ext[2, j_next]) / 2

    # Calculate lift and drag
    D_calc = np.sum(P_wall * mesh.Delta_jy[2, mesh.n_ghosts:-mesh.n_ghosts])
    L_calc = -np.sum(P_wall * mesh.Delta_jx[2, mesh.n_ghosts:-mesh.n_ghosts])
    M_calc = np.sum(-P_wall * mesh.Delta_jx[2, mesh.n_ghosts:-mesh.n_ghosts] * (x_mean[:-1] - 0.25) - (y_mean[:-1]) * P_wall * mesh.Delta_jy[2, mesh.n_ghosts:-mesh.n_ghosts])
    CD_calc = D_calc / (0.5 * rho * normV**2 * 1.0089304115)
    CL_calc = L_calc / (0.5 * rho * normV**2 * 1.0089304115)
    CM_calc = M_calc/(0.5 * rho * normV**2 * 1.0089304115*1)

    # Projection of coefficients in the direction of the freestream flow
    CLwind = CL_calc * np.cos(AOA * (np.pi) / 180) - CD_calc * np.sin(AOA * (np.pi) / 180)
    CDwind = CL_calc * np.sin(AOA * (np.pi) / 180) + CD_calc * np.cos(AOA * (np.pi) / 180)

    # Print results
    print('Nombre de Mach = ' + str(M_inf))
    print('Angle d\'attaque = ' + str(AOA))
    print('Coefficent de portance CL = ' + str(CLwind))
    print('Coefficent de trainée CD = ' + str(CDwind))
    print('Coefficient de moment CM = ' + str(CM_calc))

    # Save results to a text file
    with open(output_path_txt, 'w') as f:
        f.write("Mach Number (M_inf): {:.6f}\n".format(M_inf))
        f.write("Angle of Attack (AOA): {:.6f}\n".format(AOA))
        f.write("Lift Coefficient (CL_wind): {:.6f}\n".format(CLwind))
        f.write("Drag Coefficient (CD_wind): {:.6f}\n".format(CDwind))
        f.write("Raw Lift Coefficient (CL_calc): {:.6f}\n".format(CL_calc))
        f.write("Raw Drag Coefficient (CD_calc): {:.6f}\n".format(CD_calc))
        f.write("Moment Coefficient (CM_calc): {:.6f}\n".format(CM_calc))

def write_computational_time(computational_time, filepath):
    """Writes the computational time to a text file."""
    
    # Open the file in append mode to add new entries without overwriting existing ones
    with open(filepath, 'a') as file:
        file.write(f"Computational Time: {computational_time} seconds\n")

