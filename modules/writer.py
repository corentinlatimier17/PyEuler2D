import numpy as np
from modules.thermodynamics import GAMMA

def write4Tecplot(output_filepath, mesh, W):
    try:
        with open(output_filepath, 'w') as file:
            # Write Tecplot header
            file.write('Title = "Flow solution"\n')
            file.write('Variables = X, Y, CellVolume, I, J, rho, rhou, rhov, rhoE, p\n')
            file.write(f'ZONE T="BLOCK1", I={mesh.ni+1}, J={mesh.nj+1}, DATAPACKING=POINT\n')

            # Loop over vertices -> we need to average cell values to obtain node values
            n_ghosts = mesh.n_ghosts
            for i in range(0, mesh.ni+1):
                for j in range(0, mesh.nj+1):
                    
                    i_q = i + n_ghosts
                    j_q = j + n_ghosts
            
                    # Average values and handle NaNs
                    rho = np.nan_to_num(0.25 * (
                        W.rho[i_q][j_q] + W.rho[i_q][j_q-1] + W.rho[i_q-1][j_q-1] + W.rho[i_q-1][j_q]
                    ))
                    rhou = np.nan_to_num(0.25 * (
                        W.rhou[i_q][j_q] + W.rhou[i_q][j_q-1] + W.rhou[i_q-1][j_q-1] + W.rhou[i_q-1][j_q]
                    ))
                    rhov = np.nan_to_num(0.25 * (
                        W.rhov[i_q][j_q] + W.rhov[i_q][j_q-1] + W.rhov[i_q-1][j_q-1] + W.rhov[i_q-1][j_q]
                    ))
                    rhoE = np.nan_to_num(0.25 * (
                        W.rhoE[i_q][j_q] + W.rhoE[i_q][j_q-1] + W.rhoE[i_q-1][j_q-1] + W.rhoE[i_q-1][j_q]
                    ))

                    p = np.nan_to_num((GAMMA-1)*(rhoE-1/2*(rhou**2 + rhov**2)/rho))

                    # Calculate cell area
                    if (i == 0 and j == 0):
                        cell_area = mesh.areas[i][j]
                    elif (i == mesh.ni and j == 0):
                        cell_area = mesh.areas[i-1][j]
                    elif (i == mesh.ni-1 and j == mesh.nj-1):
                        cell_area = mesh.areas[i-1][j-1] 
                    elif (i == 0 and j == mesh.nj-1):
                        cell_area = mesh.areas[i][j-1] 
                    elif (i == 0):
                        cell_area = 0.5 * (mesh.areas[i][j] +  mesh.areas[i][j-1]) 
                    elif (i == mesh.ni-1):
                        cell_area = 0.5 * (mesh.areas[i-1][j-1] +  mesh.areas[i-1][j])  
                    elif (j == 0):
                        cell_area = 0.5 * (mesh.areas[i][j] +  mesh.areas[i-1][j])
                    elif (j == mesh.nj-1):
                        cell_area = 0.5 * (mesh.areas[i-1][j-1] +  mesh.areas[i][j-1])
                    else:
                        cell_area = 0.25 * (
                            mesh.areas[i][j] + mesh.areas[i-1][j] +
                            mesh.areas[i][j-1] + mesh.areas[i-1][j-1]
                        )
                    cell_area = np.nan_to_num(cell_area)

                    # Write data in the specified format
                    file.write(f"{mesh.x_coordinates[i][j]} {mesh.y_coordinates[i][j]} {cell_area} {i} {j} {rho} {rhou} {rhov} {rhoE} {p}\n")
    except Exception as e:
        print(f"Error opening file for writing: {e}")
        exit(1)