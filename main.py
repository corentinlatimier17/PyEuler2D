from os import makedirs
from time import time

from modules.mesh import *
from modules.conserved import *
from modules.boundaryconditions import *
from modules.numerics import *
from modules.monitor import *
from modules.writer import *
from modules.post import *

# Mesh file 
mesh_filepath = "NACA0012grids/257x257.x"

# Numerical parameters
n_ghosts = 2
itermax  = 500000
CFL      = 1.0

# Flow conditions
M_inf = 0.8 # Mach number of freestream flow
AOA   = 0 # angle of attack [deg]

# Beginning of the computation
time_start = time()

# Construct mesh
mesh = Mesh(filepath=mesh_filepath, n_ghosts=n_ghosts)

# Output paths -> defines where to store the results
folder = "examples/NACA0012_M0.8/AOA_0/"
beginByName = "MESH_" + str(mesh.ni) + "_CFL_" + str(CFL) + "_"

# Create the folder if it does not exist
makedirs(folder, exist_ok=True)

# Filenames
output_path_results = folder + beginByName + "Flow.dat"
output_path_res_fig = folder + beginByName + "Residuals.png"
output_path_res_txt = folder + beginByName + "Residuals.txt"
output_path_Cp_fig  = folder + beginByName + "Cp.png"
output_path_Cp_txt  = folder + beginByName + "Cp.txt"
output_path_CL_CD   = folder + beginByName + "CL_CD.txt"
output_comp_time    = folder + beginByName + "computational_time.txt"

# Construct conserved variables and init the flow field
W = ConservedVariables(ni=mesh.ni, nj=mesh.nj, n_ghosts=n_ghosts)
W.init_flow(M_inf, AOA)

# Apply boundary conditions
W = apply_boundary_conditions(W, mesh, M_inf, AOA)

centralscheme = CentralSchemeWithArtificialDissipation(mesh, k2=1/2, k4=1/16)
RK2 = RK2(CFL, mesh, local_time=True)
monitor = Monitor(centralscheme, RK2, itermax=itermax, eps=10**(-6))

# Iteration loop
monitor.iterate(mesh, W, M_inf, AOA)

# Get the computational time
time_end = time()
computational_time = time_end-time_start

monitor.plot_residuals(output_path_res_fig, output_path_res_txt)

# Write the results file
write4Tecplot(output_path_results, mesh, W)

# Post-processing
plot_save_CP(W, mesh, M_inf, AOA, output_path_Cp_fig, output_path_Cp_txt)
compute_save_CL_CD(M_inf, AOA, mesh, W, output_path_CL_CD)
write_computational_time(computational_time, output_comp_time)


