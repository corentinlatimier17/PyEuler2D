import numpy as np
from modules.thermodynamics import GAMMA

class ConservedVariables():
    def __init__(self, ni, nj, n_ghosts):
        self.rho  = np.zeros((ni + 2*n_ghosts, nj + 2*n_ghosts)) # Matrix that stores the rho (density) values in the extended mesh
        self.rhou = np.zeros((ni + 2*n_ghosts, nj + 2*n_ghosts)) # Matrix that stores the rhou (x-momentum) values in the extended mesh
        self.rhov = np.zeros((ni + 2*n_ghosts, nj + 2*n_ghosts)) # Matrix that stores the rhov (y-momentum) values in the extended mesh
        self.rhoE = np.zeros((ni + 2*n_ghosts, nj + 2*n_ghosts)) # Matrix that stores the rhoE (total energy) values in the extended mesh
        
    def init_flow(self, M_inf, AOA):
        """This function initialize the flow field 
        using a specified Mach number and angle of attack (deg)"""

        P_0 = 1                     # Stagnation pressure of free stream flow
        rho_0 = 1                   # Stagnation density of free stream flow
        c_0 = np.sqrt(P_0/rho_0)    # Velocity of sound 

        pressure_ratio = (1+0.5*(GAMMA-1)*M_inf**2)**(-GAMMA/(GAMMA-1))  # isentropic relation associated with Mach number for pressure
        rho_ratio = pressure_ratio**(1/GAMMA)                            # isentropic relation associated with Mach number for density
                                 
        P = pressure_ratio*P_0
        rho = rho_ratio*rho_0
        c = np.sqrt(GAMMA*P/rho)/c_0
        u = c*M_inf*np.cos(AOA*(np.pi)/180)
        v = c*M_inf*np.sin(AOA*(np.pi)/180)
        rhoE = (1/(GAMMA-1))*P+0.5*rho*(u**2+v**2)

        self.rho[:,  :] = rho
        self.rhou[:, :] = rho*u
        self.rhov[:, :] = rho*v
        self.rhoE[:, :] = rhoE
