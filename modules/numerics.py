import numpy as np
import copy
from modules.mesh import *
from modules.conserved import *
from modules.thermodynamics import GAMMA
from modules.boundaryconditions import apply_boundary_conditions


class CentralSchemeWithArtificialDissipation():
    def __init__(self, mesh, k2=1/2, k4=1/64):
        self.k2  = k2
        self.k4  = k4

        self.res_rho  = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj+2*mesh.n_ghosts))
        self.res_rhou = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj+2*mesh.n_ghosts))
        self.res_rhov = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj+2*mesh.n_ghosts))
        self.res_rhoE = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj+2*mesh.n_ghosts))
    
    def getResiduals(self, W, mesh):
        F_C_rho, F_C_rhou, F_C_rhov, F_C_rhoE = self.getFC(W, mesh)
        D_rho, D_rhou, D_rhov, D_rhoE = self.getArtificialDissipation(W, mesh)

        area = mesh.areas[mesh.n_ghosts:mesh.ni+mesh.n_ghosts,mesh.n_ghosts:mesh.nj+mesh.n_ghosts]

        self.res_rho [mesh.n_ghosts:mesh.ni+mesh.n_ghosts,mesh.n_ghosts:mesh.nj+mesh.n_ghosts] = (F_C_rho +D_rho )/area
        self.res_rhou[mesh.n_ghosts:mesh.ni+mesh.n_ghosts,mesh.n_ghosts:mesh.nj+mesh.n_ghosts] = (F_C_rhou+D_rhou)/area
        self.res_rhov[mesh.n_ghosts:mesh.ni+mesh.n_ghosts,mesh.n_ghosts:mesh.nj+mesh.n_ghosts] = (F_C_rhov+D_rhov)/area
        self.res_rhoE[mesh.n_ghosts:mesh.ni+mesh.n_ghosts,mesh.n_ghosts:mesh.nj+mesh.n_ghosts] = (F_C_rhoE+D_rhoE)/area

    def getFC(self, W, mesh):
        """This function returns the convective residuals of the conserved variables"""
        rho  = W.rho [1:-1, 1:-1]
        rhou = W.rhou[1:-1, 1:-1]
        rhov = W.rhov[1:-1, 1:-1]
        rhoE = W.rhoE[1:-1, 1:-1]

        E    = rhoE/rho
        u    = rhou/rho
        v    = rhov/rho
        p    = (GAMMA-1)*(rhoE - 1/2*(rhou**2 + rhov**2)/rho)

        Delta_ix = mesh.Delta_ix[mesh.n_ghosts:-mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]
        Delta_iy = mesh.Delta_iy[mesh.n_ghosts:-mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]
        Delta_jx = mesh.Delta_jx[mesh.n_ghosts:-mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]
        Delta_jy = mesh.Delta_jy[mesh.n_ghosts:-mesh.n_ghosts, mesh.n_ghosts:-mesh.n_ghosts]

        # Flux of density (vectorized)

        # Central scheme way (method 1 - see slides) : compute mean of the flux computed with left and right state
        Flux_rho_j_ex = 0.5*(rho[1:-1,:-1]*u[1:-1,:-1]+rho[1:-1,1:]*u[1:-1,1:])
        Flux_rho_j_ey = 0.5*(rho[1:-1,:-1]*v[1:-1,:-1]+rho[1:-1,1:]*v[1:-1,1:])
        # Central scheme way (method 1 - see slides) : compute mean of the flux computed with bottom and above state
        Flux_rho_i_ex = 0.5*(rho[:-1,1:-1]*u[:-1,1:-1]+rho[1:,1:-1]*u[1:,1:-1])
        Flux_rho_i_ey = 0.5*(rho[:-1,1:-1]*v[:-1,1:-1]+rho[1:,1:-1]*v[1:,1:-1])

        Flux_B_rho = Flux_rho_i_ex*(Delta_jy)-Flux_rho_i_ey*(Delta_jx) # we use scalar product to get the fluxes at each faces (B,D,H,G)
        Flux_D_rho = Flux_rho_j_ex*(Delta_iy)-Flux_rho_j_ey*(Delta_ix) 
        Flux_H_rho = Flux_rho_i_ex*(-Delta_jy)-Flux_rho_i_ey*(-Delta_jx)
        Flux_G_rho = Flux_rho_j_ex*(-Delta_iy)-Flux_rho_j_ey*(-Delta_ix)

        F_C_rho = Flux_B_rho[:-1,:]+Flux_D_rho[:,1:]+Flux_H_rho[1:,:]+Flux_G_rho[:,:-1]

        # Flux of x-momentum (vectorized)
        Flux_rhou_j_ex = 0.5*(rho[1:-1,:-1]*(u[1:-1,:-1]**2)+rho[1:-1,1:]*(u[1:-1,1:]**2)+p[1:-1,:-1]+p[1:-1,1:])
        Flux_rhou_j_ey = 0.5*(rho[1:-1,:-1]*v[1:-1,:-1]*u[1:-1,:-1]+rho[1:-1,1:]*v[1:-1,1:]*u[1:-1,1:])
        Flux_rhou_i_ex = 0.5*(rho[:-1,1:-1]*(u[:-1,1:-1]**2)+rho[1:,1:-1]*(u[1:,1:-1]**2)+p[:-1,1:-1]+p[1:,1:-1])
        Flux_rhou_i_ey = 0.5*(rho[:-1,1:-1]*v[:-1,1:-1]*u[:-1,1:-1]+rho[1:,1:-1]*v[1:,1:-1]*u[1:,1:-1])

        Flux_B_rhou = Flux_rhou_i_ex*(Delta_jy)-Flux_rhou_i_ey*(Delta_jx)
        Flux_D_rhou = Flux_rhou_j_ex*(Delta_iy)-Flux_rhou_j_ey*(Delta_ix)
        Flux_H_rhou = Flux_rhou_i_ex*(-Delta_jy)-Flux_rhou_i_ey*(-Delta_jx)
        Flux_G_rhou = Flux_rhou_j_ex*(-Delta_iy)-Flux_rhou_j_ey*(-Delta_ix)

        F_C_rhou = Flux_B_rhou[:-1,:]+Flux_D_rhou[:,1:]+Flux_H_rhou[1:,:]+Flux_G_rhou[:,:-1]

        # Flux of y-momentum (vectorized)
        Flux_rhov_j_ex = 0.5*(rho[1:-1,:-1]*v[1:-1,:-1]*u[1:-1,:-1]+rho[1:-1,1:]*v[1:-1,1:]*u[1:-1,1:])
        Flux_rhov_j_ey = 0.5*(rho[1:-1,:-1]*(v[1:-1,:-1]**2)+rho[1:-1,1:]*(v[1:-1,1:]**2)+p[1:-1,:-1]+p[1:-1,1:])
        Flux_rhov_i_ex = 0.5*(rho[:-1,1:-1]*v[:-1,1:-1]*u[:-1,1:-1]+rho[1:,1:-1]*v[1:,1:-1]*u[1:,1:-1])
        Flux_rhov_i_ey = 0.5*(rho[:-1,1:-1]*(v[:-1,1:-1]**2)+rho[1:,1:-1]*(v[1:,1:-1]**2)+p[:-1,1:-1]+p[1:,1:-1])

        Flux_B_rhov = Flux_rhov_i_ex*(Delta_jy)-Flux_rhov_i_ey*(Delta_jx)
        Flux_D_rhov = Flux_rhov_j_ex*(Delta_iy)-Flux_rhov_j_ey*(Delta_ix)
        Flux_H_rhov = Flux_rhov_i_ex*(-Delta_jy)-Flux_rhov_i_ey*(-Delta_jx)
        Flux_G_rhov = Flux_rhov_j_ex*(-Delta_iy)-Flux_rhov_j_ey*(-Delta_ix)

        F_C_rhov = Flux_B_rhov[:-1,:]+Flux_D_rhov[:,1:]+Flux_H_rhov[1:,:]+Flux_G_rhov[:,:-1]

        # Flux of total energy (vectorized)
        Flux_rhoE_j_ex = 0.5*((rho[1:-1,:-1]*E[1:-1,:-1]+p[1:-1,:-1])*u[1:-1,:-1]+(rho[1:-1,1:]*E[1:-1,1:]+p[1:-1,1:])*u[1:-1,1:])
        Flux_rhoE_j_ey = 0.5*((rho[1:-1,:-1]*E[1:-1,:-1]+p[1:-1,:-1])*v[1:-1,:-1]+(rho[1:-1,1:]*E[1:-1,1:]+p[1:-1,1:])*v[1:-1,1:])
        Flux_rhoE_i_ex = 0.5*((rho[:-1,1:-1]*E[:-1,1:-1]+p[:-1,1:-1])*u[:-1,1:-1]+(rho[1:,1:-1]*E[1:,1:-1]+p[1:,1:-1])*u[1:,1:-1])
        Flux_rhoE_i_ey = 0.5*((rho[:-1,1:-1]*E[:-1,1:-1]+p[:-1,1:-1])*v[:-1,1:-1]+(rho[1:,1:-1]*E[1:,1:-1]+p[1:,1:-1])*v[1:,1:-1])

        Flux_B_rhoE = Flux_rhoE_i_ex*(Delta_jy)-Flux_rhoE_i_ey*(Delta_jx)
        Flux_D_rhoE = Flux_rhoE_j_ex*(Delta_iy)-Flux_rhoE_j_ey*(Delta_ix)
        Flux_H_rhoE = Flux_rhoE_i_ex*(-Delta_jy)-Flux_rhoE_i_ey*(-Delta_jx)
        Flux_G_rhoE = Flux_rhoE_j_ex*(-Delta_iy)-Flux_rhoE_j_ey*(-Delta_ix)

        F_C_rhoE = Flux_B_rhoE[:-1,:]+Flux_D_rhoE[:,1:]+Flux_H_rhoE[1:,:]+Flux_G_rhoE[:,:-1]

        return F_C_rho, F_C_rhou, F_C_rhov, F_C_rhoE
    
    def getArtificialDissipation(self, W, mesh):
        # Compute u, v,p, c variables in each cell of the extended mesh
        p = (GAMMA-1)*(W.rhoE - 0.5*(W.rhou**2 + W.rhov**2)/W.rho)
        u = W.rhou/W.rho
        v = W.rhov/W.rho
        c = (GAMMA*p/W.rho)**(0.5)

    
        lambda_ksi_vertex = 0.5*(np.abs(u[:,1:]*mesh.Delta_iy[:,1:-1]-v[:,1:]*mesh.Delta_ix[:,1:-1])+np.abs(u[:,:-1]*mesh.Delta_iy[:,1:-1]-v[:,:-1]*mesh.Delta_ix[:,1:-1])+(c[:,1:]+c[:,:-1])*((mesh.Delta_ix[:,1:-1]**2+mesh.Delta_iy[:,1:-1]**2)**0.5))
        lambda_eta_vertex = 0.5*(np.abs(u[1:,:]*mesh.Delta_jy[1:-1,:]-v[1:,:]*mesh.Delta_jx[1:-1,:])+np.abs(u[:-1,:]*mesh.Delta_jy[1:-1,:]-v[:-1,:]*mesh.Delta_jx[1:-1,:])+(c[1:,:]+c[:-1,:])*((mesh.Delta_jx[1:-1,:]**2+mesh.Delta_jy[1:-1,:]**2)**0.5))
        
        lambda_eta_vertex[:, 0] = lambda_eta_vertex[:,  3]
        lambda_eta_vertex[:, 1] = lambda_eta_vertex[:,  2]
        lambda_eta_vertex[:,-1] = lambda_eta_vertex[:,  0]
        lambda_eta_vertex[:,-2] = lambda_eta_vertex[:,  1]

        lambda_ksi_vertex[:, 0] = lambda_ksi_vertex[:, 2]
        lambda_ksi_vertex[:,-1] = lambda_ksi_vertex[:, 0]

        lambda_ksi_centered = 0.5*(lambda_ksi_vertex[:,:-1]+lambda_ksi_vertex[:,1:])
        lambda_eta_centered = 0.5*(lambda_eta_vertex[:-1,:]+lambda_eta_vertex[1:,:])

        D_lambda_ksi = 0.5*(lambda_ksi_centered[mesh.n_ghosts:-mesh.n_ghosts,:-1]+lambda_ksi_centered[mesh.n_ghosts:-mesh.n_ghosts,1:]+lambda_eta_centered[1:-1,1:-mesh.n_ghosts]+lambda_eta_centered[1:-1,mesh.n_ghosts:-1])
        D_lambda_eta = 0.5*(lambda_ksi_centered[1:-mesh.n_ghosts,1:-1]+lambda_ksi_centered[mesh.n_ghosts:-1,1:-1]+lambda_eta_centered[:-1,mesh.n_ghosts:-mesh.n_ghosts]+lambda_eta_centered[1:,mesh.n_ghosts:-mesh.n_ghosts])
        
        delta_rho_ksi  = W.rho [mesh.n_ghosts:-mesh.n_ghosts, 1:]-W.rho [mesh.n_ghosts:-mesh.n_ghosts, :-1] 
        delta_rhou_ksi = W.rhou[mesh.n_ghosts:-mesh.n_ghosts, 1:]-W.rhou[mesh.n_ghosts:-mesh.n_ghosts, :-1]
        delta_rhov_ksi = W.rhov[mesh.n_ghosts:-mesh.n_ghosts, 1:]-W.rhov[mesh.n_ghosts:-mesh.n_ghosts, :-1]
        delta_rhoE_ksi = W.rhoE[mesh.n_ghosts:-mesh.n_ghosts, 1:]-W.rhoE[mesh.n_ghosts:-mesh.n_ghosts, :-1]

        delta_rho_eta = W.rho  [1:, mesh.n_ghosts:-mesh.n_ghosts]-W.rho [:-1, mesh.n_ghosts:-mesh.n_ghosts]
        delta_rhou_eta = W.rhou[1:, mesh.n_ghosts:-mesh.n_ghosts]-W.rhou[:-1, mesh.n_ghosts:-mesh.n_ghosts]
        delta_rhov_eta = W.rhov[1:, mesh.n_ghosts:-mesh.n_ghosts]-W.rhov[:-1, mesh.n_ghosts:-mesh.n_ghosts]
        delta_rhoE_eta = W.rhoE[1:, mesh.n_ghosts:-mesh.n_ghosts]-W.rhoE[:-1, mesh.n_ghosts:-mesh.n_ghosts]

        # Connect condition 
        delta_rho_eta [0,:] = delta_rho_eta [mesh.n_ghosts,:]
        delta_rhou_eta[0,:] = delta_rhou_eta[mesh.n_ghosts,:]
        delta_rhov_eta[0,:] = delta_rhov_eta[mesh.n_ghosts,:]
        delta_rhoE_eta[0,:] = delta_rhoE_eta[mesh.n_ghosts,:]
        delta_rho_eta [1,:] = delta_rho_eta [mesh.n_ghosts,:]
        delta_rhou_eta[1,:] = delta_rhou_eta[mesh.n_ghosts,:]
        delta_rhov_eta[1,:] = delta_rhov_eta[mesh.n_ghosts,:]
        delta_rhoE_eta[1,:] = delta_rhoE_eta[mesh.n_ghosts,:]

        mat_nu_ksi = np.abs((p[2:-2,:-2]-2*p[2:-2,1:-1]+p[2:-2,2:])/(p[2:-2,:-2]+2*p[2:-2,1:-1]+p[2:-2,2:]))
        mat_nu_eta = np.abs((p[:-2,2:-2]-2*p[1:-1,2:-2]+p[2:,2:-2])/(p[:-2,2:-2]+2*p[1:-1,2:-2]+p[2:,2:-2]))
        ''' values of the first order epsilon matrices'''
        mat_e_ksi_2 = self.k2*np.maximum(0,np.maximum.reduce([mat_nu_ksi[:,1:],mat_nu_ksi[:,:-1]]))
        mat_e_eta_2 = self.k2*np.maximum(0,np.maximum.reduce([mat_nu_eta[1:,:],mat_nu_eta[:-1,:]]))
        mat_e_ksi_4 = np.maximum(0,(self.k4-mat_e_ksi_2))
        mat_e_eta_4 = np.maximum(0,(self.k4-mat_e_eta_2))

        # Computation of the dissipation fluxes for each conserved variables

        # Density 
        D_2_ksi = delta_rho_ksi[:,2:-1]*D_lambda_ksi[:,1:]*mat_e_ksi_2[:,1:]-delta_rho_ksi[:,1:-2]*D_lambda_ksi[:,:-1]*mat_e_ksi_2[:,:-1]
        D_2_eta = delta_rho_eta[2:-1,:]*D_lambda_eta[1:,:]*mat_e_eta_2[1:,:]-delta_rho_eta[1:-2,:]*D_lambda_eta[:-1,:]*mat_e_eta_2[:-1,:]
        D_4_ksi = ((delta_rho_ksi[:,3:]-2*delta_rho_ksi[:,2:-1]+delta_rho_ksi[:,1:-2])*D_lambda_ksi[:,1:]*mat_e_ksi_4[:,1:])-((delta_rho_ksi[:,2:-1]-2*delta_rho_ksi[:,1:-2]+delta_rho_ksi[:,:-3])*D_lambda_ksi[:,:-1]*mat_e_ksi_4[:,:-1])
        D_4_eta = ((delta_rho_eta[3:,:]-2*delta_rho_eta[2:-1,:]+delta_rho_eta[1:-2,:])*D_lambda_eta[1:,:]*mat_e_eta_4[1:,:])-((delta_rho_eta[2:-1,:]-2*delta_rho_eta[1:-2,:]+delta_rho_eta[:-3,:])*D_lambda_eta[:-1,:]*mat_e_eta_4[:-1,:])
        D_rho = -(D_2_ksi-D_4_ksi+D_2_eta-D_4_eta)

        # x-momentum
        D_2_ksi = delta_rhou_ksi[:,2:-1]*D_lambda_ksi[:,1:]*mat_e_ksi_2[:,1:]-delta_rhou_ksi[:,1:-2]*D_lambda_ksi[:,:-1]*mat_e_ksi_2[:,:-1]
        D_2_eta = delta_rhou_eta[2:-1,:]*D_lambda_eta[1:,:]*mat_e_eta_2[1:,:]-delta_rhou_eta[1:-2,:]*D_lambda_eta[:-1,:]*mat_e_eta_2[:-1,:]
        D_4_ksi = ((delta_rhou_ksi[:,3:]-2*delta_rhou_ksi[:,2:-1]+delta_rhou_ksi[:,1:-2])*D_lambda_ksi[:,1:]*mat_e_ksi_4[:,1:])-((delta_rhou_ksi[:,2:-1]-2*delta_rhou_ksi[:,1:-2]+delta_rhou_ksi[:,:-3])*D_lambda_ksi[:,:-1]*mat_e_ksi_4[:,:-1])
        D_4_eta = ((delta_rhou_eta[3:,:]-2*delta_rhou_eta[2:-1,:]+delta_rhou_eta[1:-2,:])*D_lambda_eta[1:,:]*mat_e_eta_4[1:,:])-((delta_rhou_eta[2:-1,:]-2*delta_rhou_eta[1:-2,:]+delta_rhou_eta[:-3,:])*D_lambda_eta[:-1,:]*mat_e_eta_4[:-1,:])
        D_rhou = -(D_2_ksi-D_4_ksi+D_2_eta-D_4_eta)

        # y-momentum
        D_2_ksi = delta_rhov_ksi[:,2:-1]*D_lambda_ksi[:,1:]*mat_e_ksi_2[:,1:]-delta_rhov_ksi[:,1:-2]*D_lambda_ksi[:,:-1]*mat_e_ksi_2[:,:-1]
        D_2_eta = delta_rhov_eta[2:-1,:]*D_lambda_eta[1:,:]*mat_e_eta_2[1:,:]-delta_rhov_eta[1:-2,:]*D_lambda_eta[:-1,:]*mat_e_eta_2[:-1,:]
        D_4_ksi = ((delta_rhov_ksi[:,3:]-2*delta_rhov_ksi[:,2:-1]+delta_rhov_ksi[:,1:-2])*D_lambda_ksi[:,1:]*mat_e_ksi_4[:,1:])-((delta_rhov_ksi[:,2:-1]-2*delta_rhov_ksi[:,1:-2]+delta_rhov_ksi[:,:-3])*D_lambda_ksi[:,:-1]*mat_e_ksi_4[:,:-1])
        D_4_eta = ((delta_rhov_eta[3:,:]-2*delta_rhov_eta[2:-1,:]+delta_rhov_eta[1:-2,:])*D_lambda_eta[1:,:]*mat_e_eta_4[1:,:])-((delta_rhov_eta[2:-1,:]-2*delta_rhov_eta[1:-2,:]+delta_rhov_eta[:-3,:])*D_lambda_eta[:-1,:]*mat_e_eta_4[:-1,:])
        D_rhov = -(D_2_ksi-D_4_ksi+D_2_eta-D_4_eta)

        # Total energy
        D_2_ksi = delta_rhoE_ksi[:,2:-1]*D_lambda_ksi[:,1:]*mat_e_ksi_2[:,1:]-delta_rhoE_ksi[:,1:-2]*D_lambda_ksi[:,:-1]*mat_e_ksi_2[:,:-1]
        D_2_eta = delta_rhoE_eta[2:-1,:]*D_lambda_eta[1:,:]*mat_e_eta_2[1:,:]-delta_rhoE_eta[1:-2,:]*D_lambda_eta[:-1,:]*mat_e_eta_2[:-1,:]
        D_4_ksi = ((delta_rhoE_ksi[:,3:]-2*delta_rhoE_ksi[:,2:-1]+delta_rhoE_ksi[:,1:-2])*D_lambda_ksi[:,1:]*mat_e_ksi_4[:,1:])-((delta_rhoE_ksi[:,2:-1]-2*delta_rhoE_ksi[:,1:-2]+delta_rhoE_ksi[:,:-3])*D_lambda_ksi[:,:-1]*mat_e_ksi_4[:,:-1])
        D_4_eta = ((delta_rhoE_eta[3:,:]-2*delta_rhoE_eta[2:-1,:]+delta_rhoE_eta[1:-2,:])*D_lambda_eta[1:,:]*mat_e_eta_4[1:,:])-((delta_rhoE_eta[2:-1,:]-2*delta_rhoE_eta[1:-2,:]+delta_rhoE_eta[:-3,:])*D_lambda_eta[:-1,:]*mat_e_eta_4[:-1,:])
        D_rhoE = -(D_2_ksi-D_4_ksi+D_2_eta-D_4_eta)

        return D_rho, D_rhou, D_rhov, D_rhoE

class ExplicitEuler():

    def __init__(self, CFL, mesh, local_time=True):
        self.CFL = CFL
        self.local_time = local_time
        self.dt = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj + 2*mesh.n_ghosts))

    def LocalTimeStep(self, W, mesh):
        p = (GAMMA-1) *(W.rhoE - 0.5 * (W.rhou**2+W.rhov**2)/W.rho)
        u = W.rhou/W.rho
        v = W.rhov/W.rho
        c = (GAMMA*p/W.rho)**0.5

        lambda_ksi_vertex = 0.5*(np.abs(u[1:-1,1:]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,1:]*mesh.Delta_ix[1:-1,1:-1])+np.abs(u[1:-1,:-1]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,:-1]*mesh.Delta_ix[1:-1,1:-1])+(c[1:-1,1:]+c[1:-1,:-1])*((mesh.Delta_ix[1:-1,1:-1]**2+mesh.Delta_iy[1:-1,1:-1]**2)**0.5))
        lambda_eta_vertex = 0.5*(np.abs(u[1:,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[1:,1:-1]*mesh.Delta_jx[1:-1,1:-1])+np.abs(u[:-1,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[:-1,1:-1]*mesh.Delta_jx[1:-1,1:-1])+(c[1:,1:-1]+c[:-1,1:-1])*((mesh.Delta_jx[1:-1,1:-1]**2+mesh.Delta_jy[1:-1,1:-1]**2)**0.5))
        
        lambda_ksi_centered = 0.5*(lambda_ksi_vertex[:,:-1]+lambda_ksi_vertex[:,1:])
        lambda_eta_centered = 0.5*(lambda_eta_vertex[:-1,:]+lambda_eta_vertex[1:,:])

        lambda_ksi_centered[:,  0] = lambda_ksi_centered[:,  1]
        lambda_ksi_centered[:, -1] = lambda_ksi_centered[:, -2]
        lambda_eta_centered[:,  0] = lambda_eta_centered[:,  1]
        lambda_eta_centered[:, -1] = lambda_eta_centered[:, -2]

        lambda_C = lambda_ksi_centered[1:-1,1:-1]+lambda_eta_centered[1:-1,1:-1]
        self.dt[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts] = self.CFL*mesh.areas[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts]/(lambda_C)
         
    def GlobalTimeStep(self, W, mesh):
        p = (GAMMA-1) *(W.rhoE - 0.5 * (W.rhou**2+W.rhov**2)/W.rho)
        u = W.rhou/W.rho
        v = W.rhov/W.rho
        c = (GAMMA*p/W.rho)**0.5

        lambda_ksi_vertex = 0.5*(np.abs(u[1:-1,1:]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,1:]*mesh.Delta_ix[1:-1,1:-1])+np.abs(u[1:-1,:-1]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,:-1]*mesh.Delta_ix[1:-1,1:-1])+(c[1:-1,1:]+c[1:-1,:-1])*((mesh.Delta_ix[1:-1,1:-1]**2+mesh.Delta_iy[1:-1,1:-1]**2)**0.5))
        lambda_eta_vertex = 0.5*(np.abs(u[1:,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[1:,1:-1]*mesh.Delta_jx[1:-1,1:-1])+np.abs(u[:-1,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[:-1,1:-1]*mesh.Delta_jx[1:-1,1:-1])+(c[1:,1:-1]+c[:-1,1:-1])*((mesh.Delta_jx[1:-1,1:-1]**2+mesh.Delta_jy[1:-1,1:-1]**2)**0.5))
        
        lambda_ksi_centered = 0.5*(lambda_ksi_vertex[:,:-1]+lambda_ksi_vertex[:,1:])
        lambda_eta_centered = 0.5*(lambda_eta_vertex[:-1,:]+lambda_eta_vertex[1:,:])

        lambda_ksi_centered[:,  0] = lambda_ksi_centered[:,  1]
        lambda_ksi_centered[:, -1] = lambda_ksi_centered[:, -2]
        lambda_eta_centered[:,  0] = lambda_eta_centered[:,  1]
        lambda_eta_centered[:, -1] = lambda_eta_centered[:, -2]

        lambda_C = lambda_ksi_centered[1:-1,1:-1]+lambda_eta_centered[1:-1,1:-1]
        dt = self.CFL*mesh.areas[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts]/(lambda_C)
        dt_min = dt.min()
        self.dt[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts] = dt_min


    def step(self, scheme, W, mesh, M_inf, AOA):
        if self.local_time:
            self.LocalTimeStep(W, mesh)
        else :
            self.GlobalTimeStep(W, mesh)
        W.rho  -= self.dt*scheme.res_rho
        W.rhou -= self.dt*scheme.res_rhou
        W.rhov -= self.dt*scheme.res_rhov
        W.rhoE -= self.dt*scheme.res_rhoE

class RK2():

    def __init__(self, CFL, mesh, local_time=True):
        self.CFL = CFL
        self.local_time = local_time
        self.dt = np.zeros((mesh.ni+2*mesh.n_ghosts, mesh.nj + 2*mesh.n_ghosts))

    def LocalTimeStep(self, W, mesh):
        p = (GAMMA-1) *(W.rhoE - 0.5 * (W.rhou**2+W.rhov**2)/W.rho)
        u = W.rhou/W.rho
        v = W.rhov/W.rho
        c = (GAMMA*p/W.rho)**0.5

        lambda_ksi_vertex = 0.5*(np.abs(u[1:-1,1:]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,1:]*mesh.Delta_ix[1:-1,1:-1])+np.abs(u[1:-1,:-1]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,:-1]*mesh.Delta_ix[1:-1,1:-1])+(c[1:-1,1:]+c[1:-1,:-1])*((mesh.Delta_ix[1:-1,1:-1]**2+mesh.Delta_iy[1:-1,1:-1]**2)**0.5))
        lambda_eta_vertex = 0.5*(np.abs(u[1:,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[1:,1:-1]*mesh.Delta_jx[1:-1,1:-1])+np.abs(u[:-1,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[:-1,1:-1]*mesh.Delta_jx[1:-1,1:-1])+(c[1:,1:-1]+c[:-1,1:-1])*((mesh.Delta_jx[1:-1,1:-1]**2+mesh.Delta_jy[1:-1,1:-1]**2)**0.5))
        
        lambda_ksi_centered = 0.5*(lambda_ksi_vertex[:,:-1]+lambda_ksi_vertex[:,1:])
        lambda_eta_centered = 0.5*(lambda_eta_vertex[:-1,:]+lambda_eta_vertex[1:,:])

        lambda_ksi_centered[:,  0] = lambda_ksi_centered[:,  1]
        lambda_ksi_centered[:, -1] = lambda_ksi_centered[:, -2]
        lambda_eta_centered[:,  0] = lambda_eta_centered[:,  1]
        lambda_eta_centered[:, -1] = lambda_eta_centered[:, -2]

        lambda_C = lambda_ksi_centered[1:-1,1:-1]+lambda_eta_centered[1:-1,1:-1]
        self.dt[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts] = self.CFL*mesh.areas[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts]/(lambda_C)
         
    def GlobalTimeStep(self, W, mesh):
        p = (GAMMA-1) *(W.rhoE - 0.5 * (W.rhou**2+W.rhov**2)/W.rho)
        u = W.rhou/W.rho
        v = W.rhov/W.rho
        c = (GAMMA*p/W.rho)**0.5

        lambda_ksi_vertex = 0.5*(np.abs(u[1:-1,1:]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,1:]*mesh.Delta_ix[1:-1,1:-1])+np.abs(u[1:-1,:-1]*mesh.Delta_iy[1:-1,1:-1]-v[1:-1,:-1]*mesh.Delta_ix[1:-1,1:-1])+(c[1:-1,1:]+c[1:-1,:-1])*((mesh.Delta_ix[1:-1,1:-1]**2+mesh.Delta_iy[1:-1,1:-1]**2)**0.5))
        lambda_eta_vertex = 0.5*(np.abs(u[1:,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[1:,1:-1]*mesh.Delta_jx[1:-1,1:-1])+np.abs(u[:-1,1:-1]*mesh.Delta_jy[1:-1,1:-1]-v[:-1,1:-1]*mesh.Delta_jx[1:-1,1:-1])+(c[1:,1:-1]+c[:-1,1:-1])*((mesh.Delta_jx[1:-1,1:-1]**2+mesh.Delta_jy[1:-1,1:-1]**2)**0.5))
        
        lambda_ksi_centered = 0.5*(lambda_ksi_vertex[:,:-1]+lambda_ksi_vertex[:,1:])
        lambda_eta_centered = 0.5*(lambda_eta_vertex[:-1,:]+lambda_eta_vertex[1:,:])

        lambda_ksi_centered[:,  0] = lambda_ksi_centered[:,  1]
        lambda_ksi_centered[:, -1] = lambda_ksi_centered[:, -2]
        lambda_eta_centered[:,  0] = lambda_eta_centered[:,  1]
        lambda_eta_centered[:, -1] = lambda_eta_centered[:, -2]

        lambda_C = lambda_ksi_centered[1:-1,1:-1]+lambda_eta_centered[1:-1,1:-1]
        dt = self.CFL*mesh.areas[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts]/(lambda_C)
        dt_min = dt.min()
        self.dt[mesh.n_ghosts:-mesh.n_ghosts,mesh.n_ghosts:-mesh.n_ghosts] = dt_min


    def step(self, scheme, W, mesh, M_inf, AOA):
        # First step, we compute a solution at tn+1/2
        W_half = copy.deepcopy(W)
        if self.local_time:
            self.LocalTimeStep(W, mesh)
        else :
            self.GlobalTimeStep(W, mesh)
        W_half.rho  -= self.dt/2*scheme.res_rho
        W_half.rhou -= self.dt/2*scheme.res_rhou
        W_half.rhov -= self.dt/2*scheme.res_rhov
        W_half.rhoE -= self.dt/2*scheme.res_rhoE

        W_half = apply_boundary_conditions(W_half, mesh, M_inf, AOA)

        # Then we use solution at tn+1/2 to compute residuals
        scheme.getResiduals(W_half, mesh)

        # With the obtained redsiduals we update W to tn+1

        W.rho  -= self.dt*scheme.res_rho
        W.rhou -= self.dt*scheme.res_rhou
        W.rhov -= self.dt*scheme.res_rhov
        W.rhoE -= self.dt*scheme.res_rhoE




                        



