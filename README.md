# **Examples**  
The `examples/` folder provides simulation results of the solver. The cases include:  

- **Flow regimes**:  
  - Subsonic \(M_{\infty} = 0.5\)  
  - Transonic \(M = 0.8\) 
  - Supersonic \(M = 1.5\)  
- **Angles of Attack**:  
  - 0°
  - 1.25°

# **Boundary Conditions**  
- **Inflow and Outflow**: Implemented using Riemann invariants for both subsonic and supersonic flow conditions.  
- **Wall Boundary**: A no-slip condition is implemented to simulate the effects of a solid wall on the flow field.  

# **Numerical Schemes**  
The solver includes the following numerical methods:  

- **Convective Fluxes**: Central scheme with artificial dissipation.  
- **Time Integration**:  
  - Explicit Euler time integration scheme.  
  - Runge-Kutta 2nd order (RK2) integration scheme.  
- **Time Stepping**:  
  - Global time step for all grid cells.  
  - Local time step for increased efficiency (acceleration technique).  

# **Code Structure**  
- `NACA0012grids/`: Contains all the grids used.  
- `modules/`: Source files of the solver (Python).  
- `examples/`: Verification results for NACA0012 airfoil.  
- `main.py`: Main script to modify and adapt the simulation to your specific case.  
