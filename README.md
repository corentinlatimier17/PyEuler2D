Examples

The examples/ folder provides pre-configured simulation cases to test the solver. The cases include:

    Flow regimes: Subsonic, Transonic, and Supersonic.
    Two angles of attack: 0° and 1.25°.

For each case, you can find the input files and configuration details necessary to reproduce the results. These examples serve as templates for setting up your own simulations.
Boundary Conditions

    Inflow and Outflow: Implemented using Riemann invariants for both subsonic and supersonic flow conditions.
    Wall Boundary: A no-slip condition is implemented to simulate the effects of a solid wall on the flow field.

Numerical Schemes

The solver includes the following numerical methods:

    Convective Fluxes: Central scheme with artificial dissipation.
    Time Integration:
        Explicit Euler time integration scheme.
        Runge-Kutta 2nd order (RK2) integration scheme.
    Time Stepping:
        Global time step for all grid cells.
        Local time step for increased efficiency in capturing local flow dynamics.

Code Structure

    grid/ : Scripts for generating the O-type structured grid.
    solver/ : Core CFD solver implementation.
    postprocess/ : Tools for visualizing and analyzing simulation results.
    examples/ : Pre-configured cases for testing the solver.
