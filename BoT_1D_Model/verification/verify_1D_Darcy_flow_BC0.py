import numpy as np
import matplotlib.pyplot as plt

from inputs.parameters            import Parameters
from src.grid_1D_FVM              import grid_1D_FVM   
from src.flow_solver_1D_Darcy     import FlowSolver1D  

# Manufactured solution setup

L       = 1.0
P_inlet = 1.0
P_exit  = 0.0
a       = 0.1
n       = 3

def PERM_manufactured(x):
    return 2 + (x / L)**2

def P_manufactured(x):
    return P_inlet + (P_exit - P_inlet) * (x / L) + a * np.sin(n * np.pi * x / L)

def Source_manufactured(x):
    dPdx   = (P_exit - P_inlet) / L + a * n * np.pi / L * np.cos(n * np.pi * x / L)
    d2Pdx2 = -a * (n * np.pi / L)**2 * np.sin(n * np.pi * x / L)
    dkdx   = 2 * x / L**2
    return -(dkdx * dPdx + PERM_manufactured(x) * d2Pdx2)

def compute_L2_error(P_num, P_exact):
    return np.sqrt(np.mean((P_num - P_exact)**2))

# Grid refinement 

grid_sizes = [10, 100, 1_000, 10_000, 100_000]
dx_list    = []
L2_errors  = []

for N in grid_sizes:
    dx = L / N
    dx_array = np.full(N, dx)

    grid      = grid_1D_FVM(dx_array)
    x_centers = grid.x_centers
    q         = Source_manufactured(x_centers)

    # Overriding parameters to refine the grid with each run
    params = Parameters()
    params.P_left       = P_inlet
    params.P_right      = P_exit
    params.permeability = PERM_manufactured(x_centers)
    params.porosity     = np.repeat(0.3, N)

    grid.calculate_interface_properties(
        porosity    = params.porosity,
        permeability= params.permeability,  # Update interface properties using overridden permeability
        viscosity   = 1.0,  # Viscosity and core_area are taken as 1.0 to match the manufactured solution
        core_area   = 1.0   # As the manufactured solution had those simplified by taking them as 1.0
    )

    # Call Solver to solve with the input parameters and grid values
    solver       = FlowSolver1D(params, grid)
    P_numerical  = solver.solve_pressure(bc_type=0, source_term=q)

    # Calculate L2 Erro
    error        = compute_L2_error(P_numerical, P_manufactured(x_centers))
    dx_list.append(dx)
    L2_errors.append(error)

    # For the finest grid, compare profiles
    if N == grid_sizes[-1]:
        plt.figure(figsize=(8, 5))
        plt.plot(x_centers, P_manufactured(x_centers), 'g-', label='Manufactured')
        plt.plot(x_centers, P_numerical,    'r--', label='Numerical')
        plt.xlabel("Position $x$ (cm)")
        plt.ylabel("Pressure $P$ (mbar)")
        plt.title(f"P(x) Comparison at N = {N}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Convergence plot
plt.figure(figsize=(8, 5))
plt.loglog(dx_list, L2_errors, 'o-', linewidth=2)
plt.xlabel("Grid cell size $\Delta x$ (cm)")
plt.ylabel("L₂ Error")
plt.title("Grid Refinement Study: Flow Solver Verification (BC Type 0)")
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()
