import numpy as np
import matplotlib.pyplot as plt

from inputs.parameters          import Parameters
from src.grid_1D_FVM            import grid_1D_FVM   
from src.flow_solver_1D_Darcy   import FlowSolver1D 

# Manufactured solution setup 

L            = 1.0              
Q_injection  = 2.0              
P_exit       = 1.0              
A_sin        = 0.05
alpha        = 10 * np.pi

def k_manufactured(x):
    return 2 + x**2

def P_manufactured(x):
    """
    P(x)= A_sin*sin(alpha*x) + B*x + C
    with B= -1 - A_sin*alpha, C= 2+ A_sin*alpha (from derivation)
    for Q_in=2 => B + A_sin*alpha= - Q_in/2= -1
    => B= -1 - A_sin*alpha
    => C= P_right - B*L= 1- B= ...
    """
    B = -Q_injection/2.0 - A_sin*alpha
    C = P_exit - B*L
    return A_sin * np.sin(alpha*x) + B*x + C

def Source_manufactured(x):
    # q(x)= - d/dx( k(x)*P'(x) ), with k(x)= 2 + x^2.
    # P'(x)= A_sin*alpha*cos(alpha*x)+ B
    B     = -Q_injection/2.0 - A_sin*alpha
    dPdx  = A_sin*alpha*np.cos(alpha*x) + B

    # d( k(x)* dPdx )/dx
    # = k'(x)*dPdx + k(x)* d2Pdx2
    # k'(x)= 2x
    # d2Pdx2= - A_sin*(alpha^2)* sin(alpha*x)

    d2Pdx2= -A_sin*(alpha**2)*np.sin(alpha*x)
    dkdx  = 2*x
    return -(dkdx * dPdx + k_manufactured(x) * d2Pdx2)

def compute_L2_error(P_num, P_exact):
    return np.sqrt(np.mean((P_num - P_exact)**2))

# Grid refinement study

grid_sizes = [10, 100, 1_000, 10_000, 100_000]
dx_list    = []
L2_errors  = []

for N in grid_sizes:
    dx_array = np.full(N, L / N)
    grid     = grid_1D_FVM(dx_array)
    x_centers= grid.x_centers
    q        = Source_manufactured(x_centers)

    # Override parameters for this test
    params                = Parameters()
    params.Q_injection    = Q_injection
    params.P_right        = P_exit
    params.permeability   = k_manufactured(x_centers)
    params.porosity       = np.repeat(0.3, N)

    # Compute transmissibilities with mu=1.0 and A=1.0
    grid.calculate_interface_properties(
        porosity    = params.porosity,
        permeability= params.permeability,
        viscosity   = 1.0,
        core_area   = 1.0
    )

    # Solve and measure error
    solver      = FlowSolver1D(params, grid)
    P_numerical = solver.solve_pressure(bc_type=1, source_term=q)
    P_exact     = P_manufactured(x_centers)
    error       = compute_L2_error(P_numerical, P_exact)

    dx_list.append(dx_array[0])
    L2_errors.append(error)

    # Compare profiles on finest grid
    if N == grid_sizes[-1]:
        plt.figure(figsize=(8,5))
        plt.plot(x_centers, P_exact,    'g-',  label='Manufactured')
        plt.plot(x_centers, P_numerical,'r--', label='Numerical')
        plt.xlabel("Position $x$ (cm)")
        plt.ylabel("Pressure $P$ (mbar)")
        plt.title(f"P(x) Comparison at N = {N}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Convergence plot
plt.figure(figsize=(8,5))
plt.loglog(dx_list, L2_errors, 'o-', linewidth=2)
plt.xlabel("Grid cell size $\Delta x$ (cm)")
plt.ylabel("L₂ Error")
plt.title("Grid Refinement Study: Flow Solver Verification (BC Type 1)")
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()