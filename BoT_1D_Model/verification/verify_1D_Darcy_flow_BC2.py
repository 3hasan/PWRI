import numpy as np
import matplotlib.pyplot as plt

from inputs.parameters        import Parameters
from src.grid_1D_FVM          import grid_1D_FVM
from src.flow_solver_1D_Darcy import FlowSolver1D

# Manufactured solution setup 
L              = 1.0
k_const        = 2.0
por_const      = 0.3
visc, coreA    = 1.0, 1.0
P_env, P_right = 2.0, 0.5        

#  Grid sizes 
grid_sizes = [   10,  20,  40,  80, 160, 320 ]

dx_vals, L2_errs = [], []

for N in grid_sizes:
    dx_arr   = np.full(N, L/N)
    grid     = grid_1D_FVM(dx_arr)

    # Set parameters
    params              = Parameters()
    params.P_left       = P_env
    params.P_right      = P_right
    params.permeability = np.full(N, k_const)
    params.porosity     = np.full(N, por_const)

    grid.calculate_interface_properties(
        porosity     = params.porosity,
        permeability = params.permeability,
        viscosity    = visc,
        core_area    = coreA,
    )

    # Manufactured exact linear profile
    a       = (P_right - P_env) / L
    b       = P_env
    P_exact = a * grid.x_centers + b

    # Solve
    solver   = FlowSolver1D(params, grid)
    P_num    = solver.solve_pressure(bc_type=2, source_term=np.zeros(N))

    # L2 error
    err      = np.sqrt(np.mean((P_num - P_exact)**2))
    dx_vals.append(dx_arr[0])
    L2_errs.append(err)

    # Plot finest grid profile
    if N == grid_sizes[-1]:
        plt.figure(figsize=(6,4))
        plt.plot(grid.x_centers, P_exact, 'g-',  label='Exact')
        plt.plot(grid.x_centers, P_num,   'r--', label='Numeric')
        plt.xlabel('x (cm)'); plt.ylabel('Pressure (mbar)')
        plt.title(f'BC 2 profile (N={N})')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ── Convergence plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.loglog(dx_vals, L2_errs, 'o-', linewidth=2)
plt.xlabel('Δx (cm)'); plt.ylabel('L₂ error')
plt.title('BC 2 grid-refinement (Robin–Dirichlet)')
plt.grid(True, which='both', linestyle='--')
plt.tight_layout(); plt.show()

print('\nGrid   Δx        L2-error')
for N, dx, e in zip(grid_sizes, dx_vals, L2_errs):
    print(f'{N:4d}  {dx:9.4e}  {e:9.3e}')
