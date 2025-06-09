import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcx
from scipy.sparse import diags
from scipy.sparse.linalg import splu

start_time = time.time()

'Analytical Solution Function'

# from  Fetter et al. (2018), Contaminant Hydrogeology, p.77-78 2.9.3 One-Dimensional Step Change in Concentration(First-Type Boundary)
# C= C_injection/2 * [erfc((L-v_x*t)/(2*sqrt(D_L*t)) + exp (v_x*L/D_L) * erfc((L+v_x*t)/(2*sqrt(D_L_t))] = C_inj/2 * erfc(minus_term) + exp (v_x*L/D_L) * erfc(plus_term)

# Notice erfc(plus term) is scaled with erfcx to avoid overflow, so there is a -plus_term**2 in the exponential. 
# erfcx(x) = exp(x**2) * erfc(x)

def fetter_analytical_solution(L, t, C_injection, v_x, D_L):
    if t <= 0:
        return 0.0

    minus_term = (L - v_x * t) / (2 * np.sqrt(D_L * t))
    plus_term = (L + v_x * t) / (2 * np.sqrt(D_L * t))

    erfc_minus = erfc(minus_term)
    scaled_second_term = np.exp((v_x * L) / D_L - plus_term**2) * erfcx(plus_term)
    
    return 0.5 * C_injection * (erfc_minus + scaled_second_term)

'Numerical Solution (FVM) Function'

def fvm_numerical_solution(n_cells=1000, n_steps=1000, L_total=5.0, t_total=1.0, C_inj=20.0, D=0.1):
    dx = L_total / n_cells
    dt = t_total / n_steps
    x = np.linspace(dx/2.0, L_total - dx/2.0, n_cells)

    v = 5 / (5.23 * 0.3)  
    r = dt/dx

    # Internal Cells
    main_diag = np.full(n_cells, 1 + r*v + 2*r*D/dx)
    lower_diag = np.full(n_cells - 1, -r*v -r*D/dx)
    upper_diag = np.full(n_cells - 1, -r*D/dx)

    # Boundary Conditions
    main_diag[0] = 1 + r*v + 3*r*D/dx
    upper_diag[0] = -r*D/dx

    main_diag[-1] = 1 + r*v + 3*r*D/dx 
    lower_diag[-1] = - r*v - r*D/dx

    A_sparse = diags([lower_diag, main_diag, upper_diag],
                     offsets=[-1,0,1],
                     shape=(n_cells,n_cells),
                     format='csc')


    # Building and Solving the Linear System
    C_old = np.zeros (n_cells)
    A_factorized = splu(A_sparse)

    for step in range(n_steps):
        C_RBC = fetter_analytical_solution(L=L_total, t=(step+1)*dt, C_injection=C_inj, v_x=v, D_L=D)
    
        b = C_old.copy()

        b[0] += C_inj*(r*v + 2*r*D/dx)
        b[-1] += 2*r*D*C_RBC/dx

        C_new = A_factorized.solve(b)
        C_old = C_new.copy()

    C_analytical = fetter_analytical_solution(L=x, t=t_total, C_injection=C_inj, v_x=v, D_L=D)
    error = np.sqrt(np.sum(((C_new - C_analytical)**2)*dx))

    return x, dx, dt, C_new, C_analytical, error

'Constants'

# Experimental conditions
L_total = 5.0
t_total = 1.0
C_inj = 20.0

# Fixed Conditions for Error Analysis
n_steps_fixed = 1000000
n_cells_fixed = 1000000
D_fixed = 0.1

'Refinements'

# Grid Cell Refinement

n_cells_list  = [50 * 2**i for i in range(15)]
dx_values = []
errors_spatial = []

for n_cells in n_cells_list:
    x, dx, dt_val, C_numerical, C_analytical, err = fvm_numerical_solution(
        n_cells=n_cells, n_steps=n_steps_fixed, L_total=L_total, t_total=t_total, C_inj=C_inj, D=D_fixed)
    dx_values.append(dx)
    errors_spatial.append(err)


# Time Step Refinement

n_time_steps_list = [10 * 2**i for i in range(15)] 
dt_values = []
errors_temporal = []

for n_steps in n_time_steps_list:
    x_temp, dx_temp, dt_temp, C_numerical_temp, C_analytical_temp, err_temp = fvm_numerical_solution(
        n_cells=n_cells_fixed, n_steps=n_steps, L_total=L_total, t_total=t_total, C_inj=C_inj, D=D_fixed)
    dt_values.append(dt_temp)
    errors_temporal.append(err_temp)

# Dispersion Refinement

n_cells_D = int(n_cells_fixed / 100)
n_steps_D = int(n_steps_fixed / 100)
D_list    = [0.1 * 10**(-1*i) for i in range(6)]

dx_D_values = []
dt_D_values = []
errors_D = []

for D_val in D_list:
    x_D, dx_D, dt_D, C_numerical_D, C_analytical_D, err_D = fvm_numerical_solution(
        n_cells=n_cells_D, n_steps=n_steps_D, L_total=L_total, t_total=t_total, C_inj=C_inj, D=D_val)
    dx_D_values.append(dx_D)
    dt_D_values.append(dt_D)
    errors_D.append(err_D)

'Data Tables'

grid_refinement_table = pd.DataFrame({
    'n_cells': n_cells_list,
    'dx': dx_values,
    'error': errors_spatial,
    'dt (constant)': [t_total/n_steps_fixed] * len(n_cells_list),
    'D (constant)': [D_fixed] * len(n_cells_list)
})

temporal_refinement_table = pd.DataFrame({
    'n_steps': n_time_steps_list,
    'dt': dt_values,
    'error': errors_temporal,
    'dx (constant)': [L_total/n_cells_fixed] * len(n_time_steps_list),
    'D (constant)': [D_fixed] * len(n_time_steps_list)
})

D_refinement_table = pd.DataFrame({
    'D': D_list,
    'dx': dx_D_values,
    'dt': dt_D_values,
    'error': errors_D,
    'n_cells (constant)': [n_cells_D] * len(D_list),
    'n_steps (constant)': [n_steps_D] * len(D_list)
})

grid_refinement_table.to_csv("grid_refinement_table.csv", index=False)
temporal_refinement_table.to_csv("temporal_refinement_table.csv", index=False)
D_refinement_table.to_csv("D_refinement_table.csv", index=False)

'Plots'

# Numerical vs Analytical Solution (using most refined grid cell)
plt.figure()
plt.plot(x, C_numerical, 'r--', label='Numerical')
plt.plot(x, C_analytical, 'g-', label='Analytical')
plt.xlabel('x')
plt.ylabel('Concentration')
plt.title('Numerical vs Analytical Solution (t_total=%.1f, D=%.2f)' % (t_total, D_fixed))
plt.legend()
plt.grid(True)
plt.savefig("numerical_vs_analytical_solution.png", dpi=300)
plt.show()

# Grid Refinement (Error vs. dx)
plt.figure()
plt.loglog(dx_values, errors_spatial, 'o--')
plt.xlabel('dx')
plt.ylabel('Error')
plt.title('Grid Refinement: Error vs dx (dt=%.1e, D=%.2f)' % (t_total/n_steps_fixed, D_fixed))
plt.grid(True, which='both')
plt.savefig("grid_refinement_error_vs_dx.png", dpi=300)
plt.show()

# Temporal Refinement (Error vs. dt)
plt.figure()
plt.loglog(dt_values, errors_temporal, 'o--')
plt.xlabel('dt')
plt.ylabel('Error')
plt.title('Temporal Refinement: Error vs dt (dx=%.1e, D=%.2f)' % (L_total/n_cells_fixed, D_fixed))
plt.grid(True, which='both')
plt.savefig("temporal_refinement_error_vs_dt.png", dpi=300)
plt.show()

# D Refinement (Error vs. D)
plt.figure()
plt.loglog(D_list, errors_D, 'o--')
plt.xlabel('D')
plt.ylabel('Error')
plt.title('D Refinement: Error vs D (dx=%.1e, dt=%.1e)' % (dx_D_values[0], dt_D_values[0]))
plt.grid(True, which='both')
plt.savefig("D_refinement_error_vs_D.png", dpi=300)
plt.show()

# Print Elapsed Time

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.3f} seconds".format(elapsed_time))