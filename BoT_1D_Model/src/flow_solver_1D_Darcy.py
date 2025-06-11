import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from inputs.parameters import Parameters
from src.grid_1D_FVM import grid_1D_FVM

class FlowSolver1D:

    def __init__(self, params: Parameters, grid: grid_1D_FVM):
        self.params = params
        self.grid   = grid
        self.N      = self.grid.n_cells

    def solve_pressure(self, bc_type=0, source_term=None):
        """      
        BC Type
                        0 => Dirichlet-Dirichlet
                        1 => Neumann-Dirichlet
                        2 => Robin-Dirichlet
        """ 
                # FVM
        # T[i]*(P[i-1] - P[i]) - T[i+1]*(P[i] - P[i+1]) = q[i]*dx[i]

        dx = self.grid.dx
        T  = self.grid.transmissibility         
        q  = source_term if source_term is not None else np.zeros(self.N)   # can define source term instead of np.zeros

        # Assemble system A P = b
        A = lil_matrix((self.N, self.N))
        b = np.zeros(self.N)

        # Internal cells
        for i in range(1, self.N - 1):
            A[i, i - 1] = -T[i]
            A[i, i    ] =  T[i] + T[i + 1]
            A[i, i + 1] = -T[i + 1]
            b[i]        =  q[i] * dx[i]

        # LEFT boundary
        if bc_type == 0:
            # Dirichlet on left
            P_left = self.params.P_left
            A[0, 0] = T[0] + T[1]
            A[0, 1] = -T[1]
            b[0]    = q[0] * dx[0] + T[0] * P_left

        elif bc_type == 1:
            # Neumann on left (flux = injection_rate)
            # => left_flux - T[i+1]*(P[0]-P[1]) = q[0]*dx[0]
            flux = self.params.Q_injection
            A[0, 0] =  T[1]
            A[0, 1] = -T[1]
            b[0]    = q[0] * dx[0] + flux

        elif bc_type == 2:
            # Robin – Dirichlet
            #  -T0 (P0 - P_env) + T1 (P1 - P0) = q0 Δx0
            T0     = self.grid.transmissibility[0]     # 2 k / Δx  (already computed)
            T1     = self.grid.transmissibility[1]
            P_env  = self.params.P_left                # ambient pressure you set in test
            A[0,0] = T0 + T1
            A[0,1] = -T1
            b[0]   = q[0]*dx[0] + T0 * P_env


        else:
            raise ValueError("bc_type must be 0, 1, or 2")

        # RIGHT boundary (always Dirichlet)
        i = self.N - 1
        A[i, i - 1] = -T[i]
        A[i, i] =  T[i] + T[i + 1]
        b[i] =  q[i] * dx[i] + T[i + 1] * self.params.P_right

        # Solve using sparse LU
        A_csc = A.tocsc()
        lu = splu(A_csc)
        P = lu.solve(b)
        return P


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load parameters and build grid
    params = Parameters()
    grid = grid_1D_FVM(params.dx)

    grid.calculate_interface_properties(
        porosity    = params.porosity,
        permeability= params.permeability,
        viscosity   = params.viscosity_brine,
        core_area   = params.core_area
    )

    # Solve and plot
    solver = FlowSolver1D(params, grid)
    P = solver.solve_pressure(bc_type=1)

    plt.figure(figsize=(6,4))
    plt.plot(grid.x_centers, P, '-o')
    plt.xlabel("Distance (cm)")
    plt.ylabel("Pressure (mbar)")
    plt.title("1D Pressure Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
