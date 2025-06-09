import numpy as np

class Parameters:
    """ Stores default settings """

    def __init__(self):
        # Core
        self.core_area   = 5.23   # cm²
        self.core_length = 5.0    # cm
        self.core_diameter = 2.58 # cm

        # Viscosity
        self.viscosity_brine = 0.89 # cp
        self.viscosity_oil   = 2.65 # cp

        # Grid Lengths
        self.dx = np.repeat([0.01, 0.02, 0.005], [100, 100, 400])
        self.dy = np.repeat([0.01], [258])

        # Cell‐centered porosity and permeability (must match dx length)
        self.porosity = np.repeat([0.3, 0.35],
                                  [self.dx.size//2, self.dx.size//2])
        
        self.permeability = np.repeat([1.0, 5.0, 0.5],
            [self.dx.size//3, self.dx.size//3, self.dx.size//3])  # Darcy

        # Time‐stepping
        self.t_total = 2.0     # seconds
        self.dt      = 0.01    # seconds
        self.n_steps = int(self.t_total / self.dt)

        # Boundary Values 
        self.Q_injection        = 5.0   # mL/min
        self.C_injection           = 20.0  # ppm
        self.D = 0.1  # mm²/sec
        self.P_left   = 1.0 # milibar
        self.P_right  = 0.0 # milibar
        self.P_top    = 1.0 # milibar  
        self.P_bottom = 0.0 # milibar

        # ──────────── Assertions ────────────
        total_length = self.dx.sum()
        assert abs(total_length - self.core_length) < 1e-8, (
            f"sum(dx) = {total_length:.6f} ≠ core_length = {self.core_length}"
        )

        total_height = self.dy.sum()
        assert abs(total_height - self.core_diameter) < 1e-8, (
            f"sum(dy) = {total_height:.6f} ≠ core_diameter = {self.core_diameter}"
        )

        assert self.porosity.shape[0] == self.dx.shape[0], (
            f"porosity length = {self.porosity.shape[0]} ≠ dx length = {self.dx.shape[0]}"
        )
        assert self.permeability.shape[0] == self.dx.shape[0], (
            f"permeability length = {self.permeability.shape[0]} ≠ dx length = {self.dx.shape[0]}"
        )

