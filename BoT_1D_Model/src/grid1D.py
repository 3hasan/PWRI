import numpy as np

class Grid1D_FVM:
    def __init__(self, dx):
        self.dx = np.array(dx)
        self.n_cells = self.dx.size

        self.x_interfaces = np.concatenate(([0], np.cumsum(self.dx)))
        self.n_interfaces = self.x_interfaces.size
        self.x_centers = (self.x_interfaces[:-1] + self.x_interfaces[1:]) / 2.0

    def calculate_interface_properties(self, porosity, permeability, viscosity, core_area):

        #POR
        self.porosity_interfaces = np.zeros(self.n_interfaces) 
        self.porosity_interfaces[0]  = porosity[0]
        self.porosity_interfaces[-1] = porosity[-1]
        for i in range(1, self.n_interfaces - 1):
            self.porosity_interfaces[i] = (self.dx[i - 1] * porosity[i - 1] + self.dx[i] * porosity[i]) / (self.dx[i - 1] + self.dx[i])

        #PERM
        self.permeability_interfaces = np.zeros(self.n_interfaces)
        self.permeability_interfaces[0]  = permeability[0]
        self.permeability_interfaces[-1] = permeability[-1]
        for i in range(1, self.n_interfaces - 1):
            self.permeability_interfaces[i] = (self.dx[i - 1] + self.dx[i]) / (
                (self.dx[i - 1] / permeability[i - 1]) + 
                (self.dx[i]     / permeability[i])
            )

        #TRANSMISSIBILITIES
        self.transmissibility = np.zeros(self.n_interfaces)
        for i in range(self.n_interfaces):
            if i == 0:
                dx_local = self.dx[0] / 2
            elif i == self.n_interfaces - 1:
                dx_local = self.dx[-1] / 2
            else:
                dx_local = self.x_centers[i] - self.x_centers[i - 1]

            self.transmissibility[i] = (
                core_area * self.permeability_interfaces[i] /
                (viscosity * dx_local)
            )