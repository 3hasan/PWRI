import numpy as np
import pytest
from src.grid1D import Grid1D_FVM

# Tests if grid works properly on a dummy grid with 3 cells of length dx

# dx is given as [0.5, 0.8, 1.2] so, x_interfaces should be [0, 0.5, 1.3, 2.5] x_centers should be [0.25, 0.9, 1.9]
# 3 porosity and 3 permeability values are defined with area and permeability, so POR,PER,Transmissibilities at the interfaces should be calculated correctly
# Boundaries have the cell porosity while Intermediate interfaces use a weighted arithmetic average while permeability is harmonic averaged
# At the boundary cells # dx_local = dx[0]/2 and dx_local = dx[-1]/2 
# last assertions check if the values for POR,PERM,Transmissibilities are correct by comparing calculation of expected values to the grid


def test_interface_and_center_positions():
    dx = np.array([0.5, 0.8, 1.2])
    grid = Grid1D_FVM(dx)
    expected_interfaces = np.array([0.0, 0.5, 1.3, 2.5])
    expected_centers    = np.array([0.25, 0.9, 1.9])

    assert np.allclose(grid.x_interfaces, expected_interfaces), (
        f"Interface positions incorrect. Got {grid.x_interfaces}, "
        f"expected {expected_interfaces}"
    )

    assert np.allclose(grid.x_centers, expected_centers), (
        f"Cell‐center positions incorrect. Got {grid.x_centers}, "
        f"expected {expected_centers}"
    )

def test_cell_value_counts_agree_with_dx_length():
    dx = np.array([0.5, 0.8, 1.2])
    n_cells = len(dx)

    porosity     = np.array([0.2, 0.3, 0.4])
    permeability = np.array([10.0, 20.0, 40.0])

    assert len(porosity) == n_cells, (
        f"Porosity length mismatch. Got {len(porosity)}, expected {n_cells}"
    )
    assert len(permeability) == n_cells, (
        f"Permeability length mismatch. Got {len(permeability)}, expected {n_cells}"
    )


def test_interface_value_counts():
    dx = np.array([0.5, 0.8, 1.2])
    porosity     = np.array([0.2, 0.3, 0.4])
    permeability = np.array([10.0, 20.0, 40.0])
    viscosity    = 1.0
    core_area    = 1.0

    grid = Grid1D_FVM(dx)
    grid.calculate_interface_properties(porosity, permeability, viscosity, core_area)

    expected_n_interfaces = len(dx) + 1

    assert grid.porosity_interfaces.shape[0] == expected_n_interfaces, (
        f"Interface porosity count mismatch. Got {grid.porosity_interfaces.shape[0]}, "
        f"expected {expected_n_interfaces}"
    )
    assert grid.permeability_interfaces.shape[0] == expected_n_interfaces, (
        f"Interface permeability count mismatch. Got {grid.permeability_interfaces.shape[0]}, "
        f"expected {expected_n_interfaces}"
    )
    assert grid.transmissibility.shape[0] == expected_n_interfaces, (
        f"Transmissibility count mismatch. Got {grid.transmissibility.shape[0]}, "
        f"expected {expected_n_interfaces}"
    )


def test_interface_values_correctness():
    dx = np.array([0.5, 0.8, 1.2])
    porosity     = np.array([0.2, 0.3, 0.4])
    permeability = np.array([10.0, 20.0, 40.0])
    viscosity    = 1.0
    core_area    = 1.0

    grid = Grid1D_FVM(dx)
    grid.calculate_interface_properties(porosity, permeability, viscosity, core_area)
    
    expected_porosity_interfaces = np.array([
        porosity[0],
        (dx[0]*porosity[0] + dx[1]*porosity[1]) / (dx[0] + dx[1]),
        (dx[1]*porosity[1] + dx[2]*porosity[2]) / (dx[1] + dx[2]),
        porosity[-1]
    ])

    
    def harmonic_avg(k1, k2, dx1, dx2):
        return (dx1 + dx2) / ((dx1 / k1) + (dx2 / k2))

    expected_perm_interfaces = np.array([
        permeability[0],
        harmonic_avg(permeability[0], permeability[1], dx[0], dx[1]),
        harmonic_avg(permeability[1], permeability[2], dx[1], dx[2]),
        permeability[-1]
    ])


    expected_centers = grid.x_centers.copy()
    d01 = expected_centers[1] - expected_centers[0]
    d12 = expected_centers[2] - expected_centers[1]
    expected_dx_locals = np.array([
        dx[0] / 2.0,
        d01,
        d12,
        dx[-1] / 2.0
    ])

    expected_transmissibility = np.zeros(len(expected_perm_interfaces))
    for i in range(len(expected_perm_interfaces)):
        expected_transmissibility[i] = (
            core_area * expected_perm_interfaces[i]
            / (viscosity * expected_dx_locals[i])
        )

    # Assert porosity_interfaces
    assert np.allclose(grid.porosity_interfaces, expected_porosity_interfaces), (
        f"Porosity interface values incorrect.\n"
        f" Got: {grid.porosity_interfaces}\n"
        f"Expected: {expected_porosity_interfaces}"
    )

    # Assert permeability_interfaces
    assert np.allclose(grid.permeability_interfaces, expected_perm_interfaces), (
        f"Permeability interface values incorrect.\n"
        f" Got: {grid.permeability_interfaces}\n"
        f"Expected: {expected_perm_interfaces}"
    )

    # Assert transmissibility
    assert np.allclose(grid.transmissibility, expected_transmissibility), (
        f"Transmissibility values incorrect.\n"
        f" Got: {grid.transmissibility}\n"
        f"Expected: {expected_transmissibility}"
    )
