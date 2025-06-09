import pytest
from inputs.parameters import Parameters

# Tests if the assertions in the parameters work as they should when run "pytest -q"
# Those check if sum of dx and dy are equal to length and height. Also checks if the permeability and porosity shape match dx for now in 1D


def test_parameters_init_succeeds():
    params = Parameters()
    assert isinstance(params, Parameters)
