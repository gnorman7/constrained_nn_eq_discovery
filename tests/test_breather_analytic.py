from constrained_nn_eq_discovery import utils
from constrained_nn_eq_discovery import numerics
import numpy as np
import torch
import pytest

@pytest.fixture
def grid():
    n_x = 100
    n_t = 2000
    x = torch.linspace(-10, 10, n_x)
    t = torch.linspace(0, 10, n_t)
    X, T = torch.meshgrid(x, t, indexing='ij')
    return x, t, X, T

def test_breather_d_dt(grid):
    _, _, X, T = grid
    U_true = utils.breather(X, T)
    Ut = utils.d_dt_breather(X, T)
    U_fd = (utils.breather(X, T + 1e-3) - U_true) / 1e-3

    assert np.allclose(Ut, U_fd, atol=5e-3), "Breather d/dt failed finite difference test"

# parameterize with w = 2 pi / 10, and 2 pi / 10 + 0.1
@pytest.mark.parametrize("w", [2 * np.pi / 10, 2 * np.pi / 10 + 0.1])
def test_numerical_solution_of_sine_gordon(grid, w):
    x, t, X, T = grid
    U_true = utils.breather(X, T, w=w)

    uv0 = utils.breather_ic(x, w=w)
    U = numerics.solve_pde_2nd_in_time(uv0, x, t, utils.sine_gordon).T

    l2_error = utils.l2_error(U_true, U)

    assert np.allclose(U_true, U, atol=3e-2), "Failed Point-wise solution of Sine-Gordon on Breather IC"
    # l2 error should be 0.00470 or 0.0038
    assert l2_error < 5e-3, f"Failed L2 error of Sine-Gordon on Breather IC: \nL2 error {l2_error}"


