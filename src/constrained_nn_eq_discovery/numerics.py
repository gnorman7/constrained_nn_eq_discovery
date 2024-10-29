import torch
from typing import Callable

#%% Simple Schemes
def rk4(f, y0, t):
    y = torch.zeros(len(t), *y0.shape, device=y0.device)
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt / 2, y[i] + dt / 2 * k1)
        k3 = f(t[i] + dt / 2, y[i] + dt / 2 * k2)
        k4 = f(t[i] + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

def midpoint(f, y0, t):
    y = torch.zeros(len(t), *y0.shape, device=y0.device)
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt / 2, y[i] + dt / 2 * k1)
        y[i + 1] = y[i] + dt * k2
    return y

def leapfrog(f, y0, t):
    """y0 is (n, 2), f: (n, 2) -> (n, 2)"""
    y = torch.zeros(len(t), *y0.shape, device=y0.device)
    y[0] = y0 # both x and v
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        # x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
        xi = y[i, :, 0]
        vi = y[i, :, 1]
        xvi = torch.stack([xi, vi], dim=-1)
        ai = f(None, xvi)[:, 1]
        xip1 = xi + vi * dt + 0.5 * ai * dt**2
        # v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        xvip1 = torch.stack([xip1, vi], dim=-1)
        aip1 = f(None, xvip1)[:, 1] # this doesn't actually depend on vi
        vip1 = vi + 0.5 * (ai + aip1) * dt
        y[i + 1] = torch.stack([xip1, vip1], dim=-1)
    return y

def fd_centered(u_tensor, x, dim=0):
    dx = x[1] - x[0]

    # Create slices for the forward and backward differences, this is the : operator
    forward_slice = [slice(None)] * u_tensor.ndim
    backward_slice = [slice(None)] * u_tensor.ndim

    # Keep everything else as :, but set the dim to 2: and :-2
    forward_slice[dim] = slice(2, None)
    backward_slice[dim] = slice(None, -2)

    # Calculate the centered difference
    u_x = (u_tensor[tuple(forward_slice)] - u_tensor[tuple(backward_slice)]) / (2 * dx)

    return u_x

def fd_centered_2nd(u_tensor, x, dim=0):
    # See fd_centered
    dx = x[1] - x[0]

    forward_slice = [slice(None)] * u_tensor.ndim
    center_slice = [slice(None)] * u_tensor.ndim
    backward_slice = [slice(None)] * u_tensor.ndim

    forward_slice[dim] = slice(2, None)
    center_slice[dim] = slice(1, -1)
    backward_slice[dim] = slice(None, -2)

    u_xx = (u_tensor[tuple(forward_slice)] - 2 * u_tensor[tuple(center_slice)] +
            u_tensor[tuple(backward_slice)]) / (dx**2)

    return u_xx

def fd_centered_5pt(u_tensor, x, dim=0):
    dx = x[1] - x[0]

    fwd2 = [slice(None)] * u_tensor.ndim
    fwd1 = [slice(None)] * u_tensor.ndim
    bwd1 = [slice(None)] * u_tensor.ndim
    bwd2 = [slice(None)] * u_tensor.ndim
    center = [slice(None)] * u_tensor.ndim

    fwd2[dim] = slice(4, None)
    fwd1[dim] = slice(3, -1)
    center[dim ] = slice(2, -2)
    bwd1[dim] = slice(1, -3)
    bwd2[dim] = slice(None, -4)

    u_x = (u_tensor[tuple(bwd2)] - 8 * u_tensor[tuple(bwd1)] + 8 * u_tensor[tuple(fwd1)] - u_tensor[tuple(fwd2)]) / (12 * dx)
    return u_x

def fd_centered_5pt_2nd(u_tensor, x, dim=0):
    dx = x[1] - x[0]

    fwd1 = [slice(None)] * u_tensor.ndim
    fwd2 = [slice(None)] * u_tensor.ndim
    bwd1 = [slice(None)] * u_tensor.ndim
    bwd2 = [slice(None)] * u_tensor.ndim
    center = [slice(None)] * u_tensor.ndim

    fwd2[dim] = slice(4, None)
    fwd1[dim] = slice(3, -1)
    center[dim] = slice(2, -2)
    bwd1[dim] = slice(1, -3)
    bwd2[dim] = slice(None, -4)

    u_xx = (-u_tensor[tuple(fwd2)] + 16 * u_tensor[tuple(fwd1)] - 30 * u_tensor[tuple(center)] + 16 * u_tensor[tuple(bwd1)] - u_tensor[tuple(bwd2)]) / (12 * dx**2)
    return u_xx


#%% Method of Lines Discretization
def ode_2nd_in_time(x_vec: torch.Tensor, u_and_v_int: torch.Tensor,
              N: Callable[[torch.Tensor], torch.Tensor]):
    """
    u_int: (n_x, 2): [state, d/dt state]
    """
    u_and_v_vec = torch.zeros(u_and_v_int.shape[0] + 2, u_and_v_int.shape[1],
                              device=u_and_v_int.device)
    u_and_v_vec[1:-1] = u_and_v_int
    u_vec = u_and_v_vec[:, 0]

    u_x = fd_centered(u_vec, x_vec, dim=0)
    u_xx = fd_centered_2nd(u_vec, x_vec, dim=0)
    big_u = torch.stack([u_and_v_int[:,0], u_x, u_xx], dim=-1)
    u_tt = N(big_u)
    d_dt_u_and_v = torch.cat([u_and_v_int[:, 1:2], u_tt], dim=1)
    return d_dt_u_and_v

def ode_2d(x_vec: torch.Tensor, y_vec: torch.Tensor, U_int: torch.Tensor,
           N: Callable[[torch.Tensor], torch.Tensor]):
    """Use the method to construct F(Y) = dY/dt (Y is a matrix), enforcing Dirichlet BCs, NO BATCHING.

    Args:
        x_vec: (n_x,)
        y_vec: (n_y,)
        U_int: (n_x - 2, n_y - 2)
        N: function mapping (n, 6) -> (n, 1). [u, u_x, u_xx, u_y, u_yy, u_xy] -> u_t

    Return:
        U_t: (n_x - 2, n_y - 2), d/dt U_int
    """
    bcs_x = torch.zeros(1, U_int.shape[1], device=U_int.device)
    bcs_y = torch.zeros(U_int.shape[0] + 2, 1, device=U_int.device)
    U_vec_bcs = torch.cat([bcs_x, U_int, bcs_x], dim=0)
    U_vec_bcs = torch.cat([bcs_y, U_vec_bcs, bcs_y], dim=1)

    U_x = fd_centered(U_vec_bcs, x_vec, dim=0)
    U_xx = fd_centered_2nd(U_vec_bcs, x_vec, dim=0)
    U_y = fd_centered(U_vec_bcs, y_vec, dim=1)
    U_yy = fd_centered_2nd(U_vec_bcs, y_vec, dim=1)
    U_xy = fd_centered(U_x, y_vec, dim=1)
    huge_U = torch.stack([U_int,
                        U_x[:, 1:-1],
                        U_xx[:, 1:-1],
                        U_y[1:-1, :],
                        U_yy[1:-1, :],
                        U_xy], dim=-1)
    huge_u = huge_U.reshape(-1, 6)
    u_t = N(huge_u)
    U_t = u_t.reshape(U_int.shape)
    return U_t


def ode_2d_5pt(x_vec: torch.Tensor, y_vec: torch.Tensor, U_int: torch.Tensor,
           N: Callable[[torch.Tensor], torch.Tensor]):
    """Use the method to construct F(Y) = dY/dt (Y is a matrix), enforcing Dirichlet BCs, NO BATCHING.

    Args:
        x_vec: (n_x,)
        y_vec: (n_y,)
        U_int: (n_x - 4, n_y - 4)
        N: function mapping (n, 6) -> (n, 1). [u, u_x, u_xx, u_y, u_yy, u_xy] -> u_t

    Return:
        U_t: (n_x - 4, n_y - 4), d/dt U_int
    """
    bcs_x = torch.zeros(2, U_int.shape[1], device=U_int.device)
    bcs_y = torch.zeros(U_int.shape[0] + 4, 2, device=U_int.device)
    U_vec_bcs = torch.cat([bcs_x, U_int, bcs_x], dim=0)
    U_vec_bcs = torch.cat([bcs_y, U_vec_bcs, bcs_y], dim=1)

    U_x = fd_centered_5pt(U_vec_bcs, x_vec, dim=0)
    U_xx = fd_centered_5pt_2nd(U_vec_bcs, x_vec, dim=0)
    U_y = fd_centered_5pt(U_vec_bcs, y_vec, dim=1)
    U_yy = fd_centered_5pt_2nd(U_vec_bcs, y_vec, dim=1)
    U_xy = fd_centered_5pt(U_x, y_vec, dim=1)
    huge_U = torch.stack([U_int,
                        U_x[:, 2:-2],
                        U_xx[:, 2:-2],
                        U_y[2:-2, :],
                        U_yy[2:-2, :],
                        U_xy], dim=-1)
    huge_u = huge_U.reshape(-1, 6)
    u_t = N(huge_u)
    U_t = u_t.reshape(U_int.shape)
    return U_t


#%% Full PDE Solve
def solve_pde_2nd_in_time(uv0: torch.Tensor, x_vec: torch.Tensor, t_vec: torch.Tensor,
                    N: Callable[[torch.Tensor], torch.Tensor],
                    integration_scheme: Callable[
                        [Callable, torch.Tensor, torch.Tensor], torch.Tensor
                    ] = leapfrog):
    U = torch.zeros((len(t_vec), len(x_vec)), device=uv0.device)
    uv0_int = uv0[1:-1]

    integrated_fun = lambda t, uv: ode_2nd_in_time(x_vec, uv, N)
    u_and_v_int = integration_scheme(integrated_fun, uv0_int, t_vec)

    U[:, 1:-1] = u_and_v_int[:, :, 0]
    return U


def solve_pde_2d(U0: torch.Tensor, x_vec: torch.Tensor, y_vec: torch.Tensor,
                t_vec: torch.Tensor, N: Callable[[torch.Tensor], torch.Tensor],
                integration_scheme: Callable[
                    [Callable, torch.Tensor, torch.Tensor], torch.Tensor
                ] = midpoint):
    """Solve a single PDE in 2D, on rectangular domain with zero dirichlet BCs
    N = N(u, u_x, u_xx, u_y, u_yy, u_xy) -> u_t.

    Args:
        U0: (n_x, n_y), initial condition
        x_vec: (n_x,), x mesh, includes BCs (dirichlet zero)
        y_vec: (n_y,), y mesh, includes BCs (dirichlet zero)
        t_vec: (n_t,), time mesh
        N: batched dynamics mapping (n, 6) -> (n, 1), ordered as [u, u_x, u_xx, u_y, u_yy, u_xy] -> u_t
        integration_scheme: function that integrates f(t,y) = dy/dt, takes in f, y0, t_vec. Outputs with time first.

    Returns:
        U: (n_t, n_x, n_y), solution to the PDE
    """
    assert integration_scheme != leapfrog, "Attempting to use leapfrog for a PDE with one time derivative."

    U = torch.zeros((len(t_vec), len(x_vec), len(y_vec)), device=U0.device)

    integrated_fun = lambda t, U: ode_2d(x_vec, y_vec, U, N)

    U[:, 1:-1, 1:-1] = integration_scheme(integrated_fun, U0[1:-1, 1:-1], t_vec)
    return U

def solve_pde_2d_5pt(U0: torch.Tensor, x_vec: torch.Tensor, y_vec: torch.Tensor,
                     t_vec: torch.Tensor, N: Callable[[torch.Tensor], torch.Tensor],
                        integration_scheme: Callable[
                            [Callable, torch.Tensor, torch.Tensor], torch.Tensor
                        ] = rk4):

    U = torch.zeros((len(t_vec), len(x_vec), len(y_vec)), device=U0.device)

    integrated_fun = lambda t, U: ode_2d_5pt(x_vec, y_vec, U, N)

    U[:, 2:-2, 2:-2] = integration_scheme(integrated_fun, U0[2:-2, 2:-2], t_vec)
    return U

