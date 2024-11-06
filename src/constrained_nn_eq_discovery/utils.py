import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def l2_error(U_true, U_pred):
    """U_true, U_pred"""
    return torch.sqrt(torch.mean((U_true - U_pred)**2)) / torch.sqrt(torch.mean(U_true**2))

def get_data(U_true, X, T, noise_level=0.0, ss_x=1, ss_t=1):
    U_std = torch.std(U_true)
    noise_mag = noise_level * U_std
    U_full = U_true + noise_mag * torch.randn_like(U_true)
    U = U_full[::ss_x, ::ss_t]
    return U, X[::ss_x, ::ss_t], T[::ss_x, ::ss_t]

def save_data(U_data, X_data, T_data, identifier_str):
    """Saves to this directory, but so that it can be read by matlab"""
    # make into one .mat file
    to_np = lambda x : x.to(torch.float64).numpy()
    data_dict = {'U_data': to_np(U_data), 'X_data': to_np(X_data), 'T_data': to_np(T_data)}
    sio.savemat(f'{identifier_str}.mat', data_dict)


def sine_gordon(big_u):
    u = big_u[:, 0:1]
    uxx = big_u[:, 2:3]
    return -torch.sin(u) + uxx


def breather(x, t, w=2 * np.pi / 10):
    """
    Breather solution for the Sine-Gordon equation
    u(x,t) = 4 * arctan((sqrt(1 - w**2) * cos(w * t)) / (w * cosh(sqrt(1 - w**2) * x)))
    """
    # Ensure w is valid for the equation (0 < w < 1)
    # Compute terms
    sqrt_term = np.sqrt(1 - w**2)
    numerator = sqrt_term * torch.cos(w * t)
    denominator = w * torch.cosh(sqrt_term * x)

    # Final computation of the function
    u = 4 * torch.arctan(numerator / denominator)
    return u


def d_dt_breather(x, t, w=2 * np.pi / 10):
    sqrt_term = np.sqrt(1 - w**2)
    old_numerator = sqrt_term * torch.cos(w * t)
    old_denominator = w * torch.cosh(sqrt_term * x)
    old_atan_input = old_numerator / old_denominator

    datan_dinput = 1 / (1 + old_atan_input**2)
    numerator = -sqrt_term * w * torch.sin(w * t)
    dinput_dt = numerator / old_denominator

    return 4 * datan_dinput * dinput_dt


def breather_ic(x, w = 2 * np.pi / 10):
    u0 = breather(x, torch.tensor(0), w = w)
    v0 = d_dt_breather(x, torch.tensor(0), w = w)
    return torch.stack([u0, v0], dim=1)


def custom_ic(x):
    # arbitrary IC with 0 at BCs, use Gaussian too.
    u0 = 2.0 * torch.exp(-x**2 / 2)
    v0 = torch.zeros_like(x)
    return torch.stack([u0, v0], dim=1)


def porous_analytic_ic(X, Y, T=None):
    """Compute initial condition based on the analytical weak solution."""

    # Diffusivity tensor and its inverse
    D = torch.tensor([[0.3, -0.4], [-0.4, 1]], dtype=torch.float32)
    Di = torch.inverse(D) # not the best, but likely insignificant compared to other errors

    # Constant C, calculated using determinant of D
    C = (1 / 5 * torch.pi * 8 * torch.sqrt(torch.det(D)))**(-0.5)

    # Time at which we want the solution
    if T is None:
        T_f = torch.tensor(0.5, dtype=torch.float32)
    else:
        T_f = T + 0.5

    # x^T D^-1 x term
    num_term = (X * (Di[0, 0] * X + Di[0, 1] * Y) + Y * (Di[1, 0] * X + Di[1, 1] * Y))

    # Argument for max: C - (x^T D^-1 x) / (16 sqrt(t))
    max_arg = C - (num_term / (16 * torch.sqrt(T_f)))
    # max_arg = C - (num_term / (16 * t_f))

    # Return the weak solution: 1 / sqrt(t) * max(...)
    return (1 / torch.sqrt(T_f)) * torch.max(max_arg, torch.zeros_like(max_arg))


def visualize_2d_sol(sol, U_pred, n_times_to_plot=5, just_pred=False, cmap='inferno'):
    # set white color
    if cmap == 'inferno' or cmap == 'plasma' or cmap == 'magma':
        cf_kws = {'colors': '#FFFFFF50', 'levels': 7}
    else:
        cf_kws = {'colors': '#00000050', 'levels': 7}
    # cf_kws = {'colors' : '#00000050', 'levels' : 8}
    vs = {'vmax': float(sol.U.max()), 'vmin': float(sol.U.min())}
    contour_levels = np.linspace(vs['vmin'], vs['vmax'], cf_kws['levels'])
    norm = plt.Normalize(**vs)

    times = np.linspace(sol.t[0], sol.t[-1], n_times_to_plot)
    # keep close together, make x and y same size
    if just_pred:
        U_to_plot = [U_pred]
    else:
        U_to_plot = [sol.U, U_pred]
    fig, ax = plt.subplots(len(U_to_plot), n_times_to_plot, figsize=(12, 3 * len(U_to_plot)),
                           sharex=True, sharey=True, constrained_layout=True, dpi=300, squeeze=False)

    for row, U_pred_row in enumerate(U_to_plot):
        for i, time in enumerate(times):
            idx = np.argmin(np.abs(sol.t - time))
            ax[row, i].pcolor(sol.X[idx, :, :], sol.Y[idx, :, :], U_pred_row[idx, :, :], cmap=cmap, norm=norm)
            ax[row, i].set_title(f't = {time:.2f}')
            # contour plot, but only show the lines (no fill)
            ax[row, i].contour(sol.X[idx, :, :], sol.Y[idx, :, :], U_pred_row[idx, :, :],
                               levels=contour_levels, colors=cf_kws['colors'])
            # ax[i].set_xlabel('$x$')
            # ax[i].set_ylabel('$y$')
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])
        # set a colorbar for all three, just to the right of the last one
    cbar = fig.colorbar(ax[0, i].collections[0], ax=ax, orientation='vertical')
    cbar.set_label('$u(x,y,t)$')

    ax[-1, 0].set_ylabel('Predicted')
    if not just_pred:
        ax[0, 0].set_ylabel('True')
