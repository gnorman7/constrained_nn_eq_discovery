import torch
import numpy as np
import scipy.io as sio

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

