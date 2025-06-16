import torch
import torch.nn as nn
import numpy as np
from typing import Callable
from collections import OrderedDict
from ast import Raise



class DNN(nn.Module):
    def __init__(self, layers: list[int], activation: nn.Module = nn.Tanh()):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        if activation == 'sin':
            self.activation = Sine()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = activation

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layerDict)

        # # Xavier Normal Initialization
        # for layer in self.layers:
        #     if (type(layer) == nn.modules.linear.Linear):
        #         # torch.nn.init.xavier_uniform_(layer.weight)
        #         nn.init.xavier_normal_(layer.weight)
        #         print('Xavier Normal Initialization!')
        # Can just use the default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


# %% SIREN
# Taken from
# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=JMOfAQiuA0_J
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


# %% With derivatives
def get_big_u_and_utt(model: Callable[[torch.Tensor], torch.Tensor], xt_f: torch.Tensor, x_order: int = 2) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model: computes u at collocation points, using torch functions
        xt_f: tensor of collocation points, (N_f, 2)
        x_order: 1, 2, or 3, depending on how many x derivatives to take (default 2)

    Returns:
        big_u: tensor of u, u_x, u_xx, etc at collocation points, (N_f, 1 + x_order)
        u_tt: tensor of u_tt at collocation points, (N_f, 1)
    """

    assert xt_f.requires_grad
    assert x_order in [1, 2, 3]

    u = model(xt_f)
    u_xt = torch.autograd.grad(
        u, xt_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = u_xt[:, 0:1]
    u_t = u_xt[:, 1:2]
    u_tt = torch.autograd.grad(u_t, xt_f, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0][:, 1:2]
    if x_order >= 2:
        u_xx = torch.autograd.grad(u_x, xt_f, grad_outputs=torch.ones_like(
            u_x), create_graph=True)[0][:, 0:1]
        if x_order == 3:
            u_xxx = torch.autograd.grad(u_xx, xt_f, grad_outputs=torch.ones_like(
                u_xx), create_graph=True)[0][:, 0:1]
            big_u = torch.cat([u, u_x, u_xx, u_xxx], dim=1)
        else:
            big_u = torch.cat([u, u_x, u_xx], dim=1)
    else:
        big_u = torch.cat([u, u_x], dim=1)

    return big_u, u_tt


def get_big_u_and_ut_2d(model: Callable[[torch.Tensor], torch.Tensor],
                        xyt_f: torch.Tensor) -> tuple[
                            torch.Tensor, torch.Tensor]:
    assert xyt_f.requires_grad

    u = model(xyt_f)
    u_xyt = torch.autograd.grad(
        u, xyt_f, torch.ones_like(u), create_graph=True
    )[0]
    u_x = u_xyt[:, 0:1]
    u_y = u_xyt[:, 1:2]
    u_t = u_xyt[:, 2:3]
    u_x_xyt = torch.autograd.grad(
        u_x, xyt_f, torch.ones_like(u_x), create_graph=True
    )[0]
    u_xx = u_x_xyt[:, 0:1]
    u_xy = u_x_xyt[:, 1:2]
    # u_xt = u_x_xyt[:, 2:3] #unused
    u_y_xyt = torch.autograd.grad(
        u_y, xyt_f, torch.ones_like(u_y), create_graph=True
    )[0]
    u_yy = u_y_xyt[:, 1:2]

    big_u = torch.cat([u, u_x, u_xx, u_y, u_yy, u_xy], dim=1)
    return big_u, u_t


class EqDiscoveryModel(nn.Module):
    def __init__(self, u_dnn: nn.Module, N_dnn: nn.Module, is_2d: bool = False):
        super(EqDiscoveryModel, self).__init__()
        """Careful, this assumes u_tt = N(...) if 1D! I.e. don't use it with Burgers or KdV"""
        self.u_dnn = u_dnn
        self.N_dnn = N_dnn
        if is_2d:
            self.big_u_and_ut_fun = get_big_u_and_ut_2d
        else:
            self.big_u_and_ut_fun = get_big_u_and_utt

    def get_residual(self, xt_f: torch.Tensor):
        """Returns [N_f, 1] tensor of residuals at collocation points"""
        big_u, u_t = self.big_u_and_ut_fun(self.u_dnn, xt_f)
        N_eval = self.N_dnn(big_u)
        assert N_eval.shape == u_t.shape
        residual = u_t - N_eval
        return residual

    def mse(self, xt_train: torch.Tensor, u_train: torch.Tensor):
        """u_train should be [N_u, 1]"""
        u_pred = self.u_dnn(xt_train)
        # make sure the shapes are correct
        assert u_pred.shape == u_train.shape
        return torch.mean((u_pred - u_train)**2)

    def u_on_meshgrid(self, T: torch.Tensor, X: torch.Tensor, Y: torch.Tensor = None):
        """Computes u on a meshgrid of T, X"""
        if Y is not None:
            xt = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=1)
        else:
            xt = torch.stack((X.flatten(), T.flatten()), dim=1)
        u = self.u_dnn(xt)
        return u.reshape(T.shape)

    def forward(self, big_u: torch.Tensor):
        return self.N_dnn(big_u)


def simple_get_big_u_and_ut(model: Callable[[torch.Tensor], torch.Tensor],
                     xt_f: torch.Tensor,
                     x_order: int = 2,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model: computes u at collocation points, using torch functions
        xt_f: tensor of collocation points, (N_f, 2)
        x_order: 1, 2, or 3, depending on how many x derivatives to take (default 2)

    Returns:
        A tuple with two tensors:
            big_u: tensor of u, u_x, u_xx, etc at collocation points, (N_f, 1 + x_order)
            u_t: tensor of u_t at collocation points, (N_f, 1), or u_tt if is_2nd_time

    Note, this is not part of the EqDiscoveryModel class, as it can be equally
    used by the true dynamics (PINN) by prescribing model as the true dynamics.
    """

    assert xt_f.requires_grad
    assert x_order in [1, 2, 3]

    u = model(xt_f)
    u_xt = torch.autograd.grad(
        u, xt_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = u_xt[:, 0:1]
    u_t = u_xt[:, 1:2]
    if x_order >= 2:
        u_xx = torch.autograd.grad(u_x, xt_f,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0:1]
        if x_order == 3:
            u_xxx = torch.autograd.grad(u_xx, xt_f,
                                        grad_outputs=torch.ones_like(u_xx),
                                        create_graph=True)[0][:, 0:1]
            big_u = torch.cat([u, u_x, u_xx, u_xxx], dim=1)
        else:
            big_u = torch.cat([u, u_x, u_xx], dim=1)
    else:
        big_u = torch.cat([u, u_x], dim=1)

    return big_u, u_t

class SimpleEqDiscoveryModel(nn.Module):
    def __init__(self, u_dnn: nn.Module, N_dnn: nn.Module, x_order: int = 2):
        super(SimpleEqDiscoveryModel, self).__init__()
        self.u_dnn = u_dnn
        self.N_dnn = N_dnn
        self.x_order = x_order

    def get_residual(self, xt_f: torch.Tensor):
        """Returns [N_f, 1] tensor of residuals at collocation points"""
        big_u, u_t = simple_get_big_u_and_ut(self.u_dnn, xt_f, x_order=self.x_order)
        N_eval = self.N_dnn(big_u)
        assert N_eval.shape == u_t.shape
        residual = u_t - N_eval
        return residual

    def mse(self, xt_train: torch.Tensor, u_train: torch.Tensor):
        """u_train should be [N_u, 1]"""
        u_pred = self.u_dnn(xt_train)
        # make sure the shapes are correct
        assert u_pred.shape == u_train.shape
        return torch.mean((u_pred - u_train)**2)

    def u_on_meshgrid(self, T: torch.Tensor, X: torch.Tensor, Y: torch.Tensor = None):
        """Computes u on a meshgrid of T, X"""
        if Y is not None:
            xt = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=1)
        else:
            xt = torch.stack((X.flatten(), T.flatten()), dim=1)
        u = self.u_dnn(xt)
        return u.reshape(T.shape)

    def forward(self, big_u: torch.Tensor):
        return self.N_dnn(big_u)
