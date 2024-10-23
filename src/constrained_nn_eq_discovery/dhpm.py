import torch
import torch.nn as nn
from typing import Callable
from collections import OrderedDict


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


class EqDiscoveryModel(nn.Module):
    def __init__(self, u_dnn: nn.Module, N_dnn: nn.Module):
        super(EqDiscoveryModel, self).__init__()
        self.u_dnn = u_dnn
        self.N_dnn = N_dnn
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
