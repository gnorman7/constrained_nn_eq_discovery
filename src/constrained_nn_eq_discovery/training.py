import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
import pysindy as ps
from constrained_nn_eq_discovery import dhpm
from constrained_nn_eq_discovery import con_opt

# More General Class for Training
# class DHPM_Trainer():
#     def __init__(self, model: nn.Module, chosen_method = 'vanilla'):
#         self.model = model
#         self.chosen_method = chosen_method
# add loss / residual tracking here

#     def train(self, xt_train: torch.Tensor, u_train: torch.Tensor, xt_f: torch.Tensor):
#         if self.chosen_method == 'vanilla':
#             train_dhpm(self.model, xt_train, u_train, xt_f)
#         else:
#             raise NotImplementedError
class Training_Tracker():
    """Handle tracking of training progress -- do not track every epoch, but only at certain intervals."""
    def __init__(self, N_f: int, epochs: int, verbosity: int):
        # max number of entries to record
        self.verbosity = verbosity
        self.max_record = 250 # the max is actually ~ double this, if epochs is not divisible by max_record
        self.total_epochs = epochs
        self.track_interval = epochs // self.max_record
        if self.track_interval == 0:
            self.track_interval = 1
        self.num_record = epochs // self.track_interval + 1
        self.epochs = torch.ones(self.num_record) * torch.nan
        self.residual_history = torch.zeros(self.num_record, N_f)
        self.mse_history = torch.zeros(self.num_record)
        self.idx = 0

    def track_epoch(self, epoch: int, residual: torch.Tensor, mse: torch.Tensor):
        # check if the epoch should be tracked
        if self.track_interval == 0 or (epoch % self.track_interval == 0) or (epoch == self.total_epochs - 1):
            self.epochs[self.idx] = epoch
            self.residual_history[self.idx] = residual.squeeze()
            self.mse_history[self.idx] = mse
            # print status
            if self.verbosity > 0:
                print(f'Epoch {epoch}/{self.total_epochs}, Loss N: {torch.mean(residual**2):.12f}, Loss u: {mse:.12f}')
            self.idx += 1
            return True
        return False

    def _remove_untracked(self):
        self.epochs = self.epochs[~torch.isnan(self.epochs)]
        self.residual_history = self.residual_history[:self.idx]
        self.mse_history = self.mse_history[:self.idx]

    def plot_training_history(self):
        self._remove_untracked()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        residual_mse = torch.mean(self.residual_history**2, dim=1)
        ax[0].semilogy(self.epochs, residual_mse)
        ax[0].set_title('N MSE')
        ax[1].semilogy(self.epochs, self.mse_history)
        ax[1].set_title('u MSE')
        plt.show()
        return fig, ax

def train_dhpm(model: dhpm.EqDiscoveryModel, xt_train: torch.Tensor, u_train: torch.Tensor, xt_f: torch.Tensor,
               training_config: dict = {}, lams=None):
    """
    Train the model using the specified optimizer.
    """
    default_config = {
        'epochs': 100,
        'learning_rate': 0.001,
        'lam': 1.0,
        'optimizer_type': 'Adam',
        'params': ['u', 'N', 'lams'],
        'verbosity': 1
    }

    config = {**default_config, **training_config}

    epochs = config['epochs']
    learning_rate = config['learning_rate']
    lam = config['lam']
    optimizer_type = config['optimizer_type']
    params_type = config['params']
    verbosity = config['verbosity']
    N_f = xt_f.shape[0]

    tracker = Training_Tracker(N_f, epochs, verbosity)

    params = []
    if 'u' in params_type:
        params.extend(model.u_dnn.parameters())
    if 'N' in params_type:
        params.extend(model.N_dnn.parameters())
    if 'lams' in params_type:
        if lams is None:
            lams = lam * torch.randn((N_f, 1), requires_grad=True, device=xt_f.device)
        lams = nn.Parameter(lams)
        params.extend([lams])
    else:
        if lams is None:
            lams = lam*torch.ones((N_f, 1), device=xt_f.device)

    if optimizer_type == 'LBFGS':
        optimizer = torch.optim.LBFGS(params, lr=learning_rate, history_size=500, line_search_fn='strong_wolfe')
    else:
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            residual = model.get_residual(xt_f)
            loss_u = model.mse(xt_train, u_train)
            loss_N = torch.mean((lams*residual)**2)
            loss = loss_u + loss_N
            loss.backward()
            if type(lams) is nn.Parameter:
                lams.grad *= -1.0
            return loss

        if optimizer_type == 'LBFGS':
            optimizer.step(closure)
            tracker.track_epoch(epoch, model.get_residual(xt_f).detach(), model.mse(xt_train, u_train).detach())
        else:
            optimizer.zero_grad()
            residual = model.get_residual(xt_f)
            loss_u = model.mse(xt_train, u_train)
            loss_N = torch.mean((lams*residual)**2)
            loss = loss_u + loss_N
            loss.backward()
            if type(lams) is nn.Parameter:
                lams.grad *= -1.0
            optimizer.step()
            tracker.track_epoch(epoch, residual.detach(), loss_u.detach())

    if 'lams' in params_type:
        return tracker, lams.data
    else:
        return tracker


def train_dhpm_constrained(model: dhpm.EqDiscoveryModel, xt_train: torch.Tensor, u_train: torch.Tensor, xt_f: torch.Tensor,
               training_config: dict = {}):
    """
    Train the model using the specified optimizer.
    """
    default_config = {
        'epochs': 100,
        'eps': 1e-3,
        'verbosity': 1,
        'minimizer_args': {}
    }

    config = {**default_config, **training_config}
    # add 'maxiter':epochs to minimizer_args, if not already present
    if 'maxiter' not in config['minimizer_args']:
        config['minimizer_args']['maxiter'] = config['epochs']
        epochs = config['epochs']
    else:
        epochs = config['minimizer_args']['maxiter']
    verbosity = config['verbosity']
    N_f = xt_f.shape[0]
    eps = config['eps']

    print(f'{config["minimizer_args"]}')

    def con_fn():
        return model.get_residual(xt_f)

    def obj_fn():
        return model.mse(xt_train, u_train)

    minimizer_args: dict = {'maxiter': epochs}
    tracker = Training_Tracker(N_f, epochs, verbosity)
    def callback(intermediate_result: OptimizeResult):
        epoch = intermediate_result.nit
        residual = con_fn().detach()
        with torch.no_grad():
            mse = obj_fn()
        tracker.track_epoch(epoch, residual, mse)

    print(f'Beginning constrained dhpm training with {epochs} epochs and eps = {eps}')
    optimizer = con_opt.TorchScipyOptimizer(model.parameters(), minimizer_args=minimizer_args, callback=callback)
    res = optimizer.step(obj_fn, con_fn, -eps, eps)

    return tracker, res

