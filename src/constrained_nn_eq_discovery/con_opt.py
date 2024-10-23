import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Iterator, Union
import scipy.optimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

def ravel_pack(tensors):
    # Faster version of nn.utils.parameters_to_vec, modified slightly from
    # https://github.com/gngdb/pytorch-minimize/blob/master/pytorch_minimize/optim.py
    def numpyify(tensor):
        if tensor is None:
            return np.array([0.0])
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    x = np.concatenate([numpyify(tensor).ravel() for tensor in tensors], 0)
    return x

def ravel_pack_float64(tensors):
    x = ravel_pack(tensors)
    # cast to float64
    x = x.astype(np.float64)
    return x

def torch_to_np(x: Union[torch.Tensor, np.array, float]):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


class TorchScipyOptimizer(torch.optim.Optimizer):
    """
    Takes PyTorch stuff for constrained optimization and gives it to scipy.optimize.minimize
    Altogether REPLACES the traditional training loop. Extends Torch Optimizer class.
    Most of the computational cost will be dealing with gradient things: propagating via AD,
    grabbing the parameter grad attributes, flattening to vector, then converting to numpy.

    Should support using a single GPU in typical PyTorch fashion, but the constant switching
    to GPU for PyTorch and back to CPU for Numpy is quite expensive. Maybe worth it for very
    expensive PyTorch functions that greatly benefit from GPU acceleration.

    Like https://github.com/gngdb/pytorch-minimize/blob/master/pytorch_minimize/optim.py#L99,
    but for my needs (constrained, one method).
    """

    def __init__(self, parameters: Iterator[torch.Tensor],
                 minimizer_args: dict = {'maxiter': 100},
                 callback: Callable[[OptimizeResult], None] = lambda *args: None):
        """Initialize the settings passed to the optimizer.

        Args:
            parameters: iterator of torch.Tensor
            minimizer_args: dict of arguments to pass to scipy.optimize.minimize,
                (default {'maxiter':100})
            callback: function to call after each iteration of scipy.optimize.minimize,
                which should take the argument intermediate_result: OptimizeResult,
                (default lambda *args: None),
        """

        super().__init__(parameters, defaults=minimizer_args)
        self.callback = callback

    def step(self,
             obj_fn: Callable[[], torch.Tensor],
             con_fn: Callable[[], torch.Tensor],
             lower_bnd: Union[torch.Tensor, np.array, float] = -np.inf,
             upper_bnd: Union[torch.Tensor, np.array, float] = 0.0) -> OptimizeResult:
        """Minimize obj_fn subject to con_fn <= 0. Uses settings from initialization.

        Constrained optimization currently only using trust-constr method. Handles
        computation of objective gradient and constraint Jacobian via PyTorch's AD.
        Casts these to numpy arrays for scipy.optimize.minimize for each step. The
        Jacobian calculation calls the constraint function once and calls backward
        for each constraint.

        Args:
            obj_fn: function that returns a scalar tensor
            con_fn: function that returns a vector tensor of size (n_constraints, 1)
            lower_bnd: lower bound for constraints, defaults to -inf
            upper_bnd: upper bound for constraints, defaults to 0.0

        Returns:
            scipy.optimize.OptimizeResult
        """

        assert len(self.param_groups) == 1, "Only one parameter group is supported"
        parameters = self.param_groups[0]['params']

        x0 = ravel_pack(parameters)
        # get sizes for constraints and parameter sizes
        n_con = con_fn().numel()
        n_params = sum([p.numel() for p in parameters])

        np_lower_bnd = torch_to_np(lower_bnd)
        np_upper_bnd = torch_to_np(upper_bnd)

        def vec_to_params(x):
            nn.utils.vector_to_parameters(torch.tensor(x, dtype=torch.float32,
                                                       device=parameters[0].device), parameters)

        def np_obj_fun(x):
            vec_to_params(x)
            obj = obj_fn()
            return obj.item()

        def np_obj_jac(x):
            # Possibly redundant objective call. Could use obj.backward(), rather than redoing this...
            vec_to_params(x)

            self.zero_grad()
            obj = obj_fn()
            obj.backward()

            # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            grads = ravel_pack([p.grad for p in parameters])
            return grads

        def np_ineq_fun(x):
            vec_to_params(x)
            con = con_fn()
            # if con.device != torch.device('cpu'):
            #     con = con.cpu()
            # return con.detach().numpy()[:,0]
            return con.view(-1).cpu().detach().numpy()

        def np_ineq_jac(x):
            # This likely isn't efficient, and was written for PyTorch before functorch was added.
            vec_to_params(x)

            jac = np.zeros((n_con, n_params), dtype=np.float32)
            con = con_fn()
            # eeem = torch.zeros((n_con,1), dtype=torch.float32, device=con.device)
            eeem = torch.zeros_like(con)
            for i in range(n_con):
                self.zero_grad()
                eeem[i] = 1.0
                con.backward(eeem, retain_graph=True)
                eeem[i] = 0.0

                # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
                for p in parameters:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                grads = ravel_pack([p.grad for p in parameters])
                jac[i, :] = grads

            return jac

        nonlinear_constraint = scipy.optimize.NonlinearConstraint(np_ineq_fun, np_lower_bnd, np_upper_bnd,
                                                                  jac=np_ineq_jac, hess=scipy.optimize.BFGS())

        res = scipy.optimize.minimize(np_obj_fun, x0, jac=np_obj_jac,
                                      constraints=nonlinear_constraint,
                                      hess=scipy.optimize.BFGS(),
                                      method='trust-constr',
                                      options=self.defaults,
                                      callback=self.callback)

        return res



class ConstrainedOptimizerTracker():
    """
    Example usage:

    maxiter = 50
    ub = 1.0
    lb = -np.inf
    tracker = torch_scipy_wrapper.ConstrainedOptimizerTracker(maxiter, len(constraint_fun()), objective_fun, \
          constraint_fun, ub = ub, lb = lb)
    optimizer = torch_scipy_wrapper.TorchScipyOptimizer(N.parameters(), {'maxiter': maxiter}, tracker.callback)
    res = optimizer.step(tracker.obj_fn, tracker.con_fn, upper_bnd=ub, lower_bnd=lb)


    """

    def __init__(self, maxiter, n_constraints, obj, con, lb=-np.inf, ub=0.0):
        self.con_history = torch.zeros((maxiter, n_constraints))
        self.obj_history = torch.zeros((maxiter, 1))
        self.obj = obj
        self.con = con
        self.lb = lb
        self.ub = ub
        self.current_iter = 0
        self.current_obj = None
        self.current_con = None

    def constraint_violation(self, con):
        # account for lb and ub, find total constraint violation lb <= con <= ub
        ub_viol = torch.relu(con - self.ub)
        lb_viol = torch.relu(self.lb - con)
        return ub_viol + lb_viol

    def callback(self, intermediate_result):
        # if self.current_obj is None:
        #     self.current_obj = self.obj_fn()
        # if self.current_con is None:
        #     self.current_con = self.con_fn()

        self.obj_history[self.current_iter, :] = self.current_obj.detach()
        self.con_history[self.current_iter, :] = self.current_con.detach().flatten()
        self.current_iter += 1
        con_norm = torch.sum(self.constraint_violation(self.current_con))
        print(f'Iteration {self.current_iter}, Objective: {self.current_obj.item()}, Constraint Violation: {con_norm}')

    def obj_fn(self):
        self.current_obj = self.obj()
        return self.current_obj

    def con_fn(self):
        self.current_con = self.con()
        return self.current_con

    def show_history(self):
        n_quantiles = 101
        quantiles = torch.linspace(0, 1, n_quantiles)
        # plot symmetrically
        con_quants = torch.quantile(self.con_history, quantiles, axis=1)
        med = torch.median(self.con_history, dim = 1)
        min = torch.min(self.con_history, dim = 1)
        max = torch.max(self.con_history, dim = 1)
        epoch_range = range(self.current_iter)

        # get percent of constraints that are violated
        viol = self.constraint_violation(self.con_history)
        n_violated = torch.sum(viol > 0, dim=1)
        percent_violated = n_violated / self.con_history.shape[1]

        fig, ax = plt.subplots(3,1, figsize=(6,12))
        ax[0].plot(epoch_range, self.obj_history[:self.current_iter])
        ax[0].set_ylabel('Objective History')
        ax[0].set_yscale('log')

        # use quantiles to fill between symmetrically
        for quant_idx in range(n_quantiles//2):
            ax[1].fill_between(epoch_range,
                               con_quants[quant_idx, :self.current_iter], con_quants[-quant_idx-1, :self.current_iter],
                               alpha=1.0 / (n_quantiles//2), color='blue', edgecolor=None)

        # plot median as dotted line
        ax[1].plot(epoch_range, med.values[:self.current_iter], linestyle='--', color='blue', alpha=0.5)
        ax[1].plot(epoch_range, min.values[:self.current_iter], linestyle=':', color='blue', alpha=0.5)
        ax[1].plot(epoch_range, max.values[:self.current_iter], linestyle=':', color='blue', alpha=0.5)
        # plot ub and lb, if they aren't inf or -inf
        if self.lb > -np.inf:
            ax[1].axhline(self.lb, color='k', linestyle='--', alpha=0.5)
        if self.ub < np.inf:
            ax[1].axhline(self.ub, color='k', linestyle='--', alpha=0.5)
        ax[1].set_ylabel('Constraint History')
        ax[1].set_yscale('symlog')

        ax[2].plot(epoch_range, percent_violated[:self.current_iter])
        ax[2].set_ylabel('Percent of Constraints Violated')

        # layout, label bottom x
        plt.tight_layout()
        ax[2].set_xlabel('Iteration')
        return fig, ax


