# Constrained or Unconstrained? Neural-Network-Based Equation Discovery from Data

Available on [CMA](https://doi.org/10.1016/j.cma.2024.117684), with previous versions on [arXiv](https://arxiv.org/abs/2406.02581).

Grant Norman, Jacqueline Wentz, Hemanth Kolla, Kurt Maute, Alireza Doostan

## Abstract
Throughout many fields, practitioners often rely on differential equations to model systems. Yet, for many applications, the theoretical derivation of such equations and/or the accurate resolution of their solutions may be intractable. Instead, recently developed methods, including those based on parameter estimation, operator subset selection, and neural networks, allow for the data-driven discovery of both ordinary and partial differential equations (PDEs), on a spectrum of interpretability. The success of these strategies is often contingent upon the correct identification of representative equations from noisy observations of state variables and, as importantly and intertwined with that, the mathematical strategies utilized to enforce those equations. Specifically, the latter has been commonly addressed via unconstrained optimization strategies. Representing the PDE as a neural network, we propose to discover the PDE (or the associated operator) by solving a constrained optimization problem and using an intermediate state representation similar to a physics-informed neural network (PINN). The objective function of this constrained optimization problem promotes matching the data, while the constraints require that the discovered PDE is satisfied at a number of spatial collocation points. We present a penalty method and a widely used trust-region barrier method to solve this constrained optimization problem, and we compare these methods on numerical examples. Our results on several example problems demonstrate that the latter constrained method outperforms the penalty method, particularly for higher noise levels or fewer collocation points. This work motivates further exploration into using sophisticated constrained optimization methods in scientific machine learning, as opposed to their commonly used, penalty-method or unconstrained counterparts. For both of these methods, we solve these discovered neural network PDEs with classical methods, such as finite difference methods, as opposed to PINNs-type methods relying on automatic differentiation. We briefly highlight how simultaneously fitting the data while discovering the PDE improves the robustness to noise and other small, yet crucial, implementation details.

## Setup
To setup the source code, run
```
pip install -e .
```
Further requirements are in `environment.yml`.

## Project Structure
- `src/`: Source Code
- `tests/`: Tests, using pytest
- `notebooks/`: Jupyter Notebooks
- `data/`: Data
- `environment.yml`: Python package requirements

## Usage

This repository includes the main codebase for the paper, particularly for the Sine-Gordon Equation and Anisotropic Porous Medium Equation examples.

The [PyTorch constrained optimizer wrapper](./src/constrained_nn_eq_discovery/con_opt.py#L33) is the most broadly useful part of the codebase.
The function wraps SciPy's `trust-constr` [optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html), but using PyTorch's autograd for gradients of the objective function and constraints.
Hessians are replaced with BFGS approximations (through [SciPy's implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html)).
The optimizer follows PyTorch's `torch.optim.Optimizer` convention, with `step` running all constrained optimization steps.
This wrapper should support a GPU for PyTorch computations, but in our experience, the communication with the CPU for SciPy makes this slower than using only the CPU.

The [Deep Hidden Physics Model architecture](./src/constrained_nn_eq_discovery/dhpm.py#L131) may also be helpful.
The `EqDiscoveryModel.N_dnn` object can be overwritten with a function such as [the Sine-Gordon Equation](./src/constrained_nn_eq_discovery/utils.py#L25) to create a helpful object for training PINNs.


If you find this code useful, please cite the associated work:
```
@article{norman_constrained_2025,
  title = {Constrained or Unconstrained? {{Neural-network-based}} Equation Discovery from Data},
  shorttitle = {Constrained or Unconstrained?},
  author = {Norman, Grant and Wentz, Jacqueline and Kolla, Hemanth and Maute, Kurt and Doostan, Alireza},
  year = {2025},
  month = mar,
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {436},
  pages = {117684},
}
```

Please reach out if you have any questions, on the code, or on the paper!

