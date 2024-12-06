# Constrained or Unconstrained? Neural-Network-Based Equation Discovery from Data

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

## Project Description
This repository contains the source code for the paper "Constrained or Unconstrained? Neural-Network-Based Equation Discovery from Data". While we await review, the paper is available on [arXiv](https://arxiv.org/abs/2406.02581).

The [PyTorch constrained optimizer wrapper](./src/constrained_nn_eq_discovery/con_opt.py#L33) is the most broadly useful part of the codebase.
The function wraps SciPy's `trust-constr` [optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html), but using PyTorch's autograd for gradients of the objective function and constraints.
Hessians are replaced with BFGS approximations (through [SciPy's implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html)).
The optimizer follows PyTorch's `torch.optim.Optimizer` convention, with `step` running all constrained optimization steps.
This wrapper should support a GPU for PyTorch computations, but in our experience, the communication with the CPU for SciPy makes this slower than using only the CPU.

If you find this code useful, please cite the preprint:
```
@misc{norman2024constrained,
    title={Constrained or Unconstrained? Neural-Network-Based Equation Discovery from Data},
    author={Grant Norman and Jacqueline Wentz and Hemanth Kolla and Kurt Maute and Alireza Doostan},
    year={2024},
    eprint={2406.02581},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


