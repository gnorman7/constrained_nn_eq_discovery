"""
Residual vs Data Constraints Training Script

This script implements equation discovery using constrained optimization
with two main constraint modes:
- 'residual': Constraints on PDE residuals
- 'data': Constraints on data fitting

The script is converted from the residual_vs_data_constraints.ipynb notebook.
"""

import os
import time
import argparse
import sys
import glob
import shutil
from typing import Callable
import yaml

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import constrained_nn_eq_discovery.dhpm
import constrained_nn_eq_discovery.training
import constrained_nn_eq_discovery.con_opt
import constrained_nn_eq_discovery.numerics
import constrained_nn_eq_discovery.interpolate
import constrained_nn_eq_discovery.utils
from constrained_nn_eq_discovery.numerics import fd_centered, fd_centered_2nd, rk4


def true_N(big_u: torch.Tensor) -> torch.Tensor:
    """True Burgers equation operator: -u * u_x + 0.1 * u_xx.

    Args:
        big_u: (N, 3) tensor containing [u, u_x, u_xx]

    Returns:
        u_t: (N, 1) tensor containing time derivatives
    """
    u = big_u[:, 0:1]      # Shape: (N, 1)
    ux = big_u[:, 1:2]     # Shape: (N, 1)
    uxx = big_u[:, 2:3]    # Shape: (N, 1)
    return -u * ux + 0.1 * uxx  # Shape: (N, 1)


def batch_ode(x_vec: torch.Tensor, U_int: torch.Tensor,
              N: Callable[[torch.Tensor], torch.Tensor],
              num_pts: int = 3) -> torch.Tensor:
    """Use method of lines to construct ODE system f(y) = dy/dt with Dirichlet BCs.

    Args:
        x_vec: (n_x,) spatial grid
        U_int: (b, n_x - 2) interior solution values (excluding BCs)
        N: PDE operator function
        num_pts: Number of points for centered difference (3 or 5)

    Returns:
        U_t: (b, n_x - 2) time derivatives for interior points
    """
    n_bcs = (num_pts - 1) // 2
    fd_ux = fd_centered
    fd_uxx = fd_centered_2nd

    # Add homogeneous Dirichlet BCs: shape (b, n_x)
    bcs = torch.zeros(U_int.shape[0], n_bcs, device=U_int.device)
    U_vec = torch.cat([bcs, U_int, bcs], dim=1)

    # Compute spatial derivatives: shape (b, n_x - 2)
    U_x = fd_ux(U_vec, x_vec, dim=1)
    U_xx = fd_uxx(U_vec, x_vec, dim=1)

    # Stack for N operator: (b, n_x - 2, 3) -> (b*(n_x-2), 3)
    N_inputs = [U_int, U_x, U_xx]
    big_U = torch.stack(N_inputs, dim=2)
    big_u = big_U.reshape(-1, 3)  # Batch along spatial dimension too

    # Apply PDE operator and reshape back: (b*(n_x-2), 1) -> (b, n_x-2)
    u_t = N(big_u)
    U_t = u_t.reshape(U_int.shape)
    return U_t


def solve_batch_pde(U0: torch.Tensor, x_vec: torch.Tensor, t_vec: torch.Tensor,
                    N: Callable[[torch.Tensor], torch.Tensor],
                    integration_scheme: Callable = rk4,
                    num_pts: int = 3) -> torch.Tensor:
    """Solve PDE on batch of initial conditions.

    Args:
        U0: (b, n_x) initial conditions (including BCs)
        x_vec: (n_x,) spatial mesh with BCs
        t_vec: (n_t,) temporal mesh
        N: PDE operator function
        integration_scheme: Time integration function
        num_pts: Number of points for spatial derivatives

    Returns:
        U: (b, n_t, n_x) solutions including BCs
    """
    # Initialize solution tensor: (b, n_t, n_x)
    U = torch.zeros(U0.shape[0], len(t_vec), len(x_vec), device=U0.device)
    U0_int = U0[:, 1:-1]  # Extract interior points: (b, n_x-2)

    # Define time-independent dynamics with enforced BCs
    integrated_fun = lambda t, U: batch_ode(x_vec, U, N, num_pts=num_pts)

    # Integrate and transpose to get (b, n_t, n_x-2)
    U_pde_int = integration_scheme(integrated_fun, U0_int, t_vec).transpose(0, 1)
    U[:, :, 1:-1] = U_pde_int  # Insert interior solutions
    return U


def load_data(config: dict) -> tuple:
    """Load and prepare training data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (t, x, U, xt_train, u_train, xt_f)
    """
    data_config = config['data']
    datapath = data_config['datapath']
    N_u = data_config['N_u']
    N_f = data_config['N_f']
    noise_level = data_config['noise_level']
    seed = config['seed']

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data = sio.loadmat(datapath)
    t = torch.tensor(data['t']).float().squeeze()  # Shape: (n_t,)
    x = torch.tensor(data['x']).float().squeeze()  # Shape: (n_x,)
    U = torch.tensor(data['usol']).float().T       # Shape: (n_t, n_x)

    # Create meshgrid for training point selection
    T, X = torch.meshgrid(t, x, indexing='ij')     # Shapes: (n_t, n_x) each

    # Flatten for random sampling
    u_flattened = U.flatten()      # Shape: (n_t * n_x,)
    x_flattened = X.flatten()      # Shape: (n_t * n_x,)
    t_flattened = T.flatten()      # Shape: (n_t * n_x,)

    # Sample training points
    train_inds = np.random.choice(u_flattened.shape[0], N_u, replace=False)
    nf_inds = np.random.choice(u_flattened.shape[0], N_f, replace=False)

    # Add noise to training data
    data_std = torch.std(u_flattened)
    noise_std = noise_level * data_std
    u_train = u_flattened[train_inds] + noise_std * torch.randn(N_u)
    x_train = x_flattened[train_inds]
    t_train = t_flattened[train_inds]
    u_train = u_train.reshape(-1, 1)  # Shape: (N_u, 1)

    # Collocation points for residual computation
    xf = x_flattened[nf_inds]
    tf = t_flattened[nf_inds]

    # Stack coordinates: shapes (N_u, 2) and (N_f, 2)
    xt_train = torch.stack([x_train, t_train], dim=1)
    xt_f = torch.stack([xf, tf], dim=1)
    xt_f.requires_grad = True  # For automatic differentiation

    return t, x, U, xt_train, u_train, xt_f


def create_model(config: dict) -> constrained_nn_eq_discovery.dhpm.SimpleEqDiscoveryModel:
    """Create neural network model.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model_config = config['model']

    # Create N network (PDE operator): input [u, u_x, u_xx] -> output u_t
    N_dnn = constrained_nn_eq_discovery.dhpm.Siren(
        in_features=3,
        out_features=1,
        hidden_features=model_config['N_hidden_features'],
        hidden_layers=model_config['N_hidden_layers'],
        first_omega_0=model_config['N_first_omega_0'],
        hidden_omega_0=model_config['N_hidden_omega_0']
    )

    # Create u network (solution approximation): input [x, t] -> output u
    u_dnn = constrained_nn_eq_discovery.dhpm.Siren(
        in_features=2,
        out_features=1,
        hidden_features=model_config['u_hidden_features'],
        hidden_layers=model_config['u_hidden_layers'],
        first_omega_0=model_config['u_first_omega_0'],
        hidden_omega_0=model_config['u_hidden_omega_0']
    )

    model = constrained_nn_eq_discovery.dhpm.SimpleEqDiscoveryModel(u_dnn, N_dnn)
    return model


def setup_optimization_functions(model, xt_train, u_train, xt_f, config):
    """Setup objective and constraint functions based on training mode.

    Args:
        model: Neural network model
        xt_train: Training coordinates (N_u, 2)
        u_train: Training values (N_u, 1)
        xt_f: Collocation coordinates (N_f, 2)
        config: Configuration dictionary

    Returns:
        Tuple of (obj_fn, con_fn, eps_low, eps_high)
    """
    train_config = config['training']
    train_mode = train_config['mode']
    group_size = train_config['group_size']
    noise_level = config['data']['noise_level']

    # Calculate noise standard deviation
    data_std = torch.std(u_train)
    noise_std = noise_level * data_std

    def residual_fn():
        """Compute PDE residual at collocation points."""
        return model.get_residual(xt_f)  # Shape: (N_f, 1)

    def residual_fn_grouped():
        """Compute grouped residual constraints."""
        res_grouped = model.get_residual(xt_f).reshape(-1, group_size)  # Shape: (N_f//group_size, group_size)
        res_grouped_sqr = torch.mean(res_grouped**2, dim=1)  # Shape: (N_f//group_size,)
        return res_grouped_sqr

    def data_fn():
        """Compute data fitting residual."""
        u_res = model.u_dnn(xt_train) - u_train  # Shape: (N_u, 1)
        return u_res

    def data_fn_grouped():
        """Compute grouped data constraints."""
        u_dnn_grouped = model.u_dnn(xt_train).reshape(-1, group_size)  # Shape: (N_u//group_size, group_size)
        u_train_grouped = u_train.reshape(-1, group_size)  # Shape: (N_u//group_size, group_size)
        u_res_grouped = torch.abs(u_dnn_grouped - u_train_grouped)  # Shape: (N_u//group_size, group_size)
        u_res_grouped_avg = torch.mean(u_res_grouped, dim=1)  # Shape: (N_u//group_size,)
        return u_res_grouped_avg

    if train_mode == 'residual':
        # Minimize data MSE subject to residual constraints
        con_fn = residual_fn_grouped
        obj_fn = lambda: torch.mean(data_fn()**2)
        eps_low = -np.inf
        eps_high = train_config['eps_high']
    elif train_mode == 'data':
        # Minimize residual MSE subject to data constraints
        con_fn = data_fn_grouped
        obj_fn = lambda: torch.mean(residual_fn()**2)
        eps_low = -np.inf
        # For data mode, eps_high can be either direct value or multiplier of noise_std
        if 'eps_high' in train_config and train_config['eps_high'] is not None:
            eps_high = train_config['eps_high'] * noise_std
        else:
            eps_high = noise_std
    else:
        raise ValueError(f"Unknown training mode: {train_mode}")

    return obj_fn, con_fn, eps_low, eps_high


def train_model(model, obj_fn, con_fn, eps_low, eps_high, config):
    """Train model using constrained optimization.

    Args:
        model: Neural network model
        obj_fn: Objective function
        con_fn: Constraint function
        eps_low: Lower constraint bound
        eps_high: Upper constraint bound
        config: Configuration dictionary

    Returns:
        Optimization result
    """
    train_config = config['training']
    epochs = train_config['epochs']

    # Setup optimizer
    minimizer_args = {
        'maxiter': epochs,
        'verbose': 2,
        'gtol': train_config.get('gtol', 1e-10),
        'xtol': train_config.get('xtol', 1e-10)
    }

    optim = constrained_nn_eq_discovery.con_opt.TorchScipyOptimizer(
        model.parameters(),
        minimizer_args=minimizer_args
    )

    # Run optimization
    print(f"Starting constrained optimization with {epochs} iterations...")
    res = optim.step(obj_fn, con_fn, eps_low, eps_high)
    print("Optimization completed.")

    return res


def evaluate_model(model, t, x, U, config):
    """Evaluate trained model by solving PDE forward in time.

    Args:
        model: Trained neural network model
        t: Time vector
        x: Spatial vector
        U: True solution
        config: Configuration dictionary

    Returns:
        Tuple of (U_pred, U_intp, l2_error)
    """
    eval_config = config['evaluation']

    # Setup initial condition and grid
    u0_fn = lambda x_val: -torch.sin(np.pi / 8.0 * x_val)
    n_x = eval_config['n_x']
    x_vec = torch.linspace(x.min(), x.max(), n_x)

    # Setup time integration
    cfl = eval_config['cfl']
    dx = x_vec[1] - x_vec[0]
    dt = cfl * dx
    n_t = int((t.max() - t.min()) / dt) + 1
    t_vec = torch.linspace(t.min(), t.max(), n_t)

    # Initial condition: shape (1, n_x)
    U0 = u0_fn(x_vec).reshape(1, -1)
    T, X = torch.meshgrid(t, x, indexing='ij')

    with torch.no_grad():
        # Solve PDE using learned operator: shape (n_t, n_x)
        U_pred = solve_batch_pde(U0, x_vec, t_vec, model.N_dnn).squeeze(0)

        # Interpolate to original grid for comparison
        U_intp = constrained_nn_eq_discovery.interpolate.interpolate_2d(U_pred, X)

        # Compute L2 error
        l2_error = constrained_nn_eq_discovery.utils.l2_error(U, U_intp)

    return U_pred, U_intp, l2_error


def save_model_and_results(model, config, l2_error, l2_theta):
    """Save trained model and generate result plots.

    Args:
        model: Trained model
        config: Configuration dictionary
        l2_error: PDE solution L2 error
        l2_theta: Neural network L2 error

    Returns:
        model_path: Path where model was saved
    """
    train_config = config['training']
    data_config = config['data']

    # Generate model name with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = (f"{timestamp}_{train_config['mode']}_"
                 f"eps{train_config.get('eps_high', 0):.2e}_"
                 f"epochs{train_config['epochs']}_"
                 f"m{train_config['group_size']}_"
                 f"Nu{data_config['N_u']}_"
                 f"Nf{data_config['N_f']}_"
                 f"nl{data_config['noise_level']:.2f}")

    # Save model
    model_path = f"../models/residual_vs_data/{model_name}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model_path


def create_visualization(model, t, x, U, U_intp, xt_train, u_train,
                        l2_error, l2_theta, model_name):
    """Create and save visualization plots.

    Args:
        model: Trained model
        t, x: Time and space vectors
        U: True solution
        U_intp: Interpolated PDE solution
        xt_train: Training coordinates
        u_train: Training values
        l2_error: PDE solution L2 error
        l2_theta: Neural network L2 error
        model_name: Model name for saving
    """
    T, X = torch.meshgrid(t, x, indexing='ij')

    # Create subplot figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: PDE predicted solution
    ax[0].pcolor(T, X, U_intp, shading='auto', cmap='rainbow')
    ax[0].set_title(f'PDE Predicted, $\ell^2$ = {l2_error:.3f}')
    ax[0].set_ylabel('$x$')

    # Plot 2: Neural network predicted solution
    U_theta = model.u_on_meshgrid(T, X).detach()
    ax[1].pcolor(T, X, U_theta, shading='auto', cmap='rainbow')
    ax[1].set_title(rf'$u^\theta$ Predicted, $\ell^2$ = {l2_theta:.3f}')

    # Plot 3: Training data
    t_train = xt_train[:, 1].detach().numpy()
    x_train = xt_train[:, 0].detach().numpy()
    triang = tri.Triangulation(t_train, x_train)
    ax[2].tricontourf(triang, u_train.flatten(), levels=100, cmap='rainbow')
    ax[2].set_title('Training Data')

    # Set labels
    for a in ax:
        a.set_xlabel('$t$')

    # Save figure
    fig_path = f'../figs/residual_vs_data/{model_name}.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


def run_single_config(config_path):
    """Run training for a single configuration file.

    Args:
        config_path: Path to the configuration YAML file
    """
    print(f"Starting training with config: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Training mode: {config['training']['mode']}")
    print(f"Group size: {config['training']['group_size']}")

    # Load data
    t, x, U, xt_train, u_train, xt_f = load_data(config)
    print(f"Data loaded: t range [{t.min():.3f}, {t.max():.3f}], "
          f"x range [{x.min():.3f}, {x.max():.3f}]")
    print(f"Training points: {xt_train.shape[0]}, Collocation points: {xt_f.shape[0]}")

    # Create model
    model = create_model(config)
    print(f"Model created with parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup optimization
    obj_fn, con_fn, eps_low, eps_high = setup_optimization_functions(
        model, xt_train, u_train, xt_f, config)
    print(f"Constraint bounds: [{eps_low}, {eps_high}]")

    # Compute number of constraints
    train_config = config['training']
    if train_config['mode'] == 'residual':
        N_constraints = config['data']['N_f'] // train_config['group_size']
    else:
        N_constraints = config['data']['N_u'] // train_config['group_size']
    print(f"Number of constraints: {N_constraints}")

    # Train model
    res = train_model(model, obj_fn, con_fn, eps_low, eps_high, config)

    # Evaluate model
    U_pred, U_intp, l2_error = evaluate_model(model, t, x, U, config)
    print(f'PDE solution L2 error: {l2_error:.6f}')

    # Compute neural network L2 error
    T, X = torch.meshgrid(t, x, indexing='ij')
    with torch.no_grad():
        U_theta = model.u_on_meshgrid(T, X)
        l2_theta = constrained_nn_eq_discovery.utils.l2_error(U, U_theta)
    print(f'Neural network L2 error: {l2_theta:.6f}')

    # Save results
    model_path = save_model_and_results(model, config, l2_error, l2_theta)

    # Create visualization
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    create_visualization(model, t, x, U, U_intp, xt_train, u_train,
                        l2_error, l2_theta, model_name)

    print(f"Training completed successfully for {config_path}!")


def run_hpc_parallel():
    """Run HPC parallel execution by distributing config files across processes."""
    # Check for input arguments for HPC parallel execution
    if len(sys.argv) > 2 and '--' not in sys.argv[1]:
        print(f"HPC parallel execution with args: {sys.argv}")
        proc_id = int(sys.argv[1])
        proc_max = int(sys.argv[2])
    else:
        print("Single process execution")
        proc_id = 0
        proc_max = 1

    # Find all config files in the data_vs_res_configs folder
    config_folder = '../configs/data_vs_res_configs'
    if not os.path.exists(config_folder):
        print(f"Config folder not found: {config_folder}")
        print("Creating example config folder...")
        os.makedirs(config_folder, exist_ok=True)
        print(f"Please place your config files in {config_folder}")
        return

    config_files = glob.glob(os.path.join(config_folder, '*.yml'))

    if not config_files:
        print(f"No .yml config files found in {config_folder}")
        return

    print(f"Found {len(config_files)} config files")
    print(f"Process {proc_id} of {proc_max} will handle configs:")

    # Create completed folder for moving finished configs
    completed_folder = os.path.join(config_folder, 'completed')
    os.makedirs(completed_folder, exist_ok=True)

    # Process configs assigned to this process
    configs_processed = 0
    for i, config_file in enumerate(config_files):
        if i % proc_max == proc_id:
            print(f"  Processing: {os.path.basename(config_file)}")
            try:
                run_single_config(config_file)
                # Move completed config to completed folder
                completed_path = os.path.join(completed_folder, os.path.basename(config_file))
                shutil.move(config_file, completed_path)
                print(f"  Moved to completed: {completed_path}")
                configs_processed += 1
            except Exception as e:
                print(f"  ERROR processing {config_file}: {e}")
                # Continue with next config instead of stopping
                continue

    print(f"Process {proc_id} completed {configs_processed} configurations")


def main():
    """Main function that handles both single config and HPC parallel execution."""
    # Check if running in HPC parallel mode (has numeric arguments)
    if len(sys.argv) > 2 and '--' not in sys.argv[1]:
        # HPC parallel mode
        run_hpc_parallel()
    else:
        # Single config mode (original behavior)
        parser = argparse.ArgumentParser(
            description='Train equation discovery model with residual vs data constraints')
        parser.add_argument('--config', type=str, required=True,
                           help='Path to configuration YAML file')
        args = parser.parse_args()

        run_single_config(args.config)


if __name__ == "__main__":
    main()
