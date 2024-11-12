import scipy.io
import torch
import numpy as np
from dataclasses import dataclass
from typing import Union



@dataclass
class PDE_Soln_2D:
    """T, X, Y, and U are (n_t, n_x, n_y),
    t, x, and y are (n_t,), (n_x,), and (n_y,)"""
    T: torch.Tensor
    X: torch.Tensor
    Y: torch.Tensor
    U: torch.Tensor
    t: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor

    def to(self, device: torch.device):
        return PDE_Soln_2D(*(attr.to(device) for attr in
                             [self.T, self.X, self.Y, self.U, self.t, self.x, self.y]))


# %% Loading
def load_data_2D(datapath: str) -> PDE_Soln_2D:
    # these already have end points/boundaries included, don't need problem keyword
    data = scipy.io.loadmat(datapath)

    x = data['x'].flatten()
    t = data['t'].flatten()
    U = data['U_exact'][0, 0]

    y = torch.tensor(x, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    U = torch.tensor(U, dtype=torch.float32).transpose(0, 2).transpose(1, 2)

    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    out = PDE_Soln_2D(T, X, Y, U, t, x, y)
    return out


def get_noisy_U(sol: PDE_Soln_2D, noise_level: float) -> torch.Tensor:
    noise_std = noise_level * torch.std(sol.U)
    # use numpy for the random, so it can be controlled with a separate seed
    noise_instance = np.random.randn(*sol.U.shape)
    U_noisy = sol.U + torch.tensor(noise_instance, dtype=sol.U.dtype, device=sol.U.device) * noise_std
    return U_noisy


def get_training_data(sol: PDE_Soln_2D,
                      noise_level: float, N_ux: int, N_ut: int, N_f: int,
                      positive_u_split_nf: Union[float, None] = None,
                      for_validation: bool = False) -> Union[
                          tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Given a PDE_Soln or PDE_Soln_2D object, returns training data for the DHPM.
    xt tensors are [N, 3]. The u tensor is [N, 1]. Collocation points are random, training data is equispaced.

    Args:
        positive_u_split_nf: The fraction of collocation points where U is positive (according to u_train),
            if None, no information is used and the collocation points are random.

    """
    U_noisy = get_noisy_U(sol, noise_level)

    # xt_train will be equispaced, with N_ux, N_ux, and N_ut points in x, y, and t respectively

    # Replace with positive_u_split_nf sampling

    # sol.X and sol.T are 3D tensors
    # we want xt to be a 2D tensor with shape [len(t)*len(x)*len(y), 3]


    # equispaced indices, these include the boundaries!
    t_inds = torch.linspace(0, sol.T.shape[0] - 1, N_ut, dtype=torch.int)
    x_inds = torch.linspace(0, len(sol.x) - 1, N_ux, dtype=torch.int)
    y_inds = torch.linspace(0, len(sol.y) - 1, N_ux, dtype=torch.int)

    # U_train will be N_ux*N_ux*N_ut points from U_noisy
    T_inds, X_inds, Y_inds = torch.meshgrid(t_inds, x_inds, y_inds)
    U_train = U_noisy[T_inds, X_inds, Y_inds]
    X_train = sol.X[T_inds, X_inds, Y_inds]
    Y_train = sol.Y[T_inds, X_inds, Y_inds]
    T_train = sol.T[T_inds, X_inds, Y_inds]

    if for_validation:
        return X_train, Y_train, T_train, U_train

    # Take collocation points based on if U_train is positive, or randomly
    if positive_u_split_nf is None:
        xt = torch.stack((sol.X.flatten(), sol.Y.flatten(), sol.T.flatten()), dim=1)
        idx = torch.randperm(xt.shape[0])
        xt = xt[idx]
        xt_f = xt[:N_f]
    else:
        pos_inds = U_train > 0
        neg_inds = U_train <= 0
        N_f_pos = int(N_f * positive_u_split_nf)
        N_f_neg = N_f - N_f_pos
        xt_f_pos = torch.stack((X_train[pos_inds].flatten(),
                                Y_train[pos_inds].flatten(),
                                T_train[pos_inds].flatten()), dim=1)
        xt_f_neg = torch.stack((X_train[neg_inds].flatten(),
                                Y_train[neg_inds].flatten(),
                                T_train[neg_inds].flatten()), dim=1)
        # shuffle the indices and sample
        idx_pos = torch.randperm(xt_f_pos.shape[0])
        idx_neg = torch.randperm(xt_f_neg.shape[0])
        xt_f_pos = xt_f_pos[idx_pos[:N_f_pos]]
        xt_f_neg = xt_f_neg[idx_neg[:N_f_neg]]
        xt_f = torch.cat((xt_f_pos, xt_f_neg), dim=0)
    xt_f.requires_grad = True

    # not randomized
    u_train = U_train.flatten().unsqueeze(1)
    xt_train = torch.stack((X_train.flatten(), Y_train.flatten(), T_train.flatten()), dim=1)

    return xt_train, u_train, xt_f
