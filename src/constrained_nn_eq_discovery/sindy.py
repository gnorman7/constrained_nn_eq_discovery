import torch
import torch.nn as nn
import numpy as np
import pysindy as ps
from typing import Union

def get_lib_fcn(feature_ind: int, library_functions: list, derivative_order: int, include_bias: bool = False):
    n_lib_functions = len(library_functions)
    assert feature_ind >= 0, "feature index must be non-negative"
    if include_bias == False:
        feature_ind += 1
    assert feature_ind <= (n_lib_functions + 1) * (derivative_order + 1) - 1, "feature index out of range"

    # four regimes: lib function, just derivative, mixed.
    if feature_ind == 0:
        def lib_fcn(big_u):
            # trick to work for np or torch
            return (big_u[:, 0:1] * 0.0) + 1.0
    elif feature_ind <= n_lib_functions:
        def lib_fcn(big_u):
            f = library_functions[feature_ind - 1]
            return f(big_u[:, 0:1])
    elif feature_ind <= n_lib_functions + derivative_order:
        def lib_fcn(big_u):
            deriv_order = feature_ind - n_lib_functions
            return big_u[:, deriv_order:1 + deriv_order]
    else:
        def lib_fcn(big_u):
            mixed_ind = feature_ind - n_lib_functions - derivative_order - 1
            fcn_ind = mixed_ind % n_lib_functions
            deriv_order = mixed_ind // n_lib_functions + 1
            f = library_functions[fcn_ind]
            fcn_term = f(big_u[:, 0:1])
            deriv_term = big_u[:, deriv_order:1 + deriv_order]
            return fcn_term * deriv_term
    return lib_fcn


class Torch_SINDy(nn.Module):
    def __init__(self, pde_library: Union[ps.PDELibrary, ps.WeakPDELibrary], coefficients: np.ndarray,
                 param_init_val: float = 1.0, all_one_param: bool = True):
        super(Torch_SINDy, self).__init__()
        assert param_init_val != 0.0, "param_init_val cannot be zero"
        self.param_init_val = param_init_val

        coeffs = coefficients.squeeze()
        nonzero_inds = np.nonzero(coeffs)[0]
        # check if some coefficients are nonzero, if not, raise error
        assert len(nonzero_inds) > 0, "No nonzero coefficients found, check SINDy fitting errors"

        self.terms = []
        nonzero_coefficients = []

        for ind in nonzero_inds:
            # feature_ind: int, library_functions: list, derivative_order: int, include_bias: bool = False)
            lib_fcn = get_lib_fcn(ind, pde_library.functions, pde_library.derivative_order, pde_library.include_bias)
            self.terms.append(lib_fcn)
            nonzero_coefficients.append(coeffs[ind])

        self.base_coeffs = torch.tensor(nonzero_coefficients, dtype=torch.float32)

        # add parameter for each nonzero coefficient, set value to one
        if all_one_param:
            self.coefficients = nn.Parameter(torch.tensor(
                [self.param_init_val] * len(nonzero_coefficients), dtype=torch.float32))
        else:
            # add a separate parameter group for each coefficient, still just as 1.0
            self.coefficients = nn.ParameterList([
                nn.Parameter(torch.tensor([self.param_init_val], dtype=torch.float32)) for _ in nonzero_coefficients
            ])

    def get_pde_coeffs(self):
        coeffs = []
        for coeff, base_coeff in zip(self.coefficients, self.base_coeffs):
            coeffs.append(coeff / self.param_init_val * base_coeff)
        return coeffs

    def forward(self, big_u):
        out = torch.zeros_like(big_u[:, 0:1])
        for term, coeff, base_coeff in zip(self.terms, self.coefficients, self.base_coeffs):
            out += (coeff / self.param_init_val) * base_coeff * term(big_u)
        return out
