import torch

# refer to ../learning_numerics/interpolate.py for old versions
def interpolate_2d(U_locs: torch.Tensor, new_mesh_tensor: torch.Tensor, mode='bilinear'):
    """Interpolate a 2D tensor to a new mesh size (equispaced mesh in both cases)
    args:
        U_locs: 2D tensor to interpolate,
        new_mesh_tensor: tensor with desired new shape (e.g. sol.X).
        mode: interpolation mode, 'bilinear by default, could use 'nearest' or 'bicubic'
    returns:
        U_locs_interp: U_locs interpolated to new_mesh_tensor shape (equispaced)

    This operation may produce nondeterministic gradients when given tensors on a CUDA device.
    """
    assert len(U_locs.shape) == 2, "U_locs must be 2D"
    assert len(new_mesh_tensor.shape) == 2, "new_mesh_tensor must be 2D"
    U_locs_4d_input = U_locs.unsqueeze(0).unsqueeze(0)
    U_locs_4d_output = torch.nn.functional.interpolate(U_locs_4d_input, size=new_mesh_tensor.shape,
                                                       mode=mode, align_corners=True)
    return U_locs_4d_output.squeeze(0).squeeze(0)


def interpolate_3d(U_locs: torch.Tensor, new_mesh_tensor: torch.Tensor, mode='trilinear'):
    """Interpolate a 3D tensor to a new mesh size (equispaced mesh in both cases)
    args:
        U_locs: 3D tensor to interpolate,
        new_mesh_tensor: tensor with desired new shape (e.g. sol.X).
        mode: interpolation mode, 'trilinear' by default, could use 'nearest'
    returns:
        U_locs_interp: U_locs interpolated to new_mesh_tensor shape (equispaced)

    This operation may produce nondeterministic gradients when given tensors on a CUDA device.
    """
    assert len(U_locs.shape) == 3, "U_locs must be 3D"
    assert len(new_mesh_tensor.shape) == 3, "new_mesh_tensor must be 3D"
    U_locs_5d_input = U_locs.unsqueeze(0).unsqueeze(0)
    U_locs_5d_output = torch.nn.functional.interpolate(U_locs_5d_input, size=new_mesh_tensor.shape,
                                                       mode=mode, align_corners=True)
    return U_locs_5d_output.squeeze(0).squeeze(0)
