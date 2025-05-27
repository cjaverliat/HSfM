import torch


def inverse_Rt(Rt: torch.Tensor) -> torch.Tensor:
    """
    Inverse Rt matrix.

    Args:
        Rt (torch.Tensor): Tensor of shape (*, 3, 4) containg the rigid transformation matrix [R|t]

    Returns:
        Rt_inv (torch.Tensor): Tensor of shape (*, 3, 4) containing the inverse camera Rt matrix
    """
    assert Rt.ndim >= 2 and Rt.shape[-2:] == (3, 4)

    R = Rt[..., :3, :3]
    t = Rt[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = torch.einsum("...nm,...mp->...np", -R_inv, t)
    return torch.cat([R_inv, t_inv], dim=-1)


def compute_similarity_transform(
    X: torch.Tensor,
    Y: torch.Tensor,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds a similarity transformation (R, T, s) between two given sets of corresponding points sets X and Y.

    Args:
        X (torch.Tensor): Tensor of shape (B, P, 3) containing points
        Y (torch.Tensor): Tensor of shape (B, P, 3) containing points
        estimate_scale (bool): Whether to estimate the scale of the transformation
        allow_reflection (bool): Whether to allow reflections in the transformation

    Returns:
        R (torch.Tensor): Rotation matrix of shape (B, 3, 3)
        T (torch.Tensor): Translation vector of shape (B, 3)
        s (torch.Tensor): Scale factor of shape (B,)
    References:
        [1] Shinji Umeyama: Least-Squares Estimation of Transformation Parameters Between Two Point Patterns
    """
    assert X.ndim == 3 or X.ndim == 2
    batch_dims = X.shape[:-2]
    n_points = X.shape[-2]
    assert Y.shape == (*batch_dims, n_points, 3)

    has_batch_dim = X.ndim == 3

    if not has_batch_dim:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    B, P, C = X.shape

    # 1. Remove mean.
    Xmu = X.mean(dim=1, keepdim=True)
    Ymu = Y.mean(dim=1, keepdim=True)

    # Mean-center the points.
    Xc = X - Xmu
    Yc = Y - Ymu

    # Compute the covariance between the point sets Xc, Yc.
    XY_cov = torch.bmm(Xc.transpose(2, 1), Yc)

    # Decompose the covariance matrix.
    U, S, V = torch.svd(XY_cov)

    # Identity matrix used for fixing reflections.
    E = torch.eye(C, dtype=XY_cov.dtype, device=XY_cov.device)[None].repeat(B, 1, 1)

    if not allow_reflection:
        # Reflection test:
        #   Checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # Find the rotation matrix.
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # Estimate scale
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        X_cov = (Xc * Xc).sum((1, 2))
        s = trace_ES / torch.clamp(X_cov, torch.finfo(X_cov.dtype).eps)
        # Recover translation
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # Recover translation
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]
        # Unit scaling since we do not estimate scale
        s = T.new_ones(B)

    if not has_batch_dim:
        R = R.squeeze(0)
        T = T.squeeze(0)
        s = s.squeeze(0)

    return R, T, s


def apply_similarity_transform_to_points(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Apply a similarity transform to a set of points.

    Args:
        X (torch.Tensor): Tensor of shape (B, P, 3) or (P, 3) containing points
        R (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3)
        T (torch.Tensor): Translation vector of shape (3,) or (B, 3)
        s (torch.Tensor): Scale factor of shape () or (B,)

    Returns:
        X_transformed (torch.Tensor): Tensor of shape (B, P, 3) or (P, 3) containing the transformed points
    """

    assert X.ndim in [2, 3] and X.shape[-1] == 3, (
        f"Expected X to have shape (B, P, 3) or (P, 3), got {X.shape}"
    )

    batch_dims = X.shape[:-2]

    assert R.shape in [
        (*batch_dims, 3, 3),
        (1, 3, 3),
        (3, 3),
    ], (
        f"Expected R to have shape {batch_dims + (3, 3)}, (1, 3, 3) or (3, 3), got {R.shape}"
    )
    assert T.shape in [
        (*batch_dims, 3),
        (1, 3),
        (3,),
    ], f"Expected T to have shape {batch_dims + (3,)}, (1, 3) or (3,), got {T.shape}"
    assert s.shape in [
        (*batch_dims,),
        (1,),
        (),
    ], f"Expected s to have shape {batch_dims}, (1,) or (), got {s.shape}"

    # Make broadcastable
    if len(batch_dims) > 0 and R.ndim == 2:
        R = R.expand((*batch_dims, -1, -1))
    if len(batch_dims) > 0 and T.ndim == 1:
        T = T.expand((*batch_dims, -1))
    if len(batch_dims) > 0 and s.ndim == 0:
        s = s.expand((*batch_dims,))

    X = torch.einsum("...nm,...mp->...np", X, s[..., None, None] * R) + T[..., None, :]

    return X


def apply_similarity_transform_to_Rt(
    Rt: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a similarity transform to a camera Rt matrix.

    Args:
        Rt (torch.Tensor): Tensor of shape (B, 3, 4) or (3, 4) containing the camera Rt matrix
        R (torch.Tensor): Tensor of shape (B, 3, 3) or (3, 3) containing the rotation matrix
        T (torch.Tensor): Tensor of shape (B, 3) or (3,) containing the translation vector
        s (torch.Tensor): Tensor of shape (B,) or () containing the scale factor

    Returns:
        Rt_transformed (torch.Tensor): Tensor of shape (B, 3, 4) or (3, 4) containing the transformed camera Rt matrix
    """
    assert Rt.ndim in [3, 2] and Rt.shape[-2:] == (3, 4)

    batch_dims = Rt.shape[:-2]

    assert R.shape in [
        (*batch_dims, 3, 3),
        (1, 3, 3),
        (3, 3),
    ], f"Expected R to have shape {batch_dims + (3, 3)} or (3, 3), got {R.shape}"
    assert T.shape in [
        (*batch_dims, 3),
        (1, 3),
        (3,),
    ], f"Expected T to have shape {batch_dims + (3,)} or (3,), got {T.shape}"
    assert s.shape in [
        (*batch_dims,),
        (1,),
        (),
    ], f"Expected s to have shape {batch_dims}, (1,) or (), got {s.shape}"

    Rt_inv = inverse_Rt(Rt)
    cam_pose = Rt_inv[..., :3, 3]
    cam_pose_right = cam_pose + Rt_inv[..., :3, 0]
    cam_pose_up = cam_pose + Rt_inv[..., :3, 1]
    cam_pose_forward = cam_pose + Rt_inv[..., :3, 2]

    cam_pose_right, cam_pose_up, cam_pose_forward, cam_pose = (
        apply_similarity_transform_to_points(
            torch.stack([cam_pose_right, cam_pose_up, cam_pose_forward, cam_pose]),
            R,
            T,
            s,
        )
    )

    cam_right = cam_pose_right - cam_pose
    cam_up = cam_pose_up - cam_pose
    cam_forward = cam_pose_forward - cam_pose

    cam_right = torch.nn.functional.normalize(cam_right, dim=-1)
    cam_up = torch.nn.functional.normalize(cam_up, dim=-1)
    cam_forward = torch.nn.functional.normalize(cam_forward, dim=-1)

    Rt_transformed_inv = torch.zeros((*batch_dims, 3, 4), device=Rt.device)
    Rt_transformed_inv[..., :3, 0] = cam_right
    Rt_transformed_inv[..., :3, 1] = cam_up
    Rt_transformed_inv[..., :3, 2] = cam_forward
    Rt_transformed_inv[..., :3, 3] = cam_pose

    Rt_transformed = inverse_Rt(Rt_transformed_inv)

    return Rt_transformed
