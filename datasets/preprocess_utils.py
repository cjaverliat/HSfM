import torch
import torchvision.ops._utils
import torchvision
from tqdm import tqdm
from typing import Optional

# Right-handed +Z-up basis is +Z-up, +X-right, +Y-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# RH +Z-up
#
#  +Z_w
#  ^
#  |  7 +Y_w
#  | /
#  |/
#  +--------> +X_w
OPENCV_WORLD_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 1, 0],  # y
        [0, 0, 1],  # z
    ],
    dtype=torch.float32,
)

def inverse_Rt(Rt: torch.Tensor) -> torch.Tensor:
    R = Rt[..., :3, :3]
    t = Rt[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = torch.einsum("...ij,...j->...i", -R_inv, t)
    return torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)
def normalize_points_with_intrinsics(
    points: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """
    Normalize points with intrinsics.
    """
    cxcy = K[..., :2, 2]
    fxfy = K[..., :2, :2].diagonal(dim1=-2, dim2=-1)
    if len(cxcy.shape) < len(points.shape):
        cxcy, fxfy = cxcy.unsqueeze(-2), fxfy.unsqueeze(-2)
    xy = (points - cxcy) / fxfy
    return xy


def unnormalize_points_with_intrinsics(
    points: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """
    Unnormalize points with intrinsics.
    """
    x_coord = points[..., 0]
    y_coord = points[..., 1]

    # unpack intrinsics
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    if len(cx.shape) < len(x_coord.shape):
        cx, cy, fx, fy = (
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
            fx.unsqueeze(-1),
            fy.unsqueeze(-1),
        )

    u_coord = x_coord * fx + cx
    v_coord = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def convert_points_to_homogeneous(pts: torch.Tensor) -> torch.Tensor:
    """
    Convert points to homogeneous coordinates.

    Args:
        points (torch.Tensor): points to convert. Shape (*, D).

    Returns:
        torch.Tensor: points in homogeneous coordinates. Shape (*, D + 1).
    """
    batch_dims = pts.shape[:-1]
    return torch.cat([pts, torch.ones(batch_dims + (1,), device=pts.device)], dim=-1)


def convert_points_from_homogeneous(
    pts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop the homogeneous dimension of points.

    Args:
        points (torch.Tensor): points to convert. Shape (*, D + 1).

    Returns:
        torch.Tensor: points in Euclidean coordinates. Shape (*, D).
        torch.Tensor: value of the dropped dimension. Shape (*, 1).
    """
    return pts[..., :-1] / pts[..., -1:], pts[..., -1:]


def transform_points_from_world_to_camera(
    points_3d_world: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D points expressed in the world frame to the camera frame.
        X_cam = Rt @ [X_world, 1]

    Args:
        points_3d_world: 3D points to convert. Shape (P, 3) or (F, P, 3) with F being the number of frames and P being the number of points.
        Rt: Camera extrinsic matrix. Shape (3, 4), (C, 3, 4) or (F, C, 3, 4) with C being the number of cameras.

    Returns:
        points_3d_cam: 3D points in the camera frame. Shape (K, 3), (C, K, 3) or (F, C, K, 3).
    """
    points_has_frame_dim = points_3d_world.ndim == 3

    if not points_has_frame_dim:
        points_3d_world = points_3d_world.unsqueeze(0)

    F, P, _ = points_3d_world.shape

    Rt_has_camera_dim = Rt.ndim >= 3
    Rt_has_frame_dim = Rt.ndim == 4

    if not Rt_has_camera_dim:
        Rt = Rt.unsqueeze(0)
    if not Rt_has_frame_dim:
        Rt = Rt.unsqueeze(0)

    points_h = convert_points_to_homogeneous(points_3d_world)
    points_transformed = torch.einsum("fcij,fpj->fcpi", Rt, points_h)

    if not points_has_frame_dim and not Rt_has_frame_dim:
        points_transformed = points_transformed.squeeze(-4)
    if not Rt_has_camera_dim:
        points_transformed = points_transformed.squeeze(-3)

    return points_transformed

def distort_points(
    points: torch.Tensor,
    k_i: Optional[torch.Tensor] = None,
    p_i: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Distort points with radial and tangential distortion.

    Applies the Brown-Conrady distortion model:
        r = sqrt(x^2 + y^2)
        theta = atan(r)
        x_d = x / r * theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + ...)
            + 2 * p1 * x * y + p2 * (r + 2 * x^2)
        y_d = y / r * theta * (1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + ...)
            + 2 * p2 * x * y + p1 * (r + 2 * y^2)

    Args:
        points (torch.Tensor): Tensor of shape (*, P, 2) containing points in normalized coordinates [-1, 1].
        k_i (torch.Tensor): Radial distortion coefficients (k1, k2, k3, ...) of shape (*, D)
        p_i (torch.Tensor): Tangential distortion coefficients (p1, p2) of shape (*, 2)

    Returns:
        Distorted points with the same shape as input
    """

    if k_i is None and p_i is None:
        return points

    if k_i is None:
        k_i = torch.zeros((3,), device=points.device)
    if p_i is None:
        p_i = torch.zeros((2,), device=points.device)

    k_i_has_camera_dim = k_i.ndim >= 2
    k_i_has_frame_dim = k_i.ndim == 3
    p_i_has_camera_dim = p_i.ndim >= 2
    p_i_has_frame_dim = p_i.ndim == 3

    if not k_i_has_camera_dim:
        k_i = k_i.unsqueeze(0)
    if not k_i_has_frame_dim:
        k_i = k_i.unsqueeze(0)
    
    if not p_i_has_camera_dim:
        p_i = p_i.unsqueeze(0)
    if not p_i_has_frame_dim:
        p_i = p_i.unsqueeze(0)

    n_k_i = k_i.shape[-1]

    distorted_points = points.clone()

    r_sq = torch.pow(points, 2).sum(dim=-1)

    # radial_factor = 1 + k1 * r^2 + k2 * r^4 + k3 * r^6 + ...
    radial_factor = torch.ones((*points.shape[:-1],), device=points.device)
    for i in range(n_k_i):
        radial_factor += k_i[..., i] * torch.pow(r_sq, (i + 1))

    p1, p2 = p_i[..., 0], p_i[..., 1]
    x_d = (
        points[..., 0] * radial_factor
        + 2 * p1 * points[..., 0] * points[..., 1]
        + p2 * (r_sq + 2 * points[..., 0] ** 2)
    )
    y_d = (
        points[..., 1] * radial_factor
        + 2 * p2 * points[..., 0] * points[..., 1]
        + p1 * (r_sq + 2 * points[..., 1] ** 2)
    )

    distorted_points[..., 0] = x_d
    distorted_points[..., 1] = y_d
    return distorted_points

def project_points_from_camera_to_image(
    points_3d_cam: torch.Tensor,
    K: torch.Tensor,
    k_i: Optional[torch.Tensor] = None,
    p_i: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points from the camera frame to the image plane and undistort them if coefficients are provided.
    The 2D points coordinates are in the range [0, W] x [0, H] (expressed in pixels).
        X_img = K @ [X_cam, 1]

    Args:
        points_3d_cam: 3D points to project. Shape (P, 3) or (C, P, 3) or (F, C, P, 3).
        K: Camera intrinsic matrix. Shape (3, 3), (C, 3, 3), (F, C, 3, 3), (1, C, 3, 3) or (1, 1, 3, 3).
        k_i: Radial distortion coefficients. Shape (D,), (C, D), (F, C, D), (1, C, D) or (1, 1, D).
        p_i: Tangential distortion coefficients. Shape (2,), (C, 2), (F, C, 2), (1, C, 2) or (1, 1, 2).

    Returns:
        points_2d: Projected 2D points. Shape (P, 2) or (C, P, 2) or (F, C, P, 2).
        points_depth: Depth of the projected points. Shape (P, 1) or (C, P, 1) or (F, C, P, 1).
    """
    points_has_camera_dim = points_3d_cam.ndim >= 3
    points_has_frame_dim = points_3d_cam.ndim == 4

    if not points_has_camera_dim:
        points_3d_cam = points_3d_cam.unsqueeze(0)
    if not points_has_frame_dim:
        points_3d_cam = points_3d_cam.unsqueeze(0)

    F, C, P, _ = points_3d_cam.shape

    K_has_camera_dim = K.ndim >= 3
    K_has_frame_dim = K.ndim == 4

    if not K_has_camera_dim:
        K = K.unsqueeze(0)
    if not K_has_frame_dim:
        K = K.unsqueeze(0)

    projected_points = torch.einsum("fcij,fcpj->fcpi", K, points_3d_cam)
    projected_points, projected_depth = convert_points_from_homogeneous(
        projected_points
    )

    if k_i is not None or p_i is not None:
        normalized_points = normalize_points_with_intrinsics(projected_points, K)
        distorted_points = distort_points(normalized_points, k_i=k_i, p_i=p_i)
        projected_points = unnormalize_points_with_intrinsics(distorted_points, K)

    if not points_has_frame_dim and not K_has_frame_dim:
        # Remove the F dimension
        projected_points = projected_points.squeeze(-4)
        projected_depth = projected_depth.squeeze(-4)
    if not K_has_camera_dim:
        # Remove the C dimension
        projected_points = projected_points.squeeze(-3)
        projected_depth = projected_depth.squeeze(-3)

    return projected_points, projected_depth


def compute_transition_matrix(
    old_basis: torch.Tensor,
    new_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the transition matrix P to convert from old_basis to new_basis.

    Args:
        old_basis: Vectors of the old basis. Shape (*, D, D). Vectors are columns.
            This also corresponds to the change of basis matrix from the old basis to the canonical basis of R^D.
        new_basis: Vectors of the new basis. Shape (*, D, D). Vectors are columns.
            This also corresponds to the change of basis matrix from the new basis to the canonical basis of R^D.

    Returns:
        P: Transition matrix. Shape (*, D, D).
    """
    P = torch.linalg.inv(new_basis) @ old_basis
    return P


def convert_points_basis(
    points: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Change the basis of a points from src_basis to dst_basis.

    Args:
        points: Points to change. Shape (*, 3).
        src_basis: Basis vectors of the source frame. Shape (*, 3, 3). Vectors are columns.
        dst_basis: Basis vectors of the destination frame. Shape (*, 3, 3). Vectors are columns.

    Returns:
        points_transformed: Points in the destination frame. Shape (*, 3).
    """
    assert points.shape[-1] == 3
    P = compute_transition_matrix(src_basis, dst_basis).to(points.device)
    points_transformed = torch.einsum("...ij,...j->...i", P, points)
    return points_transformed


def standardize_keypoints(
    keypoints: torch.Tensor,
    src_world_basis: torch.Tensor | None = None,
    src_world_unit_in_meters: float = 1.0,
):
    if torch.cuda.is_available():
        keypoints = keypoints.cuda()

    # Scale keypoints to meters
    if src_world_unit_in_meters != 1.0:
        keypoints = keypoints * src_world_unit_in_meters

    if src_world_basis is not None:
        keypoints = convert_points_basis(
            keypoints,
            src_world_basis.to(keypoints.device),
            OPENCV_WORLD_BASIS.to(keypoints.device),
        )

    return keypoints.cpu()


def compute_keypoints_3d_cam(
    keypoints_3d_world: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform the 3D keypoints from world coordinates to camera coordinates.
    """
    keypoints_3d_world_homogeneous = torch.cat(
        [keypoints_3d_world, torch.ones_like(keypoints_3d_world[..., :1])], dim=-1
    )

    if torch.cuda.is_available():
        keypoints_3d_world_homogeneous = keypoints_3d_world_homogeneous.cuda()
        Rt = Rt.cuda()

    keypoints_3d_cam = torch.einsum(
        "ij,fnkj->fnki", Rt, keypoints_3d_world_homogeneous
    )[..., :3]

    return keypoints_3d_cam.cpu()


def compute_keypoints_2d(
    keypoints_3d_world: torch.Tensor,
    Rt: torch.Tensor,
    K: torch.Tensor,
    distortion_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Project the 3D keypoints from world coordinates to 2D image coordinates.
    """
    orig_device = keypoints_3d_world.device
    orig_shape = keypoints_3d_world.shape

    if keypoints_3d_world.ndim == 3:
        # Add the S dimension
        keypoints_3d_world = keypoints_3d_world.unsqueeze(1)

    n_frames = keypoints_3d_world.shape[0]

    if torch.cuda.is_available():
        keypoints_3d_world = keypoints_3d_world.cuda()
        Rt = Rt.cuda()
        K = K.cuda()
        distortion_coefficients = distortion_coefficients.cuda()

    # transform_points_from_world_to_camera and project_points_from_camera_to_image expect (F, P, 3) shape
    keypoints_3d_cam = transform_points_from_world_to_camera(
        keypoints_3d_world.view(n_frames, -1, 3), Rt
    )

    p_i = distortion_coefficients[..., [2, 3]]
    k_i = distortion_coefficients[
        ..., [0, 1] + list(range(4, distortion_coefficients.shape[0]))
    ]

    keypoints_2d, _ = project_points_from_camera_to_image(
        keypoints_3d_cam, K=K, k_i=k_i, p_i=p_i
    )

    return keypoints_2d.reshape(orig_shape[:-1] + (2,)).to(orig_device)


def compute_bboxes_xyxy(
    poses_2d: torch.Tensor,
    padding_x: int,
    padding_y: int,
    image_size_hw: tuple[int, int],
    clamp_to_image_size: bool = True,
) -> torch.Tensor:
    x1 = torch.min(poses_2d[..., 0], dim=-1).values
    x2 = torch.max(poses_2d[..., 0], dim=-1).values
    y1 = torch.min(poses_2d[..., 1], dim=-1).values
    y2 = torch.max(poses_2d[..., 1], dim=-1).values

    if clamp_to_image_size:
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, max=image_size_hw[1] - 1)
        y1 = torch.clamp(y1, min=0)
        y2 = torch.clamp(y2, max=image_size_hw[0] - 1)

    x1 = x1 - padding_x
    x2 = x2 + padding_x
    y1 = y1 - padding_y
    y2 = y2 + padding_y

    if clamp_to_image_size:
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, max=image_size_hw[1] - 1)
        y1 = torch.clamp(y1, min=0)
        y2 = torch.clamp(y2, max=image_size_hw[0] - 1)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def suppress_overlapping_bboxes(
    bboxes_xywh: torch.Tensor,
    distance: torch.Tensor,
    overlap_threshold: float = 0.75,
) -> torch.Tensor:
    """
    Suppress overlapping bboxes based on the overlap threshold.
    When two bboxes exceed the overlap threshold, the bbox with lower distance is kept.

    Args:
        bboxes_xywh: (F, S, 4) tensor of bounding boxes in xywh format. With F being the number of frames, S being the number of subjects.
        distance: (F, S) tensor of distance of each bounding box to the camera
        overlap_threshold: overlap threshold for suppression (default: 0.75)

    Returns:
        keep: (F, S) tensor of boolean values indicating whether the bbox should be kept
    """

    n_frames, n_subjects, _ = bboxes_xywh.shape

    bboxes_xyxy = torchvision.ops.box_convert(
        bboxes_xywh.view(-1, 4), in_fmt="xywh", out_fmt="xyxy"
    ).reshape_as(bboxes_xywh)

    keep = torch.ones(
        (n_frames, n_subjects),
        dtype=torch.bool,
        device=bboxes_xyxy.device,
    )

    # Sort subjects by distance (closest first)
    sorted_subjects_ids_by_distance = torch.argsort(distance, descending=False, dim=-1)

    pbar = tqdm(
        total=n_frames * n_subjects, desc="Suppressing overlapping bboxes", leave=False
    )

    for frame_idx in range(n_frames):
        for subject_idx in range(n_subjects):
            subject_id = sorted_subjects_ids_by_distance[frame_idx, subject_idx]

            if not keep[frame_idx, subject_id]:
                pbar.update(1)
                continue  # already suppressed

            # All the subjects that are further away from the current subject are potentially occluded.
            potentially_occluded_idxs = sorted_subjects_ids_by_distance[
                frame_idx, subject_idx + 1 :
            ]

            if potentially_occluded_idxs.numel() == 0:
                pbar.update(1)
                continue

            current_subject_bbox = bboxes_xyxy[frame_idx, subject_id].unsqueeze(0)
            potentially_occluded_bboxes = bboxes_xyxy[
                frame_idx, potentially_occluded_idxs
            ]

            lt = torch.max(
                current_subject_bbox[:, :2], potentially_occluded_bboxes[:, :2]
            )
            rb = torch.min(
                current_subject_bbox[:, 2:], potentially_occluded_bboxes[:, 2:]
            )

            intersection_bbox = torch.cat([lt, rb], dim=-1)
            intersection_bbox_area = torchvision.ops.box_area(intersection_bbox)

            # If the overlap between the potentially occluded bboxes and the intersection bbox is greater than the overlap threshold, then the current subject is occluded.
            # The simplest example for this is when the bbox of the occluded subject is completely inside the current subject bbox, in that case the overlap is 1.
            potentially_occluded_bboxes_area = torchvision.ops.box_area(
                potentially_occluded_bboxes
            )
            overlap = intersection_bbox_area / potentially_occluded_bboxes_area

            suppress = overlap > overlap_threshold
            keep[frame_idx, potentially_occluded_idxs[suppress]] = False
            pbar.update(1)

    pbar.close()
    return keep
