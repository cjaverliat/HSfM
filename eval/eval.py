import json
import pickle
import torch
from transformations import (
    inverse_Rt,
    compute_similarity_transform,
    apply_similarity_transform_to_points,
)
from typing import Literal, Any
import roma
from tqdm import tqdm
from smplx import SMPL
from smplx.lbs import vertices2joints


def compute_human_metrics(
    gt_cam2worlds: torch.Tensor,
    pred_cam2worlds: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_joints: torch.Tensor,
) -> dict[str, Any] | Literal["found nans"]:
    assert gt_joints.dim() == pred_joints.dim() == 2
    assert gt_joints.shape == pred_joints.shape
    assert gt_joints.shape[-1] == 3

    num_cams = gt_cam2worlds.shape[0]
    assert gt_cam2worlds.dim() == pred_cam2worlds.dim() == 3
    assert gt_cam2worlds.shape == pred_cam2worlds.shape == (num_cams, 4, 4)

    gt_cam_positions = gt_cam2worlds[:, :3, 3]
    pred_cam_positions = pred_cam2worlds[:, :3, 3]
    assert gt_cam_positions.shape == pred_cam_positions.shape == (num_cams, 3)

    if torch.isnan(pred_cam_positions).any():
        return "found nans"

    # W-MPJPE is computed after SE(3) alignment (rotation and translation) of camera poses
    R, t, s = compute_similarity_transform(
        X=pred_cam_positions,
        Y=gt_cam_positions,
        estimate_scale=False,
    )
    pred_joints_aligned = apply_similarity_transform_to_points(pred_joints, R, t, s)
    w_mpjpe = torch.norm(
        gt_joints - pred_joints_aligned,
        dim=-1,
    )

    # PA-MPJPE is computed after Sim(3) alignment (rotation, translation, and scale) of joints
    R, t, s = compute_similarity_transform(
        X=pred_joints,
        Y=gt_joints,
        estimate_scale=True,
    )

    pred_joints_aligned = apply_similarity_transform_to_points(pred_joints, R, t, s)
    pa_mpjpe = torch.norm(
        gt_joints - pred_joints_aligned,
        dim=-1,
    )

    return {
        "w-mpjpe": w_mpjpe,
        "pa-mpjpe": pa_mpjpe,
    }


def compute_cam_metrics(
    gt_cam2worlds: torch.Tensor,
    pred_cam2worlds: torch.Tensor,
) -> dict[str, torch.Tensor] | Literal["found nans"]:
    """
    Returns a dictionary with camera metrics. Metrics will be 1D tensors.
    Cf. https://github.com/hongsukchoi/HSfM_RELEASE/issues/1
    """
    out = dict[str, torch.Tensor]()
    num_cams = gt_cam2worlds.shape[0]
    assert gt_cam2worlds.dim() == pred_cam2worlds.dim() == 3
    assert gt_cam2worlds.shape == pred_cam2worlds.shape == (num_cams, 4, 4)

    # What metrics are used in RelPose++?
    # - Joint Rotation Accuracy @ 15 deg
    # - Camera Center Accuracy @ 0.2 (20% of scene scale)
    #     - Aligned using similarity transform
    gt_cam_positions = gt_cam2worlds[:, :3, 3]
    pred_cam_positions = pred_cam2worlds[:, :3, 3]
    assert gt_cam_positions.shape == pred_cam_positions.shape == (num_cams, 3)

    # Assumptions:
    # 1. The global coordinate system is arbitrary.
    # 2. Both the ground-truth and the predicted cameras are already in meters.

    if torch.isnan(pred_cam_positions).any():
        return "found nans"

    # Camera position accuracy, with scale alignment.
    R, t, s = compute_similarity_transform(
        X=pred_cam_positions,
        Y=gt_cam_positions,
        estimate_scale=True,
    )
    pred_cam_positions_aligned = apply_similarity_transform_to_points(
        pred_cam_positions, R, t, s
    )
    assert gt_cam_positions.shape == pred_cam_positions.shape == (num_cams, 3)
    out["s-TE"] = torch.norm((gt_cam_positions - pred_cam_positions_aligned), dim=-1)
    del pred_cam_positions_aligned

    # Camera position accuracy, without scale alignment.
    R, t, s = compute_similarity_transform(
        X=pred_cam_positions,
        Y=gt_cam_positions,
        estimate_scale=False,
    )
    pred_cam_positions_aligned = apply_similarity_transform_to_points(
        pred_cam_positions, R, t, s
    )
    assert gt_cam_positions.shape == pred_cam_positions.shape == (num_cams, 3)

    # Camera translation error
    out["TE"] = torch.norm((gt_cam_positions - pred_cam_positions_aligned), dim=-1)

    del pred_cam_positions
    del pred_cam_positions_aligned

    # Equation (7) in RelPose++.
    scene_scale = torch.max(
        torch.linalg.norm(
            gt_cam_positions - torch.mean(gt_cam_positions, dim=0, keepdim=True), dim=-1
        )
    )

    # Camera center accuracy, with scale alignment.
    out["s-CCA05"] = (out["s-TE"] < (0.05 * scene_scale)).to(torch.float32)
    out["s-CCA10"] = (out["s-TE"] < (0.10 * scene_scale)).to(torch.float32)
    out["s-CCA15"] = (out["s-TE"] < (0.15 * scene_scale)).to(torch.float32)
    out["s-CCA20"] = (out["s-TE"] < (0.20 * scene_scale)).to(torch.float32)
    out["s-CCA25"] = (out["s-TE"] < (0.25 * scene_scale)).to(torch.float32)
    out["s-CCA30"] = (out["s-TE"] < (0.30 * scene_scale)).to(torch.float32)

    # Camera center accuracy, without scale alignment.
    out["CCA05"] = (out["TE"] < (0.05 * scene_scale)).to(torch.float32)
    out["CCA10"] = (out["TE"] < (0.10 * scene_scale)).to(torch.float32)
    out["CCA15"] = (out["TE"] < (0.15 * scene_scale)).to(torch.float32)
    out["CCA20"] = (out["TE"] < (0.20 * scene_scale)).to(torch.float32)
    out["CCA25"] = (out["TE"] < (0.25 * scene_scale)).to(torch.float32)
    out["CCA30"] = (out["TE"] < (0.30 * scene_scale)).to(torch.float32)

    # Orientation accuracy.
    gt_R_world_cam = gt_cam2worlds[:, :3, :3]
    pred_R_world_cam = pred_cam2worlds[:, :3, :3]
    pred_R_world_cam = torch.einsum("ij,njk->nik", R, pred_R_world_cam)
    assert gt_R_world_cam.shape == pred_R_world_cam.shape == (num_cams, 3, 3)

    gt_cam_R_cami_camj = torch.einsum("mij,nik->mnjk", gt_R_world_cam, gt_R_world_cam)
    pred_cam_R_cami_camj = torch.einsum(
        "mij,nik->mnjk", pred_R_world_cam, pred_R_world_cam
    )

    pairwise_deltas_radians = roma.rotmat_geodesic_distance(
        gt_cam_R_cami_camj, pred_cam_R_cami_camj
    )

    # pairwise_deltas = torch.einsum(
    #     "mnij,mnkj->mnik", gt_cam_R_cami_camj, pred_cam_R_cami_camj
    # )
    # pairwise_deltas_radians = torch.norm(SO3.from_matrix(pairwise_deltas).log(), dim=-1)
    assert pairwise_deltas_radians.shape == (num_cams, num_cams)

    # Get upper-triangular terms, not including the k=0 diagonal.
    # This is because the diagonal is always 0.
    upper_triangular_mask = torch.triu(
        torch.ones_like(pairwise_deltas_radians, dtype=torch.bool), diagonal=1
    )
    pairwise_deltas_radians = pairwise_deltas_radians[upper_triangular_mask]

    # Camera angle error
    out["AE"] = torch.rad2deg(pairwise_deltas_radians)

    # Relative rotation accuracy
    out["RRA05"] = (out["AE"] < 5.0).to(torch.float32)
    out["RRA10"] = (out["AE"] < 10.0).to(torch.float32)
    out["RRA15"] = (out["AE"] < 15.0).to(torch.float32)
    out["RRA20"] = (out["AE"] < 20.0).to(torch.float32)
    out["RRA25"] = (out["AE"] < 25.0).to(torch.float32)
    out["RRA30"] = (out["AE"] < 30.0).to(torch.float32)

    return out


def compute_h36m_sequence_metrics(
    ground_truth: dict[str, Any],
    sequence_data_dir: str,
    smpl_model: SMPL,
    J_regressor: torch.Tensor,
):
    # Only one person in H3.6M sequences.
    PERSON_ID = 1
    n_frames = ground_truth["n_annotated_frames"]

    metrics = {}

    for frame_idx in tqdm(range(n_frames), leave=False):
        with open(f"{sequence_data_dir}/timing_info_{frame_idx:05d}.json", "r") as f:
            timing_info = json.load(f)

        with open(
            f"{sequence_data_dir}/hsfm_output_smpl_{frame_idx:05d}.pkl", "rb"
        ) as f:
            hsfm_output = pickle.load(f)

        if not hsfm_output:
            print(f"No hsfm output for frame {frame_idx}")
            continue

        smpl_params = hsfm_output["hsfm_people(smplx_params)"][PERSON_ID]

        gt_cam2worlds = []
        pred_cam2worlds = []

        for camera_name, camera_gt in ground_truth["cameras"].items():
            extrinsics = torch.as_tensor(camera_gt["extrinsics"])
            gt_cam2world = torch.eye(4)
            gt_cam2world[:3, :] = inverse_Rt(extrinsics)[:3, :]
            gt_cam2worlds.append(gt_cam2world)

        for camera_idx, camera in hsfm_output["hsfm_places_cameras"].items():
            cam2world = torch.from_numpy(camera["cam2world"])
            pred_cam2worlds.append(cam2world)

        gt_cam2worlds = torch.stack(gt_cam2worlds)
        pred_cam2worlds = torch.stack(pred_cam2worlds)

        gt_joints = torch.as_tensor(ground_truth["keypoints"][str(frame_idx)])
        pred_joints = get_joints_from_smpl(smpl_model, J_regressor, smpl_params)

        try:
            cam_metrics = compute_cam_metrics(
                gt_cam2worlds=gt_cam2worlds,
                pred_cam2worlds=pred_cam2worlds,
            )

            if not cam_metrics == "found nans":
                for k, v in cam_metrics.items():
                    metrics.setdefault(k, torch.empty((0,)))
                    metrics[k] = torch.cat((metrics[k], v))
        except Exception as e:
            print(f"Error computing camera metrics for frame {frame_idx}: {e}")

        try:
            human_metrics = compute_human_metrics(
                gt_cam2worlds=gt_cam2worlds,
                pred_cam2worlds=pred_cam2worlds,
                gt_joints=gt_joints,
                pred_joints=pred_joints,
            )

            if not human_metrics == "found nans":
                for k, v in human_metrics.items():
                    metrics.setdefault(k, torch.empty((0,)))
                    metrics[k] = torch.cat((metrics[k], v.unsqueeze(0)))
        except Exception as e:
            print(f"Error computing human metrics for frame {frame_idx}: {e}")

        if timing_info:
            for k, v in timing_info.items():
                if k == "n_views":
                    continue
                metrics.setdefault(k, torch.empty((0,)))
                metrics[k] = torch.cat((metrics[k], torch.as_tensor(v).unsqueeze(0)))

    for k, v in metrics.items():
        metrics[k] = v.mean()

    return metrics


def get_joints_from_smpl(
    smplx: SMPL, J_regressor: torch.Tensor, smpl_params: dict[str, Any]
) -> torch.Tensor:
    betas = torch.as_tensor(smpl_params["betas"], dtype=torch.float32)
    body_pose = torch.as_tensor(smpl_params["body_pose"], dtype=torch.float32).reshape(
        1, -1, 3, 3
    )
    global_orient = torch.as_tensor(
        smpl_params["global_orient"], dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    root_transl = torch.as_tensor(
        smpl_params["root_transl"], dtype=torch.float32
    ).reshape(1, 1, 3)

    with torch.no_grad():
        smpl_output = smplx.forward(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            pose2rot=False,
            return_verts=True,
        )
    # Similar to what's done in the original code in vis_viser_hsfm.py
    smplx_vertices = smpl_output.vertices
    smplx_j3d = smpl_output.joints  # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
    smplx_vertices = smplx_vertices - smplx_j3d[:, 0:1, :] + root_transl
    smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl

    joints = vertices2joints(J_regressor, smplx_vertices)[0]
    return joints
