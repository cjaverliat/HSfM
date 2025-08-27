import json
import torch
from tqdm import tqdm
from smplx import SMPL
import orjson
import os
import pickle
from eval.eval import get_joints_from_smpl
from eval.transformations import (
    inverse_Rt,
    compute_similarity_transform,
    apply_similarity_transform_to_points,
    apply_similarity_transform_to_Rt,
)
from eval.conversions import (
    convert_world_points_from_opencv_to_opengl,
    convert_extrinsics_from_opencv_to_opengl,
)

from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.scene.camera import OpenCVCamera, ViewerCamera, PinholeCamera
from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.scene.node import Node
import random
import colorsys
import numpy as np

C.window_type = "pyglet"

H36M_CONNECTIVITY = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


def get_subject_color_rgba(subject_id: str) -> tuple[float, float, float, float]:
    rng = random.Random(subject_id)
    subject_hue = rng.random()
    subject_color = (*colorsys.hsv_to_rgb(subject_hue, 1.0, 1.0), 1.0)
    return subject_color


def create_subject_skeleton(
    subject_id: str,
    keypoints: torch.Tensor,
    skeleton_radius: float = 0.01,
    color_rgba: tuple[float, float, float, float] = (1, 0, 0, 1),
    opencv_to_opengl: bool = True,
) -> Skeletons:
    assert keypoints.shape[1:] == (17, 3), "Keypoints must be in H3.6M format"

    # To OpenGL coordinate system
    if opencv_to_opengl:
        keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

    skeleton = Skeletons(
        joint_positions=keypoints.cpu().numpy(),
        joint_connections=H36M_CONNECTIVITY,
        color=color_rgba,
        radius=skeleton_radius,
    )
    skeleton.name = subject_id
    return skeleton


def create_camera(
    view_id: str,
    Rt: torch.Tensor,
    K: torch.Tensor,
    resolution_hw: tuple[int, int],
    color: tuple[float, float, float, float] = (1, 0, 0, 1),
    opencv_to_opengl: bool = True,
    scale: float = 1.0,
) -> OpenCVCamera:
    # To OpenGL coordinate system
    if opencv_to_opengl:
        Rt = convert_extrinsics_from_opencv_to_opengl(Rt)

    K = K.cpu().numpy()
    Rt = Rt.cpu().numpy()

    camera = OpenCVCamera(K=K, Rt=Rt, cols=resolution_hw[1], rows=resolution_hw[0])
    camera.scale = scale
    camera.mesh.color = color
    camera.color = color
    camera.active_color = color
    camera.inactive_color = color
    camera.name = view_id
    return camera


def load_h36m_joints_regressor(
    swap_legs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # From https://github.com/ubc-vision/joint-regressor-refinement/tree/master
    J_regressor = torch.load("./body_models/retrained_J_Regressor.pt").float().cpu()
    J_regressor = torch.nn.ReLU()(J_regressor)
    J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(1).expand(
        J_regressor.shape
    )
    if swap_legs:
        # Swap the legs to match GT annotations order (following MMPose's order)
        J_regressor[[1, 2, 3, 4, 5, 6]] = J_regressor[[4, 5, 6, 1, 2, 3]]
    return J_regressor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dataset_dir",
        type=str,
        default="./data/h36m/processed",
        help="Path to processed H3.6M dataset directory",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="./data/h36m/hsfm_h36_eval",
        help="Path to HSFM evaluation data directory",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="./body_models/smpl",
        help="Path to SMPL model directory",
    )
    args = parser.parse_args()

    smpl_model = SMPL(
        model_path=args.smpl_model_path,
        gender="neutral",
        num_betas=10,
        batch_size=1,
    )
    J_regressor = load_h36m_joints_regressor()

    with open(
        os.path.join(args.processed_dataset_dir, "h36m_protocol1_sequences.json"), "rb"
    ) as f:
        sequences = orjson.loads(f.read())

    sequences = [seq for seq in sequences if seq["split"] == "val"]

    for seq in tqdm(sequences, desc="Visualizing sequences"):
        subject_name = seq["subject_name"]
        subaction_name = seq["subaction_name"]
        cameras_names = seq["cameras_names"]

        with open(
            os.path.join(
                args.processed_dataset_dir,
                f"ground_truth/{subject_name}_{subaction_name}.json",
            ),
            "r",
        ) as f:
            ground_truth = json.load(f)

        n_frames = ground_truth["n_annotated_frames"]

        sequence_data_dir = os.path.join(
            args.eval_data_dir, f"{subject_name}_{subaction_name}"
        )

        for frame_idx in tqdm(range(n_frames), leave=False):
            with open(
                f"{sequence_data_dir}/hsfm_output_smpl_{frame_idx:05d}.pkl", "rb"
            ) as f:
                hsfm_output = pickle.load(f)

            # Only one person in H3.6M sequences.
            PERSON_ID = 1
            smpl_params = hsfm_output["hsfm_people(smplx_params)"][PERSON_ID]

            gt_joints = torch.as_tensor(ground_truth["keypoints"][str(frame_idx)])
            pred_joints = get_joints_from_smpl(smpl_model, J_regressor, smpl_params)

            Rt = torch.zeros((len(cameras_names), 3, 4))
            gt_Rt = torch.zeros((len(cameras_names), 3, 4))
            K = torch.zeros((len(cameras_names), 3, 3))
            gt_K = torch.zeros((len(cameras_names), 3, 3))

            world_pts_3d = torch.empty((0, 3))
            world_pts_colors = torch.empty((0, 4))

            for cam_idx, camera_name in enumerate(cameras_names):
                pts_3d = torch.as_tensor(
                    hsfm_output["hsfm_places_cameras"][cam_idx]["pts3d"]
                ).reshape(-1, 3)
                colors = torch.as_tensor(
                    hsfm_output["hsfm_places_cameras"][cam_idx]["rgbimg"]
                ).reshape(-1, 3)
                # Add alpha channel to colors
                colors = torch.cat([colors, torch.ones((colors.shape[0], 1))], dim=1)

                mask = torch.as_tensor(
                    hsfm_output["hsfm_places_cameras"][cam_idx]["msk"]
                ).reshape(-1)

                pts_3d = pts_3d[mask]
                colors = colors[mask]

                world_pts_3d = torch.cat([world_pts_3d, pts_3d])
                world_pts_colors = torch.cat([world_pts_colors, colors])

                Rt[cam_idx] = inverse_Rt(
                    torch.from_numpy(
                        hsfm_output["hsfm_places_cameras"][cam_idx]["cam2world"]
                    )[:3, :]
                )
                K[cam_idx] = torch.from_numpy(
                    hsfm_output["hsfm_places_cameras"][cam_idx]["intrinsic"]
                )

                gt_Rt[cam_idx] = torch.as_tensor(
                    ground_truth["cameras"][camera_name]["extrinsics"]
                )[:3, :]
                gt_K[cam_idx] = torch.as_tensor(
                    ground_truth["cameras"][camera_name]["intrinsics"]
                )

            # Get 3D bbox
            bbox_margin = 0.25
            xmin = pred_joints[..., 0].min() - bbox_margin
            xmax = pred_joints[..., 0].max() + bbox_margin
            ymin = pred_joints[..., 1].min() - bbox_margin
            ymax = pred_joints[..., 1].max() + bbox_margin
            zmin = pred_joints[..., 2].min() - bbox_margin
            zmax = pred_joints[..., 2].max() + bbox_margin

            pts_outside_bbox_mask = (
                (world_pts_3d[..., 0] < xmin)
                | (world_pts_3d[..., 0] > xmax)
                | (world_pts_3d[..., 1] < ymin)
                | (world_pts_3d[..., 1] > ymax)
                | (world_pts_3d[..., 2] < zmin)
                | (world_pts_3d[..., 2] > zmax)
            )

            # Remove all points inside the 3D bbox
            world_pts_3d = world_pts_3d[pts_outside_bbox_mask]
            world_pts_colors = world_pts_colors[pts_outside_bbox_mask]

            cam2world = inverse_Rt(Rt)
            gt_cam2world = inverse_Rt(gt_Rt)
            cam_pos = cam2world[:, :3, 3]
            gt_cam_pos = gt_cam2world[:, :3, 3]

            R, t, s = compute_similarity_transform(
                cam_pos, gt_cam_pos, estimate_scale=False
            )

            pred_joints = apply_similarity_transform_to_points(pred_joints, R, t, s)
            world_pts_3d = apply_similarity_transform_to_points(world_pts_3d, R, t, s)
            Rt = apply_similarity_transform_to_Rt(Rt, R, t, s)

            viewer = Viewer()
            viewer.window.init_mgl_context()

            gt_node = Node(name="Ground Truth")
            pred_node = Node(name="Predicted")

            skeleton_gt = create_subject_skeleton(
                subject_id="subject_0",
                keypoints=gt_joints.unsqueeze(0),
                color_rgba=(0, 0, 0, 1),
            )
            skeleton_gt.name = "GT"
            gt_node.add(skeleton_gt)

            skeleton = create_subject_skeleton(
                subject_id="subject_0",
                keypoints=pred_joints.unsqueeze(0),
                color_rgba=(1, 0, 0, 1),
            )
            pred_node.add(skeleton)

            camera_scale = 2.0

            for cam_idx, camera_name in enumerate(cameras_names):
                camera_Rt = Rt[cam_idx]
                camera_K = K[cam_idx]
                gt_camera_Rt = gt_Rt[cam_idx]
                gt_camera_K = gt_K[cam_idx]

                camera = create_camera(
                    view_id=camera_name,
                    Rt=camera_Rt,
                    K=camera_K,
                    resolution_hw=(1000, 1000),
                    scale=camera_scale,
                )
                gt_camera = create_camera(
                    view_id=camera_name,
                    Rt=gt_camera_Rt,
                    K=gt_camera_K,
                    resolution_hw=(1000, 1000),
                    color=(0, 0, 0, 1),
                    scale=camera_scale,
                )
                gt_node.add(gt_camera)
                pred_node.add(camera)

            world_pts_3d = convert_world_points_from_opencv_to_opengl(world_pts_3d)
            world_pcd = PointClouds(
                points=world_pts_3d.unsqueeze(0).cpu().numpy(),
                colors=world_pts_colors.unsqueeze(0).cpu().numpy(),
            )
            pred_node.add(world_pcd)

            scene_center = gt_cam_pos.mean(dim=0)
            scene_center = convert_world_points_from_opencv_to_opengl(scene_center)

            cam_pos = torch.tensor([scene_center[0], 15.0, scene_center[2]]).reshape(
                3, 1
            )
            cam_rot = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]
            )
            cam2world = torch.cat([cam_rot, cam_pos], dim=1)
            Rt = inverse_Rt(cam2world)
            K = torch.tensor(
                [
                    [1000, 0, 500],
                    [0, 1000, 500],
                    [0, 0, 1],
                ]
            )

            viewer.scene.add(gt_node)
            viewer.scene.add(pred_node)
            viewer.run()
