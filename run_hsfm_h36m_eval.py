import orjson
import os
from tqdm import tqdm
import tyro
import smplx
import pickle
import warnings
import torch
from smplx.lbs import vertices2joints
import numpy as np
import torch.nn as nn

from eval.transformations import (
    inverse_Rt,
    compute_similarity_transform,
    apply_similarity_transform_to_Rt,
    apply_similarity_transform_to_points,
)

from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds

C.window_type = "pyglet"
C.z_up = True


def main(
    processed_dataset_dir: str = "./data/h36m/processed/",
    output_dir: str = "./data_output/h36m/",
    vis: bool = False,
):
    with open(
        os.path.join(processed_dataset_dir, "h36m_protocol1_sequences.json"), "rb"
    ) as f:
        sequences = orjson.loads(f.read())

    sequences = [seq for seq in sequences if seq["split"] == "val"]

    seq_names = [f"{seq['subject_name']}_{seq['subaction_name']}" for seq in sequences]
    print(f"Running eval on {len(sequences)} sequences: {', '.join(seq_names)}")

    total_frames = sum([seq["n_annotated_frames"] for seq in sequences])

    pbar = tqdm(total=total_frames, desc="Processing frames")

    device = "cuda"

    PERSON_ID = 1

    smpl_layer = smplx.create(
        model_path="./body_models",
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        batch_size=1,
    ).to(device)

    # From https://github.com/ubc-vision/joint-regressor-refinement/tree/master
    J_regressor = (
        torch.load("./body_models/retrained_J_Regressor.pt").float().to(device)
    )
    J_regressor = nn.ReLU()(J_regressor)
    J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(1).expand(
        J_regressor.shape
    )

    # From SPIN http://visiondata.cis.upenn.edu/spin/data.tar.gz
    J_regressor_SPIN = (
        torch.from_numpy(np.load("./body_models/J_regressor_h36m.npy"))
        .float()
        .to(device)
    )

    # Swap the legs to match GT order (following MMPose's order)
    J_regressor[[1, 2, 3, 4, 5, 6]] = J_regressor[[4, 5, 6, 1, 2, 3]]
    J_regressor_SPIN[[1, 2, 3, 4, 5, 6]] = J_regressor_SPIN[[4, 5, 6, 1, 2, 3]]

    w_mpjpe = {}
    pa_mpjpe = {}

    for seq in sequences:
        total_timings = {
            "images_loading_time_seconds": [],
            "bboxes_loading_time_seconds": [],
            "world_env_time_seconds": [],
            "pose2d_time_seconds": [],
            "smpl_time_seconds": [],
            "hsfm_time_seconds": [],
        }

        total_results = {
            "TE": [],
            "w_mpjpe": [],
            "w_mpjpe_SPIN": [],
            "pa_mpjpe": [],
            "pa_mpjpe_SPIN": [],
        }

        subject_name = seq["subject_name"]
        subaction_name = seq["subaction_name"]
        n_annotated_frames = seq["n_annotated_frames"]

        ground_truth_path = os.path.join(
            processed_dataset_dir,
            "ground_truth",
            f"{subject_name}_{subaction_name}.json",
        )

        with open(ground_truth_path, "rb") as f:
            ground_truth = orjson.loads(f.read())

        cameras_gt = ground_truth["cameras"]
        cameras_names = list(cameras_gt.keys())
        cameras_K_gt = torch.stack(
            [torch.as_tensor(cameras_gt[name]["intrinsics"]) for name in cameras_names]
        ).to(device)
        cameras_Rt_gt = torch.stack(
            [torch.as_tensor(cameras_gt[name]["extrinsics"]) for name in cameras_names]
        ).to(device)

        cam_poses_gt = inverse_Rt(cameras_Rt_gt)[:, :3, 3]

        hsfm_output_dir = os.path.join(output_dir, f"{subject_name}_{subaction_name}")
        os.makedirs(hsfm_output_dir, exist_ok=True)

        for frame_idx in range(n_annotated_frames):
            frame_idx_str = str(frame_idx)

            kps3d_gt = torch.as_tensor(ground_truth["keypoints"][frame_idx_str]).to(
                device
            )

            pbar.set_postfix(
                subject_name=subject_name,
                subaction_name=subaction_name,
                frame_idx=frame_idx,
            )

            timing_info_path = os.path.join(
                hsfm_output_dir, f"timing_info_{frame_idx:05d}.json"
            )
            hsfm_output_path = os.path.join(
                hsfm_output_dir, f"hsfm_output_smpl_{frame_idx:05d}.pkl"
            )

            with open(timing_info_path, "rb") as f:
                timing_info = orjson.loads(f.read())

            with open(hsfm_output_path, "rb") as f:
                hsfm_output = pickle.load(f)

            if timing_info:
                total_timings["images_loading_time_seconds"].append(
                    timing_info["images_loading_time_seconds"]
                )
                total_timings["bboxes_loading_time_seconds"].append(
                    timing_info["bboxes_loading_time_seconds"]
                )
                total_timings["world_env_time_seconds"].append(
                    timing_info["world_env_time_seconds"]
                )
                total_timings["pose2d_time_seconds"].append(
                    timing_info["pose2d_time_seconds"]
                )
                total_timings["smpl_time_seconds"].append(
                    timing_info["smpl_time_seconds"]
                )
                total_timings["hsfm_time_seconds"].append(
                    timing_info["hsfm_time_seconds"]
                )
            else:
                warnings.warn(
                    f"No timing info found for {subject_name}_{subaction_name} frame {frame_idx}"
                )

            if hsfm_output:
                smpl_params = hsfm_output["hsfm_people(smplx_params)"][PERSON_ID]
                betas = torch.as_tensor(
                    smpl_params["betas"], dtype=torch.float32, device=device
                )
                body_pose = torch.as_tensor(
                    smpl_params["body_pose"], dtype=torch.float32, device=device
                ).reshape(1, -1, 3, 3)
                global_orient = torch.as_tensor(
                    smpl_params["global_orient"], dtype=torch.float32, device=device
                ).reshape(1, 1, 3, 3)
                root_transl = torch.as_tensor(
                    smpl_params["root_transl"], dtype=torch.float32, device=device
                ).reshape(1, 1, 3)

                cameras = hsfm_output["hsfm_places_cameras"].values()
                cameras_K = torch.stack(
                    [torch.as_tensor(cam_data["intrinsic"]) for cam_data in cameras]
                ).to(device)
                cameras_Rt_inv = torch.stack(
                    [torch.as_tensor(cam_data["cam2world"]) for cam_data in cameras]
                )[:, :3, :].to(device)
                cameras_Rt = inverse_Rt(cameras_Rt_inv)

                cam_poses = cameras_Rt_inv[:, :3, 3]

                with torch.no_grad():
                    smpl_output = smpl_layer.forward(
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

                    h36m_joints = vertices2joints(J_regressor, smplx_vertices)[0]
                    h36m_joints_SPIN = vertices2joints(
                        J_regressor_SPIN, smplx_vertices
                    )[0]

                # SE(3) similarity transform
                R1, T1, s1 = compute_similarity_transform(
                    cam_poses, cam_poses_gt, estimate_scale=False
                )

                cameras_Rt_transformed = apply_similarity_transform_to_Rt(
                    cameras_Rt, R1, T1, s1
                )
                cam_poses_transformed = apply_similarity_transform_to_points(
                    cam_poses, R1, T1, s1
                )
                h36m_joints_transformed = apply_similarity_transform_to_points(
                    h36m_joints, R1, T1, s1
                )
                h36m_joints_SPIN_transformed = apply_similarity_transform_to_points(
                    h36m_joints_SPIN, R1, T1, s1
                )
                smplx_vertices_transformed = apply_similarity_transform_to_points(
                    smplx_vertices, R1, T1, s1
                )

                if vis:
                    viewer = Viewer()
                    for camera_K, camera_Rt, camera_name in zip(
                        cameras_K, cameras_Rt_transformed, cameras_names
                    ):
                        camera = OpenCVCamera(
                            K=camera_K.cpu().numpy(),
                            Rt=camera_Rt.cpu().numpy(),
                            cols=1000,
                            rows=1000,
                        )
                        camera.name = camera_name
                        viewer.scene.add(camera)

                    for camera_K_gt, camera_Rt_gt, camera_name in zip(
                        cameras_K_gt, cameras_Rt_gt, cameras_names
                    ):
                        camera_gt = OpenCVCamera(
                            K=camera_K_gt.cpu().numpy(),
                            Rt=camera_Rt_gt.cpu().numpy(),
                            cols=1000,
                            rows=1000,
                        )
                        camera_gt.name = f"{camera_name}_gt"
                        viewer.scene.add(camera_gt)

                    smplx_mesh = Meshes(
                        vertices=smplx_vertices_transformed.reshape(-1, 3)
                        .cpu()
                        .numpy(),
                        faces=smpl_layer.faces,
                    )
                    smplx_mesh.name = "smplx_mesh"
                    viewer.scene.add(smplx_mesh)

                    # Create individual point clouds for each keypoint
                    h36m_joint_pcd = PointClouds(
                        points=h36m_joints_transformed.reshape(1, -1, 3).cpu().numpy(),
                    )
                    viewer.scene.add(h36m_joint_pcd)

                    h36m_joint_SPIN_pcd = PointClouds(
                        points=h36m_joints_SPIN_transformed.reshape(1, -1, 3)
                        .cpu()
                        .numpy(),
                    )
                    viewer.scene.add(h36m_joint_SPIN_pcd)

                    h36m_joint_gt_pcd = PointClouds(
                        points=kps3d_gt.reshape(1, -1, 3).cpu().numpy(),
                    )
                    viewer.scene.add(h36m_joint_gt_pcd)
                    viewer.run()

                # Compute translation error (TE)
                TE = torch.norm(cam_poses_transformed - cam_poses_gt, dim=-1).mean()

                # Compute W-MPJPE
                w_mpjpe_SPIN = torch.norm(
                    h36m_joints_SPIN_transformed - kps3d_gt, dim=-1
                ).mean()

                # Compute W-MPJPE with retrained J regressor
                w_mpjpe = torch.norm(h36m_joints_transformed - kps3d_gt, dim=-1).mean()

                R2, T2, s2 = compute_similarity_transform(
                    h36m_joints, kps3d_gt, estimate_scale=True
                )
                h36m_joints_pa = apply_similarity_transform_to_points(
                    h36m_joints, R2, T2, s2
                )

                R3, T3, s3 = compute_similarity_transform(
                    h36m_joints_SPIN, kps3d_gt, estimate_scale=True
                )
                h36m_joints_SPIN_pa = apply_similarity_transform_to_points(
                    h36m_joints_SPIN, R3, T3, s3
                )

                # Compute PA-MPJPE
                pa_mpjpe = torch.norm(h36m_joints_pa - kps3d_gt, dim=-1).mean()
                pa_mpjpe_SPIN = torch.norm(
                    h36m_joints_SPIN_pa - kps3d_gt, dim=-1
                ).mean()

                total_results["TE"].append(TE.item())
                total_results["w_mpjpe"].append(w_mpjpe.item())
                total_results["w_mpjpe_SPIN"].append(w_mpjpe_SPIN.item())
                total_results["pa_mpjpe"].append(pa_mpjpe.item())
                total_results["pa_mpjpe_SPIN"].append(pa_mpjpe_SPIN.item())

            pbar.update(1)

        w_mpjpe = float(np.mean(total_results["w_mpjpe"]))
        w_mpjpe_SPIN = float(np.mean(total_results["w_mpjpe_SPIN"]))
        pa_mpjpe = float(np.mean(total_results["pa_mpjpe"]))
        pa_mpjpe_SPIN = float(np.mean(total_results["pa_mpjpe_SPIN"]))
        TE = float(np.mean(total_results["TE"]))

        avg_hsfm_time_seconds = float(np.mean(total_timings["hsfm_time_seconds"]))

        result_path = os.path.join(
            output_dir, f"results_{subject_name}_{subaction_name}.json"
        )
        with open(result_path, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "w_mpjpe": w_mpjpe,
                        "w_mpjpe_SPIN": w_mpjpe_SPIN,
                        "pa_mpjpe": pa_mpjpe,
                        "pa_mpjpe_SPIN": pa_mpjpe_SPIN,
                        "TE": TE,
                        "avg_hsfm_time_seconds": avg_hsfm_time_seconds,
                    }
                )
            )
            print(f"Saved results to {result_path}")

    pbar.close()


if __name__ == "__main__":
    tyro.cli(main)
