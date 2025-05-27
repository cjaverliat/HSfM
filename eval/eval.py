import json
import pickle
import torch
import matplotlib.pyplot as plt
from transformations import (
    inverse_Rt,
    compute_similarity_transform,
    apply_similarity_transform_to_Rt,
    apply_similarity_transform_to_points,
)

if __name__ == "__main__":
    frame_idx = 0

    with open(
        "/mnt/hdd_storage/Charles_JAVERLIAT/Datasets/H3.6M/processed_hsfm/ground_truth/S9_Directions 1.json",
        "r",
    ) as f:
        ground_truth = json.load(f)

    with open(
        f"demo_output/h36m/S9_Directions 1/hsfm_output_smpl_{frame_idx:05d}.pkl", "rb"
    ) as f:
        data = pickle.load(f)

    Rts = []
    Rts_gt = []

    cam_poses = []
    cam_poses_gt = []

    for camera_name, camera_gt in ground_truth["cameras"].items():
        extrinsics = torch.as_tensor(camera_gt["extrinsics"])
        intrinsics = torch.as_tensor(camera_gt["intrinsics"])
        cam_pose = inverse_Rt(extrinsics)[:3, 3]
        Rts_gt.append(extrinsics)
        cam_poses_gt.append(cam_pose)

    for camera_idx, camera in data["hsfm_places_cameras"].items():
        img_shape = camera["rgbimg"].shape[:2]
        cam2world = torch.from_numpy(camera["cam2world"])[:3, :]
        world2cam = inverse_Rt(cam2world)
        Rts.append(world2cam)
        cam_pose = cam2world[:3, 3]
        cam_poses.append(cam_pose)

    Rts = torch.stack(Rts)
    Rts_gt = torch.stack(Rts_gt)
    cam_poses = torch.stack(cam_poses)
    cam_poses_gt = torch.stack(cam_poses_gt)

    print(Rts.shape)
    print(Rts_gt.shape)
    print(cam_poses.shape)
    print(cam_poses_gt.shape)

    # SE(3) similarity transform
    R1, T1, s1 = compute_similarity_transform(
        cam_poses, cam_poses_gt, estimate_scale=False
    )
    assert s1.item() == 1.0

    Rts_transformed = apply_similarity_transform_to_Rt(Rts, R1, T1, s1)
    cam_poses_transformed = apply_similarity_transform_to_points(cam_poses, R1, T1, s1)

    # Compare Rts_transformed and Rts_gt
    print(Rts_transformed)
    print(Rts_gt)

    # Compute translation error (TE)
    TE = torch.norm(cam_poses_transformed - cam_poses_gt, dim=-1).mean()
    print(f"Translation error: {TE}")

    # Compute scaled translation error s-TE
    # Sim(3) similarity transform
    R2, T2, s2 = compute_similarity_transform(
        cam_poses, cam_poses_gt, estimate_scale=True
    )

    cam_poses_scaled_transformed = apply_similarity_transform_to_points(
        cam_poses, R2, T2, s2
    )
    sTE = torch.norm(cam_poses_scaled_transformed - cam_poses_gt, dim=-1).mean()
    print(f"Scaled translation error: {sTE}")
    
