import os
from typing import Any, Literal
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import orjson
import roma
import roma.euler
import torch
from spacepy import pycdf
from tqdm import tqdm
import subprocess
import tyro

from preprocess_utils import compute_keypoints_2d, compute_bboxes_xyxy, inverse_Rt
from file import extract_tar_with_progress

BLACKLISTED_SEQUENCES = [
    ("S9", "Greeting"),
    ("S9", "SittingDown"),
    ("S9", "Waiting"),
]

DOWNLOADED_SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

PROTOCOL1_TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
PROTOCOL1_VAL_SUBJECTS = ["S9", "S11"]
PROTOCOL1_SELECTED_CAMERAS = ["54138969", "55011271", "58860488", "60457274"]

# Mapping from the original H3.6M format to MMPose's documented format
# References:
# - https://github.com/qxcv/pose-prediction/blob/master/H36M-NOTES.md#joint-information
# - https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html#human3-6m
H36M_JOINTS_MAPPING = [11, 1, 2, 3, 6, 7, 8, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]


def _get_video_rel_filepath(
    subject_name: str, subaction_name: str, camera_name: str
) -> str:
    return f"{subject_name}/Videos/{subaction_name}.{camera_name}.mp4"


def _serializer_fallback(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def preprocess_h36m(
    raw_dataset_dir: str,
    processed_dataset_dir: str,
    skip_extract: bool = False,
    skip_frames_extraction: bool = False,
    splits: list[Literal["train", "val"]] = ["train", "val"],
):
    """
    Preprocess the Human3.6M dataset for use in Kineo.

    This function extracts and processes the Human3.6M dataset, creating annotations in JSON format following Protocol 1.
    It also creates train/val splits for Protocol 1.

    Args:
        raw_dataset_dir: Path to the directory containing the downloaded H3.6M dataset files
        process_dataset_dir: Path where the processed data will be saved
        skip_extract: If True, skip the extraction of archive files (useful if already extracted)
    """
    os.makedirs(processed_dataset_dir, exist_ok=True)

    if not skip_extract:
        _extract_data(raw_dataset_dir)

    actions = _parse_actions_metadata(raw_dataset_dir, processed_dataset_dir)
    actions_annotations = _generate_protocol1_annotations(actions, raw_dataset_dir)

    h36m_protocol1_sequences = []

    pbar = tqdm(total=len(actions), desc="Processing actions", leave=False)
    for action, annotations in zip(actions, actions_annotations):
        subject_name = action["subject_name"]
        action_name = action["action_name"]
        subaction_name = action["subaction_name"]
        cameras_names = action["cameras_names"]
        if (
            subject_name not in PROTOCOL1_VAL_SUBJECTS
            and subject_name not in PROTOCOL1_TRAIN_SUBJECTS
        ):
            continue

        split = "train" if subject_name in PROTOCOL1_TRAIN_SUBJECTS else "val"

        if split not in splits:
            continue

        n_annotated_frames = len(annotations["3d"]["keypoints"])

        if not skip_frames_extraction:
            _extract_protocol1_frames(
                action=action,
                raw_dataset_dir=raw_dataset_dir,
                processed_dataset_dir=processed_dataset_dir,
                n_annotated_frames=n_annotated_frames,
                show_progress=True,
            )

        _export_bboxes_json(
            action=action,
            annotations=annotations,
            n_annotated_frames=n_annotated_frames,
            processed_dataset_dir=processed_dataset_dir,
        )

        _export_gt_json(
            action=action,
            annotations=annotations,
            n_annotated_frames=n_annotated_frames,
            processed_dataset_dir=processed_dataset_dir,
        )

        h36m_protocol1_sequences.append(
            {
                "subject_name": subject_name,
                "action_name": action_name,
                "subaction_name": subaction_name,
                "n_annotated_frames": n_annotated_frames,
                "cameras_names": cameras_names,
                "frames_rel_root_dir": os.path.join(
                    "video_frames",
                    f"{subject_name}_{subaction_name}",
                ),
                "bboxes_rel_root_dir": os.path.join(
                    "video_bboxes",
                    f"{subject_name}_{subaction_name}",
                ),
                "gt_rel_filepath": os.path.join(
                    "ground_truth",
                    f"{subject_name}_{subaction_name}.json",
                ),
            }
        )

        pbar.update(1)

    pbar.close()

    with open(
        os.path.join(processed_dataset_dir, "h36m_protocol1_sequences.json"),
        "wb",
    ) as f:
        f.write(
            orjson.dumps(
                h36m_protocol1_sequences,
                default=_serializer_fallback,
                option=orjson.OPT_INDENT_2,
            )
        )

    print(f"Saved {len(h36m_protocol1_sequences)} sequences to {processed_dataset_dir}")


def _export_gt_json(
    action: dict[str, Any],
    annotations: dict[str, Any],
    n_annotated_frames: int,
    processed_dataset_dir: str,
):
    subject_name = action["subject_name"]
    action_name = action["action_name"]
    subaction_name = action["subaction_name"]
    cameras_names = action["cameras_names"]

    gt_path = os.path.join(
        processed_dataset_dir,
        "ground_truth",
        f"{subject_name}_{subaction_name}.json",
    )

    gt_json = {
        "n_annotated_frames": n_annotated_frames,
        "subject_name": subject_name,
        "action_name": action_name,
        "subaction_name": subaction_name,
        "keypoints": {
            str(frame_idx): annotations["3d"]["keypoints"][str(frame_idx)]
            for frame_idx in range(n_annotated_frames)
        },
        "cameras": {
            camera_name: {
                "intrinsics": annotations["cameras"][camera_name]["intrinsics"],
                "extrinsics": annotations["cameras"][camera_name]["extrinsics"],
                "distortion_coefficients": annotations["cameras"][camera_name][
                    "distortion_coefficients"
                ],
            }
            for camera_name in cameras_names
        },
    }

    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    with open(gt_path, "wb") as f:
        f.write(orjson.dumps(gt_json, default=_serializer_fallback))


def _export_bboxes_json(
    action: dict[str, Any],
    annotations: dict[str, Any],
    n_annotated_frames: int,
    processed_dataset_dir: str,
):
    subject_name = action["subject_name"]
    subaction_name = action["subaction_name"]
    cameras_names = action["cameras_names"]

    total = n_annotated_frames * len(cameras_names)
    pbar = tqdm(total=total, desc="Exporting bboxes", leave=False)

    for camera_idx, camera_name in enumerate(cameras_names):
        for frame_idx in range(n_annotated_frames):
            bbox_path = os.path.join(
                processed_dataset_dir,
                "video_bboxes",
                f"{subject_name}_{subaction_name}",
                f"{frame_idx:05d}",
                f"mask_{camera_idx:05d}.json",
            )
            bbox = annotations["2d"]["bboxes_xyxy"][str(frame_idx)][camera_name]
            bbox_json = {
                "labels": {
                    "1": {
                        "instance_id": 1,
                        "class_name": "person",
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "logit": 0.0,
                    }
                }
            }

            os.makedirs(os.path.dirname(bbox_path), exist_ok=True)
            with open(bbox_path, "wb") as f:
                f.write(orjson.dumps(bbox_json, default=_serializer_fallback))

            pbar.update(1)

    pbar.close()


def _extract_protocol1_frames(
    action: dict[str, Any],
    raw_dataset_dir: str,
    processed_dataset_dir: str,
    n_annotated_frames: int,
    show_progress: bool = False,
    filetype: Literal["jpg", "png"] = "png",
):
    subject_name = action["subject_name"]
    subaction_name = action["subaction_name"]
    cameras_names = action["cameras_names"]

    # Extract frames from the videos at 10Hz and generate the json files for HSfM
    for camera_idx, camera_name in enumerate(cameras_names):
        video_rel_path = _get_video_rel_filepath(
            subject_name, subaction_name, camera_name
        )
        video_abs_filepath = os.path.join(raw_dataset_dir, video_rel_path)

        if not os.path.exists(video_abs_filepath):
            raise FileNotFoundError(f"Video {video_abs_filepath} does not exist")

        for frame_idx in range(n_annotated_frames):
            frame_dir = os.path.join(
                processed_dataset_dir,
                "video_frames",
                f"{subject_name}_{subaction_name}",
                f"{frame_idx:05d}",
            )
            os.makedirs(frame_dir, exist_ok=True)

        video_duration_ms = int(n_annotated_frames / 10 * 1000)

        extract_frames_cmd = [
            "ffmpeg",
            "-y",
            "-progress",
            "pipe:1",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            video_abs_filepath,
            "-start_number",
            "0",
            "-frames:v",
            str(n_annotated_frames),
            "-vf",
            "fps=10",
            os.path.join(
                processed_dataset_dir,
                "video_frames",
                f"{subject_name}_{subaction_name}",
                f"%05d/{camera_idx:04d}.{filetype}",
            ),
        ]

        p = subprocess.Popen(
            extract_frames_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        pbar = tqdm(
            desc=f"Extracting video frames for {subject_name} {subaction_name}",
            total=video_duration_ms,
            unit="ms",
            leave=False,
            disable=not show_progress,
        )

        while True:
            line = p.stdout.readline()
            if not line:
                break
            line = line.strip()

            if line.startswith("out_time_us="):
                time_ms = int(line.split("out_time_us=")[1]) // 1000
                pbar.n = time_ms
                pbar.refresh()
            if line == "progress=end":
                pbar.close()
                break
        pbar.n = pbar.total
        pbar.close()
        p.wait()

        if p.returncode != 0:
            e = p.stderr.read()
            raise Exception(f"ffmpeg command failed: {e}")


def _generate_protocol1_annotations(
    actions: list[dict[str, Any]],
    raw_dataset_dir: str,
) -> list[dict[str, Any]]:
    pbar = tqdm(total=len(actions), desc="Generating annotations", leave=False)

    all_annotations = []

    for action in actions:
        subject_name = action["subject_name"]
        subaction_name = action["subaction_name"]

        pbar.set_postfix_str(f"{subject_name} {subaction_name}")

        cameras_names = action["cameras_names"]
        cameras_intrinsics = action["cameras_intrinsics"]
        cameras_distortion_coefficients = action["cameras_distortion_coefficients"]
        cameras_extrinsics = action["cameras_extrinsics"]
        cameras_resolution_hw = action["cameras_resolution_hw"]
        cameras_n_frames = action["cameras_n_frames"]

        n_cameras = len(cameras_names)
        n_keypoints = 17

        keypoints_3d_world = _get_keypoints_3d(
            raw_dataset_dir, subject_name, subaction_name
        )

        # Downsample to 10Hz
        n_frames = min(keypoints_3d_world.shape[0], min(cameras_n_frames))
        keypoints_3d_world = keypoints_3d_world[:n_frames][::5]
        n_frames = keypoints_3d_world.shape[0]

        keypoints_2d = torch.zeros((n_frames, n_cameras, n_keypoints, 2))
        bboxes_xyxy = torch.zeros((n_frames, n_cameras, 4))

        # Project the 3D keypoints into each camera
        for camera_idx in range(n_cameras):
            keypoints_2d[:, camera_idx] = compute_keypoints_2d(
                keypoints_3d_world=keypoints_3d_world,
                K=cameras_intrinsics[camera_idx],
                Rt=cameras_extrinsics[camera_idx],
                distortion_coefficients=cameras_distortion_coefficients[camera_idx],
            )

            bboxes_xyxy[:, camera_idx] = compute_bboxes_xyxy(
                poses_2d=keypoints_2d[:, camera_idx],
                padding_x=50,
                padding_y=60,
                image_size_hw=cameras_resolution_hw[camera_idx],
                clamp_to_image_size=True,
            )

        annotations = {
            "2d": {
                "bboxes_xyxy": {
                    str(frame_idx): {
                        camera_name: bboxes_xyxy[frame_idx, camera_idx]
                        for camera_idx, camera_name in enumerate(cameras_names)
                    }
                    for frame_idx in range(n_frames)
                },
                "keypoints": {
                    str(frame_idx): {
                        camera_name: keypoints_2d[frame_idx, camera_idx]
                        for camera_idx, camera_name in enumerate(cameras_names)
                    }
                    for frame_idx in range(n_frames)
                },
            },
            "3d": {
                "keypoints": {
                    str(frame_idx): keypoints_3d_world[frame_idx]
                    for frame_idx in range(n_frames)
                },
            },
            "cameras": {
                camera_name: {
                    "intrinsics": cameras_intrinsics[camera_idx],
                    "extrinsics": cameras_extrinsics[camera_idx],
                    "distortion_coefficients": cameras_distortion_coefficients[
                        camera_idx
                    ],
                }
                for camera_idx, camera_name in enumerate(cameras_names)
            },
        }

        all_annotations.append(annotations)
        pbar.update(1)

    pbar.close()
    return all_annotations


def _extract_data(raw_dataset_dir: str):
    for subject_name in tqdm(DOWNLOADED_SUBJECTS, desc="Extracting data"):
        videos_tar_path = os.path.join(raw_dataset_dir, f"Videos_{subject_name}.tgz")
        meshes_tar_path = os.path.join(raw_dataset_dir, f"Meshes_{subject_name}.tgz")
        poses_tar_path = os.path.join(
            raw_dataset_dir, f"Poses_D3_Positions_{subject_name}.tgz"
        )

        extract_tar_with_progress(
            videos_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' videos",
            leave=False,
        )

        extract_tar_with_progress(
            poses_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' poses",
            leave=False,
        )
        extract_tar_with_progress(
            meshes_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' meshes",
            leave=False,
        )


def _parse_actions_metadata(
    raw_dataset_dir: str, processed_dataset_dir: str
) -> list[dict[str, Any]]:
    code_zip = os.path.join(raw_dataset_dir, "code-v1.2.zip")

    if not os.path.exists(code_zip):
        raise FileNotFoundError(f"Code zip file not found: {code_zip}")

    zipfile = ZipFile(code_zip, "r")

    with zipfile.open("Release-v1.2/metadata.xml") as f:
        root = ET.fromstring(f.read())

    mapping = list(root.find("mapping"))

    camera_names = [elem.text for elem in root.find("dbcameras/index2id")]
    # Subject names from S1 to S11
    subjects_name = [elem.text for elem in mapping[0]][2:]

    # Parse all the actions/subactions
    actions: list[dict[str, Any]] = []
    action_names = [elem.text for elem in root.find("actionnames")]

    # Skip header and first two rows (_ALL actions)
    for action_row in mapping[3:33]:
        action_id = int(action_row[0].text) - 1
        action_name = action_names[action_id]

        for subject_idx, action_column in enumerate(action_row[2:]):
            subject_name = subjects_name[subject_idx]
            subaction_name = action_column.text

            actions.append(
                {
                    "subject_name": subject_name,
                    "action_name": action_name,
                    "subaction_name": subaction_name,
                    "cameras_names": camera_names,
                }
            )

    n_cameras = len(camera_names)
    n_subjects = len(subjects_name)

    # The layout of the metadata is as follows:
    # - The first 264 elements are the camera extrinsics organized as [-x, y, -z, tx, ty, tz] * n_cameras * n_subjects
    # - The next 36 elements are the camera intrinsics organized as [fx, fy, cx, cy, k1, k2, p1, p2, k3] * n_cameras
    cameras_params = torch.from_numpy(
        np.fromstring(root.find("w0").text.strip("[]"), sep=" ")
    ).to(torch.float32)
    cameras_extrinsics_params = cameras_params[:264].reshape(n_cameras, n_subjects, 6)
    cameras_intrinsics_params = cameras_params[264:].reshape(n_cameras, 9)

    cameras_intrinsics = torch.zeros(n_cameras, 3, 3)
    cameras_intrinsics[..., 0, 0] = cameras_intrinsics_params[..., 0]  # fx
    cameras_intrinsics[..., 1, 1] = cameras_intrinsics_params[..., 1]  # fy
    cameras_intrinsics[..., 0, 2] = cameras_intrinsics_params[..., 2]  # cx
    cameras_intrinsics[..., 1, 2] = cameras_intrinsics_params[..., 3]  # cy
    cameras_intrinsics[..., 2, 2] = 1.0
    cameras_distortion_coefficients = torch.zeros(n_cameras, 5)
    cameras_distortion_coefficients[..., 0] = cameras_intrinsics_params[..., 4]  # k1
    cameras_distortion_coefficients[..., 1] = cameras_intrinsics_params[..., 5]  # k2
    cameras_distortion_coefficients[..., 2] = cameras_intrinsics_params[..., 6]  # p1
    cameras_distortion_coefficients[..., 3] = cameras_intrinsics_params[..., 7]  # p2
    cameras_distortion_coefficients[..., 4] = cameras_intrinsics_params[..., 8]  # k3

    # Each subject has different extrinsics
    cameras_extrinsics: dict[str, torch.Tensor] = {}
    for subject_idx in range(n_subjects):
        subject_name = subjects_name[subject_idx]

        # Camera parameters are stored as cam2world in left-handed coordinates
        # we negate angles to flip coordinate system (left-handed to right-handed coordinate system)
        angles = -cameras_extrinsics_params[..., subject_idx, :3].clone()
        cam_rot = roma.euler.euler_to_rotmat(convention="xyz", angles=angles)
        cam_pose = cameras_extrinsics_params[..., subject_idx, 3:6].unsqueeze(-1)
        cam_pose = cam_pose / 1000.0  # Convert from mm to m

        cam2world = torch.cat((cam_rot, cam_pose), dim=-1)
        world2cam = inverse_Rt(cam2world)

        cameras_extrinsics[subject_name] = world2cam

    # Remove blacklisted sequences and only keep S1, S5, S6, S7, S8, S9, S11
    actions = [
        a
        for a in actions
        if (a["subject_name"], a["action_name"]) not in BLACKLISTED_SEQUENCES
        and a["subject_name"] in DOWNLOADED_SUBJECTS
    ]

    for action in actions:
        subject_name = action["subject_name"]
        subaction_name = action["subaction_name"]
        cameras_names = action["cameras_names"]

        cameras_info = [
            _get_video_info(
                os.path.join(
                    raw_dataset_dir,
                    _get_video_rel_filepath(subject_name, subaction_name, camera_name),
                )
            )
            for camera_name in cameras_names
        ]

        action["cameras_intrinsics"] = cameras_intrinsics
        action["cameras_distortion_coefficients"] = cameras_distortion_coefficients
        action["cameras_extrinsics"] = cameras_extrinsics[subject_name]
        action["cameras_resolution_hw"] = [
            (camera_info["height"], camera_info["width"])
            for camera_info in cameras_info
        ]
        action["cameras_n_frames"] = [
            camera_info["n_frames"] for camera_info in cameras_info
        ]

    return actions


def _get_video_info(
    video_path: str,
) -> dict[str, Any]:
    assert os.path.exists(video_path), f"Video {video_path} does not exist"
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": width,
        "height": height,
        "n_frames": n_frames,
    }


def _get_keypoints_3d(
    raw_dataset_dir: str, subject_name: str, subaction_name: str
) -> torch.Tensor:
    poses_cdf_fp = os.path.join(
        raw_dataset_dir,
        subject_name,
        "MyPoseFeatures/D3_Positions/",
        f"{subaction_name}.cdf",
    )

    keypoints_3d_world = pycdf.CDF(poses_cdf_fp)["Pose"][0]
    keypoints_3d_world = torch.from_numpy(keypoints_3d_world).to(torch.float32)
    n_frames = keypoints_3d_world.shape[0]

    # Convert from mm to meters
    keypoints_3d_world = keypoints_3d_world.reshape(n_frames, -1, 3) / 1000.0
    # Map original H3.6M joints to MMPose joints
    keypoints_3d_world = keypoints_3d_world[:, H36M_JOINTS_MAPPING]
    return keypoints_3d_world.reshape(n_frames, -1, 3)


def main(
    raw_dataset_dir: str = "./data/h36m/raw",
    processed_dataset_dir: str = "./data/h36m/processed",
    skip_tar_extraction: bool = False,
    skip_frames_extraction: bool = False,
    splits: list[Literal["train", "val"]] = ["train", "val"],
):
    preprocess_h36m(
        raw_dataset_dir,
        processed_dataset_dir,
        skip_tar_extraction,
        skip_frames_extraction,
        splits,
    )


if __name__ == "__main__":
    tyro.cli(main)
