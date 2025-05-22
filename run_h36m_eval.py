import orjson
import os
import json
from tqdm import tqdm
import tyro
from get_world_env_dust3r_for_hsfm import (
    get_world_env_dust3r_for_hsfm,
    load_dust3r_model,
)
from get_pose2d_vitpose_for_hsfm import get_pose2d_vitpose_for_hsfm, load_vitpose_model
from get_smpl_hmr2_for_hsfm import get_smpl_hmr2_for_hsfm, load_smpl_hmr2_model
from align_world_env_and_smpl_hsfm_optim import (
    align_world_env_and_smpl_hsfm,
    create_smplx_layer_dict,
    show_results as show_hsfm_results,
    save_results as save_hsfm_results,
)
import warnings
from time import perf_counter
import cv2
import numpy as np

import torch


def main(
    processed_dataset_dir: str = "./data/h36m/processed/",
    output_dir: str = "./data_output/h36m/",
    sequences_range: tuple[int, int] = (0, -1),
    vis: bool = False,
):
    print("CUDA available: ", torch.cuda.is_available())
    print("Current device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.backends.cuda.matmul.allow_tf32 = True

    with open(
        os.path.join(processed_dataset_dir, "h36m_protocol1_sequences.json"), "rb"
    ) as f:
        sequences = orjson.loads(f.read())

    if sequences_range[1] == -1:
        sequences_range = (0, len(sequences))

    if sequences_range[0] < 0 or sequences_range[1] > len(sequences):
            raise ValueError(
                f"Invalid sequences range: {sequences_range}. Must be between 0 and {len(sequences)}"
            )
    if sequences_range[0] > sequences_range[1]:
        raise ValueError(
            f"Invalid sequences range: {sequences_range}. Must be between 0 and {len(sequences)}"
        )
    sequences = sequences[sequences_range[0] : sequences_range[1]]

    if len(sequences) == 0:
        raise ValueError(
            f"No sequences to process. Please check the sequences range: {sequences_range}"
        )

    seq_names = [f"{seq['subject_name']}_{seq['subaction_name']}" for seq in sequences]
    print(f"Processing {len(sequences)} sequences: {', '.join(seq_names)}")

    total_frames = sum([seq["n_annotated_frames"] for seq in sequences])

    pbar = tqdm(total=total_frames, desc="Processing frames")

    device = "cuda"

    person_ids = [1]

    smplx_layer_dict = create_smplx_layer_dict(
        body_model_name="smpl",
        person_ids=person_ids,
        device=device,
    )

    dust3r_model = load_dust3r_model(
        model_path="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device=device,
    )

    smpl_hmr2_model = load_smpl_hmr2_model(
        model_path="./checkpoints/hmr2.ckpt",
        model_config_path="./configs/hmr2/model_config.yaml",
        device=device,
    )

    vitpose_model = load_vitpose_model(
        model_path="./checkpoints/vitpose_huge_wholebody.pth",
        model_config_path="./configs/vitpose/ViTPose_huge_wholebody_256x192.py",
        device=device,
    )

    for seq in sequences:
        subject_name = seq["subject_name"]
        subaction_name = seq["subaction_name"]
        n_annotated_frames = seq["n_annotated_frames"]

        hsfm_output_dir = os.path.join(output_dir, f"{subject_name}_{subaction_name}")
        os.makedirs(hsfm_output_dir, exist_ok=True)

        for frame_idx in range(n_annotated_frames):
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

            if os.path.exists(timing_info_path) and os.path.exists(hsfm_output_path):
                print(
                    f"Sequence {subject_name} {subaction_name} frame {frame_idx} already processed. Skipping..."
                )
                pbar.update(1)
                continue

            frame_idx_str = f"{frame_idx:05d}"
            frame_inputs_dir = os.path.join(
                processed_dataset_dir, seq["frames_rel_root_dir"], frame_idx_str
            )
            bbox_inputs_dir = os.path.join(
                processed_dataset_dir, seq["bboxes_rel_root_dir"], frame_idx_str
            )

            valid_inputs = True

            images_loading_time = 0
            bboxes_loading_time = 0

            # Load the image for each camera
            images = []
            images_bboxes = []
            images_indices = []
            for camera_idx in range(len(seq["cameras_names"])):
                image_path = os.path.join(frame_inputs_dir, f"{camera_idx:05d}.png")
                bbox_path = os.path.join(bbox_inputs_dir, f"mask_{camera_idx:05d}.json")

                if not os.path.exists(image_path):
                    warnings.warn(
                        f"Image not found: {image_path}. Skipping frame {frame_idx} for subject {subject_name} and subaction {subaction_name} camera {camera_idx}"
                    )
                    valid_inputs = False
                if not os.path.exists(bbox_path):
                    warnings.warn(
                        f"Bbox not found: {bbox_path}. Skipping frame {frame_idx} for subject {subject_name} and subaction {subaction_name} camera {camera_idx}"
                    )
                    valid_inputs = False

                image_loading_start_time = perf_counter()
                image = cv2.imread(image_path)
                image_loading_end_time = perf_counter()
                images_loading_time += image_loading_end_time - image_loading_start_time

                bbox_loading_start_time = perf_counter()
                with open(bbox_path, "r") as f:
                    image_bboxes = json.load(f)
                bbox_loading_end_time = perf_counter()
                bboxes_loading_time += bbox_loading_end_time - bbox_loading_start_time

                images.append(image)
                images_indices.append(camera_idx)
                images_bboxes.append(image_bboxes)

            if not valid_inputs:
                pbar.update(1)
                continue

            world_env_start_time = perf_counter()
            torch.cuda.synchronize()
            world_env_results = get_world_env_dust3r_for_hsfm(
                model=dust3r_model,
                images=images,
                images_indices=images_indices,
                verbose=False,
            )
            torch.cuda.synchronize()
            world_env_end_time = perf_counter()

            pose2d_start_time = perf_counter()
            torch.cuda.synchronize()
            pose2d_results = get_pose2d_vitpose_for_hsfm(
                model=vitpose_model,
                images=images,
                images_bboxes=images_bboxes,
                images_indices=images_indices,
            )
            torch.cuda.synchronize()
            pose2d_end_time = perf_counter()

            smpl_start_time = perf_counter()
            torch.cuda.synchronize()
            smpl_results, _ = get_smpl_hmr2_for_hsfm(
                model=smpl_hmr2_model,
                images=images,
                images_bboxes=images_bboxes,
                images_indices=images_indices,
                person_ids=person_ids,
                batch_size=1,
                show_progress=False,
            )
            torch.cuda.synchronize()
            smpl_end_time = perf_counter()

            # Convert the bboxes
            images_bboxes_params_dict = {}
            for image_idx, image_bboxes in zip(images_indices, images_bboxes):
                images_bboxes_params_dict[image_idx] = {}
                for person_id in person_ids:
                    bbox_data = image_bboxes["labels"][str(person_id)]
                    bbox = np.array(
                        [
                            bbox_data["x1"],
                            bbox_data["y1"],
                            bbox_data["x2"],
                            bbox_data["y2"],
                            1.0,
                        ]
                    )
                    images_bboxes_params_dict[image_idx][person_id] = bbox

            hsfm_start_time = perf_counter()
            torch.cuda.synchronize()
            hsfm_results = align_world_env_and_smpl_hsfm(
                world_env=world_env_results,
                images_smplx_params_dict=smpl_results,
                images_pose2d_params_dict=pose2d_results,
                images_bboxes_params_dict=images_bboxes_params_dict,
                images_sam2_mask_params_dict={},
                smplx_layer_dict=smplx_layer_dict,
                person_ids=person_ids,
                body_model_name="smpl",
                device=device,
                verbose=False,
                show_progress=False,
            )
            torch.cuda.synchronize()
            hsfm_end_time = perf_counter()

            timing_info = {
                "n_views": len(images),
                "images_loading_time_seconds": images_loading_time,
                "bboxes_loading_time_seconds": bboxes_loading_time,
                "world_env_time_seconds": world_env_end_time - world_env_start_time,
                "pose2d_time_seconds": pose2d_end_time - pose2d_start_time,
                "smpl_time_seconds": smpl_end_time - smpl_start_time,
                "hsfm_time_seconds": hsfm_end_time - hsfm_start_time,
            }

            with open(timing_info_path, "w") as f:
                json.dump(timing_info, f)

            save_hsfm_results(results=hsfm_results, output_path=hsfm_output_path)

            if vis:
                show_hsfm_results(
                    results=hsfm_results,
                    body_model_name="smpl",
                    smplx_layer_dict=smplx_layer_dict,
                    device=device,
                )

            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    tyro.cli(main)
