# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import torch
import os
import json
import cv2
import numpy as np
import tyro
import pickle
import orjson
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, DEFAULT_CHECKPOINT, check_smpl_exists, HMR2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def load_hmr2(checkpoint_path=DEFAULT_CHECKPOINT, config_path=""):
    from pathlib import Path
    from hmr2.configs import get_config
    if config_path == "":
        model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    else:
        model_cfg = config_path
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg

def main(
    model_checkpoint: str = DEFAULT_CHECKPOINT,
    model_config: str = "",
    img_dir: str = "./demo_data/input_images/arthur_tyler_pass_by_nov20/cam01",
    bbox_dir: str = "./demo_data/input_masks/arthur_tyler_pass_by_nov20/cam01/json_data",
    output_dir: str = "./demo_data/input_3d_meshes/arthur_tyler_pass_by_nov20/cam01",
    batch_size: int = 1,
    person_ids: list = [
        1,
    ],
    timing_info_dir: str = "./demo_output/timing_info",
    vis: bool = False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Download checkpoints
    download_models(CACHE_DIR_4DHUMANS)

    load_model_start_time = perf_counter()
    # Setup HMR2.0 model
    model, model_cfg = load_hmr2(model_checkpoint, model_config)
    model = model.to(device)
    model.eval()
    load_model_end_time = perf_counter()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    output_dir = os.path.join(output_dir, os.path.basename(img_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_images_start_time = perf_counter()
    # Get all demo images that end with .jpg or .png
    img_paths = [img for img in Path(img_dir).glob("*.jpg")]
    if len(img_paths) == 0:
        img_paths = [img for img in Path(img_dir).glob("*.png")]
    img_paths.sort()

    # Iterate over all images in folder
    print("Batchifying input images...")
    dataset_list = []
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # read bbox
        frame_idx = int(img_path.stem.split("_")[-1])
        bbox_path = Path(bbox_dir) / f"mask_{frame_idx:05d}.json"
        with open(bbox_path, "r") as f:
            bbox_data = json.load(f)
        # if value of "labels" key is empty, continue
        if not bbox_data["labels"]:
            continue
        else:
            labels = bbox_data["labels"]
            # "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 454, "y1": 399, "x2": 562, "y2": 734, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 45, "y1": 301, "x2": 205, "y2": 812, "logit": 0.0}}}
            label_keys = sorted(labels.keys())

            # filter label keys by person ids
            selected_label_keys = [
                x for x in label_keys if labels[x]["instance_id"] in person_ids
            ]
            label_keys = selected_label_keys

            # get boxes
            boxes = np.array(
                [
                    [
                        labels[str(i)]["x1"],
                        labels[str(i)]["y1"],
                        labels[str(i)]["x2"],
                        labels[str(i)]["y2"],
                    ]
                    for i in label_keys
                ]
            )
            # get target person ids
            target_person_ids = np.array(
                [labels[str(i)]["instance_id"] for i in label_keys]
            )

            # sanity check; if boxes is empty, continue
            if boxes.sum() == 0:
                continue

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, target_person_ids, frame_idx)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        dataset_list.append(dataset)

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)
    load_images_end_time = perf_counter()

    dataloader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    timing_info = {
        "load_model_time": load_model_end_time - load_model_start_time,
        "load_images_time": load_images_end_time - load_images_start_time,
        "batch": [],
    }

    print("Running HMR2.0...")
    result_dict = defaultdict(
        dict
    )  # key: frame_idx, value: dictionary with keys: 'all_verts', 'all_cam_t'
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        inference_start_time = perf_counter()
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        inference_end_time = perf_counter()

        batch_timing_info = {
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "inference_time": inference_end_time - inference_start_time,
        }
        timing_info["batch"].append(batch_timing_info)

        pred_cam = out["pred_cam"]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            .detach()
            .cpu()
            .numpy()
        )

        batch_size = batch["img"].shape[0]
        batch_pred_smpl_params = out["pred_smpl_params"]
        for n in range(batch_size):
            frame_idx = int(batch["frame_idx"][n])
            person_id = int(batch["personid"][n])

            pred_smpl_params = {}
            # 'global_orient': (1, 3, 3), 'body_pose': (23, 3, 3), 'betas': (10)
            for key in batch_pred_smpl_params.keys():
                pred_smpl_params[key] = (
                    batch_pred_smpl_params[key][n].detach().cpu().numpy()
                )

            result_dict[frame_idx][person_id] = {
                "smpl_params": pred_smpl_params,
            }

            if vis:
                # Add all verts and cams to list
                verts = out["pred_vertices"][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                # update the dictionary
                result_dict[frame_idx][person_id]["verts"] = verts
                result_dict[frame_idx][person_id]["cam_t"] = cam_t
                result_dict[frame_idx][person_id]["img_size"] = img_size[n].tolist()

    print("Saving results...")
    save_start_time = perf_counter()
    # Save the result
    for frame_idx in sorted(result_dict.keys()):
        frame_result_save_path = os.path.join(
            output_dir, f"smpl_params_{frame_idx:05d}.pkl"
        )
        with open(frame_result_save_path, "wb") as f:
            pickle.dump(result_dict[frame_idx], f)
    save_end_time = perf_counter()

    timing_info["save_time"] = save_end_time - save_start_time

    with open(os.path.join(timing_info_dir, "smpl_hmr2_timing_info.json"), "wb") as f:
        f.write(orjson.dumps(timing_info, option=orjson.OPT_INDENT_2))

    print(
        f"Timing info saved to {os.path.join(timing_info_dir, 'smpl_hmr2_timing_info.json')}"
    )

    # Render front view
    if vis:
        print("Rendering result overlay...")
        for file_idx, frame_idx in enumerate(result_dict.keys()):
            all_verts = []
            all_cam_t = []
            img_size = []
            for person_id, result in result_dict[frame_idx].items():
                all_verts.append(result["verts"])
                all_cam_t.append(result["cam_t"])
                img_size.append(result["img_size"])

            if len(all_verts) > 0:
                img_size = img_size[0]
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t, render_res=img_size, **misc_args
                )

                # Overlay image
                img_fn = os.path.splitext(os.path.basename(img_paths[file_idx]))[0]
                img_cv2 = cv2.imread(str(img_paths[file_idx]))
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate(
                    [input_img, np.ones_like(input_img[:, :, :1])], axis=2
                )  # Add alpha channel
                input_img_overlay = (
                    input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                    + cam_view[:, :, :3] * cam_view[:, :, 3:]
                )

                Path(os.path.join(output_dir, "vis")).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    os.path.join(output_dir, "vis", f"{img_fn}_all.png"),
                    255 * input_img_overlay[:, :, ::-1],
                )


if __name__ == "__main__":
    tyro.cli(main)
