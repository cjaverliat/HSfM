# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import torch
import os
import cv2
import numpy as np
import tyro
import pickle
import json
from tqdm import tqdm
from pathlib import Path
from typing import Any
import glob
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
        model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    else:
        model_cfg = config_path
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and (
        "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg

def load_smpl_hmr2_model(
    model_path: str = DEFAULT_CHECKPOINT,
    model_config_path: str = "",
    device: str = "cuda",
):
    model, _ = load_hmr2(model_path, model_config_path)
    model = model.to(device)
    model.eval()
    return model

def get_smpl_hmr2_for_hsfm(
    model: HMR2,
    images: list[np.ndarray],
    images_bboxes: list[dict[str, np.ndarray]],
    images_indices: list[int],
    person_ids: list[int],
    batch_size: int = 1,
) -> tuple[dict[int, Any], dict[int, Any]]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset_list = []
    for image, image_bbox, image_idx in tqdm(
        zip(images, images_bboxes, images_indices)
    ):
        # if value of "labels" key is empty, continue
        if not image_bbox["labels"]:
            continue
        else:
            labels = image_bbox["labels"]
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
        dataset = ViTDetDataset(model.cfg, image, boxes, target_person_ids, image_idx)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        dataset_list.append(dataset)

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    dataloader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    results = {}
    vis_results = {}
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        batch_size = batch["img"].shape[0]
        batch_pred_smpl_params = out["pred_smpl_params"]

        pred_cam = out["pred_cam"]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        pred_vertices = out["pred_vertices"]

        scaled_focal_length = (
            model.cfg.EXTRA.FOCAL_LENGTH / model.cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            .detach()
            .cpu()
            .numpy()
        )

        for n in range(batch_size):
            image_idx = int(batch["frame_idx"][n])
            person_id = int(batch["personid"][n])

            pred_smpl_params = {}
            # 'global_orient': (1, 3, 3), 'body_pose': (23, 3, 3), 'betas': (10)
            for key in batch_pred_smpl_params.keys():
                pred_smpl_params[key] = (
                    batch_pred_smpl_params[key][n].detach().cpu().numpy()
                )

            verts = pred_vertices[n].detach().cpu().numpy()
            cam_t = pred_cam_t_full[n]

            results.setdefault(image_idx, {})
            results[image_idx][person_id] = {
                "smpl_params": pred_smpl_params,
            }

            vis_results.setdefault(image_idx, {})
            vis_results[image_idx][person_id] = {
                "verts": verts,
                "cam_t": cam_t,
                "img_size": img_size[n].tolist(),
            }

    return results, vis_results


def save_results(results: dict[int, Any], output_dir: str) -> None:
    for image_idx in results.keys():
        frame_result_save_path = os.path.join(
            output_dir, f"smpl_params_{image_idx:05d}.pkl"
        )
        with open(frame_result_save_path, "wb") as f:
            pickle.dump(results[image_idx], f)


def save_vis_results(
    vis_results: dict[int, Any],
    images: list[np.ndarray],
    images_indices: list[int],
    output_dir: str,
    model: HMR2,
) -> None:
    renderer = Renderer(model.cfg, faces=model.smpl.faces)

    for image, image_idx in zip(images, images_indices):
        if image_idx not in vis_results:
            continue

        vis_result = vis_results[image_idx]

        all_verts = []
        all_cam_t = []
        img_size = []
        for person_id, result in vis_result.items():
            all_verts.append(result["verts"])
            all_cam_t.append(result["cam_t"])
            img_size.append(result["img_size"])

        scaled_focal_length = (
            model.cfg.EXTRA.FOCAL_LENGTH
            / model.cfg.MODEL.IMAGE_SIZE
            * torch.as_tensor(img_size).max()
        )

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
            input_img = image.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )

            filepath = os.path.join(output_dir, f"smpl_{image_idx:05d}_all.jpg")
            os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(
                filepath,
                255 * input_img_overlay[:, :, ::-1],
            )


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
    vis: bool = False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Download checkpoints
    download_models(CACHE_DIR_4DHUMANS)

    # Setup HMR2.0 model
    model, _ = load_hmr2(model_checkpoint, model_config)
    model = model.to(device)
    model.eval()

    # Make output directory if it does not exist
    output_dir = os.path.join(output_dir, os.path.basename(img_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all demo images that end with .jpg or .png
    img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if img_path_list == []:
        img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    images = []
    images_bboxes = []
    images_indices = []

    for img_path in img_path_list:
        img_idx = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        image = cv2.imread(img_path)
        det_result_path = os.path.join(bbox_dir, f"mask_{img_idx:05d}.json")
        with open(det_result_path, "r") as f:
            det_results = json.load(f)
        images.append(image)
        images_indices.append(img_idx)
        images_bboxes.append(det_results)

    results, vis_results = get_smpl_hmr2_for_hsfm(
        model=model,
        images=images,
        images_bboxes=images_bboxes,
        images_indices=images_indices,
        person_ids=person_ids,
        batch_size=batch_size,
    )

    save_results(results, output_dir)

    if vis:
        save_vis_results(
            vis_results=vis_results,
            images=images,
            images_indices=images_indices,
            output_dir=os.path.join(output_dir, "vis"),
            model=model,
        )


if __name__ == "__main__":
    tyro.cli(main)
