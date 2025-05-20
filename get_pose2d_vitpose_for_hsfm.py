# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import os
import glob
import tyro
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import orjson

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

from time import perf_counter

os.environ["PYOPENGL_PLATFORM"] = "egl"


class ViTPoseModel:
    MODEL_DICT = {
        "ViTPose+-G (multi-task train, COCO)": {
            "config": "default_configs/thirdparty/vitpose/ViTPose_huge_wholebody_256x192.py",
            "model": "essentials/vitpose/ckpts/vitpose+_huge/wholebody.pth",
        },
    }

    def __init__(
        self,
        model_name: str = "ViTPose+-G (multi-task train, COCO)",
        model_config: str = None,
        model_checkpoint: str = None,
        device: str = "cuda",
        **kwargs,
    ):
        self.device = torch.device(device)

        self.model_name = model_name
        self.MODEL_DICT[model_name] = {
            "config": model_config,
            "model": model_checkpoint,
        }

        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic["model"]
        model = init_pose_model(dic["config"], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(
            image, out, kpt_score_threshold, vis_dot_radius, vis_line_thickness
        )
        return out, vis

    def predict_pose(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float = 0.5,
    ) -> List[Dict[str, np.ndarray]]:
        """
        det_results: a list of Dict[str, np.ndarray] 'bbox': xyxyc
        """
        out, _ = inference_top_down_pose_model(
            self.model,
            image,
            person_results=det_results,
            bbox_thr=box_score_threshold,
            format="xyxy",
        )
        return out

    def visualize_pose_results(
        self,
        image: np.ndarray,
        pose_results: List[np.ndarray],
        kpt_score_threshold: float = 0.3,
        vis_dot_radius: int = 4,
        vis_line_thickness: int = 1,
    ) -> np.ndarray:
        vis = vis_pose_result(
            self.model,
            image,
            pose_results,
            kpt_score_thr=kpt_score_threshold,
            radius=vis_dot_radius,
            thickness=vis_line_thickness,
        )
        return vis


def main(
    img_dir: str = "./demo_data/input_images/arthur_tyler_pass_by_nov20/cam01",
    bbox_dir: str = "./demo_data/input_masks/arthur_tyler_pass_by_nov20/cam01/json_data",
    output_dir: str = "./demo_data/input_2d_poses/arthur_tyler_pass_by_nov20/cam01",
    model_config: str = "./configs/vitpose/ViTPose_huge_wholebody_256x192.py",
    model_checkpoint: str = "./checkpoints/vitpose_huge_wholebody.pth",
    timing_info_dir: str = "./demo_output/timing_info",
    vis: bool = False,
):
    # Load the model
    load_model_start_time = perf_counter()
    model = ViTPoseModel(model_config=model_config, model_checkpoint=model_checkpoint)
    load_model_end_time = perf_counter()

    # Pose estimation configuration
    box_score_threshold = 0.5
    kpt_score_threshold = 0.3
    vis_dot_radius = 4
    vis_line_thickness = 1

    output_dir = os.path.join(output_dir, os.path.basename(img_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timing_info = {
        "model_loading_time": load_model_end_time - load_model_start_time,
        "images": []
    }

    # Run per image
    img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if img_path_list == []:
        img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    for img_idx, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        load_data_start_time = perf_counter()
        img_idx = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        image = cv2.imread(img_path)
        det_result_path = os.path.join(bbox_dir, f"mask_{img_idx:05d}.json")
        bboxes = []
        person_ids = []
        with open(det_result_path, "r") as f:
            det_results = json.load(f)
            # {"mask_name": "mask_00066.npy", "mask_height": 1280, "mask_width": 720, "promote_type": "mask", "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 501, "y1": 418, "x2": 711, "y2": 765, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 0, "y1": 300, "x2": 155, "y2": 913, "logit": 0.0}}}
            # 1: {"instance_id": 1, "class_name": "person", "x1": 501, "y1": 418, "x2": 711, "y2": 765, "logit": 0.0}
            # 2: {"instance_id": 2, "class_name": "person", "x1": 0, "y1": 300, "x2": 155, "y2": 913, "logit": 0.0}
            # If labels is empty, skip the frame
            if len(det_results["labels"]) == 0:
                continue
            for box in det_results["labels"].values():
                bbox_dict = {
                    "bbox": np.array([box["x1"], box["y1"], box["x2"], box["y2"], 1.0])
                }
                bboxes.append(bbox_dict)
                person_ids.append(box["instance_id"])

            # sanity check; if boxes is empty, continue
            bboxes_sum = sum([bbox["bbox"][:4].sum() for bbox in bboxes])
            if bboxes_sum == 0:
                continue
        load_data_end_time = perf_counter()

        model_inference_start_time = perf_counter()
        out = model.predict_pose(image, bboxes, box_score_threshold)
        model_inference_end_time = perf_counter()

        save_start_time = perf_counter()
        # out: List[Dict[str, np.ndarray]]; keys: bbox, keypoints. values are numpy arrays
        # convert values to lists
        save_out = {}

        for out_idx, person_id in enumerate(person_ids):
            save_out[person_id] = {}
            for key in ["bbox", "keypoints"]:
                save_out[person_id][key] = out[out_idx][key].tolist()

        # Save the pose results
        pose_result_path = os.path.join(output_dir, f"pose_{img_idx:05d}.json")
        with open(pose_result_path, "w") as f:
            json.dump(save_out, f)

        if vis:
            vis_out = model.visualize_pose_results(
                image, out, kpt_score_threshold, vis_dot_radius, vis_line_thickness
            )
            vis_out_path = os.path.join(output_dir, "vis", f"pose_{img_idx:05d}.jpg")
            Path(os.path.join(output_dir, "vis")).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(vis_out_path, vis_out)

        save_end_time = perf_counter()

        img_timing_info = {
            "img_idx": img_idx,
            "img_path": img_path,
            "load_data_time": load_data_end_time - load_data_start_time,
            "model_inference_time": model_inference_end_time
            - model_inference_start_time,
            "save_time": save_end_time - save_start_time,
        }

        timing_info["images"].append(img_timing_info)

    timing_info_file = os.path.join(timing_info_dir, "pose2d_vitpose_timing_info.json")
    os.makedirs(timing_info_dir, exist_ok=True)
    with open(timing_info_file, "wb") as f:
        f.write(orjson.dumps(timing_info, option=orjson.OPT_INDENT_2))

    print(f"Timing info saved to {timing_info_file}")


if __name__ == "__main__":
    tyro.cli(main)
