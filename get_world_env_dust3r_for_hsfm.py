# -*- coding: utf-8 -*-
# @Time    : 2025/02/12
# @Author  : Hongsuk Choi

import os
import os.path as osp
import glob
import numpy as np
import copy
import pickle
import tyro
import PIL
import cv2
import torch
from typing import Any

from dust3r.utils.image import ImgNorm
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo


def transform_keypoints(homogeneous_keypoints: np.ndarray, affine_matrix: np.ndarray):
    # Ensure keypoints is a numpy array
    homogeneous_keypoints = np.array(homogeneous_keypoints)

    # Apply the transformation
    transformed_keypoints = np.dot(affine_matrix, homogeneous_keypoints.T).T

    # Round to nearest integer for pixel coordinates
    transformed_keypoints = np.round(transformed_keypoints).astype(int)

    return transformed_keypoints


def check_affine_matrix(
    test_img: PIL.Image, original_image: PIL.Image, affine_matrix: np.ndarray
):
    assert affine_matrix.shape == (2, 3)

    # get pixels near the center of the image in the new image space
    # Sample 100 pixels near the center of the image
    w, h = test_img.size
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4  # Use a quarter of the smaller dimension as the radius

    # Generate random offsets within the circular region
    num_samples = 100
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    r = np.random.uniform(0, radius, num_samples)

    # Convert polar coordinates to Cartesian
    x_offsets = r * np.cos(theta)
    y_offsets = r * np.sin(theta)

    # Add offsets to the center coordinates and ensure they're within image bounds
    sampled_x = np.clip(center_x + x_offsets, 0, w - 1).astype(int)
    sampled_y = np.clip(center_y + y_offsets, 0, h - 1).astype(int)

    # Create homogeneous coordinates
    pixels_near_center = np.column_stack((sampled_x, sampled_y, np.ones(num_samples)))

    # draw the pixels on the image and save it
    test_img_pixels = np.asarray(test_img).copy().astype(np.uint8)
    for x, y in zip(sampled_x, sampled_y):
        test_img_pixels = cv2.circle(test_img_pixels, (x, y), 3, (0, 255, 0), -1)
    PIL.Image.fromarray(test_img_pixels).save("test_new_img_pixels.png")

    transformed_keypoints = transform_keypoints(pixels_near_center, affine_matrix)
    # Load the original image
    original_img_array = np.array(original_image)

    # Draw the transformed keypoints on the original image
    for point in transformed_keypoints:
        x, y = point[:2]
        # Ensure the coordinates are within the image bounds
        if 0 <= x < original_image.width and 0 <= y < original_image.height:
            cv2.circle(
                original_img_array,
                (int(x), int(y)),
                int(3 * affine_matrix[0, 0]),
                (255, 0, 0),
                -1,
            )

    # Save the image with drawn keypoints
    PIL.Image.fromarray(original_img_array).save("test_original_img_keypoints.png")


# hard coding to get the affine transform matrix
def preprocess_and_get_transform(image: np.ndarray, size=512, square_ok=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(image)
    original_width, original_height = img.size

    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * size / S)) for x in img.size)
    img_resized = img.resize(new_size, interp)

    # Calculate center of the resized image
    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2

    # Step 2: Crop
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3 * halfw // 4

    img_cropped = img_resized.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height

    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y

    affine_matrix = np.array(
        [[1 / scale_x, 0, translate_x], [0, 1 / scale_y, translate_y]]
    )

    return img_cropped, affine_matrix


def get_reconstructed_scene(
    model: AsymmetricCroCo3DStereo,
    images: list[np.ndarray],
    image_size: int,
    schedule: str,
    niter: int,
    scenegraph_type: str,
    winsize: int,
    refid: int,
    device: torch.device,
    verbose: bool = False,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """

    # get affine transform matrix list
    affine_matrix_list = []
    cropped_images = []
    for image in images:
        img_cropped, affine_matrix = preprocess_and_get_transform(
            image, size=image_size
        )
        affine_matrix_list.append(affine_matrix)
        cropped_images.append(img_cropped)

    imgs = []

    for cropped_image in cropped_images:
        imgs.append(
            dict(
                img=ImgNorm(cropped_image)[None],
                true_shape=np.int32([cropped_image.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    # CHECK the first image
    # test_img = img_cropped_list[0]
    # org_img = PIL.Image.open(filelist[0])
    # check_affine_matrix(test_img, org_img, affine_matrix_list[0])
    # import pdb; pdb.set_trace()

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=1, verbose=verbose)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode, verbose=verbose)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )
        print("final loss: ", loss)
    # get optimized values from scene
    # scene = scene_state.sparse_ga
    rgbimg = scene.imgs  # list of N numpy images with shape (H,W,3) , rgb
    intrinsics = to_numpy(scene.get_intrinsics())  # N intrinsics # (N, 3, 3)
    cams2world = to_numpy(scene.get_im_poses())  # (N,4,4)

    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d: list of N pointclouds, each shape(H, W,3)
    # confs: list of N confidence scores, each shape(H, W)
    # msk: boolean mask of valid points, shape(H, W)
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())
    msk = to_numpy(scene.get_masks())
    confs = to_numpy([c for c in scene.im_conf])

    return (
        rgbimg,
        intrinsics,
        cams2world,
        pts3d,
        depths,
        msk,
        confs,
        affine_matrix_list,
        output,
    )


def get_world_env_dust3r_for_hsfm(
    model: AsymmetricCroCo3DStereo,
    images: list[np.ndarray],
    images_indices: list[int],
    image_size: int = 512,
    schedule: str = "linear",
    niter: int = 300,
    scenegraph_type: str = "complete",
    winsize: int = 1,
    refid: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    (
        rgbimg,
        intrinsics,
        cams2world,
        pts3d,
        depths,
        msk,
        confs,
        affine_matrix_list,
        output,
    ) = get_reconstructed_scene(
        model=model,
        images=images,
        image_size=image_size,
        schedule=schedule,
        niter=niter,
        scenegraph_type=scenegraph_type,
        winsize=winsize,
        refid=refid,
        verbose=verbose,
        device=device,
    )

    return {
        "dust3r_network_output": output,
        "dust3r_ga_output": {
            image_idx: {
                "rgbimg": rgbimg[i],
                "intrinsics": intrinsics[i],
                "cams2world": cams2world[i],
                "pts3d": pts3d[i],
                "depths": depths[i],
                "msk": msk[i],
                "confs": confs[i],
                "affine_matrix": affine_matrix_list[i],
            }
            for i, image_idx in enumerate(images_indices)
        },
    }


def save_results(
    results: dict[str, Any],
    output_dir: str,
):
    output_file = osp.join(output_dir, "dust3r_reconstruction_results.pkl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


def load_dust3r_model(
    model_path: str = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    return model


def main(
    model_path: str = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    out_pkl_dir: str = "./demo_output/input_dust3r",
    img_dir: str = "./demo_data/input_images/bww_stairs_nov17",
):
    img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if img_path_list == []:
        img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    images = []
    images_indices = []

    for img_path in img_path_list:
        img_idx = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        image = cv2.imread(img_path)
        images.append(image)
        images_indices.append(img_idx)

    # parameters
    device = "cuda"
    verbose = False
    image_size = 512
    schedule = "linear"
    niter = 300
    scenegraph_type = "complete"
    winsize = 1
    refid = 0

    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    results = get_world_env_dust3r_for_hsfm(
        model=model,
        images=images,
        images_indices=images_indices,
        image_size=image_size,
        schedule=schedule,
        niter=niter,
        scenegraph_type=scenegraph_type,
        winsize=winsize,
        refid=refid,
        verbose=verbose,
    )

    save_results(results, out_pkl_dir)


if __name__ == "__main__":
    tyro.cli(main)
