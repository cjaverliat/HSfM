import subprocess
import orjson
import os
from tqdm import tqdm
import tyro

from time import perf_counter


def main(
    processed_dataset_dir: str = "./data/h36m/processed/",
    output_root_dir: str = "./data_output/h36m/",
):
    os.makedirs(output_root_dir, exist_ok=True)

    with open(
        os.path.join(processed_dataset_dir, "h36m_protocol1_sequences.json"), "rb"
    ) as f:
        sequences = orjson.loads(f.read())

    total_frames = sum([seq["n_annotated_frames"] for seq in sequences])

    pbar = tqdm(total=total_frames, desc="Processing frames")

    seq_timing_info = {}

    for seq in sequences:
        seq_start_time = perf_counter()

        subject_name = seq["subject_name"]
        subaction_name = seq["subaction_name"]

        n_annotated_frames = seq["n_annotated_frames"]

        for frame_idx in range(n_annotated_frames):
            pbar.set_postfix(
                subject_name=subject_name,
                subaction_name=subaction_name,
                frame_idx=frame_idx,
            )

            frame_idx_str = f"{frame_idx:05d}"
            frame_inputs_dir = os.path.join(
                processed_dataset_dir, seq["frames_rel_root_dir"], frame_idx_str
            )
            bbox_inputs_dir = os.path.join(
                processed_dataset_dir, seq["bboxes_rel_root_dir"], frame_idx_str
            )

            frame_output_dir = os.path.join(
                output_root_dir, f"{subject_name}_{subaction_name}", frame_idx_str
            )
            action_output_dir = os.path.join(
                output_root_dir, f"{subject_name}_{subaction_name}"
            )

            os.makedirs(action_output_dir, exist_ok=True)
            os.makedirs(frame_output_dir, exist_ok=True)

            subprocess.run(
                [
                    "python",
                    "get_world_env_dust3r_for_hsfm.py",
                    "--model_path",
                    "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                    "--img-dir",
                    frame_inputs_dir,
                    "--out-pkl-dir",
                    action_output_dir,
                    "--timing-info-dir",
                    frame_output_dir,
                ]
            )
            subprocess.run(
                [
                    "python",
                    "get_pose2d_vitpose_for_hsfm.py",
                    "--img-dir",
                    frame_inputs_dir,
                    "--bbox-dir",
                    bbox_inputs_dir,
                    "--output-dir",
                    action_output_dir,
                    "--model-config",
                    "./configs/vitpose/ViTPose_huge_wholebody_256x192.py",
                    "--model-checkpoint",
                    "./checkpoints/vitpose_huge_wholebody.pth",
                    "--timing-info-dir",
                    frame_output_dir,
                ]
            )
            subprocess.run(
                [
                    "python",
                    "get_smpl_hmr2_for_hsfm.py",
                    "--model-checkpoint",
                    "./checkpoints/hmr2.ckpt",
                    "--model-config",
                    "./configs/hmr2/model_config.yaml",
                    "--img-dir",
                    frame_inputs_dir,
                    "--bbox-dir",
                    bbox_inputs_dir,
                    "--output-dir",
                    action_output_dir,
                    "--timing-info-dir",
                    frame_output_dir,
                    "--batch-size",
                    "1",
                    "--person-ids",
                    "1",
                ]
            )
            subprocess.run(
                [
                    "python",
                    "align_world_env_and_smpl_hsfm_optim.py",
                    "--world-env-path",
                    os.path.join(
                        action_output_dir,
                        frame_idx_str,
                        f"dust3r_reconstruction_results_{frame_idx:05d}.pkl",
                    ),
                    "--bbox-dir",
                    bbox_inputs_dir,
                    "--pose2d-dir",
                    frame_output_dir,
                    "--smplx-dir",
                    frame_output_dir,
                    "--out-dir",
                    frame_output_dir,
                    "--person-ids",
                    "1",
                    "--body-model-name",
                    "smpl",
                    "--timing-info-dir",
                    frame_output_dir,
                ]
            )

            pbar.update(1)

        seq_end_time = perf_counter()

        seq_timing_info[f"{subject_name}_{subaction_name}"] = (
            seq_end_time - seq_start_time
        )

    pbar.close()

    with open(os.path.join(output_root_dir, "seq_timing_info.json"), "w") as f:
        f.write(orjson.dumps(seq_timing_info))


if __name__ == "__main__":
    tyro.cli(main)
