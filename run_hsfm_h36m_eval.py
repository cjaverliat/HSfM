import json
import torch
from tqdm import tqdm
from smplx import SMPL
import orjson
import os
from eval import compute_h36m_sequence_metrics


def _serializer_fallback(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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

    for seq in tqdm(sequences, desc="Computing metrics for sequences"):
        subject_name = seq["subject_name"]
        subaction_name = seq["subaction_name"]

        with open(
            os.path.join(
                args.processed_dataset_dir,
                f"ground_truth/{subject_name}_{subaction_name}.json",
            ),
            "r",
        ) as f:
            ground_truth = json.load(f)

        seq["metrics"] = compute_h36m_sequence_metrics(
            ground_truth=ground_truth,
            sequence_data_dir=os.path.join(
                args.eval_data_dir, f"{subject_name}_{subaction_name}"
            ),
            smpl_model=smpl_model,
            J_regressor=J_regressor,
        )

    metrics_per_action = {}
    avg_metrics = {}

    for seq in sequences:
        if "metrics" not in seq:
            continue
        action_name = seq["action_name"]
        metrics_per_action.setdefault(action_name, {})
        for k, v in seq["metrics"].items():
            metrics_per_action[action_name].setdefault(k, []).append(v)
            avg_metrics.setdefault(k, []).append(v)

    for action_name, metrics in metrics_per_action.items():
        for k, v in metrics.items():
            metrics_per_action[action_name][k] = torch.as_tensor(v).mean()

    for k, v in avg_metrics.items():
        avg_metrics[k] = torch.as_tensor(v).mean()

    with open(os.path.join(args.eval_data_dir, "eval_results.json"), "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "metrics_per_action": metrics_per_action,
                    "avg_metrics": avg_metrics,
                },
                default=_serializer_fallback,
            )
        )

    print(f"Saved results to {os.path.join(args.eval_data_dir, 'eval_results.json')}")
