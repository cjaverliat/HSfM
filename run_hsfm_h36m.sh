#!/bin/bash

# Default values
VIS="--vis"
PERSON_IDS="1"
OUT_DIR="demo_output/h36m/"
IMG_DIR="demo_data/h36m/"

# Remove trailing slashes from directories
IMG_DIR=${IMG_DIR%/}
OUT_DIR=${OUT_DIR%/}

# Extract the base name of the image directory
IMG_DIR_NAME=$(basename "$IMG_DIR")

echo "Running HSfM pipeline..."
echo "Image directory: $IMG_DIR"
echo "Output directory: $OUT_DIR"
echo "Visualization: ${VIS:+enabled}"
echo "Person IDs: $PERSON_IDS"

# Run the pipeline
python get_world_env_dust3r_for_hsfm.py \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --img-dir "$IMG_DIR" \
    --out-pkl-dir "$OUT_DIR"

python get_pose2d_vitpose_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --bbox-dir "$IMG_DIR" \
    --output-dir "$OUT_DIR" \
    $VIS

python get_smpl_hmr2_for_hsfm.py \
    --img-dir "$IMG_DIR" \
    --bbox-dir "$IMG_DIR" \
    --output-dir "$OUT_DIR" \
    --person-ids $PERSON_IDS \
    $VIS

# python get_mano_wilor_for_hsfm.py \
#     --img-dir "$IMG_DIR" \
#     --pose2d-dir "$OUT_DIR/$IMG_DIR_NAME" \
#     --output-dir "$OUT_DIR" \
#     --person-ids $PERSON_IDS \
#     $VIS

# python get_smplx_from_smpl_and_mano_for_hsfm.py \
#     --smpl-dir "$OUT_DIR/$IMG_DIR_NAME" \
#     --mano-dir "$OUT_DIR/$IMG_DIR_NAME" \
#     --output-dir "$OUT_DIR" \
#     $VIS

python align_world_env_and_smpl_hsfm_optim.py \
    --world-env-path "$OUT_DIR/$IMG_DIR_NAME/dust3r_reconstruction_results_${IMG_DIR_NAME}.pkl" \
    --person-ids $PERSON_IDS \
    --bbox-dir "$IMG_DIR" \
    --pose2d-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --smplx-dir "$OUT_DIR/$IMG_DIR_NAME" \
    --body-model-name smpl \
    --out-dir "$OUT_DIR/$IMG_DIR_NAME" \

echo "HSfM pipeline completed!" 