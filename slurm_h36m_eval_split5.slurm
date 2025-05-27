#!/bin/bash

#SBATCH --job-name=hsfm_h36m_eval_split5

#SBATCH --partition=gpu_p5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=8           # number of cores per task for gpu_p2 (1/8 of 8-GPUs V100 node)

#SBATCH -A rch@a100
#SBATCH -C a100

#SBATCH --hint=nomultithread         # hyperthreading deactivated
#SBATCH --time=19:50:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.out

module purge
module load arch/a100
module load pytorch-gpu/py3/2.0.1

# echo of commands
set -x

srun python -u run_hsfm_h36m.py --processed-dataset-dir \
/lustre/fsn1/projects/rech/rch/uoe13eg/datasets/h36m_processed_hsfm \
--output-dir /lustre/fsn1/projects/rech/rch/uoe13eg/hsfm_h36m_eval \
--sequences-start-idx 32 \
--sequences-end-idx 39
