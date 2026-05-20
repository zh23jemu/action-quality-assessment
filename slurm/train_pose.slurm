#!/bin/bash
# Pose Contrastive 训练脚本。
# 如需临时调整分区，可使用：sbatch --partition=目标分区 slurm/train_pose.slurm

#SBATCH --job-name=aqa_pose_train
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/pose_train_%j.out
#SBATCH --error=outputs/logs/pose_train_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

.venv/bin/python scripts/train_pose.py \
  --manifest data/processed/mtl_aqa_manifest_features.csv \
  --epochs 30 \
  --batch-size 16 \
  --checkpoint outputs/checkpoints/pose_best.pt \
  --metrics outputs/metrics/pose_train.json
