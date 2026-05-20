#!/bin/bash
# Motion Disentangling 训练脚本。
# 如需临时调整分区，可使用：sbatch --partition=目标分区 slurm/train_motion.slurm

#SBATCH --job-name=aqa_motion_train
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/motion_train_%j.out
#SBATCH --error=outputs/logs/motion_train_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# 训练前请确认 .venv 中已安装 CUDA 12.x 兼容 PyTorch，并已先运行特征抽取脚本，
# 生成带 feature_path 的 data/processed/mtl_aqa_manifest_features.csv。
.venv/bin/python scripts/train_motion.py \
  --manifest data/processed/mtl_aqa_manifest_features.csv \
  --epochs 30 \
  --batch-size 16 \
  --checkpoint outputs/checkpoints/motion_best.pt \
  --metrics outputs/metrics/motion_train.json
