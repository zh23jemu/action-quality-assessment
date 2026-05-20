#!/bin/bash
# MTL-AQA 帧统计特征抽取脚本。
# 该步骤在训练前运行，把 data/raw/whole_videos_frames 中的抽帧图片转换为
# data/processed/features/*.npy，并生成带 feature_path 的 manifest。
# 如需临时调整分区，可使用：sbatch --partition=目标分区 slurm/extract_features.slurm

#SBATCH --job-name=aqa_extract_features
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/extract_features_%j.out
#SBATCH --error=outputs/logs/extract_features_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Pillow 用于读取抽帧图片；若环境已安装则该命令会快速确认依赖。
.venv/bin/python -m pip install pillow

.venv/bin/python scripts/extract_features.py \
  --manifest data/processed/mtl_aqa_manifest.csv \
  --output data/processed/mtl_aqa_manifest_features.csv \
  --feature-dir data/processed/features \
  --num-frames 16 \
  --input-dim 128
