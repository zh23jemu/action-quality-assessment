#!/bin/bash
# Pose Contrastive 测试脚本，输出各动作属性 F1 与平均 F1。

#SBATCH --job-name=aqa_pose_eval
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/pose_eval_%j.out
#SBATCH --error=outputs/logs/pose_eval_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

.venv/bin/python scripts/eval_pose.py \
  --manifest data/processed/mtl_aqa_manifest.csv \
  --checkpoint outputs/checkpoints/pose_best.pt \
  --metrics outputs/metrics/pose_eval.json
