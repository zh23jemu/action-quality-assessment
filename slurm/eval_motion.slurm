#!/bin/bash
# Motion Disentangling 测试脚本，输出 SRC 和 R-L2。

#SBATCH --job-name=aqa_motion_eval
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/motion_eval_%j.out
#SBATCH --error=outputs/logs/motion_eval_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

.venv/bin/python scripts/eval_motion.py \
  --manifest data/processed/mtl_aqa_manifest.csv \
  --checkpoint outputs/checkpoints/motion_best.pt \
  --metrics outputs/metrics/motion_eval.json
