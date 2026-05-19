# MTL-AQA 上复现 Fitness-AQA 两个模型

本项目用于课程实践：在 MTL-AQA 跳水动作质量评估数据集上，适配 Fitness-AQA 论文中的 Motion Disentangling 与 Pose Contrastive 两个自监督表示学习方法，并输出课程要求的 SRC、R-L2 和 F1 Score 指标。

## 目录结构

- `scripts/`：数据整理、训练、测试和单样本推理入口。
- `slurm/`：Slurm GPU 训练与测试脚本。
- `docs/issues_log.md`：实验过程中遇到的问题、修复方式和 PPT 素材记录。
- `data/raw/`：手工放置 MTL-AQA 原始数据或标注文件。
- `data/processed/`：整理后的 manifest 与中间文件。
- `outputs/`：训练日志、模型权重、指标和预测结果。

## 环境准备

始终使用项目本地 `.venv`。GPU 训练建议安装 CUDA 12.x 兼容的 PyTorch：

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
.venv/bin/pip install -r requirements.txt
```

Windows 本地调试时使用：

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py --help
```

## 数据准备

先下载上游代码，便于核对原始实现和论文复现实验设置：

```bash
.venv/bin/python scripts/fetch_upstream.py
```

将 MTL-AQA 数据放入 `data/raw/`，至少需要一个包含样本、划分、分数和动作属性的标注文件。推荐整理为 CSV，字段如下：

```text
sample_id,split,feature_path,score,position,armstand,rotation_type,somersaults,twists
```

然后运行：

```bash
.venv/bin/python scripts/prepare_data.py --raw-dir data/raw --output data/processed/mtl_aqa_manifest.csv
```

如果只有原始视频，应先使用上游代码或单独的视频特征抽取流程生成 `.npy` / `.pt` 特征，并在 manifest 的 `feature_path` 中填写路径。

## 训练与测试

Motion Disentangling：

```bash
.venv/bin/python scripts/train_motion.py --manifest data/processed/mtl_aqa_manifest.csv
.venv/bin/python scripts/eval_motion.py --manifest data/processed/mtl_aqa_manifest.csv --checkpoint outputs/checkpoints/motion_best.pt
```

Pose Contrastive：

```bash
.venv/bin/python scripts/train_pose.py --manifest data/processed/mtl_aqa_manifest.csv
.venv/bin/python scripts/eval_pose.py --manifest data/processed/mtl_aqa_manifest.csv --checkpoint outputs/checkpoints/pose_best.pt
```

Slurm 环境可直接使用 `slurm/` 下脚本；默认 GPU 分区为 `aws`，账号为 `gpo-ifv7xx`，QOS 为 `normal`。如需覆盖分区，可提交时使用：

```bash
sbatch --partition=目标分区 slurm/train_motion.slurm
```

## 冒烟测试

没有真实数据时，只允许用显式冒烟模式验证代码链路：

```bash
.venv/bin/python scripts/train_motion.py --synthetic-smoke --epochs 1 --batch-size 4
.venv/bin/python scripts/train_pose.py --synthetic-smoke --epochs 1 --batch-size 4
```

冒烟结果不能作为课程正式指标。
