# action-quality-assessment 项目记忆

## 项目目标

在 MTL-AQA 跳水动作质量评估数据集上，适配并复现 Fitness-AQA 论文中的两个自监督表示学习模型：

- Motion Disentangling：输出动作质量回归指标 SRC 与 R-L2。
- Pose Contrastive：基于 MTL-AQA 的动作属性标签输出 F1 Score。

最终服务于课程实践交付，包括可运行代码、训练测试说明、汇报 PPT、课程总结报告和问题记录。

## 技术栈

- Python 3 + PyTorch。
- 数据处理：CSV/JSON/JSONL manifest，支持 `.npy` / `.pt` / `.pth` 特征文件。
- 训练运行：本地 `.venv` 与 Slurm GPU。
- 默认 Slurm：`aws` 分区、`gpo-ifv7xx` 账号、`normal` QOS。

## 当前架构

- `scripts/aqa_common.py`：公共数据读取、指标、模型、训练辅助函数。
- `scripts/prepare_data.py`：整理和校验 MTL-AQA 数据，生成统一 manifest。
- `scripts/train_motion.py` / `scripts/eval_motion.py`：Motion Disentangling 训练与测试。
- `scripts/train_pose.py` / `scripts/eval_pose.py`：Pose Contrastive 训练与测试。
- `scripts/infer_single.py`：单样本推理。
- `slurm/`：训练与测试提交脚本。
- `docs/issues_log.md`：实验问题、修复和 PPT 素材记录。

## 开发规范

- 始终使用项目本地 `.venv`，不要使用系统 Python。
- 新增代码使用较详细中文注释，说明用途、关键逻辑和重要分支。
- 默认最小修改，不做无关重构。
- 大数据、缓存、日志、权重和本地环境文件不入库。

## TODO

- 下载并整理 MTL-AQA 数据集。
- 按实际数据字段确认 manifest 的列名映射。
- 在 Slurm GPU 上执行两个模型的完整训练与测试。
- 将训练成功、测试指标、单样本推理截图补充到 PPT 和说明文档。

## Current Status

已创建 MTL-AQA + Fitness-AQA 适配复现实验的项目骨架、训练评估入口、Slurm 脚本、README 和问题记录模板。服务器 GPU 与两个模型 synthetic smoke 已验证通过；MTL-AQA 15 个原始视频已下载、通过 GitHub Release 同步到服务器并完成抽帧。当前已新增轻量帧统计特征抽取入口，下一步需要在服务器生成 `data/processed/mtl_aqa_manifest_features.csv` 后启动真实训练。

## Recent Changes

- 新增统一 manifest 数据接口，支持原始标注整理、特征文件校验和显式冒烟测试。
- 新增 Motion Disentangling 的回归训练、评估和单样本推理流程。
- 新增 Pose Contrastive 的多属性分类训练、评估和 F1 指标输出流程。
- 新增 Slurm 脚本，默认使用课程约定的 GPU 分区、账号和 QOS。
- 修复冒烟特征路径二次拼接问题，并兼容 PyTorch 2.6+ checkpoint 默认 `weights_only=True` 的加载策略变化。
- 本地验证结果：语法检查通过；Motion 冒烟测试输出 SRC/R-L2；Pose 冒烟测试输出五类动作属性 F1；两个模型的单样本推理均可生成 JSON。
- 新增 `scripts/fetch_upstream.py`，已通过 Python 标准库下载并解压 `external/MTL-AQA` 与 `external/Fitness-AQA`。
- `scripts/prepare_data.py` 已支持解析官方 `Ready_2_Use/MTL-AQA_split_0_data` pkl，生成包含分数、动作属性、起止帧和 split 的 manifest。
- 本地已通过代理和 Cookie 成功下载 MTL-AQA 全部 15 个视频，并上传到 GitHub Release `mtl-aqa-videos-480p` 供服务器下载。
- 服务器已加载 `ffmpeg/4.2.10` 完成抽帧；`prepare_data.py` 能生成 1412 条 manifest，但在特征抽取前 `missing_features=1412` 属于预期。
- 新增 `scripts/extract_features.py` 与 `slurm/extract_features.slurm`，使用真实抽帧图片生成 128 维轻量视频特征并输出带 `feature_path` 的 manifest。
- 更新训练 Slurm 脚本，默认读取 `data/processed/mtl_aqa_manifest_features.csv`。

## Next TODO

- 在服务器运行 `slurm/extract_features.slurm` 或 `.venv/bin/python scripts/extract_features.py`，生成 `data/processed/mtl_aqa_manifest_features.csv`。
- 使用带特征 manifest 启动 Motion 与 Pose 的正式训练，并保存测试指标 JSON。
- 将真实训练日志、测试指标 JSON、单样本推理输出截图加入 PPT 和说明文档。

## Open Issues

- 当前仓库不提交真实 MTL-AQA 视频、抽帧图片或特征文件；这些数据通过 GitHub Release 和服务器本地目录管理。
- Fitness-AQA 原方法主要面向健身动作，Pose Contrastive 的 F1 在 MTL-AQA 上采用动作属性多分类口径，需在报告中说明适配口径。
- Motion Disentangling 若无法获得原始上游完整实现，需要以论文思想和公开说明实现可运行复现，并在 PPT 中记录差异。
- 当前帧统计特征是最小可运行真实视频特征，不等价于论文原始 C3D/I3D/pose 特征；正式报告需要说明该适配差异。

## Architecture Decisions

- 默认不使用合成数据产出正式指标；合成数据只允许通过 `--synthetic-smoke` 显式启用，用于验证代码链路。
- 训练脚本优先读取预抽取特征，避免在训练入口中混入耗时、依赖复杂的视频解码逻辑。
- 指标统一输出 JSON，便于后续截图、汇总和写入 PPT。
- 由于上游两个仓库代码不完整，先采用真实帧统计特征打通端到端训练评估链路；后续可在不改变 manifest 接口的前提下替换为更强视频/姿态特征。
