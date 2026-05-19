# 实验问题记录

本文件用于记录 MTL-AQA + Fitness-AQA 适配复现实验中遇到的问题，后续可直接整理到汇报 PPT 和课程总结报告。

## 问题 1：当前仓库缺少真实数据和特征

- 问题现象：项目目录初始化时只有两个 Word 模板，没有 MTL-AQA 数据、特征、上游代码或训练日志。
- 原因分析：课程数据需要从指定地址或上游仓库说明中单独下载，不能从当前目录直接训练。
- 解决方案：先创建数据整理脚本和统一 manifest 规范；真实数据放入 `data/raw/` 后再运行 `scripts/prepare_data.py`。
- 是否影响结果：影响。当前不能产出正式 SRC、R-L2 或 F1。
- PPT 素材建议：可放入“环境与数据准备问题”页，说明先完成工程化适配，待数据下载后执行训练。

## 问题 2：Fitness-AQA 方法与 MTL-AQA 数据任务不完全一致

- 问题现象：Fitness-AQA 原始方法主要面向健身动作姿态错误与细粒度质量评估，课程要求将两个模型用于 MTL-AQA 跳水数据。
- 原因分析：论文 README 提到方法可用于 Diving，但公开代码和数据标签口径与 MTL-AQA 不完全一致。
- 解决方案：Motion Disentangling 使用分数回归输出 SRC/R-L2；Pose Contrastive 使用 MTL-AQA 的 position、armstand、rotation_type、somersaults、twists 五类动作属性输出 F1。
- 是否影响结果：影响指标解释，需要在 PPT 中说明这是“严格复现方向下的数据集适配”。
- PPT 素材建议：放入“模型适配方案”页，说明标签映射和评价指标口径。

## 问题 3：当前未实际运行 Slurm 训练

- 问题现象：本地 Windows 工作区没有 Slurm 调度器和 GPU 数据环境。
- 原因分析：完整训练需要集群 GPU、已下载数据和 `.venv` 中的 CUDA 版 PyTorch。
- 解决方案：已提供 `slurm/` 脚本，后续在集群上提交训练并截图日志。
- 是否影响结果：影响。当前只能做代码级准备，不能生成真实完整训练日志。
- PPT 素材建议：后续补充 Slurm `tail` 日志、测试指标 JSON 和单样本推理输出截图。

## 问题 4：本地 CUDA 版 PyTorch 安装成功但没有可用 GPU

- 问题现象：本地 `.venv` 安装 `torch-2.12.0+cu126` 后，`torch.cuda.is_available()` 输出 `False`。
- 原因分析：当前 Windows 工作区没有可被 PyTorch 使用的 NVIDIA GPU 或驱动环境，CUDA wheel 只能证明依赖安装成功，不能代表可训练环境可用。
- 解决方案：本地只执行 CPU 冒烟测试；完整训练按 Slurm GPU 脚本提交到集群。
- 是否影响结果：影响。本地不能生成正式 GPU 训练时间和完整指标。
- PPT 素材建议：放在“环境兼容问题”页，说明本地验证与集群训练的职责边界。

## 问题 5：PyTorch 新版 checkpoint 加载默认策略变化

- 问题现象：PyTorch 2.6 之后 `torch.load` 默认 `weights_only=True`，包含配置对象的 checkpoint 在评估阶段会被拒绝加载。
- 原因分析：新版 PyTorch 为安全性修改默认加载策略，训练脚本保存的 checkpoint 不只是权重，还包含配置、环境和标签映射。
- 解决方案：保存 checkpoint 前将配置转换为 JSON 安全类型；加载本项目可信 checkpoint 时显式兼容 `weights_only=False`。
- 是否影响结果：不影响模型结果，但会影响训练后测试脚本能否直接读取 checkpoint。
- PPT 素材建议：可作为“版本兼容性修复”示例，体现代码适配过程。

## 问题 6：Ready_2_Use 只有标注和划分，没有可直接训练的视频特征

- 问题现象：`data/raw/Ready_2_Use/MTL-AQA_split_0_data` 包含 `final_annotations_dict.pkl`、`train_split_0.pkl`、`test_split_0.pkl`，但没有 `.npy` / `.pt` 特征文件。
- 原因分析：官方发布的 Ready_2_Use pkl 主要用于提供分数、动作属性和划分；视频仍需根据 `Video_List.xlsx` 下载并抽帧，或另行生成 C3D/I3D 等特征。
- 解决方案：已补充 pkl 解析逻辑，可生成 manifest；正式训练前需要先完成视频下载和特征抽取，再补齐 `feature_path`。
- 是否影响结果：影响。没有特征时不能进行正式训练。
- PPT 素材建议：放入“数据处理流程”页，说明标注解析与特征抽取是两个阶段。

## 问题 7：服务器无法直连 YouTube，本地代理下载会触发 Cookie/风控问题

- 问题现象：服务器使用 `yt-dlp --list-formats` 访问 YouTube 时出现 `Connection reset by peer`；本地通过 `127.0.0.1:7897` 代理、Cookie 和 Node challenge solver 可以下载，但下载到 `01.mp4`、`02.mp4` 后后续链接提示 Cookie 失效或需要登录确认不是机器人。
- 原因分析：集群网络无法直连 YouTube；本地代理出口触发 YouTube 429/机器人校验，浏览器 Cookie 可能在导出后被轮换。
- 解决方案：已成功下载 `01.mp4` 和 `02.mp4`；后续需要重新导出 `youtube_cookies.txt`，必要时更换代理出口，继续按编号续传。已完成文件会被跳过，不需要重复下载。
- 是否影响结果：影响。真实视频未完整下载前，不能抽帧和生成正式特征。
- PPT 素材建议：放入“数据获取问题与解决方案”页，说明官方仅提供链接，实际下载受网络和平台风控限制。
