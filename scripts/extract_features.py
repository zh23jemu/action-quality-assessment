"""从 MTL-AQA 抽帧结果生成轻量视频特征。

该脚本用于弥补上游 Fitness-AQA/MTL-AQA 公开代码没有提供完整特征抽取
流水线的问题。它不依赖额外预训练模型，而是从每个样本的起止帧中均匀采样，
提取颜色、亮度、边缘和相邻帧差分统计量，生成固定 128 维 `.npy` 特征。

注意：这是“最小可运行复现”的真实视频特征入口，目的是先打通正式训练、
测试和指标保存链路。后续若接入 I3D/C3D/pose keypoint 特征，可以保持
输出 manifest 和 feature_path 约定不变，直接替换本脚本的单样本特征逻辑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from aqa_common import project_root, read_manifest, write_json, write_manifest


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="从 MTL-AQA 抽帧目录生成 128 维轻量特征")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--output", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest_features.csv")
    parser.add_argument("--feature-dir", type=Path, default=project_root() / "data" / "processed" / "features")
    parser.add_argument("--num-frames", type=int, default=16, help="每个动作片段均匀采样的帧数")
    parser.add_argument("--input-dim", type=int, default=128, help="输出特征维度，需和训练脚本 --input-dim 一致")
    parser.add_argument("--overwrite", action="store_true", help="重新生成已经存在的特征文件")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条样本，用于快速冒烟测试")
    return parser.parse_args()


def list_frame_files(frames_dir: Path) -> List[Path]:
    """列出抽帧目录中的图片文件。

    ffmpeg 抽帧脚本通常会生成 `000001.jpg` 这类按时间递增的文件名。这里按
    文件名排序，使 manifest 中的 `start_frame/end_frame` 可以直接映射到
    1-based 帧序号。若抽帧脚本改为其它图片后缀，也会被统一纳入。
    """

    if not frames_dir.exists():
        raise FileNotFoundError(f"帧目录不存在：{frames_dir}")
    files = [path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"帧目录中没有图片文件：{frames_dir}")
    return sorted(files, key=lambda path: path.name)


def safe_int(value: Any, default: int) -> int:
    """安全转换整数标注字段。"""

    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def sample_frame_paths(frame_files: Sequence[Path], start_frame: int, end_frame: int, num_frames: int) -> List[Path]:
    """按样本起止帧均匀采样图片路径。

    MTL-AQA 的起止帧是全视频帧编号，ffmpeg 输出文件名通常从 1 开始连续编号。
    因此这里把帧号裁剪到 `[1, len(frame_files)]`，再转换为 Python 的 0-based
    索引。若某个标注片段异常短，`linspace` 仍会重复采样少量帧，保证输出维度
    稳定。
    """

    if num_frames <= 0:
        raise ValueError("--num-frames 必须大于 0")
    total = len(frame_files)
    start = max(1, min(start_frame, total))
    end = max(1, min(end_frame, total))
    if end < start:
        start, end = end, start
    indices = np.linspace(start - 1, end - 1, num=num_frames)
    return [frame_files[int(round(index))] for index in indices]


def load_rgb_image(path: Path) -> np.ndarray:
    """读取图片并转换为 `[0, 1]` 范围的 RGB 数组。

    优先使用 Pillow，因为它在 CPU 环境下足够轻量；如果服务器还没安装 Pillow，
    会给出清晰错误，用户只需在 `.venv` 中安装 `pillow`。
    """

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("缺少 Pillow，请先执行：.venv/bin/python -m pip install pillow") from exc

    with Image.open(path) as image:
        resized = image.convert("RGB").resize((96, 96))
        return np.asarray(resized, dtype=np.float32) / 255.0


def frame_statistics(image: np.ndarray) -> np.ndarray:
    """提取单帧静态统计特征。

    特征包含 RGB 均值/标准差、亮度分布、简单边缘强度和四象限亮度均值。
    这些特征虽然不等价于深度视觉特征，但真实来自视频帧，足以让训练入口
    使用真实数据完成端到端复现实验。
    """

    rgb_mean = image.mean(axis=(0, 1))
    rgb_std = image.std(axis=(0, 1))
    gray = image.mean(axis=2)
    gray_stats = np.array([gray.mean(), gray.std(), gray.min(), gray.max()], dtype=np.float32)
    hist, _ = np.histogram(gray, bins=8, range=(0.0, 1.0), density=True)
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    h, w = gray.shape
    quadrants = np.array(
        [
            gray[: h // 2, : w // 2].mean(),
            gray[: h // 2, w // 2 :].mean(),
            gray[h // 2 :, : w // 2].mean(),
            gray[h // 2 :, w // 2 :].mean(),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [rgb_mean, rgb_std, gray_stats, hist.astype(np.float32), np.array([grad_x, grad_y], dtype=np.float32), quadrants]
    )


def temporal_statistics(images: Sequence[np.ndarray]) -> np.ndarray:
    """提取相邻采样帧差分统计特征。"""

    if len(images) < 2:
        return np.zeros(6, dtype=np.float32)
    diffs = [np.abs(images[index] - images[index - 1]) for index in range(1, len(images))]
    stacked = np.stack(diffs, axis=0)
    channel_mean = stacked.mean(axis=(0, 1, 2))
    gray_diff = stacked.mean(axis=3)
    return np.concatenate(
        [
            channel_mean.astype(np.float32),
            np.array([gray_diff.mean(), gray_diff.std(), gray_diff.max()], dtype=np.float32),
        ]
    )


def pad_or_trim(feature: np.ndarray, input_dim: int) -> np.ndarray:
    """把任意长度特征调整为训练脚本约定的维度。"""

    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.size == input_dim:
        return feature
    if feature.size > input_dim:
        return feature[:input_dim]
    padded = np.zeros(input_dim, dtype=np.float32)
    padded[: feature.size] = feature
    return padded


def extract_sample_feature(row: Dict[str, str], frame_cache: Dict[str, List[Path]], num_frames: int, input_dim: int) -> np.ndarray:
    """为单个 MTL-AQA 样本生成固定维度特征。"""

    frames_dir = Path(row["frames_dir"])
    cache_key = str(frames_dir)
    if cache_key not in frame_cache:
        frame_cache[cache_key] = list_frame_files(frames_dir)
    frame_files = frame_cache[cache_key]
    start_frame = safe_int(row.get("start_frame"), 1)
    end_frame = safe_int(row.get("end_frame"), start_frame)
    sampled_paths = sample_frame_paths(frame_files, start_frame, end_frame, num_frames)
    images = [load_rgb_image(path) for path in sampled_paths]
    per_frame = np.stack([frame_statistics(image) for image in images], axis=0)
    static_feature = np.concatenate(
        [
            per_frame.mean(axis=0),
            per_frame.std(axis=0),
            per_frame[-1] - per_frame[0],
            temporal_statistics(images),
        ]
    )
    return pad_or_trim(static_feature, input_dim)


def summarize(rows: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, str]], output: Path) -> Dict[str, Any]:
    """生成特征抽取摘要，方便写报告和排错。"""

    missing_features = 0
    for row in rows:
        feature_path = str(row.get("feature_path", "")).strip()
        if not feature_path or not Path(feature_path).exists():
            missing_features += 1
    return {
        "total": len(rows),
        "feature_ready": len(rows) - missing_features,
        "missing_features": missing_features,
        "failures": list(failures),
        "manifest": str(output),
    }


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    args.feature_dir.mkdir(parents=True, exist_ok=True)
    frame_cache: Dict[str, List[Path]] = {}
    failures: List[Dict[str, str]] = []
    updated_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        feature_path = args.feature_dir / f"{row['sample_id']}.npy"
        next_row: Dict[str, Any] = dict(row)
        if feature_path.exists() and not args.overwrite:
            next_row["feature_path"] = str(feature_path)
            updated_rows.append(next_row)
            continue
        try:
            feature = extract_sample_feature(row, frame_cache, args.num_frames, args.input_dim)
            np.save(feature_path, feature.astype(np.float32))
            next_row["feature_path"] = str(feature_path)
        except Exception as exc:
            failures.append({"sample_id": row.get("sample_id", str(index)), "error": str(exc)})
        updated_rows.append(next_row)
        if index % 50 == 0:
            print(json.dumps({"processed": index, "failures": len(failures)}, ensure_ascii=False))

    write_manifest(args.output, updated_rows)
    summary = summarize(updated_rows, failures, args.output)
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
