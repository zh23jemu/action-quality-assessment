"""整理和校验 MTL-AQA 数据。

脚本目标是把不同来源的 MTL-AQA 标注整理为统一 CSV manifest。训练脚本只依赖
manifest，因此后续无论数据来自官方压缩包、手工下载还是已有特征目录，都可以
通过这个入口统一检查。
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List

from aqa_common import LABEL_COLUMNS, project_root, synthetic_rows, write_json, write_manifest


CANONICAL_COLUMNS = ["sample_id", "split", "feature_path", "video_path", "score", *LABEL_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 MTL-AQA 数据并生成统一 manifest")
    parser.add_argument("--raw-dir", type=Path, default=project_root() / "data" / "raw", help="MTL-AQA 原始数据目录")
    parser.add_argument("--output", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv", help="输出 manifest 路径")
    parser.add_argument("--annotation", type=Path, default=None, help="显式指定标注文件，支持 CSV/JSON/JSONL")
    parser.add_argument("--ready-dir", type=Path, default=None, help="显式指定 Ready_2_Use/MTL-AQA_split_0_data 目录")
    parser.add_argument("--synthetic-smoke", action="store_true", help="生成仅用于冒烟测试的合成 manifest")
    parser.add_argument("--input-dim", type=int, default=128, help="冒烟测试特征维度")
    return parser.parse_args()


def find_annotation(raw_dir: Path) -> Path:
    """在原始目录中寻找最可能的标注文件。"""

    candidates: List[Path] = []
    for pattern in ("*.csv", "*.json", "*.jsonl"):
        candidates.extend(raw_dir.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {raw_dir} 找到 CSV/JSON/JSONL 标注文件。请下载 MTL-AQA 数据后放入 data/raw，"
            "或使用 --annotation 指定手工整理的标注文件。"
        )
    # 优先选择文件名中带 annotation/label/manifest 的文件，降低误选 README 等文件的概率。
    scored = sorted(
        candidates,
        key=lambda path: (
            0 if any(token in path.stem.lower() for token in ("annotation", "label", "manifest", "aqa")) else 1,
            len(path.parts),
            path.name,
        ),
    )
    return scored[0]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """读取常见标注格式。"""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("data", "annotations", "samples"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise ValueError(f"无法从 JSON 中识别样本列表：{path}")
    raise ValueError(f"不支持的标注格式：{path}")


def find_ready_dir(raw_dir: Path) -> Path | None:
    """自动发现 MTL-AQA 官方 Ready_2_Use 划分目录。

    GitHub 发布包中通常包含：
    `Ready_2_Use/MTL-AQA_split_0_data/final_annotations_dict.pkl`。
    该目录已经提供课程需要的分数、动作属性和训练/测试划分，是生成 manifest
    的首选来源。
    """

    candidates = [
        raw_dir / "Ready_2_Use" / "MTL-AQA_split_0_data",
        raw_dir / "MTL-AQA_split_0_data",
    ]
    for candidate in candidates:
        if (candidate / "final_annotations_dict.pkl").exists():
            return candidate
    return None


def load_pickle(path: Path) -> Any:
    """读取官方 pkl 标注文件。"""

    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_ready_to_use(ready_dir: Path, raw_dir: Path) -> List[Dict[str, Any]]:
    """把官方 Ready_2_Use pkl 转换为统一 manifest。

    注意：官方 Ready_2_Use 只提供标注和划分，不包含可直接训练的视频特征。
    因此这里的 `feature_path` 留空，同时写出视频、帧目录和起止帧，后续需要先
    抽取特征，再把特征路径补回 manifest。
    """

    annotations: Dict[Any, Dict[str, Any]] = load_pickle(ready_dir / "final_annotations_dict.pkl")
    train_keys = set(load_pickle(ready_dir / "train_split_0.pkl"))
    test_keys = set(load_pickle(ready_dir / "test_split_0.pkl"))
    rows: List[Dict[str, Any]] = []

    for key, annotation in sorted(annotations.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        video_id = int(key[0])
        dive_id = int(key[1])
        if key in train_keys:
            split = "train"
        elif key in test_keys:
            split = "test"
        else:
            # 少数 caption 字典和 annotation 字典数量可能不完全一致，未在 split 中
            # 出现的样本不参与训练测试，避免泄漏或划分不明确。
            continue

        sample_id = f"{video_id:02d}_{dive_id:04d}"
        video_name = f"{video_id:02d}.mp4"
        frames_dir = raw_dir / "whole_videos_frames" / f"{video_id:02d}"
        video_path = raw_dir / "whole_videos" / video_name
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "feature_path": "",
                "video_path": str(video_path),
                "frames_dir": str(frames_dir),
                "video_id": video_id,
                "dive_id": dive_id,
                "start_frame": annotation.get("start_frame", ""),
                "end_frame": annotation.get("end_frame", ""),
                "score": annotation.get("final_score", ""),
                "difficulty": annotation.get("difficulty", ""),
                "position": annotation.get("position", ""),
                "armstand": annotation.get("armstand", ""),
                "rotation_type": annotation.get("rotation_type", ""),
                "somersaults": annotation.get("ss_no", ""),
                "twists": annotation.get("tw_no", ""),
            }
        )

    if not rows:
        raise ValueError(f"未能从 Ready_2_Use 目录生成样本：{ready_dir}")
    return rows


def first_present(row: Dict[str, Any], names: Iterable[str], default: str = "") -> str:
    """从多个候选字段中取第一个非空值。"""

    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def normalize_row(row: Dict[str, Any], index: int, raw_dir: Path) -> Dict[str, Any]:
    """把不同命名习惯的标注字段规范化为训练脚本需要的列。"""

    sample_id = first_present(row, ("sample_id", "id", "video_id", "name", "file_name"), f"sample_{index:06d}")
    split = first_present(row, ("split", "subset", "phase"), "train").lower()
    if split in {"validation", "valid"}:
        split = "val"
    if split not in {"train", "test", "val"}:
        # 官方标注若没有明确 split，默认作为训练集，后续由用户补充划分。
        split = "train"

    feature_path = first_present(row, ("feature_path", "features", "feat_path", "feature"))
    video_path = first_present(row, ("video_path", "video", "path", "file"))
    score = first_present(row, ("score", "final_score", "target", "label", "quality_score"), "0")

    normalized: Dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "feature_path": feature_path,
        "video_path": video_path,
        "score": score,
        "position": first_present(row, ("position", "body_position", "pos"), ""),
        "armstand": first_present(row, ("armstand", "arm_stand"), ""),
        "rotation_type": first_present(row, ("rotation_type", "rotation", "rot_type"), ""),
        "somersaults": first_present(row, ("somersaults", "somersault", "som", "num_somersaults"), ""),
        "twists": first_present(row, ("twists", "twist", "num_twists"), ""),
    }

    # 如果特征或视频路径是相对路径，统一转为相对 raw_dir 的形式，便于跨机器迁移。
    for column in ("feature_path", "video_path"):
        value = str(normalized[column]).strip()
        if value:
            path = Path(value)
            if path.is_absolute():
                normalized[column] = str(path)
            else:
                normalized[column] = str((raw_dir / path).resolve())
    return normalized


def validate_manifest(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成数据校验摘要。"""

    split_counts: Dict[str, int] = {}
    missing_features = 0
    missing_labels = {column: 0 for column in LABEL_COLUMNS}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        feature_path = str(row.get("feature_path", "")).strip()
        if not feature_path or not Path(feature_path).exists():
            missing_features += 1
        for column in LABEL_COLUMNS:
            if str(row.get(column, "")).strip() == "":
                missing_labels[column] += 1
    return {
        "total": len(rows),
        "split_counts": split_counts,
        "missing_features": missing_features,
        "missing_labels": missing_labels,
        "official_metric_warning": "缺失特征或标签时不能产出正式训练指标，请先补齐数据。",
    }


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        feature_dir = args.output.parent / "synthetic_features"
        rows = synthetic_rows(feature_dir=feature_dir, input_dim=args.input_dim)
    elif args.ready_dir or find_ready_dir(args.raw_dir):
        ready_dir = args.ready_dir or find_ready_dir(args.raw_dir)
        assert ready_dir is not None
        rows = parse_ready_to_use(ready_dir, args.raw_dir)
    else:
        annotation = args.annotation or find_annotation(args.raw_dir)
        source_rows = read_rows(annotation)
        rows = [normalize_row(row, index, args.raw_dir) for index, row in enumerate(source_rows)]

    write_manifest(args.output, rows)
    summary = validate_manifest(rows)
    summary["manifest"] = str(args.output)
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
