"""单样本推理入口。

该脚本用于课程要求中的“单样本推理截图”。它可以读取 manifest 中某个样本，
分别使用 Motion 或 Pose checkpoint 输出预测结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqa_common import (
    FeatureDataset,
    load_checkpoint,
    make_motion_components,
    make_pose_components,
    project_root,
    read_manifest,
    runtime_env,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 MTL-AQA 单个样本执行推理")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Motion 或 Pose 模型 checkpoint")
    parser.add_argument("--model", choices=["motion", "pose"], required=True)
    parser.add_argument("--sample-id", default="", help="指定样本 ID；为空时使用 split 中第一个样本")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=project_root() / "outputs" / "predictions" / "single_infer.json")
    parser.add_argument("--allow-metadata-fallback", action="store_true", help="允许用样本元信息生成占位特征，正式实验不要开启")
    return parser.parse_args()


def choose_row(rows, sample_id: str, split: str):
    """从 manifest 中选择待推理样本。"""

    filtered = [row for row in rows if row.get("split", "").lower() == split.lower()]
    if sample_id:
        matched = [row for row in rows if row.get("sample_id") == sample_id]
        if not matched:
            raise ValueError(f"未找到 sample_id={sample_id!r} 的样本")
        return matched[0]
    if not filtered:
        raise ValueError(f"manifest 中没有 split={split!r} 的样本")
    return filtered[0]


def main() -> None:
    args = parse_args()
    torch = __import__("aqa_common").require_torch()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config", {})
    input_dim = int(config.get("input_dim", 128))
    hidden_dim = int(config.get("hidden_dim", 256))
    rows = read_manifest(args.manifest)
    row = choose_row(rows, args.sample_id, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "motion":
        dataset = FeatureDataset([row], args.manifest.parent, input_dim, allow_metadata_fallback=args.allow_metadata_fallback)
        item = dataset.item(0)
        model = make_motion_components(input_dim, hidden_dim).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            feature = torch.tensor(item["feature"], dtype=torch.float32).unsqueeze(0).to(device)
            pred, _, _ = model(feature)
        payload = {
            "model": "Motion Disentangling",
            "sample_id": item["sample_id"],
            "true_score": item["score"],
            "pred_score": float(pred.squeeze().cpu()),
            "checkpoint": str(args.checkpoint),
            "env": runtime_env(),
        }
    else:
        label_maps = checkpoint["label_maps"]
        dataset = FeatureDataset([row], args.manifest.parent, input_dim, label_maps=label_maps, allow_metadata_fallback=args.allow_metadata_fallback)
        item = dataset.item(0)
        model = make_pose_components(input_dim, hidden_dim, label_maps).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        reverse_maps = {
            column: {index: label for label, index in mapping.items()}
            for column, mapping in label_maps.items()
        }
        with torch.no_grad():
            feature = torch.tensor(item["feature"], dtype=torch.float32).unsqueeze(0).to(device)
            _, logits = model(feature)
        predicted = {
            column: reverse_maps[column][int(torch.argmax(value, dim=1).cpu())]
            for column, value in logits.items()
        }
        payload = {
            "model": "Pose Contrastive",
            "sample_id": item["sample_id"],
            "predicted_attributes": predicted,
            "checkpoint": str(args.checkpoint),
            "env": runtime_env(),
        }

    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
