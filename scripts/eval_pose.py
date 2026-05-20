"""评估 Pose Contrastive 适配模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqa_common import (
    FeatureDataset,
    as_torch_dataset,
    load_checkpoint,
    macro_f1,
    make_pose_components,
    project_root,
    read_manifest,
    runtime_env,
    split_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 Pose Contrastive 并输出动作属性 F1")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "outputs" / "checkpoints" / "pose_best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--metrics",
        "--output",
        dest="metrics",
        type=Path,
        default=project_root() / "outputs" / "metrics" / "pose_eval.json",
        help="指标 JSON 输出路径；--output 是 --metrics 的等价别名，便于和训练/推理脚本保持直观一致",
    )
    parser.add_argument("--allow-metadata-fallback", action="store_true", help="允许用样本元信息生成占位特征，正式实验不要开启")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = __import__("aqa_common").require_torch()
    checkpoint = load_checkpoint(args.checkpoint)
    label_maps = checkpoint["label_maps"]
    config = checkpoint.get("config", {})
    input_dim = int(config.get("input_dim", args.input_dim))
    hidden_dim = int(config.get("hidden_dim", args.hidden_dim))

    rows = split_rows(read_manifest(args.manifest), args.split)
    dataset = as_torch_dataset(
        FeatureDataset(rows, args.manifest.parent, input_dim, label_maps=label_maps, allow_metadata_fallback=args.allow_metadata_fallback)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_pose_components(input_dim, hidden_dim, label_maps).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    true_by_column = {column: [] for column in label_maps}
    pred_by_column = {column: [] for column in label_maps}
    preview = []
    with torch.no_grad():
        for batch in loader:
            _, logits = model(batch["feature"].to(device))
            labels = batch["labels"].to(device)
            for idx, column in enumerate(label_maps):
                pred = torch.argmax(logits[column], dim=1)
                true_by_column[column].extend(labels[:, idx].cpu().numpy().tolist())
                pred_by_column[column].extend(pred.cpu().numpy().tolist())
            for row_idx, sample_id in enumerate(batch["sample_id"][:5]):
                preview.append(
                    {
                        "sample_id": sample_id,
                        "predicted_labels": {
                            column: int(torch.argmax(logits[column], dim=1)[row_idx].cpu())
                            for column in label_maps
                        },
                    }
                )

    f1_by_column = {
        column: macro_f1(true_by_column[column], pred_by_column[column])
        for column in label_maps
    }
    payload = {
        "model": "Pose Contrastive",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "F1": f1_by_column,
        "mean_f1": sum(f1_by_column.values()) / max(len(f1_by_column), 1),
        "num_samples": len(rows),
        "predictions_preview": preview[:10],
        "env": runtime_env(),
    }
    write_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
