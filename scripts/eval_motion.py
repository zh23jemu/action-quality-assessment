"""评估 Motion Disentangling 适配模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqa_common import (
    FeatureDataset,
    as_torch_dataset,
    make_motion_components,
    project_root,
    read_manifest,
    relative_l2,
    runtime_env,
    spearman_corr,
    split_rows,
    load_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 Motion Disentangling 并输出 SRC/R-L2")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "outputs" / "checkpoints" / "motion_best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--metrics", type=Path, default=project_root() / "outputs" / "metrics" / "motion_eval.json")
    parser.add_argument("--allow-metadata-fallback", action="store_true", help="允许用样本元信息生成占位特征，正式实验不要开启")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = __import__("aqa_common").require_torch()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config", {})
    input_dim = int(config.get("input_dim", args.input_dim))
    hidden_dim = int(config.get("hidden_dim", args.hidden_dim))

    rows = split_rows(read_manifest(args.manifest), args.split)
    dataset = as_torch_dataset(
        FeatureDataset(rows, args.manifest.parent, input_dim, allow_metadata_fallback=args.allow_metadata_fallback)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_motion_components(input_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_pred, sample_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, _, _ = model(batch["feature"].to(device))
            y_true.extend(batch["score"].cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])

    payload = {
        "model": "Motion Disentangling",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "SRC": spearman_corr(y_true, y_pred),
        "R-L2": relative_l2(y_true, y_pred),
        "num_samples": len(y_true),
        "predictions_preview": [
            {"sample_id": sid, "true_score": true, "pred_score": pred}
            for sid, true, pred in list(zip(sample_ids, y_true, y_pred))[:10]
        ],
        "env": runtime_env(),
    }
    write_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
