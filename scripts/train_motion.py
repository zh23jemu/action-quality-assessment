"""训练 Motion Disentangling 适配模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqa_common import (
    FeatureDataset,
    as_torch_dataset,
    checkpoint_payload,
    make_motion_components,
    project_root,
    read_manifest,
    relative_l2,
    runtime_env,
    set_seed,
    spearman_corr,
    split_rows,
    synthetic_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Motion Disentangling 分数回归模型")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "outputs" / "checkpoints" / "motion_best.pt")
    parser.add_argument("--metrics", type=Path, default=project_root() / "outputs" / "metrics" / "motion_train.json")
    parser.add_argument("--synthetic-smoke", action="store_true", help="只用于验证代码链路的合成数据")
    parser.add_argument("--allow-metadata-fallback", action="store_true", help="允许用样本元信息生成占位特征，正式实验不要开启")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch = __import__("aqa_common").require_torch()

    if args.synthetic_smoke:
        rows = synthetic_rows(count=32, feature_dir=args.metrics.parent / "motion_synthetic_features", input_dim=args.input_dim)
        manifest_dir = args.metrics.parent
        allow_fallback = False
    else:
        rows = read_manifest(args.manifest)
        manifest_dir = args.manifest.parent
        allow_fallback = args.allow_metadata_fallback

    train_rows = split_rows(rows, "train")
    test_rows = split_rows(rows, "test")
    train_dataset = as_torch_dataset(FeatureDataset(train_rows, manifest_dir, args.input_dim, allow_metadata_fallback=allow_fallback))
    test_dataset = as_torch_dataset(FeatureDataset(test_rows, manifest_dir, args.input_dim, allow_metadata_fallback=allow_fallback))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_motion_components(args.input_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_src = -2.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features = batch["feature"].to(device)
            score = batch["score"].to(device)
            optimizer.zero_grad()
            pred, motion_latent, quality_latent = model(features)
            # 轻量正交约束用于分离 motion latent 与 quality latent，贴合 disentangling 的核心思想。
            orthogonal_loss = torch.mean(torch.abs(torch.sum(motion_latent * quality_latent, dim=1)))
            loss = criterion(pred, score) + 0.01 * orthogonal_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * features.shape[0]

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                features = batch["feature"].to(device)
                pred, _, _ = model(features)
                y_true.extend(batch["score"].cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
        src = spearman_corr(y_true, y_pred)
        r_l2 = relative_l2(y_true, y_pred)
        item = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_dataset), 1),
            "SRC": src,
            "R-L2": r_l2,
        }
        history.append(item)
        print(json.dumps(item, ensure_ascii=False))
        if src > best_src:
            best_src = src
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    vars(args),
                    extra={"best_metric": {"SRC": src, "R-L2": r_l2}, "model_type": "motion"},
                ),
                args.checkpoint,
            )

    payload = {
        "model": "Motion Disentangling",
        "task": "MTL-AQA score regression",
        "synthetic_smoke": bool(args.synthetic_smoke),
        "checkpoint": str(args.checkpoint),
        "best_SRC": best_src,
        "history": history,
        "env": runtime_env(),
    }
    write_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
