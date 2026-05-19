"""训练 Pose Contrastive 适配模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqa_common import (
    FeatureDataset,
    as_torch_dataset,
    build_label_maps,
    checkpoint_payload,
    macro_f1,
    make_pose_components,
    project_root,
    read_manifest,
    runtime_env,
    set_seed,
    split_rows,
    supervised_contrastive_loss,
    synthetic_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Pose Contrastive 多属性 F1 模型")
    parser.add_argument("--manifest", type=Path, default=project_root() / "data" / "processed" / "mtl_aqa_manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "outputs" / "checkpoints" / "pose_best.pt")
    parser.add_argument("--metrics", type=Path, default=project_root() / "outputs" / "metrics" / "pose_train.json")
    parser.add_argument("--synthetic-smoke", action="store_true", help="只用于验证代码链路的合成数据")
    parser.add_argument("--allow-metadata-fallback", action="store_true", help="允许用样本元信息生成占位特征，正式实验不要开启")
    return parser.parse_args()


def evaluate(model, loader, label_maps, device):
    """评估每个动作属性的宏 F1。"""

    torch = __import__("aqa_common").require_torch()
    model.eval()
    true_by_column = {column: [] for column in label_maps}
    pred_by_column = {column: [] for column in label_maps}
    with torch.no_grad():
        for batch in loader:
            _, logits = model(batch["feature"].to(device))
            labels = batch["labels"].to(device)
            for idx, column in enumerate(label_maps):
                pred = torch.argmax(logits[column], dim=1)
                true_by_column[column].extend(labels[:, idx].cpu().numpy().tolist())
                pred_by_column[column].extend(pred.cpu().numpy().tolist())
    f1_by_column = {
        column: macro_f1(true_by_column[column], pred_by_column[column])
        for column in label_maps
    }
    f1_by_column["mean_f1"] = sum(f1_by_column.values()) / max(len(label_maps), 1)
    return f1_by_column


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch = __import__("aqa_common").require_torch()

    if args.synthetic_smoke:
        rows = synthetic_rows(count=40, feature_dir=args.metrics.parent / "pose_synthetic_features", input_dim=args.input_dim)
        manifest_dir = args.metrics.parent
        allow_fallback = False
    else:
        rows = read_manifest(args.manifest)
        manifest_dir = args.manifest.parent
        allow_fallback = args.allow_metadata_fallback

    train_rows = split_rows(rows, "train")
    test_rows = split_rows(rows, "test")
    label_maps = build_label_maps(train_rows)
    train_dataset = as_torch_dataset(
        FeatureDataset(train_rows, manifest_dir, args.input_dim, label_maps=label_maps, allow_metadata_fallback=allow_fallback)
    )
    test_dataset = as_torch_dataset(
        FeatureDataset(test_rows, manifest_dir, args.input_dim, label_maps=label_maps, allow_metadata_fallback=allow_fallback)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_pose_components(args.input_dim, args.hidden_dim, label_maps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features = batch["feature"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            projection, logits = model(features)
            # 多属性分类损失：每个 MTL-AQA 动作属性分别计算交叉熵后求平均。
            cls_loss = sum(
                criterion(logits[column], labels[:, idx])
                for idx, column in enumerate(label_maps)
            ) / max(len(label_maps), 1)
            con_loss = supervised_contrastive_loss(projection, labels)
            loss = cls_loss + args.contrastive_weight * con_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * features.shape[0]

        f1_payload = evaluate(model, test_loader, label_maps, device)
        item = {"epoch": epoch, "train_loss": total_loss / max(len(train_dataset), 1), **f1_payload}
        history.append(item)
        print(json.dumps(item, ensure_ascii=False))
        if f1_payload["mean_f1"] > best_f1:
            best_f1 = f1_payload["mean_f1"]
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    vars(args),
                    extra={"label_maps": label_maps, "best_metric": f1_payload, "model_type": "pose"},
                ),
                args.checkpoint,
            )

    payload = {
        "model": "Pose Contrastive",
        "task": "MTL-AQA multi-attribute classification",
        "synthetic_smoke": bool(args.synthetic_smoke),
        "checkpoint": str(args.checkpoint),
        "best_mean_f1": best_f1,
        "label_maps": label_maps,
        "history": history,
        "env": runtime_env(),
    }
    write_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
