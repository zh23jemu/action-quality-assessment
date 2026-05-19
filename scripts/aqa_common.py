"""MTL-AQA 复现实验公共工具。

本模块集中放置数据读取、指标计算、模型定义和日志输出等共享逻辑。
这样训练、测试和单样本推理脚本可以保持一致的数据口径，避免同一个
指标在不同脚本中被重复实现后产生偏差。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LABEL_COLUMNS = ["position", "armstand", "rotation_type", "somersaults", "twists"]


def project_root() -> Path:
    """返回项目根目录。

    脚本可能从 Slurm 工作目录、本地 PowerShell 或 IDE 中启动，因此这里
    通过当前文件位置反推根目录，避免依赖调用者的当前工作目录。
    """

    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """把 manifest 中的相对路径解析为绝对路径。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_parent(path: Path) -> None:
    """确保输出文件的父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """设置常见随机源，保证小规模调试结果尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # prepare_data 等无 PyTorch 场景不应因为未安装 torch 而失败。
        pass


def load_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 文件。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """以可读格式写出 JSON，便于截图和后续报告引用。"""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def json_safe(value: Any) -> Any:
    """把配置对象转换为 JSON/checkpoint 友好的基础类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """读取统一 manifest。

    当前实现使用 CSV 作为最低依赖格式，方便人工检查和在 Excel 中打开。
    字段名由 prepare_data 统一生成；如果用户手工整理，也只需保持字段名一致。
    """

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """写出统一 manifest，并自动汇总所有字段。"""

    ensure_parent(path)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_rows(rows: Sequence[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    """按 split 字段筛选样本，支持 train/test/val 等划分。"""

    selected = [row for row in rows if row.get("split", "").strip().lower() == split.lower()]
    if not selected:
        raise ValueError(f"manifest 中没有 split={split!r} 的样本")
    return selected


def coerce_float(value: Any, default: float = 0.0) -> float:
    """把标注字段安全转换为浮点数。"""

    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def stable_hash_vector(text: str, dim: int) -> np.ndarray:
    """为冒烟测试或显式 fallback 生成稳定向量。

    该函数不能用于正式指标，只用于 `--synthetic-smoke` 或用户明确允许
    metadata fallback 的场景。这样可以验证训练代码，但不会伪造真实视频特征。
    """

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8)
    return (values.astype(np.float32) / 255.0) * 2.0 - 1.0


def load_feature(row: Dict[str, str], manifest_dir: Path, input_dim: int, allow_metadata_fallback: bool) -> np.ndarray:
    """读取样本特征。

    正式训练优先要求 `feature_path` 指向 `.npy`、`.pt` 或 `.pth` 文件。
    如果没有特征且未显式允许 fallback，会直接报错，防止把占位特征误当作结果。
    """

    feature_path = row.get("feature_path", "").strip()
    if feature_path:
        path = resolve_path(feature_path, manifest_dir)
        if not path.exists():
            raise FileNotFoundError(f"特征文件不存在：{path}")
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        elif path.suffix.lower() in {".pt", ".pth"}:
            import torch

            tensor = torch.load(path, map_location="cpu")
            array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        else:
            raise ValueError(f"暂不支持的特征格式：{path.suffix}，请使用 .npy/.pt/.pth")
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if array.size == input_dim:
            return array
        if array.size > input_dim:
            return array[:input_dim]
        padded = np.zeros(input_dim, dtype=np.float32)
        padded[: array.size] = array
        return padded

    if allow_metadata_fallback:
        identity = row.get("sample_id") or row.get("video_path") or json.dumps(row, sort_keys=True)
        return stable_hash_vector(identity, input_dim)

    raise ValueError(
        "manifest 缺少 feature_path。正式训练需要先抽取视频特征；"
        "仅冒烟测试可使用 --synthetic-smoke 或 --allow-metadata-fallback。"
    )


def build_label_maps(rows: Sequence[Dict[str, str]], label_columns: Sequence[str] = LABEL_COLUMNS) -> Dict[str, Dict[str, int]]:
    """根据训练集构建动作属性到类别编号的映射。"""

    maps: Dict[str, Dict[str, int]] = {}
    for column in label_columns:
        values = sorted({str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip() != ""})
        if not values:
            raise ValueError(f"训练集缺少动作属性字段：{column}")
        maps[column] = {value: idx for idx, value in enumerate(values)}
    return maps


def encode_labels(row: Dict[str, str], label_maps: Dict[str, Dict[str, int]]) -> List[int]:
    """把一行样本的多个动作属性编码为类别编号。"""

    encoded: List[int] = []
    for column, mapping in label_maps.items():
        value = str(row.get(column, "")).strip()
        if value not in mapping:
            # 测试集中出现训练集未见标签时，映射到 0 并在评估中保留风险。
            encoded.append(0)
        else:
            encoded.append(mapping[value])
    return encoded


def spearman_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """计算 Spearman 等级相关系数。

    优先使用 scipy；如果环境中没有 scipy，则使用 numpy 的简化 rank 计算。
    """

    if len(y_true) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr

        value = spearmanr(y_true, y_pred).correlation
        return float(0.0 if math.isnan(value) else value)
    except Exception:
        true_rank = np.argsort(np.argsort(np.asarray(y_true)))
        pred_rank = np.argsort(np.argsort(np.asarray(y_pred)))
        value = np.corrcoef(true_rank, pred_rank)[0, 1]
        return float(0.0 if math.isnan(value) else value)


def relative_l2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """计算相对 L2 距离，分母加极小值避免全零标签导致除零。"""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.linalg.norm(true - pred, ord=2) / (np.linalg.norm(true, ord=2) + 1e-8))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """计算宏平均 F1，优先使用 scikit-learn，缺失时使用轻量实现。"""

    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        labels = sorted(set(y_true) | set(y_pred))
        values = []
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            values.append(2 * precision * recall / (precision + recall + 1e-8))
        return float(np.mean(values)) if values else 0.0


def runtime_env() -> Dict[str, Any]:
    """记录运行环境摘要，便于报告和问题排查。"""

    payload: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root()),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }
        )
    except Exception as exc:
        payload["torch_error"] = str(exc)
    return payload


def synthetic_rows(count: int = 32, feature_dir: Optional[Path] = None, input_dim: int = 128) -> List[Dict[str, Any]]:
    """生成显式冒烟测试样本。

    该数据只用于验证 dataloader、模型、loss、checkpoint 和评估输出，不应写入报告
    的正式实验结果。
    """

    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        split = "train" if idx < int(count * 0.75) else "test"
        score = 40.0 + (idx % 10) * 3.0 + (idx // 10)
        row: Dict[str, Any] = {
            "sample_id": f"synthetic_{idx:04d}",
            "split": split,
            "score": score,
            "position": str(idx % 3),
            "armstand": str(idx % 2),
            "rotation_type": str(idx % 4),
            "somersaults": str(idx % 5),
            "twists": str(idx % 4),
        }
        if feature_dir is not None:
            # 冒烟特征写入绝对路径，避免训练脚本用 manifest 目录二次拼接相对路径。
            feature_dir = feature_dir.resolve()
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_path = feature_dir / f"{row['sample_id']}.npy"
            feature = stable_hash_vector(row["sample_id"], input_dim) + score / 100.0
            np.save(feature_path, feature.astype(np.float32))
            row["feature_path"] = str(feature_path)
        rows.append(row)
    return rows


@dataclass
class DatasetConfig:
    """数据集配置，供训练和测试脚本共享。"""

    manifest: Path
    split: str
    input_dim: int
    allow_metadata_fallback: bool = False


def require_torch() -> Any:
    """延迟导入 PyTorch，并给出更清晰的错误信息。"""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("缺少 PyTorch，请先在项目 .venv 中安装 CUDA 12.x 兼容版本的 torch。") from exc


def make_motion_components(input_dim: int, hidden_dim: int):
    """创建 Motion Disentangling 的轻量复现模型。

    模型保留“共享编码 + 运动/质量相关潜变量 + 分数回归头”的核心思想，
    用于在 MTL-AQA 特征上输出动作质量分数。
    """

    torch = require_torch()
    nn = torch.nn

    class MotionDisentanglingNet(nn.Module):
        """基于特征向量的动作质量回归网络。"""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.motion_head = nn.Linear(hidden_dim, hidden_dim // 2)
            self.quality_head = nn.Linear(hidden_dim, hidden_dim // 2)
            self.score_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, features):
            encoded = self.encoder(features)
            motion_latent = self.motion_head(encoded)
            quality_latent = self.quality_head(encoded)
            score = self.score_head(quality_latent).squeeze(-1)
            return score, motion_latent, quality_latent

    return MotionDisentanglingNet()


def make_pose_components(input_dim: int, hidden_dim: int, label_maps: Dict[str, Dict[str, int]]):
    """创建 Pose Contrastive 的轻量复现模型。

    模型包含共享编码器、投影头和多个动作属性分类头。训练脚本会同时使用
    分类损失和监督式对比损失，使表征更贴近同类动作属性。
    """

    torch = require_torch()
    nn = torch.nn

    class PoseContrastiveNet(nn.Module):
        """多属性动作分类与对比学习网络。"""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )
            self.classifiers = nn.ModuleDict(
                {
                    column: nn.Linear(hidden_dim, len(mapping))
                    for column, mapping in label_maps.items()
                }
            )

        def forward(self, features):
            encoded = self.encoder(features)
            projection = self.projector(encoded)
            logits = {column: head(encoded) for column, head in self.classifiers.items()}
            return projection, logits

    return PoseContrastiveNet()


class FeatureDataset:
    """基于 manifest 的特征数据集。

    该类不继承 PyTorch Dataset，避免 prepare_data 等无 torch 场景导入失败；
    训练脚本中会通过 `as_torch_dataset` 包装为真正的 Dataset。
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        manifest_dir: Path,
        input_dim: int,
        label_maps: Optional[Dict[str, Dict[str, int]]] = None,
        allow_metadata_fallback: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.manifest_dir = manifest_dir
        self.input_dim = input_dim
        self.label_maps = label_maps
        self.allow_metadata_fallback = allow_metadata_fallback

    def __len__(self) -> int:
        return len(self.rows)

    def item(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        payload: Dict[str, Any] = {
            "sample_id": row.get("sample_id", str(index)),
            "feature": load_feature(row, self.manifest_dir, self.input_dim, self.allow_metadata_fallback),
            "score": coerce_float(row.get("score")),
        }
        if self.label_maps is not None:
            payload["labels"] = encode_labels(row, self.label_maps)
        return payload


def as_torch_dataset(dataset: FeatureDataset):
    """把轻量数据集包装成 PyTorch Dataset。"""

    torch = require_torch()

    class TorchFeatureDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            item = dataset.item(index)
            payload = {
                "sample_id": item["sample_id"],
                "feature": torch.tensor(item["feature"], dtype=torch.float32),
                "score": torch.tensor(item["score"], dtype=torch.float32),
            }
            if "labels" in item:
                payload["labels"] = torch.tensor(item["labels"], dtype=torch.long)
            return payload

    return TorchFeatureDataset()


def supervised_contrastive_loss(projection, labels, temperature: float = 0.2):
    """计算监督式对比损失。

    这里用第一个动作属性作为正样本分组依据，避免多属性标签组合过稀导致一个
    batch 内没有正样本。若 batch 太小或没有正样本，则返回 0，使训练仍可继续。
    """

    torch = require_torch()
    if projection.shape[0] < 2:
        return projection.sum() * 0.0
    projection = torch.nn.functional.normalize(projection, dim=1)
    sim = torch.matmul(projection, projection.T) / temperature
    label = labels[:, 0].view(-1, 1)
    positive_mask = torch.eq(label, label.T).float()
    self_mask = torch.eye(label.shape[0], device=projection.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    if positive_mask.sum() <= 0:
        return projection.sum() * 0.0
    exp_sim = torch.exp(sim) * (1.0 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    return loss.mean()


def checkpoint_payload(model, optimizer, epoch: int, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造 checkpoint 内容。"""

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": json_safe(config),
        "env": runtime_env(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_checkpoint(path: Path):
    """加载本项目生成的可信 checkpoint。

    PyTorch 2.6 起 `torch.load` 默认启用 weights_only，包含配置字典的 checkpoint
    会被拒绝。这里的文件来自本项目训练脚本，因此显式关闭 weights_only。
    """

    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版 PyTorch：旧版本没有 weights_only 参数。
        return torch.load(path, map_location="cpu")
