"""下载上游 MTL-AQA 与 Fitness-AQA 代码。

为了遵守“少用 shell”的项目约束，本脚本不依赖 `git clone`，而是使用 Python
标准库下载 GitHub zip 包并解压到 `external/`。下载失败时会给出明确错误，
便于把网络或上游地址问题记录到 PPT。
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from aqa_common import project_root


REPOS = {
    "MTL-AQA": "https://github.com/ParitoshParmar/MTL-AQA/archive/refs/heads/master.zip",
    "Fitness-AQA": "https://github.com/ParitoshParmar/Fitness-AQA/archive/refs/heads/main.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载课程实验所需的两个上游代码仓库")
    parser.add_argument("--external-dir", type=Path, default=project_root() / "external")
    parser.add_argument("--force", action="store_true", help="已存在目标目录时重新覆盖下载")
    return parser.parse_args()


def download_zip(url: str, target: Path) -> None:
    """下载 zip 文件。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载：{url}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_repo(zip_path: Path, output_dir: Path, final_name: str, force: bool) -> None:
    """解压 GitHub zip，并把自动生成的顶层目录重命名为固定名称。"""

    target_dir = output_dir / final_name
    if target_dir.exists():
        if not force:
            print(f"已存在，跳过：{target_dir}")
            return
        shutil.rmtree(target_dir)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
        top_level = archive.namelist()[0].split("/")[0]
    extracted = output_dir / top_level
    extracted.rename(target_dir)
    print(f"已解压：{target_dir}")


def main() -> None:
    args = parse_args()
    args.external_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.external_dir / "_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, url in REPOS.items():
        zip_path = cache_dir / f"{name}.zip"
        download_zip(url, zip_path)
        extract_repo(zip_path, args.external_dir, name, args.force)


if __name__ == "__main__":
    main()
