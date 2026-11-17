"""运行产物的目录约定与旧检查点路径兼容。

新检查点放在运行目录的 checkpoints/ 下，评估与导出使用各自子目录。
旧的运行目录/迭代号参数仍可定位迁移后的检查点，不修改历史配置。
JSON 元信息先写临时文件再替换；本模块不加载模型或仿真器。
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def resolve_checkpoint(directory: Path) -> Path:
    """兼容原地保留的旧检查点，以及已移入 checkpoints/ 的旧路径。"""
    if (directory / "config.json").is_file():
        return directory
    for parent in (directory.parent, directory.parent / "legacy"):
        candidate = parent / "checkpoints" / directory.name
        if (candidate / "config.json").is_file():
            return candidate
    return directory


def run_directory(checkpoint: Path) -> Path:
    """从新旧检查点路径得到所属运行目录。"""
    checkpoint = resolve_checkpoint(checkpoint)
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def write_json(path: Path, value: dict[str, Any]) -> None:
    """完整写入后替换 JSON，读取方不会看到半写内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as temporary:
        stage = Path(temporary) / path.name
        stage.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.replace(path)
