"""运行产物的目录约定与 JSON 原子写入。

检查点放在运行目录的 checkpoints/ 下，评估与导出使用各自子目录。
JSON 元信息先写临时文件再替换。
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def run_directory(checkpoint: Path) -> Path:
    """校验检查点目录结构并返回所属运行目录。"""
    if checkpoint.parent.name != "checkpoints" or not checkpoint.name.isdecimal():
        raise ValueError(f"检查点目录应为 <运行目录>/checkpoints/<迭代号>：{checkpoint}")
    return checkpoint.parent.parent


def write_json(path: Path, value: dict[str, Any]) -> None:
    """完整写入后替换 JSON，读取方不会看到半写内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as temporary:
        stage = Path(temporary) / path.name
        stage.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.replace(path)
