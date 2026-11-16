"""导入规则测试：核心模块必须与仿真器、日志工具解耦。

递归扫描全部子包；common、models、rollout 不得导入 mjlab、mujoco、wandb。
仅 runtime/sim.py 与后续的 runtime/viewer.py 可以导入 mjlab。
规则以完整相对路径判断，其他目录的同名文件不能获得例外。
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "nanomimic"

CORE_PACKAGES = frozenset({"common", "models", "rollout"})
CORE_FORBIDDEN_ROOTS = frozenset({"mjlab", "mujoco", "wandb"})
SIM_MODULES = frozenset({"runtime/sim.py", "runtime/viewer.py"})


def _package_modules() -> list[Path]:
    """递归返回包内全部模块，包含各子包的初始化文件。"""
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """返回模块顶层导入的根包名，例如 ``import mjlab.tasks`` 记作 ``mjlab``。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_modules_are_scanned() -> None:
    """确保子包内的关键模块被扫描，防止移动后测试空转。"""
    paths = {path.relative_to(PACKAGE_DIR).as_posix() for path in _package_modules()}
    required = {
        "common/specs.py",
        "models/model.py",
        "models/policy.py",
        "rollout/sensor.py",
        "rollout/imagine.py",
        "rollout/buffer.py",
        "runtime/sim.py",
    }
    assert required <= paths, f"导入规则扫描遗漏关键模块：{sorted(required - paths)}"


def test_core_modules_do_not_import_simulator_or_logging() -> None:
    """公共构件、模型和轨迹模块不得依赖仿真器与 wandb。"""
    offenders: list[str] = []
    for path in _package_modules():
        relative = path.relative_to(PACKAGE_DIR)
        if relative.parts[0] not in CORE_PACKAGES:
            continue
        forbidden = _imported_roots(path) & CORE_FORBIDDEN_ROOTS
        if forbidden:
            offenders.append(f"{relative.as_posix()}: {sorted(forbidden)}")
    assert not offenders, "以下核心模块违反导入规则：\n" + "\n".join(offenders)


def test_only_sim_modules_import_mjlab() -> None:
    """只有 runtime/sim.py 与 runtime/viewer.py 可以导入 mjlab。"""
    offenders = [
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in _package_modules()
        if path.relative_to(PACKAGE_DIR).as_posix() not in SIM_MODULES and "mjlab" in _imported_roots(path)
    ]
    assert not offenders, f"只有 runtime/sim.py / runtime/viewer.py 可以导入 mjlab，违规：{offenders}"
