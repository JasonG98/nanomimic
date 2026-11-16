"""导入规则测试：核心模块必须与仿真器、日志工具解耦。

规则来自 `docs/SPEC.md` 第 1 节：model、policy、sensor、imagine、buffer 不得导入
mjlab、mujoco、wandb；只有 sim.py 与 viewer.py 可以导入 mjlab。
核心模块由后续任务创建，文件出现后自动纳入检查。
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "nanomimic"

CORE_MODULES = ("model.py", "policy.py", "sensor.py", "imagine.py", "buffer.py")
CORE_FORBIDDEN_ROOTS = frozenset({"mjlab", "mujoco", "wandb"})
SIM_MODULES = frozenset({"sim.py", "viewer.py"})


def _package_modules() -> list[Path]:
    """返回包内全部顶层模块。"""
    return sorted(PACKAGE_DIR.glob("*.py"))


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
    """防止包路径写错导致导入规则测试空转。"""
    assert _package_modules(), f"未在 {PACKAGE_DIR} 找到任何模块"


def test_core_modules_do_not_import_simulator_or_logging() -> None:
    """model / policy / sensor / imagine / buffer 不得依赖仿真器与 wandb。"""
    offenders: list[str] = []
    for name in CORE_MODULES:
        path = PACKAGE_DIR / name
        if not path.is_file():
            # 文件由后续任务创建；创建后本测试自动覆盖。
            continue
        forbidden = _imported_roots(path) & CORE_FORBIDDEN_ROOTS
        if forbidden:
            offenders.append(f"{name}: {sorted(forbidden)}")
    assert not offenders, "以下核心模块违反导入规则：\n" + "\n".join(offenders)


def test_only_sim_modules_import_mjlab() -> None:
    """只有 sim.py 与 viewer.py 可以导入 mjlab。"""
    offenders = [
        path.name for path in _package_modules() if path.name not in SIM_MODULES and "mjlab" in _imported_roots(path)
    ]
    assert not offenders, f"只有 sim.py / viewer.py 可以导入 mjlab，违规：{offenders}"
