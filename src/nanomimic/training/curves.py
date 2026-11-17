"""把训练已记录的标量绘制为可反复更新的曲线。

只消费 CPU 标量，不接触模型、仿真器或 GPU，也不创建交互窗口。
缺失损失不补零；浅线显示原始点，深线显示向后看的 21 点均值。
PNG 与 SVG 先写临时目录再替换，读取图片时不会看到半写文件。
"""

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_training_curves(rows: Sequence[dict[str, float]], output: Path, labels: dict[str, str]) -> None:
    """按实际存在的指标绘图，覆盖运行目录内的 training-curves.png / svg。"""
    names = [name for name in labels if any(name in row for row in rows)]
    if not names:
        return
    latest = int(rows[-1]["iteration"])
    with mpl.rc_context(
        {
            "font.sans-serif": [
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "PingFang SC",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        height = (len(names) + 1) // 2
        figure = Figure(figsize=(13, height * 3.2), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(height, 2, squeeze=False).ravel()
        figure.suptitle(f"训练指标 · 截至迭代 {latest:,}（浅线：逐轮原始数据；深线：滑动均值）")
        for axis, name in zip(axes, names, strict=False):
            points = np.array([(row["iteration"], row[name]) for row in rows if name in row])
            x, y = points.T
            axis.plot(x, y, color="#277da1", alpha=0.25, linewidth=0.8, label="原始数据")
            window = min(21, len(y))
            average = np.convolve(y, np.ones(window) / window, mode="valid")
            axis.plot(x[window - 1 :], average, color="#17647f", linewidth=1.6, label=f"{window} 点均值")
            axis.set_title(f"{labels[name]} · 最新 {y[-1]:.4g}", loc="left")
            axis.set_xlabel("采集迭代数")
            axis.set_xlim(0, max(1, latest * 1.02))
            axis.grid(alpha=0.18)
            if name in {"wm_loss", "critic_loss"} and np.all(y > 0):
                axis.set_yscale("log")
            axis.legend(frameon=False, fontsize=8)
        for axis in axes[len(names) :]:
            axis.set_visible(False)
        with TemporaryDirectory(prefix=".curves-", dir=output) as temporary:
            for extension in ("png", "svg"):
                path = Path(temporary) / f"training-curves.{extension}"
                figure.savefig(path, dpi=140, metadata={"Creator": "nanomimic"})
                path.replace(output / path.name)
        figure.clear()
