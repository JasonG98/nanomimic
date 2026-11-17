"""从中文训练日志绘制损失、真实奖励和完整动作评估曲线。

只读取日志与评估 JSON，不接触训练进程、模型或 GPU。
浅线保留原始采样点，深线是向后看的滑动均值；缺失损失不补零。
PNG 用于查看，SVG 用于放大；重新执行可更新同一输出目录。
"""

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import tyro
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Config:
    # ScalarSink 输出的中文训练日志。
    log: Path
    # 包含 evaluations/ 的运行目录，或直接包含 eval-*.json 的评估目录。
    evaluations: Path | None = None
    # nvidia-smi 每 200 ms 采样的整卡资源 CSV。
    vram: Path | None = None
    # 图片输出目录；重复运行更新曲线。
    out: Path = Path("outputs/training-curves")
    # 每条平滑线使用的已记录点数，不跨未来数据。
    smooth: int = 21


def read_metrics(path: Path) -> list[dict[str, float]]:
    """只读取完整日志行，按迭代去重，保留当次实际存在的标量。"""
    labels = {
        "世界模型损失": "wm_loss",
        "策略损失": "actor_loss",
        "价值损失": "critic_loss",
        "仿真奖励": "sim_reward",
        "重置比例": "done_fraction",
        "累计训练次数": "updates",
    }
    records: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        match = re.match(r"迭代 (\d+) \|", line)
        if match is None or not line.endswith("\n"):
            continue
        iteration = int(match[1])
        row = {"iteration": float(iteration)}
        for term in line.split("|")[1:]:
            label, value = term.strip().rsplit(" ", 1)
            if label in labels:
                row[labels[label]] = float(value)
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError(f"迭代 {iteration} 的日志包含非有限值")
        records[iteration] = row
    if not records:
        raise ValueError("日志尚无完整的训练标量行")
    return [records[key] for key in sorted(records)]


def _line(ax: Axes, rows: list[dict[str, float]], name: str, title: str, smooth: int, scale: float = 1) -> None:
    points = np.array([(row["iteration"], row[name] * scale) for row in rows if name in row])
    if points.size == 0:
        ax.set_title(f"{title}：暂无数据")
        return
    x, y = points.T
    ax.plot(x, y, color="#277da1", alpha=0.28, linewidth=0.8, label="原始日志点")
    window = min(smooth, len(y))
    average = np.convolve(y, np.ones(window) / window, mode="valid")
    ax.plot(x[window - 1 :], average, color="#17647f", linewidth=1.8, label=f"{window} 点滑动均值")
    ax.set_title(f"{title}  ·  最新 {y[-1]:.4g}", loc="left")
    ax.set_xlabel("采集迭代数")
    ax.set_xlim(0, rows[-1]["iteration"] * 1.02)
    ax.grid(alpha=0.18)


def _save(fig: Figure, out: Path, name: str) -> list[Path]:
    FigureCanvasAgg(fig)
    paths = [out / f"{name}.{extension}" for extension in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=160, metadata={"Creator": "nanomimic"})
    return paths


def _training(cfg: Config, rows: list[dict[str, float]], evaluations: list[dict[str, Any]]) -> list[Path]:
    fig = Figure(figsize=(13, 11.5), layout="constrained")
    axes = fig.subplots(3, 2).flatten()
    latest = int(rows[-1]["iteration"])
    fig.suptitle(
        f"T7 真实动作训练 · {latest:,} 次迭代\n浅线：每 10 次迭代的原始日志；深线：{cfg.smooth} 点滑动均值",
        fontsize=16,
    )
    for ax, name, title in zip(
        axes[:5],
        ("wm_loss", "actor_loss", "critic_loss", "sim_reward", "done_fraction"),
        (
            "世界模型损失",
            "Actor 损失（含末端价值的负回报）",
            "Critic 损失",
            "真实仿真奖励 / 控制步",
            "采集时每步重置比例（%）",
        ),
        strict=True,
    ):
        _line(ax, rows, name, title, cfg.smooth, 100 if name == "done_fraction" else 1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Bryson 加权均方误差 · 对数轴")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("越负表示模型预测的回报越高")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("价值目标均方误差 · 对数轴")
    axes[3].axhline(0.1, color="#888888", linestyle="--", linewidth=1, label="正奖励上界 0.1")
    axes[3].set_ylim(top=max(0.108, max(row.get("sim_reward", 0) for row in rows) * 1.08))
    axes[3].legend(frameon=False)
    axes[4].set_ylabel("单步 done 行数 / 环境数 × 100")
    evaluation = axes[5]
    evaluation.set_title("完整动作评估 · 从第 0 帧开始", loc="left")
    evaluation.set_xlabel("检查点迭代数")
    evaluation.set_ylabel("平均首次尝试存活时间（秒）")
    if evaluations:
        x = [row["iteration"] for row in evaluations]
        y = [row["mean_survival_seconds"] for row in evaluations]
        duration = evaluations[0]["expected_steps"] * 0.02
        evaluation.plot(x, y, "o-", color="#17647f", linewidth=1.8)
        evaluation.axhline(duration, color="#888888", linestyle="--", linewidth=1, label=f"完整动作 {duration:.2f} 秒")
        for row in evaluations:
            above = row["mean_survival_seconds"] < 0.65 * duration
            evaluation.annotate(
                f"{row['mean_survival_seconds']:.1f} 秒\n全程无终止 {row['successes']}/{row['trials']}",
                (row["iteration"], row["mean_survival_seconds"]),
                xytext=(0, 10 if above else -15),
                textcoords="offset points",
                ha="center",
                va="bottom" if above else "top",
                fontsize=10,
                bbox={"facecolor": fig.get_facecolor(), "edgecolor": "none", "alpha": 0.92, "pad": 2},
            )
        evaluation.set_ylim(0, duration * 1.23)
        evaluation.legend(frameon=False, loc="upper left")
    else:
        evaluation.text(0.5, 0.5, "暂无完整动作评估结果", ha="center", transform=evaluation.transAxes)
    evaluation.set_xlim(0, latest * 1.05)
    evaluation.grid(alpha=0.18)
    fig.supxlabel("损失描述训练目标；每步重置比例来自训练采集。效果验收仍需完整动作成功率及跟踪误差。", fontsize=10)
    return _save(fig, cfg.out, "training-curves")


def _resources(cfg: Config) -> list[Path]:
    assert cfg.vram is not None
    with cfg.vram.open(encoding="utf-8") as stream:
        samples = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    times = [datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for row in samples]
    elapsed = [(time - times[0]).total_seconds() / 60 for time in times]
    memory = [float(row["memory.used [MiB]"].split()[0]) * 1024**2 / 1e9 for row in samples]
    utilization = [float(row["utilization.gpu [%]"].split()[0]) for row in samples]
    fig = Figure(figsize=(13, 5.5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    fig.suptitle("整卡资源采样 · 包含桌面、训练与独立评估", fontsize=15)
    axes[0].plot(elapsed, memory, color="#17647f", linewidth=1)
    axes[0].axhline(15, color="#999999", linestyle="--", linewidth=1, label="SPEC 上限 15 GB")
    axes[0].set_ylabel("已用显存（十进制 GB）")
    axes[0].set_ylim(0, max(16, max(memory) * 1.08))
    axes[0].set_title(f"观测峰值 {max(memory):.3f} GB；采样间隔 200 ms", loc="left")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(elapsed, utilization, color="#277da1", alpha=0.7, linewidth=0.7)
    axes[1].set_ylabel("GPU 利用率（%）")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("自首次采样经过的时间（分钟）")
    for ax in axes:
        ax.grid(alpha=0.18)
    return _save(fig, cfg.out, "gpu-resources")


def plot(cfg: Config) -> list[Path]:
    """生成静态训练和资源曲线，返回 PNG / SVG 的绝对路径。"""
    if cfg.smooth < 1:
        raise ValueError("滑动均值窗口必须大于零")
    rows = read_metrics(cfg.log)
    evaluation_dir = cfg.evaluations
    if evaluation_dir is not None and (evaluation_dir / "evaluations").is_dir():
        evaluation_dir = evaluation_dir / "evaluations"
    evaluations = (
        [] if evaluation_dir is None else [json.loads(path.read_text()) for path in evaluation_dir.glob("eval-*.json")]
    )
    evaluations.sort(key=lambda row: row["iteration"])
    cfg.out.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(
        {
            "font.sans-serif": [
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "PingFang SC",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        paths = _training(cfg, rows, evaluations)
        if cfg.vram is not None:
            paths.extend(_resources(cfg))
    print(f"已绘制至迭代 {int(rows[-1]['iteration'])}，{len(rows)} 个日志点、{len(evaluations)} 个评估点。")
    for path in paths:
        print(f"曲线：{path.resolve()}")
    return [path.resolve() for path in paths]


if __name__ == "__main__":
    plot(tyro.cli(Config, description="从训练日志绘制可复现的损失、奖励、完整动作评估与显存曲线。"))
