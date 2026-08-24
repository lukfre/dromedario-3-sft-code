import ast
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_STEPS = {
    "control": 9784,
    "sub_50": 9784,
    "sub_100": 9784,
    "add_50": 13074,
    "add_100": 16366,
}

OFFSET = 300


def parse_record(line: str) -> Optional[Dict[str, Any]]:
    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            record = parser(line)
        except Exception:
            continue
        if isinstance(record, dict):
            return record
    return None


def parse_losses(log_path: Path) -> Tuple[List[float], List[float]]:
    losses: List[float] = []
    steps: List[float] = []
    started = False

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not started:
                if line.startswith("{") and "loss" in line:
                    started = True
                else:
                    continue

            if not line.startswith("{"):
                continue

            record = parse_record(line)
            if not record or "loss" not in record:
                continue

            try:
                loss_value = float(record["loss"])
            except (TypeError, ValueError):
                continue

            epoch_value: Optional[float] = None
            if "epoch" in record:
                try:
                    epoch_value = float(record["epoch"])
                except (TypeError, ValueError):
                    epoch_value = None

            if epoch_value is None:
                epoch_value = float(len(losses) + 1)

            losses.append(loss_value)
            steps.append(epoch_value)

    return losses, steps


def parse_run_key(file_name: str) -> Optional[str]:
    parts = file_name.split("___")
    if len(parts) < 2:
        return None

    model = parts[0].lower()
    config_raw = parts[1].split(".", 1)[0]
    config = config_raw.lower()
    allowed_configs = {"control", "sub_50", "sub_100", "add_50", "add_100"}
    if config not in allowed_configs:
        return None

    if not (model.startswith("llama") or model.startswith("minerva")):
        return None

    return f"{model}___{config}"


def _extract_timestamp(path: Path) -> datetime:
    stem = path.name.removesuffix(".out.log")
    timestamp_part = "_".join(stem.split("_")[:2])
    return datetime.strptime(timestamp_part, "%Y-%m-%d_%H-%M-%S")


def gather_log_segments(logs_root: Path) -> Dict[str, List[Path]]:
    segments: Dict[str, List[Path]] = {}
    if not logs_root.exists():
        return segments
    for run_dir in logs_root.iterdir():
        if not run_dir.is_dir():
            continue
        if "___" not in run_dir.name:
            continue
        log_files = sorted(run_dir.glob("*.out.log"), key=_extract_timestamp)
        if log_files:
            segments[run_dir.name] = log_files
    return segments


def order_segment(
    losses: List[float], steps: List[float]
) -> Tuple[List[float], List[float]]:
    if not steps or len(steps) != len(losses):
        return losses, steps

    ordered = sorted(zip(steps, losses), key=lambda pair: pair[0])
    sorted_steps = [step for step, _ in ordered]
    sorted_losses = [loss for _, loss in ordered]
    return sorted_losses, sorted_steps


def plot_loss(run_name: str, segment_paths: List[Path], output_path: Path) -> bool:
    segments: List[Tuple[List[float], List[float]]] = []
    for log_path in segment_paths:
        losses, steps = parse_losses(log_path)
        if not losses:
            continue
        segments.append(order_segment(losses, steps))

    if not segments:
        return False

    def to_steps(epochs: List[float]) -> List[float]:
        parts = run_name.split("___", 1)
        config = parts[1] if len(parts) == 2 else ""
        max_steps = NUM_STEPS.get(config)
        if not max_steps:
            return epochs
        steps_per_epoch = max_steps / 2.0
        return [epoch * steps_per_epoch for epoch in epochs]

    def epoch_markers() -> List[float]:
        parts = run_name.split("___", 1)
        config = parts[1] if len(parts) == 2 else ""
        max_steps = NUM_STEPS.get(config)
        if not max_steps:
            return []
        steps_per_epoch = max_steps / 2.0
        return [steps_per_epoch, steps_per_epoch * 2.0]

    segments.sort(key=lambda segment: segment[1][0] if segment[1] else float("inf"))
    step_segments = [(losses, to_steps(steps)) for losses, steps in segments]
    all_losses = [loss for losses, _ in step_segments for loss in losses]
    all_steps = [step for _, steps in step_segments for step in steps]
    valid_losses = [loss for loss in all_losses if math.isfinite(loss)]
    if not valid_losses:
        return False

    break_threshold = 1
    if len(valid_losses) > 1:
        sorted_losses = sorted(valid_losses)
        index = int(0.95 * (len(sorted_losses) - 1))
        percentile_95 = sorted_losses[index]
        break_threshold = max(break_threshold, percentile_95)

    def smooth_series(values: List[float], window: int = 25) -> List[float]:
        if window <= 1 or len(values) < 2:
            return values
        half = window // 2
        smoothed: List[float] = []
        for idx in range(len(values)):
            start = max(0, idx - half)
            end = min(len(values), idx + half + 1)
            slice_vals = values[start:end]
            smoothed.append(sum(slice_vals) / len(slice_vals))
        smoothed[0] = values[0]
        return smoothed

    name_lower = run_name.lower()
    if "llama" in name_lower:
        colors = ["steelblue", "steelblue"]
    elif "minerva" in name_lower:
        colors = ["#7b2d2f", "#7b2d2f"]
    else:
        colors = ["tab:gray", "tab:olive"]

    needs_break = max(valid_losses) > break_threshold
    if needs_break and "minerva___" in run_name.lower():
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8, 5),
            gridspec_kw={"height_ratios": [1, 5], "hspace": 0.05},
        )
        markers = epoch_markers()
        for ax in (ax_top, ax_bottom):
            for idx, (losses, steps) in enumerate(step_segments):
                color = colors[idx % len(colors)]
                smoothed = smooth_series(losses)
                ax.plot(steps, losses, linewidth=2.5, color=color, alpha=0.5,
                        label="raw" if idx == 0 else None)
                ax.plot(steps, smoothed, linewidth=1.5, color=color, alpha=1.0,
                        label="smoothed" if idx == 0 else None)
            for marker in markers:
                ax.axvline(marker, color="forestgreen", linestyle="-",
                           linewidth=1.5, alpha=0.3)
            ax.grid("x", linestyle="--", alpha=0.7)

        ax_bottom.set_ylim(min(valid_losses), break_threshold)
        upper_max = max(valid_losses)
        upper_pad = max(0.05, 0.05 * upper_max)
        ax_top.set_ylim(break_threshold, upper_max + upper_pad)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_bottom.xaxis.tick_bottom()

        diag_kwargs = dict(color="gray", clip_on=False, linewidth=1.0)
        dx = 0.02
        top_box = ax_top.get_position()
        bottom_box = ax_bottom.get_position()
        dy_top = dx * (top_box.width / top_box.height)
        dy_bottom = dx * (bottom_box.width / bottom_box.height)
        ax_top.plot((-dx, dx), (-dy_top, dy_top), transform=ax_top.transAxes, **diag_kwargs)
        ax_top.plot((1 - dx, 1 + dx), (-dy_top, dy_top), transform=ax_top.transAxes, **diag_kwargs)
        ax_bottom.plot((-dx, dx), (1 - dy_bottom, 1 + dy_bottom), transform=ax_bottom.transAxes, **diag_kwargs)
        ax_bottom.plot((1 - dx, 1 + dx), (1 - dy_bottom, 1 + dy_bottom), transform=ax_bottom.transAxes, **diag_kwargs)

        ax_top.set_title(run_name)
        ax_bottom.set_xlabel("step")
        ax_bottom.set_ylabel("loss")
        config = run_name.split("___", 1)[1] if "___" in run_name else ""
        max_steps = NUM_STEPS.get(config)
        if max_steps:
            ax_top.set_xlim(0 - OFFSET, max_steps + OFFSET)
            ax_bottom.set_xlim(0 - OFFSET, max_steps + OFFSET)
        ax_bottom.legend(loc="upper right")
        fig.tight_layout()
    else:
        plt.figure(figsize=(8, 5))
        markers = epoch_markers()
        for idx, (losses, steps) in enumerate(step_segments):
            color = colors[idx % len(colors)]
            smoothed = smooth_series(losses)
            plt.plot(steps, losses, linewidth=2.5, color=color, alpha=0.5,
                     label="raw" if idx == 0 else None)
            plt.plot(steps, smoothed, linewidth=1.5, color=color, alpha=1.0,
                     label="smoothed" if idx == 0 else None)
        for marker in markers:
            plt.axvline(marker, color="forestgreen", linestyle="-", linewidth=1.5, alpha=0.3)
        plt.title(run_name)
        plt.xlabel("step")
        plt.ylabel("loss")
        config = run_name.split("___", 1)[1] if "___" in run_name else ""
        max_steps = NUM_STEPS.get(config)
        if max_steps:
            plt.xlim(0 - OFFSET, max_steps + OFFSET)
        plt.tight_layout()
        plt.grid("x", linestyle="--", alpha=0.7)
        plt.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True


def main() -> None:
    code_dir = Path(__file__).resolve().parent.parent
    logs_dir = code_dir / "logs"
    plots_dir = code_dir / "plots"

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    run_segments = gather_log_segments(logs_dir)
    if not run_segments:
        raise SystemExit(f"No .out.log files found in {logs_dir}")

    for run_name, segment_paths in run_segments.items():
        output_path = plots_dir / run_name.split("___")[0] / f"{run_name}.loss.png"
        plotted = plot_loss(run_name, segment_paths, output_path)
        if plotted:
            print(f"Saved {output_path}")
        else:
            print(f"No loss records found in {run_name}")
