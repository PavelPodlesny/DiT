"""Contact-sheet grids, metrics CSV, and summary plots."""
import csv
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.utils import make_grid, save_image


def save_run_config(output_dir, config):
    """Snapshots the resolved config (including M/K) into output_dir so a
    results folder is self-describing without the original input YAML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def save_contact_sheet(path, reference_image: torch.Tensor, member_images: torch.Tensor, nrow: int = 4):
    """reference_image: (1, C, H, W), member_images: (K, C, H, W). Reference
    tile is placed first, followed by the K ensemble members."""
    grid_input = torch.cat([reference_image, member_images], dim=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid_input, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def archive_results(output_dir, archive_dir=None) -> Path:
    """Compresses `output_dir` into a timestamped .tar.gz, written to
    `archive_dir` (defaults to output_dir's parent, so the archive doesn't
    end up nested inside the directory it's archiving)."""
    output_dir = Path(output_dir)
    archive_dir = Path(archive_dir) if archive_dir else output_dir.parent
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = archive_dir / f"{output_dir.name}_{timestamp}"
    archive_path = shutil.make_archive(str(base_name), "gztar", root_dir=str(output_dir))
    return Path(archive_path)


class MetricsCSVWriter:
    FIELDS = ["class", "base", "branch_point", "type", "metric", "scope", "mean", "std"]

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, class_id, base_idx, branch_point, type_name, metrics: dict):
        """metrics: {(metric_name, scope): (mean, std)}."""
        for (metric_name, scope), (mean, std) in metrics.items():
            self._writer.writerow({
                "class": class_id,
                "base": base_idx,
                "branch_point": branch_point,
                "type": type_name,
                "metric": metric_name,
                "scope": scope,
                "mean": mean,
                "std": std,
            })

    def close(self):
        self._file.close()


def plot_summary_from_csv(csv_path, output_dir):
    """One PNG per (class, type, metric, scope), overlaying all M bases as
    separate lines, x = branch point, y = mean (error bars = std)."""
    rows = defaultdict(list)  # (class, type, metric, scope) -> [(base, branch_point, mean, std), ...]
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["class"], row["type"], row["metric"], row["scope"])
            rows[key].append((
                int(row["base"]), int(row["branch_point"]), float(row["mean"]), float(row["std"])
            ))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (class_id, type_name, metric, scope), entries in rows.items():
        by_base = defaultdict(list)
        for base_idx, branch_point, mean, std in entries:
            by_base[base_idx].append((branch_point, mean, std))

        fig, ax = plt.subplots(figsize=(6, 4))
        for base_idx, points in sorted(by_base.items()):
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            means = [p[1] for p in points]
            stds = [p[2] for p in points]
            ax.errorbar(xs, means, yerr=stds, marker="o", label=f"base {base_idx}")

        ax.set_xlabel("branch point (step index)")
        ax.set_ylabel(f"{metric} ({scope})")
        ax.set_title(f"class {class_id} — {type_name}")
        ax.legend()
        fig.tight_layout()

        out_path = output_dir / f"class{class_id}_{type_name}_{metric}_{scope}.png"
        fig.savefig(out_path)
        plt.close(fig)
