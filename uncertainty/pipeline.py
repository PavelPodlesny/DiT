"""Full (non-dry-run) pipeline: base trajectory capture, per-branch-point
perturbation ensembles, metrics, contact sheets, CSV, summary plots.
"""
import logging
from pathlib import Path

import torch

from uncertainty.metrics import MetricSuite, compute_ensemble_metrics
from uncertainty.perturbations import build_ensemble
from uncertainty.reporting import (
    MetricsCSVWriter, archive_results, plot_summary_from_csv, save_contact_sheet, save_run_config,
)
from uncertainty.trajectory import resume_from_branch, run_base_trajectory

logger = logging.getLogger("uncertainty.pipeline")


def run_pipeline(adapter, config):
    output_dir = Path(config.output_dir)
    latents_dir = output_dir / "latents"
    contact_sheets_dir = output_dir / "contact_sheets"
    plots_dir = output_dir / "plots"
    csv_path = output_dir / "metrics.csv"

    suite = MetricSuite(
        adapter.device,
        lpips_net=config.metrics.lpips_net,
        clip_model=config.metrics.clip_model,
        clip_pretrained=config.metrics.clip_pretrained,
        dino_model=config.metrics.dino_model,
    )
    csv_writer = MetricsCSVWriter(csv_path)
    save_run_config(output_dir, config)

    for class_id in config.classes:
        for base_idx in range(config.M):
            seed = config.base_seed + base_idx
            logger.info("class=%s base=%s: capturing base trajectory (seed=%s)", class_id, base_idx, seed)
            base = run_base_trajectory(adapter, class_id, base_idx, seed, config.branch_points)

            base_dir = latents_dir / f"class{class_id}" / f"base{base_idx}"
            base_dir.mkdir(parents=True, exist_ok=True)
            torch.save(base.final_latent.cpu(), base_dir / "final.pt")
            for branch_point, latent in base.branch_latents.items():
                torch.save(latent.cpu(), base_dir / f"branch{branch_point}.pt")

            final_image = adapter.decode(base.final_latent)

            for branch_point in config.branch_points:
                branch_latent = base.branch_latents[branch_point]
                reference_image = adapter.decode(branch_latent)

                for pert_spec in config.perturbations:
                    logger.info(
                        "class=%s base=%s branch=%s type=%s: generating ensemble (K=%s)",
                        class_id, base_idx, branch_point, pert_spec.type, config.K,
                    )
                    ensemble_latents = build_ensemble(
                        pert_spec.type, branch_latent, pert_spec.params, config.perturbation_seed,
                        class_id, base_idx, branch_point, config.K,
                    )
                    final_latents, _ = resume_from_branch(
                        adapter, class_id, ensemble_latents, base.step_noises, branch_point,
                    )
                    final_images = adapter.decode(final_latents)

                    ensemble_dir = base_dir / f"branch{branch_point}" / pert_spec.type
                    ensemble_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(final_latents.cpu(), ensemble_dir / "ensemble_final.pt")

                    save_contact_sheet(
                        contact_sheets_dir / f"class{class_id}" / f"base{base_idx}"
                        / f"branch{branch_point}" / f"{pert_spec.type}.png",
                        final_image, reference_image, final_images,
                    )

                    metrics = compute_ensemble_metrics(
                        suite, final_images, final_latents, reference_image, branch_latent,
                    )
                    csv_writer.write(class_id, base_idx, branch_point, pert_spec.type, metrics)

    csv_writer.close()
    plot_summary_from_csv(csv_path, plots_dir)
    logger.info("Done. Metrics: %s, plots: %s", csv_path, plots_dir)

    if config.archive:
        archive_path = archive_results(output_dir, config.archive_dir)
        logger.info("Archived results to: %s", archive_path)
