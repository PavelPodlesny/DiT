"""--dry-run verification: confirms the p_sample noise hook + resume logic
reproduce a base trajectory's own continuation exactly (up to fp32 rounding)
when the `mock` (identity) perturbation is used.
"""
import logging

import torch.nn.functional as F

from uncertainty.trajectory import run_base_trajectory, resume_from_branch

logger = logging.getLogger("uncertainty.dryrun")


def run_dry_run(adapter, config) -> bool:
    tol = config.dry_run_tolerances
    all_ok = True

    for class_id in config.classes:
        for base_idx in range(config.M):
            seed = config.base_seed + base_idx
            base = run_base_trajectory(
                adapter, class_id, base_idx, seed, config.branch_points, record_full=True
            )
            base_final_image = adapter.decode(base.final_latent)

            for branch_point in config.branch_points:
                branch_latent = base.branch_latents[branch_point]
                final_latents, per_step_latents = resume_from_branch(
                    adapter, class_id, branch_latent, base.step_noises, branch_point,
                    record_steps=True,
                )

                first_bad_step = None
                for offset, resumed_latent in enumerate(per_step_latents):
                    step_idx = branch_point + offset
                    expected = base.full_latents[step_idx]
                    mse = F.mse_loss(resumed_latent, expected).item()
                    if mse > tol.latent_mse and first_bad_step is None:
                        first_bad_step = (step_idx, mse)

                final_image = adapter.decode(final_latents)
                image_mse = F.mse_loss(final_image, base_final_image).item()

                status = "OK"
                if first_bad_step is not None or image_mse > tol.image_mse:
                    status = "FAIL"
                    all_ok = False

                logger.info(
                    "class=%s base=%s branch=%s: image_mse=%.3e (tol=%.3e) status=%s",
                    class_id, base_idx, branch_point, image_mse, tol.image_mse, status,
                )
                if first_bad_step is not None:
                    step_idx, mse = first_bad_step
                    logger.info(
                        "  first tolerance breach at step=%s latent_mse=%.3e (tol=%.3e)",
                        step_idx, mse, tol.latent_mse,
                    )

    return all_ok
