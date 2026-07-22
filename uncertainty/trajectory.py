"""Base trajectory capture and branch-point ensemble resumption.

Internally we only ever track the *kept* (class-conditioned) half of the
CFG-doubled batch. This is mathematically equivalent to carrying the full
2N-row doubled tensor through diffusion.p_sample every step and discarding
the second half at the very end (as sample.py does): model.forward_with_cfg
always rebuilds its "combined" input from the first half of whatever tensor
it's given (`half = x[: len(x) // 2]`), so the second half's stored value
never feeds back into the kept output. Re-deriving it fresh each step
(`torch.cat([x, x])`) instead of carrying it forward produces byte-identical
results for the rows we keep, and lets us record per-step noise (and branch
latents) at batch size N instead of 2N.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class BaseTrajectory:
    class_id: int
    base_idx: int
    seed: int
    final_latent: torch.Tensor
    branch_latents: dict  # step_idx -> (1, C, H, W) tensor
    step_noises: list  # per-step (1, C, H, W) tensor, indexed 0..num_steps-1
    full_latents: list = field(default_factory=list)  # only populated if record_full=True


def _double(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t, t], dim=0)


def run_base_trajectory(adapter, class_id: int, base_idx: int, seed: int,
                         branch_points, record_full: bool = False) -> BaseTrajectory:
    diffusion = adapter.diffusion
    device = adapter.device
    branch_points = set(branch_points)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    shape = adapter.latent_shape(1)
    x = torch.randn(shape, generator=gen, device=device)

    indices = list(range(diffusion.num_timesteps))[::-1]
    model_kwargs = adapter.model_kwargs(class_id, batch=1)
    forward = adapter.forward_with_cfg()

    branch_latents = {}
    step_noises = []
    full_latents = []

    for step_idx, t_val in enumerate(indices):
        if step_idx in branch_points:
            branch_latents[step_idx] = x.clone()

        t = torch.full((1,), t_val, dtype=torch.long, device=device)
        step_noise = torch.randn(shape, generator=gen, device=device)

        out = diffusion.p_sample(
            forward,
            _double(x),
            _double(t),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            noise=_double(step_noise),
        )
        x = out["sample"][:1]
        step_noises.append(step_noise.clone())
        if record_full:
            full_latents.append(x.clone())

    return BaseTrajectory(
        class_id=class_id,
        base_idx=base_idx,
        seed=seed,
        final_latent=x,
        branch_latents=branch_latents,
        step_noises=step_noises,
        full_latents=full_latents,
    )


def resume_from_branch(adapter, class_id: int, latents: torch.Tensor, step_noises: list,
                        start_step: int, record_steps: bool = False):
    """Resumes denoising an ensemble of K perturbed latents from `start_step`
    onward, reusing the base's own recorded per-step noise (broadcast across
    the K members) and the same class conditioning."""
    diffusion = adapter.diffusion
    k = latents.shape[0]
    indices = list(range(diffusion.num_timesteps))[::-1]
    model_kwargs = adapter.model_kwargs(class_id, batch=k)
    forward = adapter.forward_with_cfg()

    x = latents
    per_step_latents = []
    for step_idx in range(start_step, len(indices)):
        t_val = indices[step_idx]
        t = torch.full((k,), t_val, dtype=torch.long, device=adapter.device)
        noise = step_noises[step_idx].expand(k, -1, -1, -1).contiguous()

        out = diffusion.p_sample(
            forward,
            _double(x),
            _double(t),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            noise=_double(noise),
        )
        x = out["sample"][:k]
        if record_steps:
            per_step_latents.append(x.clone())

    return x, per_step_latents
