# Latent-Perturbation Uncertainty Estimation

Quantifies a pretrained DiT model's predictive uncertainty by measuring how sensitive the final
generated image is to perturbations injected into the latent at chosen points along the denoising
trajectory.

## Directory contents

- `config.py` — YAML config loader (classes, M/K, seeds, branch points, perturbation specs,
  dry-run tolerances, metric backends).
- `adapter.py` — wraps the model/VAE/diffusion objects, reproducing the reference snippet's
  class-conditioning / classifier-free guidance batching.
- `trajectory.py` — base trajectory capture and branch-point ensemble resumption.
- `perturbations.py` — perturbation implementations (`gaussian_noise`, `gaussian_blur`,
  `nxn_mask`, `scattered_mask`, `mock`) plus per-ensemble-member seed derivation.
- `metrics.py` — latent MSE, LPIPS, CLIP/DINO cosine distance, both pairwise-among-ensemble and
  vs-base.
- `reporting.py` — contact-sheet grids, metrics CSV writer, summary plots.
- `dryrun.py` — step-by-step verification of the noise-replay hook against tolerances.
- `pipeline.py` — ties the above together for a full (non-dry-run) run.
- `config.example.yaml` — annotated example config to copy and edit.
- `README.md` — this file.

(`run_uncertainty.py`, the CLI entry point, lives at the repo root alongside `train.py`/`sample.py`.)

## How it works

1. For each configured ImageNet class, `M` **base trajectories** are sampled with fixed,
   per-base seeds (`base_seed + base_idx`), using the same class-conditioning / classifier-free
   guidance wiring as `sample.py` (`model.forward_with_cfg`, doubled class/null-class batch,
   `cfg_scale`). The stochastic noise drawn at every reverse-diffusion step is recorded.
2. At each configured **branch point** (a respaced sampling-step index, not a raw diffusion
   timestep `t`), the base's latent is saved, and `K` perturbed copies of it are produced per
   perturbation type (see `uncertainty/perturbations.py`).
3. Each of the `K` copies resumes denoising from the branch point to completion, reusing the
   **base's own recorded per-step noise** (not fresh randomness) for every remaining step. This
   isolates the perturbation's effect: any divergence in the final images is attributable to the
   perturbation itself, not to independent sampling stochasticity.
4. Spread metrics (latent MSE, LPIPS, CLIP cosine distance, DINO cosine distance) are computed
   two ways per ensemble: mean pairwise distance *among* the K members, and each member's distance
   *to the base's own unperturbed continuation* ("vs_base"). Both are reported as mean+std, keyed
   by `(class, base, branch_point, type)` — there is no cross-base or cross-class aggregation.

### Only permitted source change

`diffusion/gaussian_diffusion.py`'s `p_sample` gained an optional `noise=None` kwarg: if given, it
is used instead of drawing `torch.randn_like(x)` internally. This is what lets step 3 above replay
an exact trajectory continuation. No other change was made to `models.py`, `diffusion/`, or any
other pipeline source file.

### Internal batching note

The tool never carries the CFG-doubled (class + null-class) batch across steps. Because
`model.forward_with_cfg` always rebuilds its "combined" model input from the first half of
whatever tensor it receives (`half = x[: len(x) // 2]`), the second half's stored value never
feeds back into the kept (class-conditioned) output. Re-deriving the doubled tensor fresh each
step (`torch.cat([x, x])`) instead of carrying the real second half forward is therefore
mathematically equivalent for the rows we keep, and lets branch latents / per-step noise be stored
at batch size N (or K) instead of 2N (or 2K).

## Setup

Beyond `environment.yml`, the metrics module needs:

```bash
pip install pyyaml lpips open_clip_torch matplotlib
```

DINO is loaded via `timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, ...)`,
which requires network access on first use (downloads the checkpoint to the timm/HF cache). LPIPS
(`lpips.LPIPS`) and CLIP (`open_clip.create_model_and_transforms(..., pretrained="openai")`)
likewise download pretrained weights over the network the first time they're instantiated.

A GPU is not strictly required (the tool falls back to CPU like `sample.py` does), but running
DiT-XL/2 plus the VAE, LPIPS, CLIP, and DINO models on CPU is impractically slow in practice.

As with `train.py`/`sample.py`, the VAE (`AutoencoderKL.from_pretrained`) and the auto-downloaded
DiT checkpoint (`download.py`) also require outbound network access to Hugging Face Hub / Meta's
checkpoint host — not new to this tool, but worth restating since this is the first thing you'll
hit if running offline.

## Usage

```bash
cp uncertainty/config.example.yaml my_config.yaml   # edit classes, branch points, perturbations, etc.

# Verify the noise-replay hook is exact before trusting real results:
python run_uncertainty.py --config my_config.yaml --dry-run

# Full run:
python run_uncertainty.py --config my_config.yaml
```

`--dry-run` only exercises the `mock` (identity) perturbation. It re-runs each base's own
continuation from every branch point and checks, step by step, that the resumed latents match the
base's originally recorded latents at that step (tolerance `dry_run_tolerances.latent_mse`), and
that the final decoded image matches the base's own final image (tolerance
`dry_run_tolerances.image_mse`). Tolerances are fp32-only — the repo does not use true mixed
precision (only TF32 matmul acceleration), so no fp16 tolerance is defined. On failure it logs the
first step at which the discrepancy exceeded tolerance, to help localize the bug.

## Outputs (under `output_dir`)

- `run_config.yaml` — a snapshot of the fully-resolved config for this run (including `M` and `K`),
  written at run start so results are self-describing without the original input YAML.
- `latents/class{c}/base{b}/final.pt`, `branch{p}.pt` — base trajectory latents.
- `latents/class{c}/base{b}/branch{p}/{type}/ensemble_final.pt` — the K perturbed members' final
  latents, shape `(K, 4, H, W)`.
- `contact_sheets/class{c}/base{b}/branch{p}/{type}.png` — grid image: the decoded branch-point
  latent (reference tile, first) followed by the K decoded ensemble members.
- `metrics.csv` — columns `class, base, branch_point, type, metric, scope, mean, std`, where
  `metric` is one of `latent_mse, lpips, clip_cosine_distance, dino_cosine_distance` and `scope` is
  `pairwise` or `vs_base`.
- `plots/class{c}_{type}_{metric}_{scope}.png` — one plot per (class, type, metric, scope),
  overlaying all `M` bases as separate lines (x = branch point, y = mean, error bars = std).
- On successful completion, `output_dir` is compressed into a timestamped
  `{output_dir_name}_{YYYYmmdd_HHMMSS}.tar.gz` (default location: alongside `output_dir`, i.e. its
  parent directory). Controlled by `archive` (bool, default `true`) and `archive_dir` (defaults to
  `output_dir`'s parent).

## Config reference

See `uncertainty/config.example.yaml` for a complete annotated example. Key fields:

- `classes`, `M`, `base_seed`: which ImageNet classes to analyze and how many fixed-seed base
  trajectories per class.
- `branch_points`: respaced sampling-step indices (`0..num_sampling_steps-1`) at which to branch.
- `perturbations`: list of `{type, params}`. Supported types: `gaussian_noise` (`std`),
  `gaussian_blur` (`kernel_size`, `sigma`), `nxn_mask` (`size`, `fill_value`), `scattered_mask`
  (`fraction`, `fill_value`), `mock` (identity, dry-run only).
- `K`, `perturbation_seed`: ensemble size and the seed each member's perturbation draw is
  deterministically derived from (via `(perturbation_seed, class, base, branch_point, type, k)`),
  so any single ensemble member can be regenerated independently and reproducibly.
