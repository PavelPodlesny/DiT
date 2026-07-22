# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official PyTorch implementation of DiT (Diffusion Transformers) — latent diffusion models that use a transformer
backbone instead of a U-Net. Operates on VAE latents (Stable Diffusion's `sd-vae-ft-{ema,mse}`), not raw pixels.

## Environment setup

```bash
conda env create -f environment.yml
conda activate DiT
```

No test suite, linter, or build step exists in this repo — it's a research codebase of standalone scripts.

## Common commands

Train (requires `torchrun` + at least one GPU; DDP is mandatory, not optional):
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-XL/2 --data-path /path/to/imagenet/train
```

Sample a small grid from a pretrained or custom checkpoint (writes `sample.png`):
```bash
python sample.py --image-size 512 --seed 1
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

Sample many images in parallel for FID/Inception Score evaluation (produces a folder of PNGs + an `.npz` for
ADM's TensorFlow eval suite):
```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

## Architecture

- `models.py` — the entire DiT model. Key pieces:
  - `DiT` — main class. Patchifies latents with `PatchEmbed` (from `timm`), adds fixed sin-cos position
    embeddings (`get_2d_sincos_pos_embed`), runs a stack of `DiTBlock`s, then a `FinalLayer`, then
    `unpatchify`s back to a latent-shaped tensor.
  - `DiTBlock` — transformer block using **adaLN-Zero** conditioning: timestep + class embeddings are summed
    into a single conditioning vector `c`, which an MLP (`adaLN_modulation`) turns into 6 chunks
    (shift/scale/gate for attention and MLP). All adaLN modulation layers and the final output layer are
    zero-initialized (`initialize_weights`) — this zero-init is load-bearing for training stability, don't
    "clean up" the init calls that look like no-ops.
  - `LabelEmbedder` — implements classifier-free guidance by embedding an extra "null" class and randomly
    dropping labels to it during training (`class_dropout_prob`).
  - `forward_with_cfg` — batches conditional + unconditional passes together for CFG at sampling time; only
    applies guidance to the first 3 channels by default (see comment in that method before changing it).
  - `DiT_models` dict at the bottom maps config strings like `"DiT-XL/2"` to constructor functions
    (depth/hidden_size/num_heads/patch_size presets: XL/L/B/S × patch sizes 2/4/8). This dict is what
    `--model` CLI args index into across all scripts.
- `diffusion/` — a vendored, largely unmodified copy of OpenAI's ADM/IDDPM Gaussian diffusion machinery
  (`gaussian_diffusion.py`, `respace.py`, `timestep_sampler.py`). `diffusion/__init__.py` exposes
  `create_diffusion(...)`, the single entry point used by every script to build a `SpacedDiffusion` object
  (handles noise schedule, loss type, timestep respacing, learned-vs-fixed variance). Treat this directory as
  a stable dependency, not a place to add DiT-specific logic.
- `train.py` — single-file DDP training loop: VAE encodes images to latents (scaled by `0.18215`) under
  `torch.no_grad()`, then `diffusion.training_losses(model, x, t, model_kwargs)` computes the loss. Maintains
  an EMA copy of the model (`update_ema`) that is what checkpoints are typically sampled from. No checkpoint
  resume support currently exists (see README's "Enhancements" TODO list — resume-from-checkpoint, AMP, and
  FID monitoring are known gaps, not accidental omissions).
- `sample.py` — loads one checkpoint, samples a fixed small grid of class labels via
  `diffusion.p_sample_loop` + CFG, decodes through the VAE, saves `sample.png`.
- `sample_ddp.py` — parallel bulk sampling across ranks for quantitative evaluation (FID/IS), not for eyeballing
  outputs.
- `download.py` — `find_model()` resolves a checkpoint path/name to a state dict, auto-downloading the two
  released DiT-XL/2 checkpoints (256×256, 512×512) if no `--ckpt` is given.

## Conventions to preserve

- `torch.backends.cuda.matmul.allow_tf32 = True` / `cudnn.allow_tf32 = True` are set at the top of `train.py`
  and `sample.py` intentionally (large speedup on Ampere/A100). Keep this if editing those files; it does mean
  results can differ slightly from strict FP32.
- Model config strings follow `"DiT-{XL,L,B,S}/{2,4,8}"` (size/patch-size) and are used verbatim as `--model`
  values and as keys into `DiT_models`. When adding a new size or patch config, follow this naming pattern and
  register it in `DiT_models`.
- VAE latents are always scaled by `0.18215` when encoding and divided by it before decoding — this constant
  is Stable Diffusion's latent scaling factor and appears in `train.py`, `sample.py`, and `sample_ddp.py`.
