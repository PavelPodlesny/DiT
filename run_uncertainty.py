"""
CLI entry point for latent-perturbation uncertainty estimation.
See uncertainty/README.md for config format and usage.
"""
import argparse
import logging
import sys

import torch

from uncertainty.adapter import ModelAdapter
from uncertainty.config import Config
from uncertainty.dryrun import run_dry_run
from uncertainty.pipeline import run_pipeline


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    config = Config.load(args.config)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    adapter = ModelAdapter(
        model_name=config.model,
        image_size=config.image_size,
        vae_name=config.vae,
        ckpt=config.ckpt,
        num_classes=config.num_classes,
        cfg_scale=config.cfg_scale,
        num_sampling_steps=config.num_sampling_steps,
        device=device,
    )

    if args.dry_run:
        ok = run_dry_run(adapter, config)
        raise SystemExit(0 if ok else 1)

    run_pipeline(adapter, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    main(parser.parse_args())
