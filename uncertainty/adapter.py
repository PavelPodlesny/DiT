"""Thin adapter around a pretrained DiT model + VAE + diffusion process.

Wires up class-conditioning / classifier-free guidance exactly the way
sample.py / the reference notebook snippet does: model.forward_with_cfg is
called on a batch doubled along dim 0 (real class labels + null-class
labels), and only the model.forward_with_cfg method is invoked -- no
DiT/pipeline source code is touched here beyond the p_sample noise hook
patched in diffusion/gaussian_diffusion.py.
"""
import torch

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models


class ModelAdapter:
    def __init__(
        self,
        model_name: str,
        image_size: int,
        vae_name: str,
        ckpt: str,
        num_classes: int,
        cfg_scale: float,
        num_sampling_steps: int,
        device: str,
    ):
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_classes = num_classes
        self.latent_size = image_size // 8

        self.model = DiT_models[model_name](
            input_size=self.latent_size, num_classes=num_classes
        ).to(device)
        ckpt_path = ckpt or f"DiT-XL-2-{image_size}x{image_size}.pt"
        state_dict = find_model(ckpt_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
        self.diffusion = create_diffusion(str(num_sampling_steps))

    def latent_shape(self, batch: int):
        return (batch, 4, self.latent_size, self.latent_size)

    def model_kwargs(self, class_label: int, batch: int) -> dict:
        """Builds the doubled (class, null-class) label batch forward_with_cfg expects."""
        y = torch.full((batch,), class_label, dtype=torch.long, device=self.device)
        y_null = torch.full((batch,), self.num_classes, dtype=torch.long, device=self.device)
        y_combined = torch.cat([y, y_null], dim=0)
        return dict(y=y_combined, cfg_scale=self.cfg_scale)

    def forward_with_cfg(self):
        return self.model.forward_with_cfg

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents / 0.18215).sample
