"""Spread metrics: pairwise-among-ensemble and vs-base(unperturbed) distance,
each as mean+std over the relevant set of item pairs.

Backends (LPIPS / CLIP / DINO) are lazily imported so that config-only usage
(e.g. --dry-run, which only needs latent MSE) doesn't require them installed.
See uncertainty/README.md for the extra pip packages needed.
"""
from itertools import combinations

import torch
import torch.nn.functional as F


def latent_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a, b).item()


class MetricSuite:
    def __init__(self, device, lpips_net="alex", clip_model="ViT-B-32",
                 clip_pretrained="openai", dino_model="vit_small_patch14_dinov2.lvd142m"):
        self.device = device
        self.lpips_net = lpips_net
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.dino_model_name = dino_model

        self._lpips = None
        self._clip = None
        self._clip_preprocess = None
        self._dino = None

    def _get_lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()
        return self._lpips

    def _get_clip(self):
        if self._clip is None:
            import open_clip
            # OpenAI's original CLIP checkpoints use QuickGELU; open_clip's
            # default configs use standard GELU, so loading "openai" weights
            # without this flag silently mismatches the activation function.
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.clip_pretrained,
                force_quick_gelu=(self.clip_pretrained == "openai"),
            )
            self._clip = model.to(self.device).eval()
            self._clip_preprocess = preprocess
        return self._clip, self._clip_preprocess

    def _get_dino(self):
        if self._dino is None:
            import timm
            self._dino = timm.create_model(self.dino_model_name, pretrained=True, num_classes=0)
            self._dino = self._dino.to(self.device).eval()
        return self._dino

    @torch.no_grad()
    def lpips_distance(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """img_a, img_b: (1, 3, H, W) tensors in [-1, 1]."""
        return self._get_lpips()(img_a, img_b).item()

    @torch.no_grad()
    def _embedding_cosine_distance(self, embed_fn, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        emb_a = embed_fn(img_a)
        emb_b = embed_fn(img_b)
        cos_sim = F.cosine_similarity(emb_a, emb_b).item()
        return 1.0 - cos_sim

    def _to_unit_range(self, img: torch.Tensor) -> torch.Tensor:
        return (img.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def clip_distance(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        model, preprocess = self._get_clip()

        def embed(img):
            img = F.interpolate(self._to_unit_range(img), size=224, mode="bicubic", align_corners=False)
            return model.encode_image(img)

        return self._embedding_cosine_distance(embed, img_a, img_b)

    @torch.no_grad()
    def dino_distance(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        model = self._get_dino()

        def embed(img):
            img = F.interpolate(self._to_unit_range(img), size=224, mode="bicubic", align_corners=False)
            return model(img)

        return self._embedding_cosine_distance(embed, img_a, img_b)


def _mean_std(values: list):
    if not values:
        return float("nan"), float("nan")
    t = torch.tensor(values)
    if len(values) == 1:
        return t.mean().item(), 0.0
    return t.mean().item(), t.std().item()


def pairwise_stats(items, metric_fn):
    """items: list of single-sample tensors. Returns (mean, std) over all
    (K choose 2) unordered pairs."""
    values = [metric_fn(a, b) for a, b in combinations(items, 2)]
    return _mean_std(values)


def vs_reference_stats(items, reference, metric_fn):
    """items: list of single-sample tensors, reference: one single-sample
    tensor. Returns (mean, std) over each item's distance to `reference`."""
    values = [metric_fn(item, reference) for item in items]
    return _mean_std(values)


def compute_ensemble_metrics(suite: MetricSuite, member_images: torch.Tensor, member_latents: torch.Tensor,
                              base_image: torch.Tensor, base_latent: torch.Tensor) -> dict:
    """member_images/member_latents: (K, C, H, W). base_image/base_latent: (1, C, H, W).
    Returns a dict of {(metric, scope): (mean, std)}."""
    k = member_images.shape[0]
    image_items = [member_images[i:i + 1] for i in range(k)]
    latent_items = [member_latents[i:i + 1] for i in range(k)]

    metric_fns = {
        "latent_mse": lambda a, b: latent_mse(a, b),
        "lpips": suite.lpips_distance,
        "clip_cosine_distance": suite.clip_distance,
        "dino_cosine_distance": suite.dino_distance,
    }

    results = {}
    for name, fn in metric_fns.items():
        latent_or_image_items = latent_items if name == "latent_mse" else image_items
        latent_or_image_reference = base_latent if name == "latent_mse" else base_image

        results[(name, "pairwise")] = pairwise_stats(latent_or_image_items, fn)
        results[(name, "vs_base")] = vs_reference_stats(latent_or_image_items, latent_or_image_reference, fn)
    return results
