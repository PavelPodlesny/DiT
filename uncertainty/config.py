"""Config loading for the latent-perturbation uncertainty estimation tool."""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class PerturbationSpec:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class DryRunTolerances:
    latent_mse: float = 1e-6
    image_mse: float = 1e-4


@dataclass
class MetricsConfig:
    lpips_net: str = "alex"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    dino_model: str = "dino_vits16"


@dataclass
class Config:
    # Model / sampling
    model: str = "DiT-XL/2"
    image_size: int = 256
    vae: str = "mse"
    ckpt: Optional[str] = None
    num_classes: int = 1000
    cfg_scale: float = 4.0
    num_sampling_steps: int = 250
    device: Optional[str] = None

    # Trajectory / branching
    classes: list = field(default_factory=lambda: [207])
    M: int = 2
    base_seed: int = 0
    branch_points: list = field(default_factory=lambda: [50, 125, 200])

    # Perturbations
    perturbations: list = field(default_factory=list)
    K: int = 4
    perturbation_seed: int = 1000

    # Output
    output_dir: str = "uncertainty_results"

    # Dry-run
    dry_run_tolerances: DryRunTolerances = field(default_factory=DryRunTolerances)

    # Metrics backends
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        raw = dict(raw)
        perturbations = [PerturbationSpec(**p) for p in raw.pop("perturbations", [])]
        dry_run_tolerances = DryRunTolerances(**raw.pop("dry_run_tolerances", {}))
        metrics = MetricsConfig(**raw.pop("metrics", {}))

        cfg = Config(
            perturbations=perturbations,
            dry_run_tolerances=dry_run_tolerances,
            metrics=metrics,
            **raw,
        )
        return cfg
