# Plan: Latent-Perturbation Uncertainty Estimation for DiT Diffusion Models

## Goal
Quantify a DiT diffusion model's predictive uncertainty by measuring how
sensitive final generated images are to perturbations injected into the
latent at different points along the denoising trajectory, across a set of
ImageNet classes.

## Desired Output
- CLI tool that: for each given ImageNet class, generates M base
  trajectories, saves latents at chosen branching points, applies K-sized
  ensembles of perturbations (per type) at each branch point per base,
  resumes denoising with the *base's own saved per-step noise* to isolate
  the perturbation's effect, and computes spread metrics per ensemble.
- Outputs: saved latents (branch points + finals), contact-sheet grids keyed
  by (class, base, branch point, type) — including the decoded branch-point
  latent as a reference tile alongside the K ensemble members — CSV of
  metrics (mean pairwise CLIP/DINO cosine distance, pairwise LPIPS, latent
  MSE — mean+std) keyed by (class, base, branch point, type), and summary
  plots overlaying all M base trajectories per class (uncertainty vs.
  branch point, per type, per class).

## Implementation Steps
1. **Pipeline patch**: modify `p_sample` to accept per-step noise as an
   explicit argument instead of sampling internally. This is the *only*
   permitted modification to the source DiT/pipeline code — no other
   source changes, workarounds, or refactors of the underlying model code.
   You will find `p_sample` in `diffusion/gaussian_diffusion.py`
2. **Scaffolding**: `config.yaml` (classes, model path, steps, branch
   points, perturbation types+params, K, M, seeds, dry-run tolerances);
   `ModelAdapter` interface, using the class-conditioning mechanism from the
   provided reference code snippet below. Branch points are specified as
   **respaced step indices** (0..num_sampling_steps-1), matching the order
   in which per-step noise is recorded during base trajectory capture — not
   raw diffusion timesteps `t`.
3. **Trajectory capture**: for each class, run M fixed-seed base
   generations; save latents at branch points + per-step noise tensors in
   memory/scratch per base; discard noise after all that base's branches
   finish.
4. **Perturbation module**: Gaussian noise, Gaussian blur, NxN mask,
   scattered 1×1 mask, `mock` (identity). Separate perturbation seed, derived
   per-member: each ensemble member's draw is seeded from
   `perturbation_seed` plus a deterministic offset keyed by
   (class, base, branch point, type, k), so any single ensemble member can be
   regenerated independently and reproducibly.
5. **Ensemble generation**: per (class, base, branch point, type), generate
   K variants, resuming with the base's saved per-step noise and the same
   class conditioning; save images + latents.
6. **Dry-run verification** (`--dry-run`, off by default): runs `mock`
   only; logs per-step latent MSE and final image MSE; asserts against
   threshold (fp32 only: latent 1e-6 / image 1e-4 — fp16 is out of scope,
   the repo only uses TF32 matmul acceleration, not true mixed precision);
   logs exact step/value where it first exceeds tolerance for debugging.
7. **Metrics module**: for each ensemble, computes both (a) mean pairwise
   LPIPS, latent MSE, and mean pairwise CLIP/DINO cosine distance among the
   K perturbed members, and (b) each member's LPIPS, latent MSE, and
   CLIP/DINO cosine distance against the base's own unperturbed
   continuation (divergence-from-truth, not just relative spread);
   mean+std of both keyed by (class, base, branch point, type) — no
   cross-M or cross-class aggregation.
8. **Reporting**: contact sheets keyed by (class, base, branch point, type),
   with ensemble members included; overlay plots (all M shown together per
   class, uncertainty vs. branch point, per type, covering both the
   pairwise-spread and vs-base metrics); metrics CSV with both metric
   families.
9. **README**: adapter wiring (incl. class-conditioning snippet usage),
   patch notes, dry-run usage, metric defs.

## Constraints
- No modifications to source DiT/pipeline code beyond the `p_sample`
  noise-argument hook in step 1.
## Code snippets
### Setup
```python
import DiT, os
os.chdir('DiT')
os.environ['PYTHONPATH'] = '/env/python:/content/DiT'
!pip install diffusers timm --upgrade
# DiT imports:
import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
```
### Download model
```python
image_size = 256
vae_model = "stabilityai/sd-vae-ft-ema"
latent_size = int(image_size) // 8
# Load model:
model = DiT_XL_2(input_size=latent_size).to(device)
state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
model.load_state_dict(state_dict)
model.eval() # important!
vae = AutoencoderKL.from_pretrained(vae_model).to(device)
```
### Sample from pre-trained DiT model
```python
# Set user inputs:
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}

# Create diffusion object:
diffusion = create_diffusion(str(num_sampling_steps))

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, 
    model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

```