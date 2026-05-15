# Geometric Decoupling: Diagnosing the Structural Instability of Latent (ICML 2026)

Latent Diffusion Models (LDMs) achieve high-fidelity synthesis but suffer from latent space brittleness, causing discontinuous semantic jumps during editing. We introduce a Riemannian framework to diagnose this instability by analyzing the generative Jacobian, decomposing geometry into Local Scaling (capacity) and Local Complexity (curvature). Our study uncovers a Geometric Decoupling: while curvature in normal generation functionally encodes image detail, OOD generation exhibits a functional decoupling where extreme curvature is wasted on unstable semantic boundaries rather than perceptible details. This geometric misallocation identifies Geometric Hotspots as the structural root of instability, providing a robust intrinsic metric for diagnosing generative reliability.

## Overview

The core of this research is measuring how tiny perturbations in the latent noise vector $z$ influence the diffusion model's generation trajectory. It extracts these properties using randomized low-dimensional subspace projections and finite difference approximations.

### Key Metrics Computed

1. **LS (Local Scaling):** Measures the maximum sensitivity (stretching) of the generation output with respect to local perturbations in the latent space. It is equivalent to the spectral norm (largest singular value) of the local Jacobian matrix.
2. **LC (Local Complexity):** Measures the curvature or non-linearity of the latent space by assessing how the principal singular vector ($V_1$) shifts within a small neighborhood.
3. **PHFE (Persistent Homology Feature Entropy):** Measures the topological complexity and spatial variance of the principal directional derivative using a Laplacian filter.
4. **Axis_Cos:** Calculates the cosine similarity of the principal stretching directions between adjacent points, indicating the stability of the dominant transformation axis.

## Model Support

The metric evaluation and heatmap generation are currently adapted for multiple diffusion architectures:
- **Stable Diffusion 3.5 (SD3.5):** Adapted heatmap generation located in the `sd35/` directory.

## File Structure

- `Geometry_Diffusion.py`: Main script for producing a 2x2 diagnostic heatmap grid (Image, LS_map, LC_map, PHFE_map) for base SD/SD3.5 architectures.
- `ood_id_prompt_pairs_*.txt`: TSV/Text files containing paired In-Distribution and Out-of-Distribution prompts used for comparative evaluation.

## Usage

### Generating Diagnostic Heatmaps
To generate heatmaps for a specific model, run the corresponding heatmap script. The scripts are compatible with single-GPU and multi-GPU (via `torchrun`) setups.

```bash
# Single-GPU execution
python Geometry_Diffusion.py

# Multi-GPU execution (Distributed Data Parallel)
torchrun --nproc_per_node=N Geometry_Diffusion.py
```

The script will read prompts (from `ood_id_prompt_pairs_x.txt`), iteratively solve for the local geometric properties, and output:
- Compressed Numpy arrays (`.npz`) containing the raw map tensors (LS_map, LC_map, PHFE_map) and metrics (LS, LC, PHFE, Axis_Cos) for downstream statistical analysis.
- (Optional) If enabled, it can also output raw image generations and individual heatmaps for Local Scaling, Local Complexity, and Persistent Homology Feature Entropy.

Additionally, the `sd35/based_stepwise_sd.py` script computes metrics directly and outputs `metrics_rank{rank}.csv` and JSON reports, alongside generated sample images.

## Requirements

Dependencies are specified in `environment.yml` and `requirement.txt`.
Key requirements include:
- `torch`, `torchvision`
- `numpy`, `matplotlib`, `Pillow`
- `transformers`, `diffusers`

```bash
conda env create -f environment.yml
conda activate geo_diffusion
```
