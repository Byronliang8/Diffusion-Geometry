#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LC_heatMap.py  ―  SD3.5 版（对齐 LC_heatMap_flux.py 结构）

依赖：
  sd35/based_stepwise_sd.py  →  SD35GeometricAnalyzer
    提供：encode_prompt_cfg, F_sampling_latents_cfg_embeds,
          compute_Jsub_and_V1_F, get_laplacian_kernel

用法：
  单卡：
    python LC_heatMap.py

  多卡（torchrun）：
    torchrun --nproc_per_node=N LC_heatMap.py
"""

import os
import csv
import math
import json
import random
import socket
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import torch.distributed as dist

# SD35 base class (provides model loading, encode_prompt_cfg, F_sampling_latents_cfg_embeds, etc.)
from sd35.based_stepwise_sd import SD35GeometricAnalyzer, get_laplacian_kernel


# -------------------------
# IPv4 only (optional)
# -------------------------
_old_getaddrinfo = socket.getaddrinfo
def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _getaddrinfo_ipv4_only


# -------------------------
# DDP utils
# -------------------------
def get_rank_world_localrank():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank       = int(os.environ.get("RANK",       "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


rank, world, local_rank = get_rank_world_localrank()
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

all_seeds = list(range(10))
seeds = all_seeds[rank::world]


# -------------------------
# misc utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_norm01(x: np.ndarray, q_lo=0.01, q_hi=0.99, eps=1e-8):
    lo = np.quantile(x, q_lo)
    hi = np.quantile(x, q_hi)
    x  = np.clip(x, lo, hi)
    x  = x - x.min()
    return x / (x.max() + eps)


def plain_norm01(x: np.ndarray, eps=1e-8):
    x = x - x.min()
    return x / (x.max() + eps)


def heatmap_img(x01: np.ndarray, upscale_size: int = 512, cmap_name: str = "jet"):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(x01)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((upscale_size, upscale_size), resample=Image.BILINEAR)


def add_title(img: Image.Image, title: str):
    draw = ImageDraw.Draw(img)
    pad  = 6
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), title, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = (len(title) * 6, 11)

    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), title, fill=(255, 255, 255), font=font)
    return img


def make_grid_2x2(img00, img01, img10, img11):
    w, h = img00.size
    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(img00, (0,   0))
    canvas.paste(img01, (w,   0))
    canvas.paste(img10, (0,   h))
    canvas.paste(img11, (w,   h))
    return canvas


# -------------------------
# SD35 heatmap mapper
# -------------------------
class SD35SpatialMapper(SD35GeometricAnalyzer):
    """
    生成 LS_map / LC_map / PHFE_map（与 LC_heatMap_flux.py 的 FluxSpatialMapper 对齐）。

    继承 SD35GeometricAnalyzer，复用：
      - encode_prompt_cfg
      - F_sampling_latents_cfg_embeds
      - compute_Jsub_and_V1_F
    """

    # ------------------------------------------------------------------
    # 核心计算：compute_maps_fullF
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_maps_fullF(
        self,
        z: torch.Tensor,           # (1, 16, 64, 64)
        pos_c: torch.Tensor,
        neg_c: torch.Tensor,
        pos_p: torch.Tensor,
        neg_p: torch.Tensor,
        D_low: int   = 64,
        epsilon: float = 0.02,
        num_inference_steps_F: int  = 12,
        guidance_scale_F: float     = 7.0,
        fd_batch_size: int          = 4,
    ) -> dict:
        """
        返回 dict（与 FluxSpatialMapper.compute_maps_fullF 对齐）：
          LS, LC, PHFE, Axis_Cos,
          LS_map  (H_lat, W_lat)  np.ndarray
          LC_map  (H_lat, W_lat)  np.ndarray
          PHFE_map (H_lat, W_lat) np.ndarray
        """
        dev  = self.device
        # ---- 随机低维子空间 ----
        z_dim = z.numel()                                              # 16*64*64
        Q, _  = torch.linalg.qr(torch.randn(z_dim, D_low, device=dev, dtype=torch.float32))
        W_sub = Q[:, :D_low]                                          # (D, D_low)

        # ---- center 输出 & J_sub ----
        out_center, J_unpacked_stack, V1_low, LS = self._compute_center_and_Jsub(
            z, pos_c, neg_c, pos_p, neg_p, W_sub, epsilon, num_inference_steps_F, guidance_scale_F, fd_batch_size
        )

        # ---- LS_map ----
        # J_unpacked_stack: (D_low, C, H, W)  →  sum over channels & directions
        J4    = J_unpacked_stack.permute(1, 2, 3, 0).contiguous()    # (C, H, W, D_low)
        LS_map = torch.sqrt(torch.sum(J4 ** 2, dim=(0, 3)))           # (H, W)

        # ---- P1 along V1 ----
        V1_full = (W_sub @ V1_low).view(z.shape).to(self.dtype)
        V1_full = V1_full / (torch.norm(V1_full.float()) + 1e-8)

        out_plus = self.F_sampling_latents_cfg_embeds(
            z + epsilon * V1_full, pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
        )
        P1 = (out_plus.float() - out_center.float()) / epsilon        # (1, C, H, W)

        # ---- PHFE_map ----
        C          = P1.shape[1]
        lap_kernel = get_laplacian_kernel(dev, C)
        P1_lap     = F.conv2d(P1, lap_kernel, padding=1, groups=C)
        PHFE_scalar = torch.var(P1_lap).item()
        PHFE_map   = torch.sqrt(torch.sum(P1_lap[0] ** 2, dim=0))    # (H, W)

        # ---- neighbor for LC ----
        n_low  = torch.randn(D_low, device=dev, dtype=torch.float32)
        n_low  = (n_low / (torch.norm(n_low) + 1e-8)) * epsilon
        z_prime = (z.float() + (W_sub @ n_low).view(z.shape)).to(self.dtype)

        out_center_p, J_unpacked_stack_p, V1_low_p, _ = self._compute_center_and_Jsub(
            z_prime, pos_c, neg_c, pos_p, neg_p, W_sub, epsilon, num_inference_steps_F, guidance_scale_F, fd_batch_size
        )

        cos_sim = torch.dot(V1_low, V1_low_p).item()
        if cos_sim < 0:
            V1_low_p = -V1_low_p
            cos_sim  = -cos_sim
        Axis_Cos  = abs(cos_sim)
        LC_scalar = torch.norm(V1_low - V1_low_p).item() / epsilon

        # P1 neighbor → LC_map
        V1_full_p = (W_sub @ V1_low_p).view(z.shape).to(self.dtype)
        V1_full_p = V1_full_p / (torch.norm(V1_full_p.float()) + 1e-8)

        out_plus_p = self.F_sampling_latents_cfg_embeds(
            z_prime + epsilon * V1_full_p, pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
        )
        P1_p   = (out_plus_p.float() - out_center_p.float()) / epsilon  # (1, C, H, W)
        LC_map = torch.norm((P1_p[0] - P1[0]), dim=0) / epsilon          # (H, W)

        return {
            "LS":       float(LS),
            "LC":       float(LC_scalar),
            "PHFE":     float(PHFE_scalar),
            "Axis_Cos": float(Axis_Cos),
            "LS_map":   LS_map.detach().cpu().numpy(),
            "LC_map":   LC_map.detach().cpu().numpy(),
            "PHFE_map": PHFE_map.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # 内部辅助：运行 F_sampling + 有限差分建 J_sub（与 based_stepwise_sd 对齐）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_center_and_Jsub(
        self,
        z_center: torch.Tensor,
        pos_c, neg_c, pos_p, neg_p,
        W_sub: torch.Tensor,
        epsilon: float,
        num_inference_steps: int,
        guidance_scale: float,
        fd_batch_size: int,
    ):
        """
        返回:
          out_center      (1,C,H,W)  float32
          J_unpacked_stack (D_low,C,H,W) float32   — 各扰动方向的输出差
          V1_low          (D_low,)   float32
          LS              float
        """
        D_low  = W_sub.shape[1]
        dev    = self.device

        # center 输出
        out_center = self.F_sampling_latents_cfg_embeds(
            z_center, pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).float()                                                      # (1,C,H,W)

        W_full = W_sub.T.view(D_low, *z_center.shape[1:]).to(dev)    # (D_low, C, H, W)

        J_cols        = []
        J_unpacked_list = []

        for i in range(0, D_low, fd_batch_size):
            end     = min(i + fd_batch_size, D_low)
            batch_w = W_full[i:end]                                   # (bs, C, H, W)

            z_batch = (z_center.float() + epsilon * batch_w.float()).to(self.dtype)
            out_batch = self.F_sampling_latents_cfg_embeds(
                z_batch, pos_c, neg_c, pos_p, neg_p,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).float()                                                  # (bs, C, H, W)

            diff = (out_batch - out_center) / epsilon                  # (bs, C, H, W)
            J_cols.append(diff.flatten(start_dim=1))                  # (bs, D_out)
            J_unpacked_list.append(diff)

        # (D_out, D_low)
        J_matrix = torch.cat(J_cols, dim=0).T
        A         = J_matrix.T @ J_matrix                             # (D_low, D_low)
        L, V      = torch.linalg.eigh(A)
        V1_low    = V[:, -1]                                          # (D_low,)
        LS        = math.sqrt(max(L[-1].item(), 0.0))

        J_unpacked_stack = torch.cat(J_unpacked_list, dim=0)          # (D_low, C, H, W)

        return out_center, J_unpacked_stack, V1_low, LS

    # ------------------------------------------------------------------
    # 主入口：analyze_and_map（对齐 FluxSpatialMapper.analyze_and_map）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def analyze_and_map(
        self,
        seed: int,
        prompt: str,
        label: str,
        outdir: str,
        negative_prompt: str      = "",
        D_low: int                = 64,
        epsilon: float            = 0.02,
        num_inference_steps_F: int = 12,
        guidance_scale_F: float   = 7.0,
        fd_batch_size: int        = 4,
        upscale_size: int         = 512,
        cmap_name: str            = "jet",
        robust_vis: bool          = True,
        save_npz: bool            = True,
        save_image: bool          = False,
        viz_steps: int            = 28,
        viz_guidance_scale: float = 7.0,
    ):
        """
        保存：
          {label}_S{seed}_LS{...}_LC{...}_PHFE{...}-seed={seed}.png
          {label}_S{seed}_... -seed={seed}_ls.png
          {label}_S{seed}_... -seed={seed}_lc.png
          {label}_S{seed}_... -seed={seed}_phfe.png
          [可选] .npz
        """
        os.makedirs(outdir, exist_ok=True)
        set_seed(seed)

        # prompt embeddings
        pos_c, neg_c, pos_p, neg_p = self.encode_prompt_cfg(prompt, negative_prompt)

        # initial noise latent: SD35 uses (1, 16, 64, 64)
        z = torch.randn(1, 16, 64, 64, device=self.device, dtype=self.dtype)

        # compute maps
        metrics = self.compute_maps_fullF(
            z=z,
            pos_c=pos_c, neg_c=neg_c, pos_p=pos_p, neg_p=neg_p,
            D_low=D_low,
            epsilon=epsilon,
            num_inference_steps_F=num_inference_steps_F,
            guidance_scale_F=guidance_scale_F,
            fd_batch_size=fd_batch_size,
        )

        # generate visualization image via full pipeline
        with torch.no_grad():
            img = self.pipe(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                latents=z,
                num_inference_steps=viz_steps,
                guidance_scale=viz_guidance_scale,
                output_type="pil",
            ).images[0]
        img = img.resize((upscale_size, upscale_size))

        # normalize maps for visualisation
        norm_fn = robust_norm01 if robust_vis else plain_norm01
        ls01 = norm_fn(metrics["LS_map"])
        lc01 = norm_fn(metrics["LC_map"])
        ph01 = norm_fn(metrics["PHFE_map"])

        ls_img = heatmap_img(ls01, upscale_size, cmap_name)
        lc_img = heatmap_img(lc01, upscale_size, cmap_name)
        ph_img = heatmap_img(ph01, upscale_size, cmap_name)

        # save
        base     = f"{label}_S{seed}_LS{metrics['LS']:.3f}_LC{metrics['LC']:.3f}_PHFE{metrics['PHFE']:.3f}"
        if save_image:
            png_path = os.path.join(outdir, base)

            img.save(   png_path + f"-seed={seed}.png")
            ls_img.save(png_path + f"-seed={seed}_ls.png")
            lc_img.save(png_path + f"-seed={seed}_lc.png")
            ph_img.save(png_path + f"-seed={seed}_phfe.png")

        if save_npz:
            npz_path = os.path.join(outdir, base + ".npz")
            np.savez_compressed(
                npz_path,
                LS_map=metrics["LS_map"],
                LC_map=metrics["LC_map"],
                PHFE_map=metrics["PHFE_map"],
                LS=metrics["LS"], LC=metrics["LC"], PHFE=metrics["PHFE"], Axis_Cos=metrics["Axis_Cos"],
            )

        print(
            f"[Saved] {png_path}\n"
            f"  LS={metrics['LS']:.4f} | LC={metrics['LC']:.4f} | "
            f"PHFE={metrics['PHFE']:.4f} | Axis_Cos={metrics['Axis_Cos']:.4f}"
        )


# -------------------------
# prompt-pairs loader（与 LC_heatMap_flux.py 相同）
# -------------------------
def load_prompt_pairs_tsv(path: str):
    """
    TSV columns: pair_id, label, prompt
    label in {id, ood} (case-insensitive)
    Returns: dict[int, dict[str, str]]  e.g. {1: {"ID": "...", "OOD": "..."}, ...}
    """
    pairs = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"pair_id", "label", "prompt"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"TSV must contain columns {required}, got {reader.fieldnames}")

        for row in reader:
            pid   = int(row["pair_id"])
            lab   = row["label"].strip().lower()
            prompt = row["prompt"].strip()

            if lab == "id":
                pairs[pid]["ID"] = prompt
            elif lab == "ood":
                pairs[pid]["OOD"] = prompt
            else:
                raise ValueError(f"Unknown label '{row['label']}' in row: {row}")

    for pid, d in pairs.items():
        if "ID" not in d or "OOD" not in d:
            raise ValueError(f"pair_id={pid} missing ID or OOD prompt: {d}")

    return dict(pairs)


# -------------------------
# Entry points
# -------------------------
def main_single(normal_prompt: str, ood_prompt: str):
    mapper = SD35SpatialMapper(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        device=device,
    )

    prompts      = {"ID": normal_prompt, "OOD": ood_prompt}
    base_outdir  = "./heat_map_sd35_result_single"
    os.makedirs(base_outdir, exist_ok=True)

    for label, prompt in prompts.items():
        outdir = os.path.join(base_outdir, label, f"rank_{rank:02d}")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "prompt.txt"), "w", encoding="utf-8") as pf:
            pf.write(prompt + "\n")

        for s in seeds:
            torch.cuda.empty_cache()
            try:
                mapper.analyze_and_map(
                    seed=s,
                    prompt=prompt,
                    label=label,
                    outdir=outdir,
                    negative_prompt="",
                    D_low=64,
                    epsilon=0.02,
                    num_inference_steps_F=12,
                    guidance_scale_F=7.0,
                    fd_batch_size=1,
                    upscale_size=512,
                    cmap_name="jet",
                    robust_vis=True,
                    save_npz=True,
                    viz_steps=28,
                    viz_guidance_scale=7.0,
                )
            except Exception as e:
                print(f"[Error] {label} seed={s}: {e}")


def main_pairs(prompt_file: str, base_outdir: str, seeds_small=None):
    """
    对齐 LC_heatMap_flux.py 的目录结构：
      base_outdir/pair_XX/ID/rank_XX/*.png|npz
      base_outdir/pair_XX/OOD/rank_XX/*.png|npz
    """
    mapper = SD35SpatialMapper(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        device=device,
    )

    prompt_pairs = load_prompt_pairs_tsv(prompt_file)
    last_pids    = sorted(prompt_pairs.keys())

    if seeds_small is None:
        seeds_small = list(range(10))

    seeds_use = seeds_small[rank::world] if world > 1 else seeds_small

    for pid in last_pids:
        for seed in seeds_use:
            for label in ("ID", "OOD"):
                prompt = prompt_pairs[pid][label]
                outdir = os.path.join(base_outdir, f"pair_{pid:02d}", label, f"rank_{rank:02d}")
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, "prompt.txt"), "w", encoding="utf-8") as pf:
                    pf.write(prompt + "\n")

                torch.cuda.empty_cache()
                try:
                    mapper.analyze_and_map(
                        seed=seed,
                        prompt=prompt,
                        label=label,
                        outdir=outdir,
                        negative_prompt="",
                        D_low=64,
                        epsilon=0.02,
                        num_inference_steps_F=12,
                        guidance_scale_F=7.0,
                        fd_batch_size=1,
                        upscale_size=512,
                        cmap_name="jet",
                        robust_vis=True,
                        save_npz=True,
                        viz_steps=28,
                        viz_guidance_scale=7.0,
                    )
                except Exception as e:
                    print(f"[Error] pid={pid} {label} seed={seed}: {e}")


if __name__ == "__main__":
    # --- 方式 A：单对 ID/OOD prompt 测试 ---
    # normal_prompt = "A freshwater fish swimming underwater in a river, silver scales, flowing fins, natural aquatic environment with plants, sunlight filtering through water, photorealistic"
    # ood_prompt    = "A fish with four muscular legs walking on sand in a desert, resembling a salamander but with fish scales and fins, speculative biology concept, photorealistic, midday sun, clear shadows."
    # main_single(normal_prompt, ood_prompt)

    # --- 方式 B：完整跑 pair ---
    prompt_file = "./ood_id_prompt_pairs_3.txt"
    base_outdir = "./main_output_sd35"
    main_pairs(prompt_file, base_outdir, seeds_small=list(range(100)))

    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


"""
使用指南：
  单卡：
    python LC_heatMap.py

  多卡：
    torchrun --nproc_per_node=N LC_heatMap.py
"""