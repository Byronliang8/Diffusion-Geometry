import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
import json
import random
from PIL import Image
import math
from diffusers import StableDiffusion3Pipeline
import torch.distributed as dist


def get_rank_world_localrank():
    #
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


rank, world, local_rank = get_rank_world_localrank()
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"

# seeds
all_seeds = list(range(500))
seeds = all_seeds[rank::world]


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_laplacian_kernel(device, channels: int):
    """laplacian kernel for PHFE"""
    k = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], device=device, dtype=torch.float32)
    k = k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return k


# -------------------------
# Analyzer
# -------------------------
class SD35GeometricAnalyzer:
    def __init__(self, model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda"):
        print(f"[Load] {model_id} ...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(device)

        self.device = device
        self.dtype = torch.bfloat16

        # freeze params (we do NOT train; FD only)
        self.pipe.transformer.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder_2"):
            self.pipe.text_encoder_2.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder_3"):
            self.pipe.text_encoder_3.requires_grad_(False)

        print("[Load] model success.")

    @torch.no_grad()
    def encode_prompt_cfg(self, prompt: str, negative_prompt: str = ""):
        """
        Return SD3 embeddings for CFG.
        diffusers SD3 encode_prompt returns:
          prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        """
        (pos_c, neg_c, pos_p, neg_p) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        # ensure dtype
        return (pos_c.to(self.dtype), neg_c.to(self.dtype), pos_p.to(self.dtype), neg_p.to(self.dtype))

    @torch.no_grad()
    def F_sampling_latents_cfg_embeds(
            self,
            z0: torch.Tensor,  # (B,16,64,64)
            pos_c: torch.Tensor,  # (B or 1, seq, dim)
            neg_c: torch.Tensor,  # (B or 1, seq, dim)
            pos_p: torch.Tensor,  # (B or 1, dim)
            neg_p: torch.Tensor,  # (B or 1, dim)
            num_inference_steps: int = 28,
            guidance_scale: float = 7.0,
    ):
        """
        Full sampling mapping F with CFG:
          z0 -> z_final  (N steps of transformer + scheduler.step)
        Output: final latents (B,16,64,64)
        """
        device = self.device
        dtype = self.dtype
        scheduler = self.pipe.scheduler
        transformer = self.pipe.transformer

        latents = z0.to(device=device, dtype=dtype)

        # match diffusers common convention (if exists)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        B = latents.shape[0]

        # expand embeddings to batch size if they are (1, ...)
        if pos_c.shape[0] == 1 and B > 1:
            pos_c_b = pos_c.expand(B, -1, -1).contiguous()
            neg_c_b = neg_c.expand(B, -1, -1).contiguous()
            pos_p_b = pos_p.expand(B, -1).contiguous()
            neg_p_b = neg_p.expand(B, -1).contiguous()
        else:
            pos_c_b, neg_c_b, pos_p_b, neg_p_b = pos_c, neg_c, pos_p, neg_p

        do_cfg = guidance_scale is not None and guidance_scale > 1.0

        for t in timesteps:
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                enc = torch.cat([neg_c_b, pos_c_b], dim=0)
                pool = torch.cat([neg_p_b, pos_p_b], dim=0)
            else:
                latent_model_input = latents
                enc = pos_c_b
                pool = pos_p_b

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # timestep tensor: keep float32 often safer
            t_val = t
            if isinstance(t_val, torch.Tensor):
                # ensure 0-dim
                t_scalar = t_val
            else:
                t_scalar = torch.tensor(t_val, device=device)

            t_batch = t_scalar.to(device=device, dtype=torch.float32).view(1).repeat(latent_model_input.shape[0])

            model_output = transformer(
                hidden_states=latent_model_input,
                timestep=t_batch,
                encoder_hidden_states=enc,
                pooled_projections=pool,
                return_dict=False,
            )[0]

            if do_cfg:
                uncond, cond = model_output.chunk(2, dim=0)
                model_output = uncond + guidance_scale * (cond - uncond)

            step_out = scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=latents,
                return_dict=True
            )
            latents = step_out.prev_sample

        return latents

    def compute_Jsub_and_V1_F(
            self,
            z_center: torch.Tensor,
            pos_c: torch.Tensor, neg_c: torch.Tensor, pos_p: torch.Tensor, neg_p: torch.Tensor,
            W_subspace: torch.Tensor,  # (D, D_low) orthonormal
            epsilon: float,
            num_inference_steps: int,
            guidance_scale: float,
            fd_batch_size: int = 4,  # NOTE: for full F, keep small to avoid OOM
    ):
        """
        FD on full mapping F in a low-dim subspace:
          J[:,i] ~ (F(z+eps*w_i)-F(z))/eps
        Then get top eigenvector of J^T J => V1_low, and LS = sqrt(lambda_max)
        """
        D_low = W_subspace.size(1)

        # center output: F(z_center)
        out_center = self.F_sampling_latents_cfg_embeds(
            z_center,
            pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # (1,16,64,64)

        # (D_low, C, H, W)
        W_full = W_subspace.T.view(D_low, *z_center.shape[1:]).to(self.device)

        J_cols = []
        for i in range(0, D_low, fd_batch_size):
            end = min(i + fd_batch_size, D_low)
            batch_w = W_full[i:end]  # (bs, C,H,W) float32

            # build perturbed batch (bs,C,H,W)
            z_batch = (z_center.float() + epsilon * batch_w.float()).to(self.dtype)

            out_batch = self.F_sampling_latents_cfg_embeds(
                z_batch,
                pos_c, neg_c, pos_p, neg_p,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )  # (bs,16,64,64)

            diff = (out_batch.float() - out_center.float()) / epsilon  # (bs,16,64,64)
            J_cols.append(diff.flatten(start_dim=1))  # (bs, D_out)

        # stack to (D_low, D_out) then transpose -> (D_out, D_low)
        J_matrix = torch.cat(J_cols, dim=0).T
        A = J_matrix.T @ J_matrix  # (D_low, D_low)

        L, V = torch.linalg.eigh(A)
        V1_low = V[:, -1]  # (D_low,)
        ls = math.sqrt(max(L[-1].item(), 0.0))
        return V1_low, ls

    def analyze_sample(
            self,
            seed: int,
            prompt: str,
            negative_prompt: str = "",
            D_low: int = 256,
            epsilon: float = 0.02,
            num_inference_steps_F: int = 12,  # <= 建议先小一点
            guidance_scale_F: float = 7.0,
            fd_batch_size: int = 4,
    ):
        set_seed(seed)

        # 1) sample initial noise latent
        z = torch.randn(1, 16, 64, 64, device=self.device, dtype=self.dtype)

        # 2) embeddings for CFG (computed ONCE per sample)
        pos_c, neg_c, pos_p, neg_p = self.encode_prompt_cfg(prompt, negative_prompt=negative_prompt)

        # 3) random low-dim orthonormal subspace
        z_dim = z.numel()
        Q, _ = torch.linalg.qr(torch.randn(z_dim, D_low, device=self.device, dtype=torch.float32))
        W_sub = Q[:, :D_low]

        print(f"--- Seed {seed}: compute LS & V1 on full F (CFG) ...")
        V1_low_center, ls = self.compute_Jsub_and_V1_F(
            z_center=z,
            pos_c=pos_c, neg_c=neg_c, pos_p=pos_p, neg_p=neg_p,
            W_subspace=W_sub,
            epsilon=epsilon,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
            fd_batch_size=fd_batch_size,
        )

        # 4) PHFE on full F: directional derivative of F along V1
        V1_full = (W_sub @ V1_low_center).view(z.shape).to(self.dtype)
        V1_full = V1_full / (torch.norm(V1_full) + 1e-8)

        out_center = self.F_sampling_latents_cfg_embeds(
            z, pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
        )
        out_plus = self.F_sampling_latents_cfg_embeds(
            z + epsilon * V1_full, pos_c, neg_c, pos_p, neg_p,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
        )

        P1_latent = (out_plus.float() - out_center.float()) / epsilon  # (1,16,64,64)
        lap_kernel = get_laplacian_kernel(self.device, 16)
        P1_lap = F.conv2d(P1_latent, lap_kernel, padding=1, groups=16)
        phfe = torch.var(P1_lap).item()

        # 5) LC on full F: compare principal direction at z and z'
        print(f"--- Seed {seed}: compute LC on full F (neighbor) ...")
        n_low = torch.randn(D_low, device=self.device, dtype=torch.float32)
        n_low = (n_low / torch.norm(n_low)) * epsilon
        z_prime = (z.float() + (W_sub @ n_low).view(z.shape)).to(self.dtype)

        V1_low_prime, _ = self.compute_Jsub_and_V1_F(
            z_center=z_prime,
            pos_c=pos_c, neg_c=neg_c, pos_p=pos_p, neg_p=neg_p,
            W_subspace=W_sub,
            epsilon=epsilon,
            num_inference_steps=num_inference_steps_F,
            guidance_scale=guidance_scale_F,
            fd_batch_size=fd_batch_size,
        )

        cos_sim = torch.dot(V1_low_center, V1_low_prime).item()
        if cos_sim < 0:
            V1_low_prime = -V1_low_prime
            cos_sim = -cos_sim

        lc = torch.norm(V1_low_center - V1_low_prime).item() / epsilon

        # 6) save a visualization image (use full pipeline generation)
        # NOTE: this is just visualization; it does not affect metrics.
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                latents=z,
                num_inference_steps=28,
                guidance_scale=7.0,
                output_type="pil",
            ).images[0]

        return {
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "LS": ls,
            "LC": lc,
            "PHFE": phfe,
            "Axis_Cos": abs(cos_sim),
            "image": image
        }


# -------------------------
# main
# -------------------------
def append_row_to_csv(csv_path: str, row: dict, fieldnames: list):

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    #
    safe_row = {k: row.get(k, None) for k in fieldnames}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_row)
        f.flush()  #


def main(normal_prompt, OOD_prompt, input_class=None):
    analyzer = SD35GeometricAnalyzer(device=device)  # ✅ 显式传 cuda:{local_rank}

    prompts = {
        "ID_Normal": normal_prompt,
        "OOD_Hallucination": OOD_prompt,
        "zero_prompt":""
    }
    # seeds = range(500)

    save_dir = "sd35_"+input_class+"_geo_results"
    os.makedirs(save_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(save_dir, f"metrics_rank{rank}.csv")
    json_path = os.path.join(save_dir, f"metrics_report_rank{rank}.json")
    fieldnames = [
        "label", "filename",
        "seed", "prompt", "negative_prompt",
        "D_low", "epsilon",
        "num_inference_steps_F", "guidance_scale_F", "fd_batch_size",
        "LS", "LC", "PHFE", "Axis_Cos",
    ]

    all_metrics = []  # save json

    for label, text in prompts.items():
        print(f"--- Processing {label}...")
        for s in seeds:
            torch.cuda.empty_cache()
            try:
                #
                D_low = 64
                epsilon = 0.02
                num_inference_steps_F = 12
                guidance_scale_F = 7.0
                fd_batch_size = 1
                negative_prompt = ""

                res = analyzer.analyze_sample(
                    seed=s,
                    prompt=text,
                    negative_prompt=negative_prompt,
                    D_low=D_low,
                    epsilon=epsilon,
                    num_inference_steps_F=num_inference_steps_F,
                    guidance_scale_F=guidance_scale_F,
                    fd_batch_size=fd_batch_size,
                )

                fn = f"{label}_S{s}_LC{res['LC']:.1f}_PHFE{res['PHFE']:.2f}.png"
                res["image"].save(os.path.join(save_dir, fn))

                row = {
                    "label": label,
                    "filename": fn,
                    "seed": res["seed"],
                    "prompt": res["prompt"],
                    "negative_prompt": res["negative_prompt"],
                    "D_low": D_low,
                    "epsilon": epsilon,
                    "num_inference_steps_F": num_inference_steps_F,
                    "guidance_scale_F": guidance_scale_F,
                    "fd_batch_size": fd_batch_size,
                    "LS": float(res["LS"]),
                    "LC": float(res["LC"]),
                    "PHFE": float(res["PHFE"]),
                    "Axis_Cos": float(res["Axis_Cos"]),
                }

                append_row_to_csv(csv_path, row, fieldnames)

                all_metrics.append(row)

                print(f"DONE: {label} | LC: {row['LC']:.3f} | LS: {row['LS']:.3f} | PHFE: {row['PHFE']:.3f}")
                del res["image"]
                del res

            except Exception as e:
                print(f"Error on {label} Seed {s}: {e}")

    #
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    normal_prompt = "A real sunflower with bright yellow petals and a dark brown center, standing tall in a sunny field, macro photography, natural colors, sharp focus, photorealistic."
    OOD_prompt = "A sunflower with vivid blue petals and a dark brown center, standing in a field, macro photography, botanically incorrect but artistically rendered, sharp focus."
    input_class = "sunflower"
    main(normal_prompt, OOD_prompt,input_class)
    print("It's done.")

"""
torchrun --nproc_per_node=2 cul_.py

"""