import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import json

import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score
import os

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset import Lung_Pair_DM
import argparse
from ldm.util import instantiate_from_config

import math
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [642, 115, 893, 23, 42, 403, 756, 401, 912, 55, 330, 888, 147, 679, 92, 450, 713, 28, 599, 974]


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config"
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="configs/evaluator/vit.yaml",
        help="Path to evaluation model config"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="Whether to save heatmaps as PNG images"
    )
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default="heatmaps",
        help="Directory to save heatmaps"
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save comparison grids of cond_img, generated, and gt_img"
    )
    parser.add_argument(
        "--max_grid_images",
        type=int,
        default=16,
        help="Max number of grid images to save per batch"
    )
    parser.add_argument(
        "--save_variance_examples",
        action="store_true",
        help="Save qualitative examples: GT image, mean error map, and pixel-wise variance map"
    )
    parser.add_argument(
        "--num_variance_examples",
        type=int,
        default=20,
        help="Number of qualitative variance examples to save (default picks 10 lowest-variance + 10 highest-variance)"
    )
    parser.add_argument(
        "--variance_examples_dir",
        type=str,
        default="",
        help="Directory to save qualitative variance example figures (default: results/<project>/variance_examples)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of sampling steps for diffusion/flow models"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--include_lpips",
        action="store_true",
        help="Also compute LPIPS metric"
    )
    parser.add_argument(
        "--include_fid",
        action="store_true",
        help="Also compute FID metric"
    )

    return parser

def sample_from_model(model, batch_size, batch, num_steps=20, guidance=1.0):
    """Generate samples from LatentDiffusionCond_LLM."""
    from ldm.models.diffusion.ddpm import LatentDiffusionCond_LLM
    from ldm.models.diffusion.ddim import DDIMSampler

    model.eval()

    with torch.no_grad():
        z0, ehr_tok, _, _ = model.get_input(batch)
        contexts = model.model.diffusion_model.contexts
        if contexts.device != z0.device or contexts.dtype != torch.float32:
            contexts = contexts.to(device=z0.device, dtype=torch.float32)
        emb_txt = model.text_encoder(ehr_tok.to(z0.device), contexts)
        cond = {'c_concat': [z0], 'c_crossattn': [emb_txt]}

        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=0.0, verbose=False)
        ddim_start = torch.randn(
            batch_size, 4, z0.shape[-2], z0.shape[-1], device=z0.device, dtype=model.dtype
        )
        samples_ddim = sampler.decode(ddim_start, t_start=num_steps - 1, cond=cond)
        samples = model.decode_first_stage(samples_ddim)

    samples = samples.to(device)
    return samples


def get_heatmap(attention_map):
    cls_attn = attention_map[0, :, 0, 1:]
    cls_attn_mean = cls_attn.mean(0)

    num_patches = cls_attn_mean.shape[0]
    grid_size = int(math.sqrt(num_patches))
    attn_map = cls_attn_mean.reshape(grid_size, grid_size)

    heatmap = attn_map.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(64,64), mode="bilinear").squeeze()
    heatmap = heatmap.detach().cpu()

    img_min = heatmap.min()
    img_max = heatmap.max()
    heatmap = (heatmap - img_min) / (img_max - img_min)

    return heatmap

def save_heatmap(attention_map, img, batch_idx, heatmap_dir, name="heatmap"):
    heatmap = get_heatmap(attention_map)
    img = (img[0].detach().cpu() + 1) / 2
    img = img.squeeze(0)
    heatmap_concat = torch.cat([img, heatmap], dim=1)
    save_image = transforms.ToPILImage()(heatmap_concat)
    save_image.save(os.path.join(heatmap_dir, f"{name}_{batch_idx}.png"))

def save_multiseed_grid(cond_imgs, all_gen_imgs, gt_imgs, batch_idx, grid_dir, max_images=16):
    """
    Save a grid with rows = samples, columns = cond | gen_seed1 | ... | gen_seedN | gt.

    Args:
        cond_imgs: [B, C, H, W] tensor, condition images for this batch
        all_gen_imgs: list of [B, C, H, W] tensors, one per seed
        gt_imgs: [B, C, H, W] tensor, ground truth images for this batch
        batch_idx: batch index for filename
        grid_dir: output directory
        max_images: max rows per PNG
    """
    n_seeds = len(all_gen_imgs)
    n_samples = min(cond_imgs.shape[0], max_images)
    n_cols = 1 + n_seeds + 1  # cond + seeds + gt

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2 * n_cols, 2 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        # Cond image
        img_np = (cond_imgs[i][0].numpy() + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_title("Cond" if i == 0 else "")
        axes[i, 0].axis("off")

        # Generated images per seed
        for s in range(n_seeds):
            img_np = (all_gen_imgs[s][i][0].numpy() + 1) / 2
            img_np = np.clip(img_np, 0, 1)
            axes[i, 1 + s].imshow(img_np, cmap="gray")
            axes[i, 1 + s].set_title(f"Seed {s+1}" if i == 0 else "")
            axes[i, 1 + s].axis("off")

        # GT image
        img_np = (gt_imgs[i][0].numpy() + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        axes[i, -1].imshow(img_np, cmap="gray")
        axes[i, -1].set_title("GT" if i == 0 else "")
        axes[i, -1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(grid_dir, f"multiseed_batch_{batch_idx}.png"), dpi=150)
    plt.close(fig)


def save_variance_examples(all_seed_gen_imgs, saved_gt_imgs, predictive_variance,
                           labels, output_dir, num_examples=20, mean_pred_prob=None):
    """
    Save qualitative figures with:
      1) ground-truth X2 image,
      2) mean absolute error map vs GT across seeds,
      3) pixel-wise variance map across seeds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten per-batch tensors to align with logits order [num_samples, ...]
    gt_flat = torch.cat(saved_gt_imgs, dim=0).cpu().numpy()[:, 0]  # [N, H, W]
    gen_stack = np.stack(
        [torch.cat(seed_batches, dim=0).cpu().numpy()[:, 0] for seed_batches in all_seed_gen_imgs],
        axis=0
    )  # [S, N, H, W]

    num_samples = predictive_variance.shape[0]

    # Deterministically pick variance extremes:
    # - 10 samples with the lowest predictive variance
    # - 10 samples with the highest predictive variance
    # If fewer than 20 total samples exist, keep unique indices only.
    sorted_indices = np.argsort(predictive_variance)
    low_count = min(10, num_samples)
    high_count = min(10, num_samples)
    low_var_indices = sorted_indices[:low_count]
    high_var_indices = sorted_indices[-high_count:][::-1]

    sample_indices = []
    seen = set()
    for idx in np.concatenate([low_var_indices, high_var_indices]):
        idx_int = int(idx)
        if idx_int not in seen:
            seen.add(idx_int)
            sample_indices.append(idx_int)

    # Keep backward compatibility with num_examples by truncating if requested.
    if num_examples is not None and num_examples > 0:
        sample_indices = sample_indices[:num_examples]
    labels_int = labels.astype(int)

    saved_paths = []
    for rank, sample_idx in enumerate(sample_indices):
        gt_img = gt_flat[sample_idx]               # [-1, 1]
        mean_error = np.mean(np.abs(gen_stack[:, sample_idx] - gt_img[None, ...]), axis=0)
        pixel_var = np.var(gen_stack[:, sample_idx], axis=0)
        gt_vis = np.clip((gt_img + 1.0) / 2.0, 0.0, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        title = (
            f"Sample {sample_idx} | label={labels_int[sample_idx]} | "
            f"pred var={predictive_variance[sample_idx]:.6f}"
        )
        if mean_pred_prob is not None:
            title += f" | mean p={mean_pred_prob[sample_idx]:.6f}"
        fig.suptitle(title, fontsize=11)

        axes[0].imshow(gt_vis, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground truth X2")
        axes[0].axis("off")

        im1 = axes[1].imshow(mean_error, cmap="hot")
        axes[1].set_title("Mean error map")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(pixel_var, cmap="magma")
        axes[2].set_title("Pixel-wise variance map")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"variance_example_rank_{rank+1:02d}_sample_{sample_idx:04d}.png")
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def evaluate_single(model, loader, eval_model, num_steps=20, guidance=1.0,
                    save_heatmaps=False, heatmap_dir="heatmaps",
                    lpips_metric=None, fid_metric=None,
                    return_images=False):
    """Run a single evaluation pass: sample once per input, return predictions, labels, and image metrics."""
    model.eval()
    eval_model.eval()

    result_cond = []
    result_gt = []
    result_model = []
    labels = []
    all_lpips = []

    # For returning raw images
    batch_cond_imgs = []
    batch_gen_imgs = []
    batch_gt_imgs = []

    if save_heatmaps:
        os.makedirs(heatmap_dir, exist_ok=True)

    batch_idx = 0
    for batch in tqdm(loader, desc="Evaluating"):
        # Batch format: (cond_img, gt_img, sentence_encoded, label)
        cond_img, gt_img, sentence_encoded, label = batch
        cond_img = cond_img.to(device)
        gt_img = gt_img.to(device)
        sentence_encoded = sentence_encoded.to(device)
        label = label.to(device)
        batch = (cond_img, gt_img, sentence_encoded, label)

        # Sample once per input
        model_output = sample_from_model(model, cond_img.shape[0], batch, num_steps=num_steps, guidance=guidance)

        def repeat_channels(img):
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            return img

        model_output_3ch = repeat_channels(model_output)
        cond_img = repeat_channels(cond_img)
        gt_img_3ch = repeat_channels(gt_img)

        with torch.no_grad():
            output_model, attn_model = eval_model(model_output_3ch, return_map=True)
            output_A, attention_map = eval_model(cond_img, return_map=True)
            output_B, _ = eval_model(gt_img_3ch, return_map=True)

        # Compute LPIPS (per-sample)
        if lpips_metric is not None:
            samples_01 = (model_output_3ch.clamp(-1, 1) + 1) / 2
            gt_01 = (gt_img_3ch.clamp(-1, 1) + 1) / 2
            with torch.no_grad():
                for i in range(samples_01.shape[0]):
                    lpips_val = lpips_metric(samples_01[i:i+1], gt_01[i:i+1])
                    all_lpips.append(lpips_val.item())

        # Update FID
        if fid_metric is not None:
            samples_01 = (model_output_3ch.clamp(-1, 1) + 1) / 2
            gt_01 = (gt_img_3ch.clamp(-1, 1) + 1) / 2
            fid_metric.update(gt_01, real=True)
            fid_metric.update(samples_01, real=False)

        if save_heatmaps:
            save_heatmap(attention_map, gt_img_3ch, batch_idx, heatmap_dir, name="heatmap_gt")
            save_heatmap(attn_model, model_output_3ch, batch_idx, heatmap_dir, name="heatmap_model")
        if return_images:
            batch_cond_imgs.append(cond_img.detach().cpu())
            batch_gen_imgs.append(model_output_3ch.detach().cpu())
            batch_gt_imgs.append(gt_img_3ch.detach().cpu())
        batch_idx += 1

        result_cond.append(output_A.detach().cpu().numpy())
        result_gt.append(output_B.detach().cpu().numpy())
        result_model.append(output_model.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    result_cond = np.concatenate(result_cond).reshape(-1)
    result_gt = np.concatenate(result_gt).reshape(-1)
    result_model = np.concatenate(result_model).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    # Compute FID for this run
    fid_score = None
    if fid_metric is not None:
        fid_score = fid_metric.compute().item()
        fid_metric.reset()

    lpips_mean = None
    if all_lpips:
        lpips_mean = float(np.mean(all_lpips))

    if return_images:
        return result_cond, result_gt, result_model, labels, lpips_mean, fid_score, batch_gen_imgs, batch_cond_imgs, batch_gt_imgs

    return result_cond, result_gt, result_model, labels, lpips_mean, fid_score


def compute_metrics(labels, predictions):
    """Compute ROC-AUC and AUPRC for a single set of predictions."""
    fpr, tpr, _ = roc_curve(labels, predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(labels, predictions)
    return roc_auc, auprc


def main():

    '''
    Usage:
        python eval_all.py --config configs/model.yaml --ckpt ckpts/model/last.ckpt
        python eval_all.py --config configs/model.yaml --ckpt ckpts/model/last.ckpt --include_lpips --include_fid
    '''
    parser = get_parser()
    args = parser.parse_args()

    project_dir = args.ckpt.split("/")[1]
    args.grid_dir = f"comparison_grids/{project_dir}"
    if not args.variance_examples_dir:
        args.variance_examples_dir = f"results/{project_dir}/variance_examples_{args.guidance}"

    config = OmegaConf.load(args.config)
    if args.ckpt:
        config.model.params.ckpt_path = None

    eval_config = OmegaConf.load(args.eval_config)

    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt,)

    eval_model = instantiate_from_config(eval_config.model)

    model.to(device)
    eval_model.to(device)

    dm = instantiate_from_config(config.data)
    dm.set_retriever(model)
    dm.hparams.batch_size = args.batch_size
    dm.setup('test')

    # Initialize optional image quality metrics
    lpips_metric = None
    if args.include_lpips:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    fid_metric = None
    if args.include_fid:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Run evaluation num_samples times
    model_preds_runs = []
    auc_model_list = []
    auprc_model_list = []
    lpips_list = []
    fid_list = []
    result_cond = None
    result_gt = None
    labels = None

    # Collect generated images per seed for optional qualitative visualizations
    # all_seed_gen_imgs[seed_idx][batch_idx] = [B, C, H, W] cpu tensor
    all_seed_gen_imgs = []
    saved_cond_imgs = None  # list of [B, C, H, W] per batch (same across seeds)
    saved_gt_imgs = None

    for i, seed in enumerate(SEEDS):
        seed_everything(seed)
        print(f"\n=== Evaluation run {i+1}/{len(SEEDS)} (seed={seed}) ===")

        ret = evaluate_single(
            model, dm.test_dataloader(), eval_model,
            num_steps=args.num_steps,
            guidance=args.guidance,
            save_heatmaps=args.save_heatmaps and i == 0,
            heatmap_dir=args.heatmap_dir,
            lpips_metric=lpips_metric,
            fid_metric=fid_metric,
            return_images=(args.save_grid or args.save_variance_examples),
        )

        if args.save_grid or args.save_variance_examples:
            r_cond, r_gt, r_model, r_labels, lpips_mean, fid_score, gen_imgs, cond_imgs, gt_imgs = ret
            all_seed_gen_imgs.append(gen_imgs)
            if saved_cond_imgs is None:
                saved_cond_imgs = cond_imgs
                saved_gt_imgs = gt_imgs
        else:
            r_cond, r_gt, r_model, r_labels, lpips_mean, fid_score = ret

        # Cond/gt predictions are deterministic, keep from first run
        if result_cond is None:
            result_cond = r_cond
            result_gt = r_gt
            labels = r_labels

        model_preds_runs.append(r_model)

        roc_model, ap_model = compute_metrics(r_labels, r_model)
        auc_model_list.append(roc_model)
        auprc_model_list.append(ap_model)
        print(f"  ROC-AUC  model={roc_model:.4f}  AUPRC  model={ap_model:.4f}")
        if lpips_mean is not None:
            print(f"  LPIPS    model={lpips_mean:.4f}")
            lpips_list.append(lpips_mean)
        if fid_score is not None:
            print(f"  FID      model={fid_score:.4f}")
            fid_list.append(fid_score)

    # Save multi-seed comparison grids
    if args.save_grid and saved_cond_imgs is not None:
        os.makedirs(args.grid_dir, exist_ok=True)
        n_batches = len(saved_cond_imgs)
        print(f"\nSaving multi-seed comparison grids ({n_batches} batches, {len(SEEDS)} seeds)...")
        for b in range(n_batches):
            # Gather generated images for this batch across all seeds
            gens_for_batch = [all_seed_gen_imgs[s][b] for s in range(len(SEEDS))]
            save_multiseed_grid(
                saved_cond_imgs[b], gens_for_batch, saved_gt_imgs[b],
                b, args.grid_dir, max_images=args.max_grid_images,
            )

    def summarize_runs(values):
        arr = np.array(values)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": [float(v) for v in arr]}

    # [num_seeds, num_samples]
    model_preds_matrix = np.stack(model_preds_runs, axis=0)
    # [num_samples, num_seeds] so each row is one datapoint across seeds
    model_preds_per_sample = model_preds_matrix.T

    os.makedirs(f"results/{project_dir}", exist_ok=True)
    logits_path = f"results/{project_dir}/vit_logits_all_seeds_{args.guidance}.npy"
    np.save(logits_path, model_preds_per_sample)
    print(
        f"Saved per-seed ViT logits to: {logits_path} "
        f"(shape={model_preds_per_sample.shape}; rows=samples, cols=seeds)"
    )

    # Convert logits to probabilities in [0, 1], then compute predictive variance
    model_probs_per_sample = 1.0 / (1.0 + np.exp(-model_preds_per_sample))
    probs_path = f"results/{project_dir}/vit_probs_all_seeds_{args.guidance}.npy"
    np.save(probs_path, model_probs_per_sample)
    print(
        f"Saved per-seed ViT probabilities to: {probs_path} "
        f"(shape={model_probs_per_sample.shape}; rows=samples, cols=seeds)"
    )

    predictive_variance = np.var(model_probs_per_sample, axis=1)
    variance_path = f"results/{project_dir}/vit_predictive_variance_probs_{args.guidance}.npy"
    np.save(variance_path, predictive_variance)
    print(
        f"Saved predictive variance to: {variance_path} "
        f"(shape={predictive_variance.shape}; one value per sample; expected range [0, 0.25])"
    )

    # Calibration-style uncertainty analysis: error vs variance
    mean_pred_prob = np.mean(model_probs_per_sample, axis=1)
    abs_error = np.abs(labels.astype(float) - mean_pred_prob)
    abs_error_path = f"results/{project_dir}/vit_abs_error_meanprob_{args.guidance}.npy"
    np.save(abs_error_path, abs_error)
    print(
        f"Saved absolute error of marginalized prediction to: {abs_error_path} "
        f"(shape={abs_error.shape})"
    )

    # Jittered strip-style scatter: x=true label (0/1), y=predictive variance
    labels_int = labels.astype(int)
    rng = np.random.default_rng(42)
    x_jittered = labels_int.astype(float) + rng.uniform(-0.1, 0.1, size=labels_int.shape[0])
    variance_plot_path = f"results/{project_dir}/predictive_variance_strip_{args.guidance}.png"

    plt.figure(figsize=(8, 5))
    benign_mask = labels_int == 0
    malignant_mask = labels_int == 1
    plt.scatter(
        x_jittered[benign_mask], predictive_variance[benign_mask],
        c="tab:blue", alpha=0.5, s=18, edgecolors="none", label="Benign (0)"
    )
    plt.scatter(
        x_jittered[malignant_mask], predictive_variance[malignant_mask],
        c="tab:red", alpha=0.5, s=18, edgecolors="none", label="Malignant (1)"
    )
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.005, 0.255)
    plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    plt.xlabel("True Label")
    plt.ylabel("Predictive Variance (sigmoid probabilities)")
    plt.title("Jittered strip plot: probability variance by true label")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(variance_plot_path, dpi=200)
    plt.close()
    print(f"Saved predictive variance strip plot to: {variance_plot_path}")

    # Scatter plot: x=|y-mu_pred|, y=predictive variance
    calibration_plot_path = f"results/{project_dir}/error_vs_variance_{args.guidance}.png"

    if np.std(abs_error) > 0 and np.std(predictive_variance) > 0:
        corr = float(np.corrcoef(abs_error, predictive_variance)[0, 1])
    else:
        corr = None

    plt.figure(figsize=(7, 5))
    benign_mask = labels_int == 0
    malignant_mask = labels_int == 1
    plt.scatter(
        abs_error[benign_mask], predictive_variance[benign_mask],
        c="tab:blue", alpha=0.5, s=18, edgecolors="none", label="Benign (0)"
    )
    plt.scatter(
        abs_error[malignant_mask], predictive_variance[malignant_mask],
        c="tab:red", alpha=0.5, s=18, edgecolors="none", label="Malignant (1)"
    )
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.005, 0.255)
    plt.xlabel("Absolute error |y - mu_pred|")
    plt.ylabel("Predictive variance (sigmoid probabilities)")
    title = "Calibration check: error vs predictive variance"
    if corr is not None:
        title += f" (Pearson r={corr:.3f})"
    plt.title(title)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(calibration_plot_path, dpi=200)
    plt.close()
    print(f"Saved error-vs-variance plot to: {calibration_plot_path}")

    qualitative_paths = []
    if args.save_variance_examples:
        qualitative_paths = save_variance_examples(
            all_seed_gen_imgs=all_seed_gen_imgs,
            saved_gt_imgs=saved_gt_imgs,
            predictive_variance=predictive_variance,
            labels=labels,
            output_dir=args.variance_examples_dir,
            num_examples=args.num_variance_examples,
            mean_pred_prob=mean_pred_prob,
        )
        print(
            f"Saved {len(qualitative_paths)} qualitative variance examples to: "
            f"{args.variance_examples_dir}"
        )

    if (args.save_grid or args.save_variance_examples) and saved_gt_imgs is not None:
        # Free large cached image tensors once all visualization exports are done
        del all_seed_gen_imgs, saved_cond_imgs, saved_gt_imgs

    # === Option A : average predictions, then compute metrics --- ensemble ===
    avg_model_preds = np.mean(model_preds_matrix, axis=0)

    roc_cond, ap_cond = compute_metrics(labels, result_cond)
    roc_gt, ap_gt = compute_metrics(labels, result_gt)
    roc_model_avg, ap_model_avg = compute_metrics(labels, avg_model_preds)
    auc_cond = {"mean": float(roc_cond)}
    auc_gt = {"mean": float(roc_gt)}
    auc_model_avg = {"mean": float(roc_model_avg)}
    auprc_cond = {"mean": float(ap_cond)}
    auprc_gt = {"mean": float(ap_gt)}
    auprc_model_avg = {"mean": float(ap_model_avg)}

    def fmt(d):
        return f'{d["mean"]:.4f}' + (f' ± {d["std"]:.4f}' if "std" in d else '')

    print(f"\n=== Main results: averaged predictions (num_seeds={len(SEEDS)}) ===")
    print(f'ROC-AUC  cond:  {fmt(auc_cond)}')
    print(f'ROC-AUC  gt:    {fmt(auc_gt)}')
    print(f'ROC-AUC  model: {fmt(auc_model_avg)}')
    print(f'AUPRC    cond:  {fmt(auprc_cond)}')
    print(f'AUPRC    gt:    {fmt(auprc_gt)}')
    print(f'AUPRC    model: {fmt(auprc_model_avg)}')

    # === Option B (supplementary): average per-run scores ===
    auc_model_runs = summarize_runs(auc_model_list)
    auprc_model_runs = summarize_runs(auprc_model_list)

    print(f"\n=== Supplementary: per-run score average (num_seeds={len(SEEDS)}) ===")
    print(f'ROC-AUC  model: {auc_model_runs["mean"]:.4f} ± {auc_model_runs["std"]:.4f}')
    print(f'AUPRC    model: {auprc_model_runs["mean"]:.4f} ± {auprc_model_runs["std"]:.4f}')

    # Save results to json
    results = {
        "seeds": SEEDS,
        "batch_size": args.batch_size,
        "config": args.config,
        "checkpoint": args.ckpt,
        "split": "test",
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "auc_cond": auc_cond,
        "auc_gt": auc_gt,
        "auc_model_avg_pred": auc_model_avg,
        "auprc_cond": auprc_cond,
        "auprc_gt": auprc_gt,
        "auprc_model_avg_pred": auprc_model_avg,
        "auc_model_avg_score": auc_model_runs,
        "auprc_model_avg_score": auprc_model_runs,
    }

    if lpips_list:
        lpips_summary = summarize_runs(lpips_list)
        print(f'LPIPS    model: {lpips_summary["mean"]:.4f} ± {lpips_summary["std"]:.4f}')
        results["lpips"] = lpips_summary

    if fid_list:
        fid_summary = summarize_runs(fid_list)
        print(f'FID      model: {fid_summary["mean"]:.4f} ± {fid_summary["std"]:.4f}')
        results["fid"] = fid_summary

    results["vit_logits_all_seeds_path"] = logits_path
    results["vit_logits_all_seeds_shape"] = list(model_preds_per_sample.shape)
    results["vit_probs_all_seeds_path"] = probs_path
    results["vit_probs_all_seeds_shape"] = list(model_probs_per_sample.shape)
    results["predictive_variance_path"] = variance_path
    results["predictive_variance_shape"] = list(predictive_variance.shape)
    results["predictive_variance_type"] = "variance_of_sigmoid_probabilities_across_seeds"
    results["predictive_variance_plot_path"] = variance_plot_path
    results["abs_error_path"] = abs_error_path
    results["error_vs_variance_plot_path"] = calibration_plot_path
    results["error_vs_variance_pearson_r"] = corr
    if qualitative_paths:
        results["qualitative_variance_examples_dir"] = args.variance_examples_dir
        results["qualitative_variance_examples_count"] = len(qualitative_paths)
        results["qualitative_variance_example_paths"] = qualitative_paths
    results["predictive_variance_summary"] = {
        "overall": {
            "mean": float(np.mean(predictive_variance)),
            "std": float(np.std(predictive_variance)),
            "n": int(predictive_variance.shape[0]),
        },
        "benign": {
            "mean": float(np.mean(predictive_variance[labels_int == 0])) if np.any(labels_int == 0) else None,
            "std": float(np.std(predictive_variance[labels_int == 0])) if np.any(labels_int == 0) else None,
            "n": int(np.sum(labels_int == 0)),
        },
        "malignant": {
            "mean": float(np.mean(predictive_variance[labels_int == 1])) if np.any(labels_int == 1) else None,
            "std": float(np.std(predictive_variance[labels_int == 1])) if np.any(labels_int == 1) else None,
            "n": int(np.sum(labels_int == 1)),
        },
    }

    os.makedirs(f"results/{project_dir}", exist_ok=True)
    with open(f"results/{project_dir}/eval_all_{args.guidance}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: results/{project_dir}/eval_all_{args.guidance}.json")

if __name__ == "__main__":
    main()
