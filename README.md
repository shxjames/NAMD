# NAMD: Nodule-Aware Multimodal Diffusion

> Nodule-Aligned Latent Space Learning with LLM-Driven Multimodal Diffusion for Longitudinal Lung LDCT Prediction

---

## Training and Testing NAMD

### Downloading the Dataset
Due to data-sharing regulations, we cannot directly distribute the datasets in this repository. You can download them from:

- [NLST](https://www.cancerimagingarchive.net/collection/nlst/)
- [DLCS](https://zenodo.org/records/13799069)
- [LUNA25](https://luna25.grand-challenge.org/)
- [LUNA16](https://luna16.grand-challenge.org/)

Place the downloaded datasets in the following folders:

- `NLST_with_second_large_cleaned`
- `Luna25_nodule_2D_Checked`
- `DLCS_patches`
- `Luna16_patches`

Note that NLST is the only one with paired images. The others are unpaired images and do not provide the level of patient/nodule features that NLST has; hence, those images are used to only train the unconditional unet prior. NLST is used to train the conditional model. 

### Step 1 &mdash; Download SD 1.5 Pretrained Weights

Download the SD 1.5 checkpoint into `preprocess/`:

```bash
wget -P preprocess/ https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
```

Then extract the autoencoder weights:

```bash
python preprocess/preprocess.py \
    --ckpt_path preprocess/v1-5-pruned.ckpt \
    --output_path preprocess/v1-5-autokl.ckpt \
    --type autoencoder
```

---

### Step 2 &mdash; Train the Autoencoder

Fine-tune the autoencoder on the lung nodule dataset:

```bash
python run.py --config configs/autoencoder/contrastive-VAE.yaml --train --project autoencoder
```

Checkpoints are saved under `checkpoints/`.

---

### Step 3 &mdash; Compute the Scale Factor

Run `eval_autoencoder.py` to compute the scale factor that normalizes the latent space:

```bash
python eval_autoencoder.py \
    --config configs/autoencoder/contrastive-VAE.yaml \
    --ckpt YOUR_AE_CKPT
```

The script prints a `Recommended scale_factor` at the end. Update `scale_factor` in both LDM configs:
- `configs/ldm/v1.5-unconditional-unet.yaml`
- `configs/ldm/v1.5-conditional-unet-llm.yaml`

Also set `ckpt_path` under `first_stage_config` in both configs to your trained autoencoder checkpoint.

---

### Step 4 &mdash; Train the Unconditional Diffusion Model

```bash
python run.py --config configs/ldm/v1.5-unconditional-unet.yaml --train --project v1.5-unconditional
```

---

### Step 5 &mdash; Train the Conditional Diffusion Model

Set `ckpt_path` in `configs/ldm/v1.5-conditional-unet-llm.yaml` to the unconditional checkpoint from Step 4, then run:

```bash
python run.py --config configs/ldm/v1.5-conditional-unet-llm.yaml --train --project v1.5-conditional
```

---

### Step 6 &mdash; Evaluation

Run the full multi-seed evaluation pipeline:

```bash
python eval_all.py \
    --config configs/ldm/v1.5-conditional-unet-llm.yaml \
    --ckpt YOUR_CONDITIONAL_CKPT \
    --eval_config configs/evaluator/vit.yaml \
    --num_steps 50
```

| Flag | Description |
|------|-------------|
| `--include_lpips` | Compute LPIPS |
| `--include_fid` | Compute FID |
| `--save_grid` | Save multi-seed comparison grids |

Results are saved to `results/<project>/eval_all_<guidance>.json`.

For reproducibility, we also provide the pretrained ViT checkpoint used for evaluation. Download it as follows:
```
hf download FlyingFlower/vit 42.pth --local-dir checkpoints/vit
```

---

## Ablation Study

Repeat Steps 2&ndash;6 using `configs/autoencoder/ablation-VAE.yaml` instead of `contrastive-VAE.yaml`.

To train the autoencoder from scratch (without SD 1.5 initialization), comment out the following line in the config:

```yaml
# ckpt_path: preprocess/v1-5-autokl.ckpt
```

---

## Data Format

Place your dataset under `NLST_with_second_large_cleaned/`. Each `.npy` file contains a `32×1` array — the first 16 entries represent the first-year LDCT scan and the last 16 the second-year LDCT scan.

### Demographic Features

| Feature | Description |
|---------|-------------|
| `SCT_PRE_ATT` | Predominant attenuation |
| `SCT_EPI_LOC` | Location of nodule in the lung |
| `SCT_LONG_DIA` | Longest diameter |
| `SCT_PERP_DIA` | Longest perpendicular diameter |
| `SCT_MARGINS` | Margin of the nodule |
| `unique_ids` | Patient ID |
| `year` | Year |
| `age` | Age |
| `diagemph` | Diagnosis of Emphysema |
| `gender` | Gender |
| `famfather` | Family history — Father |
| `fammother` | Family history — Mother |
| `fambrother` | Family history — Brother |
| `famsister` | Family history — Sister |
| `famchild` | Family history — Child |
| `can_scr` | Malignancy label (B/M) — not a feature |

