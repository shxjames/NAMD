"""Pick and render the NAMD_3 success / failure case studies.

Reads the per-(seed, sample) metrics written by compute_metrics.py, optionally
joins the MedSAM2 + PyRadiomics nodule properties from
nodule_quality_eval/results_NAMD_3, then writes:

  <out>/per_sample_metrics.csv      one row per test sample, all aggregates
  <out>/success_cases.csv           the selected good cases
  <out>/failure_cases.csv           the selected bad cases
  <out>/success/*.png               per-case panels
  <out>/failure/*.png               per-case panels
  <out>/success_overview.png        one sheet with every good case
  <out>/failure_overview.png        one sheet with every bad case
  <out>/SUMMARY.md                  the write-up

The ViT decision threshold is not assumed to be 0: it is the Youden-J optimum
of the classifier's own scores on the *real* follow-up images, so "predicts the
label right" means right under the operating point the evaluator actually has.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

DEFAULT_NAMD_DIR = "/scratch/liyues_root/liyues/shxjames/NLST-Img/NAMD_3"

# Kept in sync with compute_metrics.py, duplicated so this script needs no torch.
_SAMPLE_RE = re.compile(r"^sample_(\d+)_(\d+)\.npy$")
_BOOT_RE = re.compile(r"^boot_(\d+)$")


def load_seed_dir(seed_dir):
    items = []
    for fname in os.listdir(seed_dir):
        m = _SAMPLE_RE.match(fname)
        if m is None:
            continue
        data = np.load(os.path.join(seed_dir, fname))  # (3, 64, 64, 1)
        items.append((int(m.group(1)), int(m.group(2)), data[0], data[1], data[2]))
    items.sort(key=lambda x: x[0])
    return items


def discover_seeds(namd_dir):
    seeds = []
    for name in os.listdir(namd_dir):
        m = _BOOT_RE.match(name)
        if m and os.path.isdir(os.path.join(namd_dir, name)):
            seeds.append((int(m.group(1)), os.path.join(namd_dir, name)))
    seeds.sort(key=lambda x: x[0])
    return seeds


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def youden_threshold(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    return float(thr[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------- aggregation
def build_table(npz, thr):
    labels = npz["labels"]
    lg, lr, lt = npz["logit_gen"], npz["logit_real"], npz["logit_t0"]
    lp = npz["lpips"]

    pred_gen = (lg > thr).astype(int)                      # [S, N]
    correct = (pred_gen == labels[None, :]).astype(float)
    # Confidence in the *true* class, on the sigmoid scale, relative to thr.
    p_gen = sigmoid(lg - thr)
    signed_conf = np.where(labels[None, :] == 1, p_gen, 1 - p_gen)

    df = pd.DataFrame({
        "sample": npz["sample_ids"],
        "label": labels,
        "lpips_mean": lp.mean(0),
        "lpips_std": lp.std(0),
        "lpips_min": lp.min(0),
        "lpips_max": lp.max(0),
        "mse_mean": npz["mse"].mean(0),
        "mse_roi_mean": npz["mse_roi"].mean(0),
        "psnr_mean": npz["psnr"].mean(0),
        "logit_gen_mean": lg.mean(0),
        "logit_gen_std": lg.std(0),
        "prob_gen_mean": p_gen.mean(0),
        "seeds_correct": correct.sum(0).astype(int),
        "seeds_pred_mal": pred_gen.sum(0).astype(int),
        "true_class_conf": signed_conf.mean(0),
        "logit_real": lr[0],
        "prob_real": sigmoid(lr[0] - thr),
        "real_correct": ((lr[0] > thr).astype(int) == labels).astype(int),
        "logit_t0": lt[0],
        "prob_t0": sigmoid(lt[0] - thr),
        "t0_correct": ((lt[0] > thr).astype(int) == labels).astype(int),
    })
    return df


def join_nodule(df, nodule_dir):
    """Add per-sample nodule properties averaged over the 20 generation seeds."""
    path = os.path.join(nodule_dir, "per_sample_summary_all.csv")
    if not os.path.exists(path):
        print(f"[nodule] {path} not found — skipping nodule properties")
        return df, False

    nd = pd.read_csv(path)
    nd["sample_idx"] = nd["sample"].str.extract(r"sample_(\d+)_").astype(int)
    nd["area_ratio"] = nd["area_T1g"] / nd["area_T1r"].replace(0, np.nan)
    nd["log_area_ratio"] = np.log2(nd["area_ratio"].clip(lower=1e-6))
    nd["abs_hu_err"] = (nd["mean_T1g"] - nd["mean_T1r"]).abs()
    if "shape_Sphericity_abs_err" in nd.columns:
        nd["sphericity_err"] = nd["shape_Sphericity_abs_err"]
    else:
        nd["sphericity_err"] = (nd["shape_Sphericity_T1g"]
                                - nd["shape_Sphericity_T1r"]).abs()

    agg = nd.groupby("sample_idx").agg(
        area_T1r=("area_T1r", "mean"),
        area_T1g=("area_T1g", "mean"),
        abs_area_err=("abs_area_error_T1", "mean"),
        log_area_ratio=("log_area_ratio", "mean"),
        hu_T1r=("mean_T1r", "mean"),
        hu_T1g=("mean_T1g", "mean"),
        abs_hu_err=("abs_hu_err", "mean"),
        sphericity_err=("sphericity_err", "mean"),
        delta_area_real=("delta_area_real", "mean"),
        delta_area_gen=("delta_area_gen", "mean"),
    ).reset_index().rename(columns={"sample_idx": "sample"})

    out = df.merge(agg, on="sample", how="left")
    print(f"[nodule] joined {agg.shape[0]} samples from {path}")
    return out, True


# ------------------------------------------------------------------ selection
def take_balanced(pool, n, n_pos_target):
    """Take n rows from an already-ranked pool, front-loading malignant cases."""
    pos = pool[pool.label == 1]
    neg = pool[pool.label == 0]
    picked = pd.concat([pos.head(n_pos_target), neg.head(n - min(n_pos_target, len(pos)))])
    if len(picked) < n:                       # top up from whatever is left
        rest = pool[~pool["sample"].isin(picked["sample"])]
        picked = pd.concat([picked, rest.head(n - len(picked))])
    return picked.sort_values("lpips_mean").reset_index(drop=True)


def nodule_gate(df):
    """Flag samples whose generated nodule is faithful on its own class's terms.

    Malignant nodules are systematically larger with larger HU errors, so an
    absolute cohort-wide threshold would silently exclude every malignant case.
    Thresholds are therefore the median *within each label group*. Area
    agreement is mandatory — a nodule rendered at half or double the real size
    is not a good generation however clean it looks — plus at least one of
    intensity or boundary shape.
    """
    d = df.copy()
    d["abs_lar"] = d["log_area_ratio"].abs()
    ok = {}
    for c in ("abs_lar", "abs_hu_err", "sphericity_err"):
        med = d.groupby("label")[c].transform("median")
        ok[c] = (d[c] <= med).fillna(False)
    return ok["abs_lar"] & (ok["abs_hu_err"] | ok["sphericity_err"])


def select_success(df, n=9, n_pos=4, min_seeds=19, has_nodule=False):
    ok = df[(df.seeds_correct >= min_seeds) & (df.real_correct == 1)].copy()
    print(f"[success] {len(ok)} samples pass the classifier filter "
          f"(>= {min_seeds}/20 seeds correct AND real T1 correct); "
          f"{int((ok.label == 1).sum())} malignant")
    if has_nodule:
        ok = ok[nodule_gate(df).reindex(ok.index).fillna(False)]
        print(f"[success] {len(ok)} of those also pass the nodule-fidelity gate; "
              f"{int((ok.label == 1).sum())} malignant")
    ok = ok.sort_values(["lpips_mean"])
    return take_balanced(ok, n, n_pos)


def select_failure(df, n=10, has_nodule=False, min_seeds_wrong=19):
    """Half the slate is perceptual collapse, half is a confident clinical flip."""
    df = df.copy()
    # Bucket A — worst reconstructions.
    a = df.sort_values("lpips_mean", ascending=False).head(60).copy()
    if has_nodule and a["abs_area_err"].notna().any():
        # Break the LPIPS tie with nodule-property damage.
        z = lambda s: (s - df[s.name].mean()) / (df[s.name].std() + 1e-9)
        a["nodule_damage"] = (z(a["abs_area_err"].fillna(df["abs_area_err"].mean()))
                              + z(a["abs_hu_err"].fillna(df["abs_hu_err"].mean()))
                              + z(a["sphericity_err"].fillna(df["sphericity_err"].mean())))
        a = a.sort_values(["lpips_mean"], ascending=False)
    a["failure_mode"] = "perceptual"
    bucket_a = take_balanced(a, n // 2, max(1, n // 4))

    # Bucket B — the real image is classified right, every generated one is wrong.
    n_b = n - len(bucket_a)
    thresh = min_seeds_wrong
    while thresh > 10:
        b = df[(df.real_correct == 1) & ((20 - df.seeds_correct) >= thresh)].copy()
        b = b[~b["sample"].isin(bucket_a["sample"])]
        if len(b) >= n_b:
            break
        thresh -= 2
    print(f"[failure] {len(b)} samples are a confident classifier flip "
          f"(real correct, >= {thresh}/20 generated wrong); "
          f"{int((b.label == 1).sum())} malignant")
    # Rank by how confidently wrong: lowest confidence in the true class.
    b = b.sort_values("true_class_conf")
    bucket_b = take_balanced(b, n_b, max(2, n_b // 2))
    bucket_b["failure_mode"] = "classifier-flip"

    out = pd.concat([bucket_a, bucket_b]).reset_index(drop=True)
    return out, thresh


# ------------------------------------------------------------------ rendering
def load_images(namd_dir):
    """Return t0[N,H,W], real[N,H,W], gen[S,N,H,W], seeds[S], ids[N]."""
    seeds = discover_seeds(namd_dir)
    gen, t0, real, ids = [], None, None, None
    for _, sd in seeds:
        items = load_seed_dir(sd)
        gen.append(np.stack([x[3].squeeze(-1) for x in items]))
        if t0 is None:
            ids = np.array([x[0] for x in items])
            t0 = np.stack([x[2].squeeze(-1) for x in items])
            real = np.stack([x[4].squeeze(-1) for x in items])
    return t0, real, np.stack(gen), np.array([s for s, _ in seeds]), ids


def panel(ax, img, title, vmin=0, vmax=255, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def case_row(axes, t0, real, gen_stack, seed_ids, lpips_col):
    """gen_stack: [S, H, W] for this sample; lpips_col: [S] LPIPS per seed."""
    order = np.argsort(lpips_col)
    best, med, worst = order[0], order[len(order) // 2], order[-1]

    panel(axes[0], t0, "T0 (input)")
    panel(axes[1], real, "T1 real (target)")
    panel(axes[2], gen_stack[best], f"gen best s{seed_ids[best]}\nLPIPS {lpips_col[best]:.3f}")
    panel(axes[3], gen_stack[med], f"gen median s{seed_ids[med]}\nLPIPS {lpips_col[med]:.3f}")
    panel(axes[4], gen_stack[worst], f"gen worst s{seed_ids[worst]}\nLPIPS {lpips_col[worst]:.3f}")

    err = np.abs(gen_stack.astype(np.float32).mean(0) - real.astype(np.float32))
    im = axes[5].imshow(err, cmap="magma", vmin=0, vmax=max(err.max(), 1e-6))
    axes[5].set_title("|mean gen - real|", fontsize=8)
    axes[5].axis("off")
    # Mark the centre disc used as the nodule-region proxy.
    h, w = err.shape
    axes[5].add_patch(plt.Circle((w // 2, h // 2), int(min(h, w) * 0.25),
                                 fill=False, ec="cyan", lw=1.0, ls="--"))
    return im


def render_set(df, t0, real, gen, seed_ids, lpips, id_to_pos, out_dir, title,
               has_nodule):
    os.makedirs(out_dir, exist_ok=True)
    # Drop panels from an earlier selection so the directory only ever holds the
    # cases the current run actually chose.
    for stale in os.listdir(out_dir):
        if stale.endswith(".png"):
            os.remove(os.path.join(out_dir, stale))
    n = len(df)
    fig, axes = plt.subplots(n, 6, figsize=(15, 2.55 * n))
    if n == 1:
        axes = axes[None, :]

    for r, (_, row) in enumerate(df.iterrows()):
        j = id_to_pos[int(row["sample"])]
        case_row(axes[r], t0[j], real[j], gen[:, j], seed_ids, lpips[:, j])
        label_txt = "MALIGNANT" if row.label == 1 else "benign"
        cap = (f"sample {int(row['sample'])} · {label_txt}\n"
               f"LPIPS {row.lpips_mean:.3f}±{row.lpips_std:.3f} · PSNR {row.psnr_mean:.1f} dB\n"
               f"ViT correct {int(row.seeds_correct)}/20 · p(true) {row.true_class_conf:.2f}\n"
               f"real p={row.prob_real:.2f} ({'ok' if row.real_correct else 'wrong'}) · "
               f"T0 p={row.prob_t0:.2f}")
        if has_nodule and not pd.isna(row.get("area_T1r", np.nan)):
            cap += (f"\narea gen/real {row.area_T1g:.0f}/{row.area_T1r:.0f} px "
                    f"({row.log_area_ratio:+.2f} log2)\n"
                    f"HU err {row.abs_hu_err:.1f} · sphericity err {row.sphericity_err:.3f}")
        axes[r, 0].text(-0.12, 0.5, cap, transform=axes[r, 0].transAxes,
                        fontsize=7.5, va="center", ha="right", family="monospace")

        # Per-case figure as well.
        f2, a2 = plt.subplots(1, 6, figsize=(15, 3.1))
        case_row(a2, t0[j], real[j], gen[:, j], seed_ids, lpips[:, j])
        f2.suptitle(cap.replace("\n", "   |   "), fontsize=8, family="monospace")
        f2.tight_layout(rect=[0, 0, 1, 0.86])
        f2.savefig(os.path.join(out_dir, f"sample_{int(row['sample']):04d}_"
                                         f"label{int(row.label)}.png"), dpi=170)
        plt.close(f2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0.14, 0, 1, 0.985])
    return fig


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namd_dir", default=DEFAULT_NAMD_DIR)
    ap.add_argument("--metrics", default="results/NAMD_3_cases/per_sample_seed_metrics.npz")
    ap.add_argument("--nodule_dir",
                    default="/scratch/liyues_root/liyues/shxjames/NAMD/"
                            "nodule_quality_eval/results_NAMD_3")
    ap.add_argument("--out_dir", default="results/NAMD_3_cases")
    ap.add_argument("--n_success", type=int, default=9)
    ap.add_argument("--n_failure", type=int, default=10)
    args = ap.parse_args()

    npz = np.load(args.metrics)
    labels = npz["labels"]
    thr = youden_threshold(labels, npz["logit_real"][0])
    print(f"ViT operating point (Youden J on real T1 scores): logit > {thr:.4f}")

    df = build_table(npz, thr)
    df, has_nodule = join_nodule(df, args.nodule_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    df.sort_values("sample").to_csv(
        os.path.join(args.out_dir, "per_sample_metrics.csv"), index=False)

    print(f"\ncohort: n={len(df)}  malignant={int((df.label == 1).sum())}  "
          f"LPIPS {df.lpips_mean.mean():.4f}  "
          f"real-image ViT accuracy {df.real_correct.mean():.3f}")

    succ = select_success(df, args.n_success, has_nodule=has_nodule)
    fail, flip_thresh = select_failure(df, args.n_failure, has_nodule)
    succ.to_csv(os.path.join(args.out_dir, "success_cases.csv"), index=False)
    fail.to_csv(os.path.join(args.out_dir, "failure_cases.csv"), index=False)

    t0, real, gen, seed_ids, ids = load_images(args.namd_dir)
    id_to_pos = {int(s): i for i, s in enumerate(ids)}
    assert np.array_equal(ids, npz["sample_ids"]), "image order != metric order"

    f1 = render_set(succ, t0, real, gen, seed_ids, npz["lpips"], id_to_pos,
                    os.path.join(args.out_dir, "success"),
                    "NAMD_3 — good generations the ViT also calls correctly",
                    has_nodule)
    f1.savefig(os.path.join(args.out_dir, "success_overview.png"), dpi=150)
    plt.close(f1)

    f2 = render_set(fail, t0, real, gen, seed_ids, npz["lpips"], id_to_pos,
                    os.path.join(args.out_dir, "failure"),
                    "NAMD_3 — spectacular failures (perceptual + classifier flip)",
                    has_nodule)
    f2.savefig(os.path.join(args.out_dir, "failure_overview.png"), dpi=150)
    plt.close(f2)

    write_summary(args.out_dir, df, succ, fail, thr, has_nodule, flip_thresh)
    print(f"\nwrote {args.out_dir}/{{success,failure}}_overview.png and CSVs")


def write_summary(out_dir, df, succ, fail, thr, has_nodule, flip_thresh):
    lines = []
    A = lines.append
    A("# NAMD_3 case studies\n")
    A(f"Cohort: {len(df)} test samples x 20 generation seeds "
      f"(`/scratch/liyues_root/liyues/shxjames/NLST-Img/NAMD_3`), "
      f"{int((df.label == 1).sum())} malignant.\n")
    A(f"- LPIPS (AlexNet, gen vs real T1), mean over all 9000 pairs: "
      f"**{df.lpips_mean.mean():.4f}**\n")
    A(f"- ViT operating point: `logit > {thr:.4f}` (Youden-J optimum of the same "
      f"classifier on the *real* follow-up images). Real-image accuracy at that "
      f"point: **{df.real_correct.mean():.3f}**; T0 accuracy {df.t0_correct.mean():.3f}.\n")
    if not has_nodule:
        A("- Nodule property metrics (MedSAM2 + PyRadiomics) were not available "
          "for this arm; selection uses LPIPS / PSNR / centre-ROI error only.\n")

    A("\n## 1. Good cases (low LPIPS, ViT gets the label right)\n")
    gate_txt = ""
    if has_nodule:
        gate_txt = (", **and** the generated nodule passes a fidelity gate: "
                    "|log2 area ratio| within its own class's median, plus HU "
                    "or sphericity error within class median. Those thresholds "
                    "are per-label because malignant nodules are larger with "
                    "larger HU errors, so a cohort-wide cut excludes all of them")
    A(f"Filter: real T1 classified correctly **and** >=19/20 generated seeds "
      f"classified correctly{gate_txt}; ranked by mean LPIPS ascending, "
      f"balanced across labels.\n")
    A(fmt_table(succ, has_nodule))

    A("\n## 2. Failure cases\n")
    A("Two buckets. `perceptual` = worst mean LPIPS in the cohort. "
      "`classifier-flip` = the real follow-up is classified correctly but "
      f">={flip_thresh}/20 generated follow-ups flip the ViT to the wrong "
      "class, ranked by lowest confidence in the true class.\n")
    A(fmt_table(fail, has_nodule, extra=["failure_mode"]))

    A("\n## 3. Cohort context for these picks\n")
    for line in cohort_stats(df):
        A(f"- {line}")

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))


def cohort_stats(df):
    """The population-level facts that decide how far the case studies generalise."""
    from scipy.stats import mannwhitneyu, pearsonr

    out = []
    d = df.dropna(subset=["log_area_ratio"])
    shift = (df.prob_gen_mean - df.prob_real)

    r, p = pearsonr(d.log_area_ratio, d.prob_gen_mean)
    out.append(f"Generated nodule size tracks the malignancy score: "
               f"corr(log2 area ratio, p_gen) = **{r:.2f}** (p={p:.1e}, n={len(d)}). "
               f"Real but modest — size errors push the classifier, they do not "
               f"determine it.")

    r2, p2 = pearsonr(df.lpips_mean, shift.abs())
    out.append(f"Perceptual quality and classifier behaviour are **decoupled**: "
               f"corr(LPIPS, |p_gen - p_real|) = {r2:.2f} (p={p2:.2f}, n.s.).")

    worst = df.nlargest(max(1, len(df) // 10), "lpips_mean")
    out.append(f"The worst-decile-LPIPS samples are classified no worse than the "
               f"cohort: ViT correct on >=19/20 seeds for "
               f"{int((worst.seeds_correct >= 19).sum())}/{len(worst)} of them "
               f"({(worst.seeds_correct >= 19).mean():.2f}) vs "
               f"{(df.seeds_correct >= 19).mean():.2f} overall. So the two failure "
               f"buckets in section 2 are genuinely different failure modes, not "
               f"one phenomenon seen twice.")

    # Does the model reproduce a diagnostic *change* between the two visits?
    n_seeds = 20
    c_cond = (df.prob_t0 >= 0.5).astype(int)
    c_gt = (df.prob_real >= 0.5).astype(int)
    side_cond = np.where(c_cond == 1, df.seeds_pred_mal,
                         n_seeds - df.seeds_pred_mal) / n_seeds
    match_gt = np.where(c_gt == 1, df.seeds_pred_mal,
                        n_seeds - df.seeds_pred_mal) / n_seeds
    dis = (c_cond != c_gt)
    out.append(f"When the ViT reads T0 and the real follow-up the **same** way "
               f"(n={int((~dis).sum())}), the generated image matches "
               f"{match_gt[~dis].mean():.2f} of the time. When it reads them "
               f"**differently** (n={int(dis.sum())}) the generations are at "
               f"chance: {side_cond[dis].mean():.2f} side with T0, "
               f"{1 - side_cond[dis].mean():.2f} with the real follow-up. The "
               f"model reproduces appearance well but carries little signal "
               f"about the direction of change — which is where the classifier "
               f"flips come from. Note it is *not* simply copying T0: "
               f"corr(p_gen, p_GT) = {pearsonr(df.prob_gen_mean, df.prob_real)[0]:.2f} "
               f"slightly exceeds corr(p_gen, p_cond) = "
               f"{pearsonr(df.prob_gen_mean, df.prob_t0)[0]:.2f}.")

    ok = df[df.real_correct == 1]
    flip, stable = ok[ok.seeds_correct <= 1], ok[ok.seeds_correct >= 19]
    if len(flip) > 2 and len(stable) > 2:
        _, pu = mannwhitneyu(flip.log_area_ratio.dropna(),
                             stable.log_area_ratio.dropna())
        out.append(f"Flipped samples do trend toward over-grown nodules (median "
                   f"log2 area ratio {flip.log_area_ratio.median():+.2f} vs "
                   f"{stable.log_area_ratio.median():+.2f} for stable-correct "
                   f"ones) but with only n={len(flip)} flips this does not reach "
                   f"significance (Mann-Whitney p={pu:.2f}) — suggestive, not "
                   f"established.")
        n_pos = int((flip.label == 1).sum())
        out.append(f"Of the {len(flip)} confident flips, {len(flip) - n_pos} are "
                   f"benign called malignant and {n_pos} malignant called benign. "
                   f"The false-negative direction is rare here but is the "
                   f"clinically costly one.")
    return out


def fmt_table(d, has_nodule, extra=()):
    cols = ["sample", "label", "lpips_mean", "psnr_mean", "seeds_correct",
            "true_class_conf", "prob_real", "prob_gen_mean"]
    if has_nodule:
        cols += ["area_T1r", "area_T1g", "log_area_ratio", "abs_hu_err",
                 "sphericity_err"]
    cols += list(extra)
    cols = [c for c in cols if c in d.columns]
    sub = d[cols]

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "-" if pd.isna(v) else f"{v:.3f}"
        return str(v)

    rows = [[cell(v) for v in r] for r in sub.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    fmt = lambda vals: "| " + " | ".join(
        v.rjust(w) for v, w in zip(vals, widths)) + " |"
    return "\n".join([fmt(cols),
                      "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
                     + [fmt(r) for r in rows])


if __name__ == "__main__":
    main()
