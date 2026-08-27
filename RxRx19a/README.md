# RxRx19a — ADIGen counterfactual cell-image experiment

Trains a **conditional latent diffusion / flow-matching DiT** on the [RxRx19a](https://www.rxrx.ai/rxrx19a) COVID-19 microscopy screen under a **doubly-robust (DR) training risk**. Evaluation is done on the ATE's derived via embeddings of the images.

Each well in RxRx19a received exactly one `(compound, concentration)`treatment. 

To estimate the *counterfactual* image distribution`p(image | do(compound, conc))`, we use ADIGen.

---

## 1. Environment

Python **3.8**, CUDA **12.1**, PyTorch **2.4.1**. The cluster env is a conda env named `torch-cuda`; every `scripts/*.slurm` starts with

```bash
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate torch-cuda
```

To rebuild it from scratch:

```bash
conda create -n torch-cuda python=3.8 -y
conda activate torch-cuda

# torch first, from the CUDA-matched wheel index
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### OpenPhenom encoder (optional, for `--encoder openphenom`)

`src/eval/openphenom_encoder.py`

```bash
git clone https://github.com/recursionpharma/maes_microscopy third_party/maes_microscopy
```

Weights (`recursionpharma/OpenPhenom`) and the VAE (`ostris/vae-kl-f8-d16`) are pulled with `huggingface_hub.snapshot_download`. **Compute nodes are offline**, so warm the HF cache on the login node once, then run jobs with `HF_HUB_OFFLINE=1`.

### Paths

`src/spec.py` derives everything from `PROJECT_ROOT`, which defaults to this
directory (`.../DM-ADIGen/RxRx19a`) and is overridable with `RXRX19A_ROOT`.

> `RXRX19A_ROOT` must point at **this** directory (the one holding `src/`), not  
> at the repo root — the slurm scripts `cd` into it and then run `python -m src...`.

---

## 2. Data layout

Unpack the Recursion RxRx19a download **as-is** inside this directory, so the distributed folder name nests once:

```
DM-ADIGen/RxRx19a/            <- PROJECT_ROOT (code: src/, scripts/)
└── RxRx19a/                  <- the download, verbatim
    ├── metadata.csv          (46 MB, 305,520 sites)
    ├── images/
    ├── LICENSE
    └── README.md
```

---

## 2b. What on disk still matches the current spec

`src/spec.py` holds information on (`FIELDS` → A/C/E roles, the context tensor's column order, `adjustment_set`).

## 3. Pipeline

Run in order. Each step is a SLURM launcher under `scripts/`.

### 3.0 — Build the tabular dataset  →  `data/rxrx19a_tabular/`

```bash
sbatch scripts/build_dataset.slurm
#   python -m src.data.build_dataset --check-paths
#   --limit N  builds only the first N rows (smoke test)
```

Reads `metadata.csv`, builds the compound vocab, encodes covariates, attaches the per-site 5-channel image paths, saves an HF Arrow dataset.

### 3.1 — Train/holdout split  →  `data/nuisances*/splits.json`

```bash
python -m src.data.splits --holdout-frac 0.20
```

Stratified on `(disease_condition, compound_idx)`, we get splits that`fit_urr` and `train_diffusion` see `train_idx`, while the eval component is separate.

### 3.2 — Encode VAE latents  →  `data/latents.npy`

```bash
sbatch scripts/encode_latents.slurm
#   python scripts/encode_latents.py --batch_size 32 --num_workers 12
```

Encodes all 305,520 sites once to `(N, 80, 16, 16)` fp16 with the 16-channel `ostris/vae-kl-f8-d16` VAE. Required by every `--latent 1` training arm.

### 3.3 — Nuisances: Riesz alpha, then DR weights

```bash
sbatch scripts/fit_urr.slurm          # -> data/nuisances/alpha_urr_fold{0,1}.pt
sbatch scripts/fit_knn_dr.slurm       # -> data/nuisances/dr_weights_knn.npz
#   python -m src.nuisances.fit_urr    --device cuda
#   python -m src.nuisances.fit_knn_dr --device cuda
```

`fit_urr` must run first — `fit_knn_dr` scores its cross-fitted alpha nets.
`fit_knn_dr` is only needed for the `knn_dr` arm; the `conditional` arm skips both.
For a rarity ablation: `sbatch scripts/fit_knn_dr.slurm <compound_frac> <keep_frac> <seed>`.

### 3.4 — Train the generator arms  →  `runs/<subdir>/checkpoint-NNNN/`

```bash
# sbatch scripts/train_dit_arm.slurm <dr_mode> <base_subdir> [cfrac kfrac seed]
sbatch scripts/train_dit_arm.slurm conditional dit_conditional   # comparison arm
sbatch scripts/train_dit_arm.slurm knn_dr      dit_dr        # DR arm
```

Latent DiT-B/2, adaLN-Zero conditioning on `(t, compound, dose, is_control, infected, …)`, CFG dropout 0.1, 4×A6000, 100 epochs, batch 128. Useful env vars:


| var                   | effect                                                            |
| --------------------- | ----------------------------------------------------------------- |
| `DIFFUSION_METHOD=fm` | flow matching instead of DDPM (own subdir; not resume-compatible) |
| `POS_GAMMA=1`         | confounded ablation (dose × plate-edge) instead of MCAR           |
| `TRAIN_SEED=1`        | replicate run, tagged `_ts1` (a separate output dir)              |
| `DRW=<file>`          | alternate kNN weight file (IPW / AIPW-raw ablations)              |


### 3.5 — Eval instruments (once, cached)

```bash
sbatch scripts/train_fe.slurm 20 16          # domain ResNet18 (~2.3 h/epoch)
sbatch scripts/precompute_embeddings.slurm   # embed all real rows -> feat_cache/
```

### 3.6 — Evaluate

Real-data oracle first (no generator, minutes), then each arm against it:

```bash
# oracle -> runs/eval_artifacts/rescue_panel.json
sbatch scripts/eval.slurm

# a generator arm, scored against that oracle
sbatch scripts/eval.slurm "" "" "" --source generated --dit_subdir dit_conditional \
       --truth runs/eval_artifacts/rescue_panel.json

# quick pipeline check
sbatch scripts/eval.slurm "" "" "" --smoke
```



```bash
python -m src.eval.evaluate --device cuda --num_workers 8 --source real
```

## 4. Evaluation, in more detail

`src/eval/evaluate.py` runs:

1. **EFFECTS** — per-compound treatment effect on the rescue axis, plus hits-vs-negatives separation and AUROC over the curated 16-compound panel (`--all_compounds` scores all 1,669 instead).
2. **ACCURACY** (`--truth`) — MSE / bias / Spearman of those effects against the real-data oracle. *This* is the number that says whether a generator reproduces the causal quantity.
3. **QUALITY** — FID + KID (Inception) and FID + MMD (domain ResNet18), marginal and per dose-bin, on row-matched (real, generated) pairs, plus the TRTS conditioning check.

Useful flags: 

`--encoder {domain,openphenom,inception}` 

`--pool {all,holdout,kept}`,
`--guidance_scale`, `--num_inference_steps`, `--gen_epoch`.

Output (atomic): `runs/eval_artifacts/rescue_panel*.json`.

---

## 5. Repo layout

- `src/spec.py` — the contract: treatment columns, covariates, image channels/resolution, dose grid, population restriction, and all I/O paths.
- `src/data/`
  - `build_dataset.py` — metadata/images → HF tabular dataset.
  - `splits.py` — stratified train/holdout split.
  - `dataset.py` — images/latents + tabular rows → training batches; VAE loading.
  - `rarity.py` — the scarcity / confounding ablation (`--rare-*` flags, dir tags).
- `src/nuisances/`
  - `alpha_net.py` — Riesz `alpha` architecture.
  - `fit_urr.py` — URR Riesz fitting and loss
  - `knn_dr.py` / `fit_knn_dr.py` — NN AIPW weight. Derive `dr_weights_knn.npz`, the only nuisance artifact the trainer reads.
  - `alpha_truth_check.py` — known-truth check on the estimator.
- `src/models/` — `dit.py` (DiT with adaLN-Zero), `conditioning.py` (`CondSpec`).
- `src/processes/` — `ddpm.py` , `flow_matching.py`
- `src/train/train_diffusion.py` — the trainer; writes `arch.json` next to the checkpoints so eval reconstructs the model with no flags to remember.
- `src/eval/` — `evaluate.py` (entry point), `generation.py`, `dist_metrics.py`,
`feature_extractor.py` (domain ResNet18), `openphenom_encoder.py` (+ TVN),
`prediction_transfer.py`.
- `scripts/` — SLURM launchers plus standalone analysis scripts  
(`rxrx19a_multiarm_dr.py`, `sdedit_probe.py`, `xy_effect.py`, …).

