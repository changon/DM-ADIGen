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

Weights (`recursionpharma/OpenPhenom`) and the VAE (`ostris/vae-kl-f8-d16`) are pulled with `huggingface_hub.snapshot_download`.

---

## 2. Data layout

Folder layout looks like the following, noting repeated name

```
DM-ADIGen/RxRx19a/            <- PROJECT_ROOT (code: src/, scripts/)
└── RxRx19a/                  <- the download, verbatim
    ├── metadata.csv          (46 MB, 305,520 sites)
    ├── images/
    ├── LICENSE
    └── README.md
```

---

## 2b. specs.

`src/spec.py` holds information on (`FIELDS` → A/C/E roles, the context tensor's column order, `adjustment_set`).

## 3. Pipeline

Run in order. Each step is a SLURM launcher under `scripts/`.

### 3.0 — Build the tabular dataset  →  `data/rxrx19a_tabular/`

```bash
sbatch scripts/build_dataset.slurm
#   python -m src.data.build_dataset --check-paths --limit N  # builds only the first N rows for testing
```

Reads `metadata.csv`, builds the compound vocab, encodes covariates, attaches the per-site 5-channel image paths, saves an HF Arrow dataset.

### 3.1 — Build the the experimental setup  →  `data/nuisances_tier_k<k>_pg<γ>_s<seed>/`

```bash
python -m src.data.build_tiered_split --plan       # power table, writes nothing
python -m src.data.build_tiered_split --gamma 0    # unconfounded instance
python -m src.data.build_tiered_split --gamma 2    # plate-confounded instance
```

Above k is the number of wells kept for holdout eval.

This produces:

- **test data** (k wells per scored evaluations)
- **train+validation** (well-grouped, stratified 80/20 train/holdout)

Above, the **experiment** introduces confounding by sampling prob of keeping, with probability π ∝ exp(−γ·z_position); α = 1/π recorded as `dr_weights_design.npz`.

### 3.2 — Encode VAE latents  →  `data/latents.npy`

```bash
sbatch scripts/encode_latents.slurm
#   python scripts/encode_latents.py --batch_size 32 --num_workers 12
```

Encodes all 305,520 sites once to `(N, 80, 16, 16)` fp16 with the 16-channel `ostris/vae-kl-f8-d16` VAE.

### 3.3 — Nuisances: alpha weights and cmean targets (all into the split dir)

The **learned alpha, URR,** has two estimators:

```bash
# closed-form discrete URR: smoothed count ratio nu(a)/f_train(a). CPU.
python -m src.nuisances.export_urr_weights --nuisance_dir <dir> --mode counts \
    --out dr_weights_counts.npz          # validates against the design truth in-run

# Riesz fit, then export:
sbatch scripts/fit_urr.slurm --nuisance_dir <dir> --nu_rows <dir>/nu_rows_design.npy
python -m src.nuisances.export_urr_weights --nuisance_dir <dir> --mode net
```

We can incorporate a regularization for causal learning: add auxiliary loss encouraging that **conditional mean targets align** (`--cmean_lambda`):

```bash
sbatch scripts/cmean.slurm --nuisance_dir <dir>    # -> <dir>/cmean.npz
```

### 3.4 — Train the generator arms  →  `runs/<subdir>/checkpoint-NNNN/`

```bash
# the launcher 
NUIS=data/nuisances_tier_k2_pg0_s42 CONDMODE=xattn ENVSET=experiment INVLAM=1.0 \
EPOCHS=100 CKPT_EVERY=25 sbatch scripts/dose_enc_arm.slurm scalar full
```

The generator implied by this: latent DiT-B, patch 2, flow matching, **xattn conditioning** (role-A fields — compound, dose, is_control — enter as cross-attention tokens on top of the adaLN path `include_env=0`), scalar dose, uniform tau, V-REx over experiment.

Some useful env vars:


| var                                           | effect                                       |
| --------------------------------------------- | -------------------------------------------- |
| `NUIS=<dir>`                                  | the split dir                                |
| `DRMODE=knn_dr DRWFILE=dr_weights_design.npz` | α-weighted loss (design / counts / net file) |
| `CMEAN=3.0`                                   | conditional-mean auxiliary loss weight       |
| `CONDMODE=xattn`                              | cross-attention conditioning (vs `adaln`)    |
|                                               |                                              |


Resume is automatic on requeue; `scripts/warm_restart_arm.slurm` restarts the LR schedule from a checkpoint (`--reset_lr`).

### 3.5 — Eval setup. Caches the encoding data.

```bash
sbatch scripts/train_fe.slurm 20 16          # domain ResNet18 (~2.3 h/epoch)
sbatch scripts/precompute_embeddings.slurm   # embed all real rows -> feat_cache/
```

### 3.6 — Evaluate

The train/test pair per arm:

```bash
NUIS=data/nuisances_tier_k2_pg0_s42
ARM=dose_scalar_full_xattn_Eexperiment_lam1.0_tierk2pg0s42
T=runs/eval_artifacts/rescue_panel_openphenom_tvn-vehicle-experiment_HRCE

# true effect computations: compute the in-sample (train wells) and out-of-sample (reserve wells) 'true effects'
sbatch --array=0-0 scripts/panel_seeds.slurm --pool train   --nuisance_dir $NUIS
sbatch --array=0-0 scripts/panel_seeds.slurm --pool reserve --nuisance_dir $NUIS

# the generated pairings, compared to the true effects above.
sbatch --array=0-0 scripts/panel_seeds.slurm --pool train   --nuisance_dir $NUIS \
    --source generated --dit_subdir $ARM --gen_epoch 99 --truth ${T}_pooltrain.json
sbatch --array=0-0 scripts/panel_seeds.slurm --pool reserve --nuisance_dir $NUIS \
    --source generated --dit_subdir $ARM --gen_epoch 99 --truth ${T}_poolreserve.json
```

`--array=0-N` runs N+1 seeds (generation noise); aggregate with`scripts/panel_agg.py`. 

Latent-space scoring: `scripts/score_reserve.slurm` (reserve truth) and `scripts/rank_library.py`(all 1,669 compounds).

## 4. Evaluation, in more detail

`src/eval/evaluate.py` runs:

1. **EFFECTS** — per-compound (and per-dose) ATE on the infection axis.
2. **ACCURACY** — `vs_literature` (separation of curated actives from inactive controls) and `vs_oracle` (`--truth`): pearson /  spearman / bias / calibration slope per statistic
3. **QUALITY** — FID/KID (Inception) + FID/MMD (domain), marginal and per dose-bin, plus the classifier conditioning probe.

- `--pool {all,train,holdout,reserve}` picks the rows
- `train` vs`reserve` is the in-sample/out-of-sample pair
- `--encoder {domain,openphenom,inception}`
- `--nuisance_dir` supplies every split-dependent input. 

Output: `runs/eval_artifacts/rescue_panel*.json`; `--save_centroids` adds a `_centroids.npz` sidecar (full displacement vectors, for`scripts/vector_ate.py`).

---

## 5. Repo layout

- `src/spec.py` — the contract: FIELDS (the single role declaration, A/C/E), action/covariate/image specs, population, all I/O paths.
- `src/data/`
  - `build_dataset.py` — metadata/images → HF tabular dataset.
  - `build_tiered_split.py` — **the experiment builder**: reserve, split, tiers, design weights, ν pool → in one dir.
  - `dataset.py` — latents/images + rows → batches; cond spec + tensors (`cond_from_arrays` owns the dose→NaN-for-controls rule); VAE loading.
- `src/nuisances/`
  - `alpha_net.py` / `fit_urr.py` —  Riesz α (with `--shrink_to_one`).
  - `export_urr_weights.py` — fitted or **counts** α to an npz → `dr_weights_*.npz`.
  - `precompute_cmean.py` — per-action latent means for the τ=0 aux loss.
- `src/models/` — `dit.py` (DiT, adaLN-Zero + optional cross-attention conditioning)
- `src/processes/` — `ddpm.py`, `flow_matching.py`.
- `src/train/train_diffusion.py` — the trainer including α-weighted loss, cmean aux loss, invariance loss;  writes `arch.json` so eval reconstructs the model based on one file
- `src/eval/` — `evaluate.py` (entry point), `generation.py`, `dist_metrics.py`, `feature_extractor.py` (domain ResNet18),`openphenom_encoder.py` (+ TVN), `prediction_transfer.py`.

