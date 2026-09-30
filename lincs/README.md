# LINCS L1000: ADIGen

This directory runs the ADIGen experiment group of `RxRx19a/` on Phase II
LINCS L1000 (`GSE70138`). The population is MCF7 at 24 h (`mcf7_24h`). The
outcome $Y$ is the 978 landmark genes of Level 3, centred on each plate's DMSO
wells and z-scored.

- The dataset and the causal definition of $X, A, Y$ are in [`SPEC.md`](SPEC.md).
- The engineering plan, decisions, TODO and review log are in
  [`IMPLEMENT.md`](IMPLEMENT.md).

Run every command from `lincs/` as `python -m src....`. `lincs/src` is a copy
of the RxRx19a code, adapted for LINCS; it never imports `RxRx19a`.

- **Environment:** conda env `adi` (Python 3.11, torch 2.5.1+cu121). The pinned
  packages are in [`requirements.txt`](requirements.txt).
- **Where to run:** anything that imports torch runs on a GPU node. Data and
  count steps are numpy-only CPU jobs.
- **Batch scripts:** the SLURM wrappers are in `scripts/`, which is local and
  not tracked by git.

## Layout

```text
src/spec.py                  FIELDS / roles, OutcomeSpec, PopulationSpec, Paths (the frozen decisions)
src/data/                    build_dataset (GCTX -> table + expr.npy), build_tiered_split (splits),
                             expr_stats (plate centres + z-score), dataset (in-memory LincsDataset)
src/nuisances/               fit_urr (cross-fitted URR alpha), export_urr_weights (dr_weights_*.npz),
                             precompute_cmean (mu_hat for the FM cmean loss)
src/models/                  conditioning (CondSpec), layers, mlp (MLPDenoiser), dit1d (DiT1DModel),
                             __init__ (build_generator, arch.json)
src/processes/               DDPM (zero-SNR, v-prediction) and flow matching
src/train/train_diffusion.py the trainer
src/tests/                   read-back checks (numpy) and GPU smoke tests
data/<build>/                built artifacts; data/<build>/nuisances/ is the v1 split dir
runs/<build>/<run>/          arch.json, loss_history.jsonl, checkpoint-NNNN/ (model_1 = EMA), wandb/
```

## v1 pipeline

**1. Ingest (CPU).** Build the table, `expr.npy`, the gene order, plate QC, and
the encoders:

```bash
python -m src.data.build_dataset                  # -> data/mcf7_24h/
python -m src.data.build_dataset --limit 1500     # smoke build -> data/mcf7_24h_limit1500/
```

**2. Split, outcome statistics and nuisances (CPU).** The `net` weights are the
v1 primary. The `counts` weights are the fallback, and are exactly 1 in v1.

```bash
python -m src.data.build_tiered_split                                # splits.json, nu_rows.npy
python -m src.data.expr_stats                                        # expr_meta.json (+ centring gate)
python -m src.nuisances.export_urr_weights --mode counts             # dr_weights_counts.npz
python -m src.nuisances.fit_urr --nu_rows nu_rows.npy --device cpu   # alpha_urr_fold{0,1}.pt
python -m src.nuisances.export_urr_weights --mode net                # dr_weights_urr.npz
python -m src.nuisances.precompute_cmean [--min_n 8]                 # cmean.npz (FM cmean loss only)
```

**3. Train the four v1 arms (GPU).** All four use the same split and seed.

| arm | command |
|---|---|
| `mlp_conditional` | `python -m src.train.train_diffusion --arch mlp --num_epochs 500` |
| `mlp_dr` | `python -m src.train.train_diffusion --arch mlp --num_epochs 500 --dr_mode weighted` |
| `dit1d_conditional` | `python -m src.train.train_diffusion --arch dit1d --num_epochs 500` |
| `dit1d_dr` | `python -m src.train.train_diffusion --arch dit1d --num_epochs 500 --dr_mode weighted` |

The **weighted (`dr`) arm needs two or three jobs**, depending on where its
weights come from:

1. **v1, `net` weights (primary):** run `fit_urr`, then
   `export_urr_weights --mode net`, then
   `train_diffusion --dr_mode weighted`. The weights file defaults to
   `dr_weights_urr.npz`.
2. **v1, `counts` weights (fallback):** run `export_urr_weights --mode counts`,
   then `train_diffusion --dr_mode weighted --dr_weights_file dr_weights_counts.npz`.
3. **Steps C and A (thinning instances):** on the tier's split dir, run
   `export_urr_weights --mode counts --adjustment_set <C>` to get the
   positivity-cell weights. Then run the same `train_diffusion` command with
   `--nuisance_dir <tier dir> --adjustment_set <C> --dr_weights_file dr_weights_counts.npz`.
   `fit_urr` refuses a thinning instance.

The trainer checks that the weights file's `row_id` equals `train_idx`, and it
normalises `w` to mean 1.

**Trainer options:**
- `--diffusion_method fm` selects flow matching.
- `--cmean_lambda 0.1 --cmean_file cmean.npz` turns on the cmean ablation. It
  runs with FM only, an empty C and `--include_env 0`, because μ̂ is keyed on
  the action alone.
- `--invariance_lambda` turns on V-REx over plate.
- `--plate_center none` is the centring ablation. It needs the matching
  `expr_meta_none.json`.
- `--mlp_size` / `--dit_size` / `--patch_size` set the model size.
- `--data_dir data/mcf7_24h_limit1500` trains on the smoke build, into
  `runs/mcf7_24h_limit1500/`.

**Run directories.** The run dir name is derived from the settings, for example
`runs/mcf7_24h/mlp-B_weighted-urr_ddpm_s0`.
- A fresh start is refused when the dir already has checkpoints, or holds the
  `arch.json` of a different run (e.g. a concurrent launch with the same
  name). To resume, pass the same arguments plus `--resume_epoch N`; every
  identity field in `arch.json` must match, and a missing checkpoint is refused
  before anything is written.
- Eval rebuilds the model from `arch.json` alone (`build_generator_from_ckpt`).

**Tracking.** Runs log to wandb (`--wandb_project lincs-adigen`,
`--wandb_entity` defaults to `493302570`, credentials from `~/.netrc`). Use
`--wandb_mode offline` or `--wandb_mode disabled` to log locally or not at all.
`loss_history.jsonl` is written either way, and it is the record eval and the
checks read.
- **Online resume** continues the same wandb run, because the run id is kept in
  `wandb_run_id.txt`. wandb drops any step below the run's current step, so
  epochs replayed after a crash keep their first logged values. Those values
  are identical when the replay is deterministic, but not after `--reset_lr`.
- **Offline resume** starts a new local run with the same id. wandb ignores
  `resume` offline, so sync the runs separately.

## Checks

| check | what it covers | where to run |
|---|---|---|
| `python -m src.tests.check_build [--data_dir ...]` | the ingest | CPU |
| `python -m src.tests.check_phase1 [--data_dir ...]` | split, `nu`, `expr_meta` and the centring gate, URR gate, weights, tier | CPU |
| `python -m src.tests.smoke_phase1_torch --device cuda` | dataset, cond spec, AlphaNet | GPU |
| `python -m src.tests.test_lincs_shapes --device cuda [--data_dir ...] [--ckpt_dir RUN ...]` | both backbones, the trainer's loss and batcher, cmean, rebuild from `arch.json` | GPU |

`scripts/phase2_gpu.sub` runs `test_lincs_shapes` and a two-epoch trainer
matrix on the smoke build, covering resume, the refusals and the rebuild
checks.
