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
src/eval/                    evaluate (gene-space ATE), generation (CFG sampler),
                             dist_metrics (MMD / Frechet on Y)
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
`runs/mcf7_24h/` also holds eight flow-matching arms (the same four, with
`--diffusion_method fm`, at `--cmean_lambda` 0 and 0.1) built for the P9
`cmean` ablation, so twelve arms are scored in Phase 4.

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

## v1 evaluation (Phase 4)

The estimand is the dose-specific ATE in gene space,
`tau_hat(c, d) = mu_hat(c, d) - mu_hat(0)`, on the plate-centred, z-scored
outcome. Arms are keyed on `(compound_idx, dose_level)`. The settings are fixed
in `IMPLEMENT.md` §3.14 (E1-E12) and are the defaults: `checkpoint-0499` EMA
weights, no guidance, 16 samples per real row, 100 sampler steps,
`--min_dose_n 2`.

**1. The real-data oracle (CPU).** `--source real` is numpy only -- it never
imports torch -- so it runs as a CPU job. It also measures the two references a
per-arm number has to be read against (E11): the DMSO **noise floor** per arm
size, and the **split-half reliability ceiling**.

```bash
python -m src.eval.evaluate --source real --pool all
# -> runs/mcf7_24h/eval_artifacts/oracle_mcf7_24h_poolall{.json,_tau.npz}
```

**2. Score one arm (GPU).** `--pool all` also reports the `holdout` sub-pool
from the same generation pass.

```bash
O=runs/mcf7_24h/eval_artifacts/oracle_mcf7_24h_poolall.json
python -m src.eval.evaluate --source generated --device cuda --truth "$O" \
    --run_dir runs/mcf7_24h/mlp-B_conditional_ddpm_s0 --pool all
```

The eight FM `cmean` arms are scored on `--pool holdout` instead (E12): mu_hat(a)
is the mean of each arm's ~2 train wells, and the `--pool all` oracle contains
those wells, so a model that memorises them is flattered there.

**Outputs**, under `<run_dir>/eval_artifacts/` (oracle:
`runs/<build>/eval_artifacts/`):

| file | contents |
|---|---|
| `<tag>.json` | aggregates, the curated-compound panel, per-compound responder status |
| `<tag>_tau.npz` | the `(K, 978)` tau matrix per sub-pool, plus per-arm scalars. `--truth` reads this |
| `<tag>_gen.npz` | per-row generated means and the sample reservoir, so metrics can be recomputed without re-sampling (`--no_save_gen` to skip) |

**Options:** `--n_per_row` (samples per real row), `--num_inference_steps`,
`--sampler {ddim,ddpm,dpm}` (DDPM arms; FM always uses the flow Euler sampler),
`--guidance_scale` (E2 keeps the headline at 1.0), `--gen_anchors` (take
mu_hat(0) from generated vehicle wells instead of real ones), `--extra_metrics`
(KID, PRDC), `--probe` (a linear TRTS probe on `dose_level`), `--pool`,
`--min_dose_n` / `--min_dose_n_holdout`.

`--truth` refuses an oracle built on another population, table, split, centring
or normalisation, and a generated run may not be written onto the oracle's path.

**Batch scripts:** `scripts/eval_oracle.sub` (CPU) and `scripts/eval_arm.sub`
(one arm per job on `zabih`; every argument is passed through).

## Step C: the semi-synthetic confounder (Phase 5)

Step C injects a known effect along a random direction `v` for the rows with
`syn_c = 1`, then thins the training wells of responder compounds on
(dose half x `syn_c`) so that a naive generator is confounded and a correctly
adjusted one is not. `syn_c` itself is already in the table from ingest.

```bash
# 1. which compounds may be thinned: responders, from the uninjected oracle
python -m src.data.responders                  # -> <nuisance_dir>/responders.json

# 2. resolve the injection ONCE (beta and v); both the trainer and the oracle read it
python -m src.data.synthetic --syn_effect 1.0   # -> <nuisance_dir>/syn_meta.json

# 3. size gamma before building anything (writes nothing)
python -m src.data.build_tiered_split --scored_compounds \
    data/mcf7_24h/nuisances/responders.json --confounder syn_c --plan

# 4. the tiered instances: gamma = 1 confounds, gamma = 0 is the MCAR control
for G in 0 1; do
  python -m src.data.build_tiered_split --scored_compounds \
      data/mcf7_24h/nuisances/responders.json --confounder syn_c \
      --gamma $G --keep_frac 0.4
done

# 5. the positivity-cell DR weights (P12; numpy only, no torch)
T=data/mcf7_24h/nuisances_tier_Csyn_c_k0_g1_s42
python -m src.nuisances.export_urr_weights --mode counts --adjustment_set syn_c --nuisance_dir $T

# 6. the step-C oracle -- a DIFFERENT oracle from v1's, tagged _syn1
python -m src.eval.evaluate --source real --pool all --syn_effect 1.0
```

The thinning seed stays at `cfg.seed`: the split layer runs inside the builder,
so another seed would draw another holdout, and §3.8.1 requires v1 and step C to
share one. The "at least 2 seeds per arm" is `train_diffusion --seed`.

**The arms** (MLP only, same split and seed; `naive` shows the bias, `dr_design`
is the known-weights reference):

| arm | `--adjustment_set` | weights |
|---|---|---|
| `naive` | `''` | conditional |
| `conditional` | `syn_c` | conditional |
| `dr` | `syn_c` | `dr_weights_counts.npz` |
| `dr_design` | `syn_c` | `dr_weights_design.npz` |

Each is `train_diffusion --nuisance_dir $T --adjustment_set syn_c --syn_effect 1.0
...`, then scored with `evaluate --source generated --syn_effect 1.0 --truth
<step-C oracle>`. A tiered arm is scored on the **unablated** pool, so eval
verifies its unthinned pool (`nu_rows.npy`) reproduces the eval split rather
than matching its thinned fingerprint.

Two offsets the injection creates are expected, reported, and not errors:
mu_hat(0) moves by `beta * E[syn_c]` ~ `beta/2` (the vehicles are injected too,
and eval subtracts the known offset before reporting the centring), and each arm
carries a `beta/6` residue because a 3-well arm splits `syn_c` 2/1. See
`IMPLEMENT.md` §5 for the measured numbers and one open decision about them.

## Checks

| check | what it covers | where to run |
|---|---|---|
| `python -m src.tests.check_build [--data_dir ...]` | the ingest | CPU |
| `python -m src.tests.check_phase1 [--data_dir ...]` | split, `nu`, `expr_meta` and the centring gate, URR gate, weights, tier | CPU |
| `python -m src.tests.smoke_phase1_torch --device cuda` | dataset, cond spec, AlphaNet | GPU |
| `python -m src.tests.check_phase4 --oracle ORACLE.json [--arm ARM.json ...]` | the eval artifacts: tau recomputed from `expr.npy`, arm counts, mu_hat(0), the noise floor against its analytic value, the reliability ceiling, responder share | CPU |
| `python -m src.tests.smoke_phase4_torch --device cuda --ckpt_dir RUN ...` | the sampler: shapes, no clamp, per-row seeding, both schedule directions, guidance forward counts, the identity guard | GPU |
| `python -m src.tests.check_phase5 [--tier DIR ...]` | step C: syn_meta, the injection recomputed, tau unchanged on the unablated pool, responder eligibility, the planned vs realised thinning | CPU |
| `python -m src.tests.smoke_phase5_torch --device cuda --tier DIR` | step C on the torch path: the injection through `LincsDataset`, its refusals, `syn_c`-as-C conditioning, the DR legs | GPU |
| `python -m src.tests.test_lincs_shapes --device cuda [--data_dir ...] [--ckpt_dir RUN ...]` | both backbones, the trainer's loss and batcher, cmean, rebuild from `arch.json` | GPU |

`scripts/phase2_gpu.sub` runs `test_lincs_shapes` and a two-epoch trainer
matrix on the smoke build, covering resume, the refusals and the rebuild
checks.

`scripts/phase4_gpu.sub` does the same for Phase 4 on the smoke build: the
sampler checks, the oracle, two generated arms, the read-back checks, and the
refusals. `scripts/phase5_cpu.sub` builds the step-C data layer, and
`scripts/phase5_gpu.sub` smokes step C end to end (it runs on the `gpu`
partition, so it may be preempted; it is idempotent, so just resubmit).
