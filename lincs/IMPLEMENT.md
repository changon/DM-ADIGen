# LINCS L1000 — ADIGen implementation spec

This document is the build plan for running the same **ADIGen** experiment group
that `RxRx19a/` implements, on Phase II LINCS L1000 (`GSE70138`). Dataset facts
and the causal `X, A, Y` definition live in [`SPEC.md`](SPEC.md). This file is
the engineering spec: what differs, what to copy, what to rewrite, and in what
order.

The RxRx19a tree is the **algorithmic template**, not a multi-dataset library.
`RxRx19a/src/spec.py` claims to be case-agnostic, but `FIELDS`, paths, image I/O,
and encodings are RxRx-only. LINCS gets its own package under `lincs/` rather
than growing `if dataset == ...` branches inside RxRx19a.

**Reuse rule:** when LINCS needs existing RxRx19a logic, **always copy** the
file from `RxRx19a/src/...` to a new file under `lincs/src/...`, then edit the
copy. Never `import` `RxRx19a.src`. The two trees may diverge. See §3.1.1.

Numbers quoted below as *measured* come from metadata scans of the files on
disk on 2026-09-28, plus small h5py reads of Level 3 values: the 2,084 DMSO
wells, the two proteasome controls, and 400 random arms. §5 lists what the
2026-09-28 review changed, and the decisions taken after it.

---

## 1. Goal

Reproduce the ADIGen experiment group on LINCS:

1. Estimate the **counterfactual expression law**
   $p(Y \mid do(A))$, $A = (\text{compound}, \log_{10}\text{dose})$,
   $Y \in \mathbb{R}^{978}$ landmark genes (Level 3, centred per plate on its
   DMSO wells — §3.2), on a **fixed cell line and exposure time**
   (MCF7, 24 h — §3.12).
2. Train two generator arms on the same population and split:
   - **`conditional`**: unweighted factual denoising risk.
   - **`dr`** (`--dr_mode weighted`; RxRx calls it `knn_dr`): the same net,
     loss weighted per training row by the URR Riesz weight
     $w_i = \hat\alpha(X_i, A_i)$ from `export_urr_weights` (ADIGen). Vehicle
     rows keep $w = 1$; the trainer normalises $w$ to mean 1. The kNN AIPW
     estimator (`fit_knn_dr`) is retired (§5, 2026-09-29).
3. Ablate the **denoiser backbone** with two architectures that share the
   same conditioning contract (`CondSpec` + adaLN from $t, A, C, E$):
   - **MLP** — treat $Y$ as a 978-d vector.
   - **1D-DiT** — treat $Y$ as a 1-d sequence over gene patches.
4. Evaluate whether the generator recovers the **dose-specific ATE**
   $\tau(d) = \mathbb{E}[Y(d) - Y(0)]$ (and the multi-compound analogue)
   against a real-data oracle, plus a gene-space quality check. With ~3 wells
   per (compound, dose) arm, per-arm comparisons use the full pool and are
   read in aggregate (§3.7).
5. Test whether ADIGen's DR risk removes confounding bias, in two steps after
   v1 (§3.8):
   - **Step C:** a semi-synthetic confounder on MCF7, as a sanity check with
     known ground truth.
   - **Step A:** cell line as a real confounder, on a 5-line population, for
     real-data results.

Optional later (same scaffolding): V-REx invariance over plate, pathway-score
$Y$, DDPM vs FM.

Success for v1 is not a paper-ready number. It is: one filtered population on
disk, URR weights that pass the existing ESS / tail gate, both backbones
trainable under both risks, and an eval JSON that reports $\hat\tau$ vs the
oracle.

**v1 is a null check for DR.** On the base population $C = \varnothing$ and no
role-E field reaches $\alpha$ (§3.3), so $\alpha$ depends on $A$ only.
With `export_urr_weights --mode counts` and $\nu$ = the train pool, every
weight is exactly 1, so `dr` *is* `conditional`. With `--mode net` the
normalised weights are ≈1.004 on treated wells and ≈0.942 on DMSO (analytic,
§3.3). v1's `dr` arms use the `net` weights, with `counts` as the fallback
(P1, §3.13). Expect the two arms to agree within seed noise. A DR-vs-conditional *difference* needs a confounding mechanism. The
RxRx one does not transfer, so §3.8 replaces it with steps C and A. RxRx19a is
in the same position: its base arm measured $\alpha = 1.000$, ESS 100%.

---

## 2. Gap between LINCS and RxRx19a

Same experimental *shape* (one well, one assigned $(compound, dose)$),
different **outcome object**, **files**, **population**, **plate design**, and
**estimand**.

### 2.1 Scientific / data

| | RxRx19a | LINCS L1000 Phase II (`SPEC.md`) |
|---|---|---|
| Unit | imaging **site** (well × FOV) | Level 3 **well** (`inst_id`) |
| $Y$ | 5×128×128 fluorescence (or 80×16×16 image-VAE latents) | **978 landmark genes**, log2 expression (~1–15). Centred per plate on DMSO wells (§3.2). **Drop** the 11,350 inferred genes; **do not** use Level 5 signatures |
| On disk | `metadata.csv` + PNG tree | four `.txt.gz` TSVs + GCTX/HDF5: `.gctx.gz` 12.6 GiB and the decompressed `.gctx` 15.9 GiB, both in `lincs/lincs/GSE70138/` |
| Join keys | experiment / plate / well / site | `inst_id` → GCTX column ids (**not** in `inst_info` row order); `pr_gene_id` → GCTX row ids (same order as `gene_info`); `pert_id` → `pert_info` |
| $A$ | `treatment`, `treatment_conc`; control = **empty string** | `pert_id` + `pert_dose` (µM); control = `pert_type == ctl_vehicle` (all DMSO in this slice), $A = 0$; vehicle dose is the LINCS `-666` sentinel |
| Extra axes | `disease_condition` (Active / Mock / UV), `cell_type` (HRCE / VERO), `site` | **`pert_time`** (string `"24.0"`), **`cell_id`**. Primary spec **fixes both** (MCF7, 24 h) |
| $X$ / $E$ | experiment, plate, well row/col, edge, site | thin: `det_plate` (101 plates = 33 plate maps × replicates `X1`–`X3`, batches `B17`…`B29`), `det_well` (384-well, `A01`–`P24`); no rich pretreatment |
| Plate design | treatments spread over plates | each (compound, dose) arm ≈ **3 wells**, one per replicate plate, **same well position**; 1,647 / 1,750 compounds sit on a single plate map; every plate has 18–28 DMSO wells |
| Population | HRCE ∩ Active SARS-CoV-2 | MCF7 ∩ 24 h ∩ (`trt_cp` ∪ `ctl_vehicle`): **37,707** wells (35,623 treated, 2,084 DMSO) |
| Dose grid | hardcoded `(0.003 … 3.0)` µM | 3-fold series **0.041, 0.12, 0.37, 1.11, 3.33, 10 µM**, plus **20 µM** for the proteasome plate controls (bortezomib, MG-132); 99.5% of treated wells lie within 0.05 log10 of these 7 levels. 97 raw float values (§3.12) |
| Estimand | morphological **rescue** on Mock ↔ untreated-infected in an **image embedding** | $\tau(d)=\mathbb{E}[Y(d)-Y(0)]$ in **gene (or pathway) space** |
| Eval instruments | domain ResNet18, Inception, OpenPhenom, FID/KID, TRTS, 16-compound COVID panel | **none of those**. ATE on $Y$, optional pathway scores, MMD/FID in gene space |

### 2.2 What that implies for code

The ADIGen **algorithm** (URR $\alpha$, $\alpha$-weighted denoising risk,
CFG on role-$A$, `arch.json`) does not care that $Y$ is an image. Almost
every **I/O and architecture** module in RxRx19a does:

- **Ingest** builds PNG `channel_paths`, not a gene matrix.
- **`ImageSpec` + 2D DiT + image VAE** assume `(B, C, H, W)`. 978 genes are
  not a 128×128 (or 16×16) grid. Do **not** reshape genes onto a fake square
  to reuse `DiT2DModel`.
- **`AlphaNet` and `fit_urr`** concatenate an **`infected`** bit and then
  **drop all non-infected train rows**. On LINCS that filter is meaningless
  and would empty or distort the sample. `export_urr_weights` reads the same
  bit to pick the rows it scores, and passes it into `AlphaNet` in `net`
  mode.
- **`ContextEncoder`** hardcodes `disease_condition`, `site`, well-letter
  `edge`, and lazily builds its levels from the raw `metadata.csv`. LINCS
  columns are `cell_id`, `det_plate`, `det_well`, `pert_time`.
- **`evaluate.py`** is a COVID rescue panel. There is no Mock / untreated-
  infected morphology axis. Its `--min_dose_n 20` default would keep 12 of
  ~10.4k arms: the two proteasome controls plus a few arms of three
  multi-plate compounds. A typical arm has 3 wells.
- **`generation._generate_batch`** ends in `x.clamp(-1, 1)` and feeds
  `channels_last`. Both are image-only: the clamp would truncate ~32% of
  z-scored gene values, and `channels_last` raises on a rank-2 tensor.

**Plate / batch dominates Level 3.** Among the 2,084 MCF7 24 h DMSO wells,
`det_plate` explains a median **65%** of per-gene variance (≈5% expected by
chance with 101 plates), the batch suffix alone 44%, the plate map 49%, and the
replicate index only 2% (measured). Compounds are placed by plate map, so plate
is a common cause of $A$ and $Y$ with **zero overlap**: no (compound, dose)
arm appears on every plate. It cannot enter $\alpha$ (§3.3). It is handled in
the outcome instead (§3.2).

`processes/` (DDPM / flow matching) already broadcast over `x0.ndim`, so the
noising path can stay. The trainer's per-sample MSE currently does
`.mean(dim=(1, 2, 3))`, in the training loop **and** in `_validation_loss`, and
both must become rank-agnostic.

### 2.3 What is *not* a gap

Keep these contracts identical so nuisances and the trainer compose:

- $A =$ categorical compound + continuous log-dose + `is_control`; vehicle
  dose is **NaN / learned null**, never `0.0` (0 can be a real dose) and
  never the raw `-666`.
- One `FIELDS` declaration with roles $A / C / E$ and `adjustable`;
  `adjustment_set` is shared by `fit_urr`, `export_urr_weights`, and the
  generator.
- Tabular HF dataset with `compound_idx`, `log10_conc`, `is_control`,
  `cov_vec`; nuisances never need $Y$.
- `dr_weights_<source>.npz` `{row_id, w}` aligned to `train_idx`, with
  source `counts` / `urr` (net) from `export_urr_weights`, or `design`
  (known thinning weights, §3.8). The trainer picks one with
  `--dr_weights_file`.
- Denoiser call `model(sample, timestep, cond, drop=None) -> .sample`.
- Two risks: `--dr_mode {conditional, weighted}` (RxRx: `knn_dr`; renamed,
  P7).

---

## 3. Implementation plan

### 3.1 Package layout

New tree, parallel to RxRx19a. The working-tree `.gitignore` (change not yet
committed) un-ignores `lincs/src/**/*.py`, `lincs/*.md` and
`lincs/requirements.txt`. Everything else under `lincs/` (data, runs,
`scripts/`) stays local, as in RxRx19a.

```text
lincs/                    # LINCS_ROOT: holds src/, as RxRx19a/ is RXRX19A_ROOT
  SPEC.md                 # dataset + causal definition (exists)
  IMPLEMENT.md            # this file
  README.md               # v1 pipeline commands (§3.6)
  requirements.txt        # exists: RxRx pins + h5py (cmapPy / tables optional) + wandb
  scripts/download.sh     # exists (untracked); run from lincs/
  lincs/GSE70138/         # the download, nested like RxRx19a/RxRx19a/ (gitignored)
  data/                   # built artifacts (gitignored)
  runs/                   # checkpoints + eval JSON (gitignored)
  src/
    __init__.py           # + one per subpackage (python -m src....)
    spec.py               # LINCS CaseConfig / FIELDS / OutcomeSpec / paths
    data/
      build_dataset.py    # GCTX → HF table + expr.npy; plate QC; syn_c; ContextEncoder from the FILTERED table
      splits.py           # copy: load_splits + stratification engine; cache key = population + table fingerprint
      build_tiered_split.py  # rewrite of RxRx's: reserve (k=0 default) / split / thinning -> splits.json, dr_weights_design.npz
      expr_stats.py       # NEW: plate DMSO centres + per-gene stats from TRAIN rows
      synthetic.py        # NEW: syn_c assignment + injected effect (step C, §3.8)
      dataset.py          # in-memory vector loader + copied build_cond_spec / cond_from_* / dose_probe
      build_nu_rows.py    # nu pool = unthinned train rows minus reserve (v1: the train rows)
    nuisances/
      alpha_net.py        # drop infected input
      fit_urr.py          # drop infected==1 restriction, --cell_type, experiment reads
      knn_dr.py           # copy: fit_urr / export import ess, tail_index (knn_dr_weights unused)
      export_urr_weights.py  # copy: net (primary) | counts (fallback) -> dr_weights_{urr,counts}.npz; no infected
      precompute_cmean.py # copy, optional (P9): per-arm train means of y for the FM cmean loss
    models/
      conditioning.py     # copy as-is
      layers.py           # NEW (Phase 2): TimestepEmbedder / modulate copied from dit.py, vec_modulate, CFG-drop helper
      mlp.py              # NEW vector denoiser
      dit1d.py            # NEW 1-d DiT
      __init__.py         # build_generator / build_generator_from_ckpt dispatch on arch
    processes/            # copy __init__.py + ddpm.py + flow_matching.py
    train/
      train_diffusion.py  # adapt: y rank, --arch {mlp,dit1d}
    eval/
      evaluate.py         # NEW gene-space ATE: oracle, arms, --truth, floor + ceiling
      generation.py       # sample vectors, not images; no clamp; streaming driver
      dist_metrics.py     # MMD / Frechet on Y (no Inception, no torchvision)
    tests/
      test_lincs_shapes.py  # GPU-box smoke test; under src/ so git tracks it
      check_build.py, check_phase1.py, check_phase4.py       # numpy read-back checks
      smoke_phase1_torch.py, smoke_phase4_torch.py           # GPU smoke tests
```

Python entry points from day one (`python -m src....`, run from `lincs/`); do
not depend on the missing RxRx `scripts/*.slurm` until a LINCS launcher is
written. Copied modules keep their `from src....` imports unchanged: run from
`lincs/`, `src` resolves to `lincs/src`.

`LINCS_ROOT` (default: `lincs/`, the directory holding `src/`) replaces
`RXRX19A_ROOT`. The download lives at `LINCS_ROOT/lincs/GSE70138/` because
`scripts/download.sh` was run from `lincs/`. That mirrors RxRx's nested
`RxRx19a/RxRx19a/`. `SPEC.md` §1 assumes the repo root instead, so the two
disagree about where the files land.

#### 3.1.1 Rule of thumb: always copy reused code

**ALWAYS copy** reused code as a **new file** from `RxRx19a/src` to
`lincs/src`. The two packages may evolve independently (different `FIELDS`,
outcome rank, population filters, eval). Sharing a module at runtime would
couple them.

Concretely:

- Allowed: `cp RxRx19a/src/processes/ddpm.py lincs/src/processes/ddpm.py`
  (then adapt imports / comments if needed).
- Allowed: copy `conditioning.py`, `knn_dr.py`, trainer loop, etc., even
  when the first LINCS version is byte-identical.
- Forbidden: `from RxRx19a.src...` or putting `RxRx19a` on `sys.path` so
  LINCS can import it.
- Forbidden: a shared `common/` package that both trees import, unless a
  later decision explicitly extracts one.
- After the copy, LINCS edits stay in `lincs/src`. Do not keep the files
  in sync by patching RxRx19a “for both.”

Section 3.9’s “copy” column means this procedure, not a live dependency.

> **Aside — known bug in RxRx19a (recorded only; RxRx19a is not maintained
> from this plan, so it is not patched here).**
> `RxRx19a/src/nuisances/export_urr_weights.py:90` scores fold-f rows with
> `nets[1 - f]`. `fit_urr` saves `alpha_urr_fold{f}.pt` as the net fit on the
> rows *not* in fold f, i.e. the net meant to score fold f; `fold{1-f}.pt` was
> fit on fold f itself. The exported `--mode net` weights are therefore
> in-sample, not cross-fitted, and `fit_urr`'s gate (computed out-of-fold)
> certifies different weights from the ones exported. It is invisible when α
> is near-constant and matters once α varies with (X, A). The fix is
> `net = nets[f]` (or read `eval_model_for_fold[str(f)]` from the meta).
> The LINCS copy is fixed and checks each fold's exported mean against
> `fit_urr`'s out-of-fold mean (§5, Phase 1 implementation and review).

### 3.2 Data pipeline (rewrite ingest; keep table schema)

**Download** as in `SPEC.md` (done: all five files plus the decompressed
`.gctx` are on disk). Read the GCTX with **raw h5py**. cmapPy is optional.
Measured layout:

- `/0/DATA/0/matrix` is float32 `(345976, 12328)` = **(wells, genes)**,
  contiguous, uncompressed. With h5py a well is a row: **no transpose**.
  (cmapPy's `parse` returns a genes × wells DataFrame; transpose only on that
  path.)
- `/0/META/COL/id` (`inst_id`) is **not** in `inst_info` row order (same set,
  different order). Map `inst_id` → column position; never align by position.
- `/0/META/ROW/id` equals `gene_info.pr_gene_id` in order. The 978 landmarks
  are scattered (rows 0, 1, 25, 43, …), not a block.
- Read the retained wells as **sorted** row indices in blocks of full rows
  (49 KB each, ~1.9 GB for 37,707 wells). Slice the landmark columns in
  NumPy, then reorder to table order. Avoid h5py fancy indexing on both
  axes of a contiguous dataset.
- Ingest is a CPU job (cluster, not the dev box).

**Filters (v1 population)** — applied in `build_dataset`, recorded in
`PopulationSpec` so splits/nuisances/eval cannot drift:

| Filter | Rule |
|---|---|
| Perturbation type | `pert_type ∈ {trt_cp, ctl_vehicle}` |
| Time | `float(pert_time) == 24` (the column is the string `"24.0"`; compare numerically) |
| Cell line | `cell_id == "MCF7"` (frozen, §3.12) |
| Genes | `gene_info.pr_is_lm == 1` → 978 columns, **fixed order** persisted in `gene_order.json` |
| Plate QC | drop plates whose DMSO spread > 3× the population median (below) |
| Outcome | Level 3 only. No Level 5. Plate-centred downstream (below) |

MCF7 has 1,750 compounds × ~6 doses ≈ 10.9k arms, the same scale as RxRx's
10,078 arms (~9% ESS). If ESS collapses, restrict `--population_compounds`
the same way RxRx does: it **redefines the estimand** and must be reported.

**Control and action encoding**

- `is_control = 1` iff `pert_type == ctl_vehicle` (not empty `pert_iname`).
  All 2,084 vehicles in this slice are DMSO.
- `compound_idx = 0` for all vehicles (`__control__`). Build the vocab on
  **`pert_id`** (1,750 compounds), not `pert_iname`: 17 inames map to more
  than one `pert_id`. Keep `pert_iname` as a display label.
- LINCS writes `-666` for "not applicable" (vehicle `pert_dose` and
  `pert_dose_unit`). Map controls explicitly: `conc = 0`, `log10_conc` =
  sentinel (`-10` as RxRx), generator sees dose **NaN**. Never let `-666`
  reach `conc` or `log10`.
- All treated doses in this slice are `um`. Assert that rather than convert.
- `log10_conc` stays continuous (nuisance and model input). Add `dose_level`:
  the nearest of the 7 nominal levels (0.0412, 0.1235, 0.3704, 1.1111,
  3.3333, 10, 20 µM) when within 0.05 log10, else the raw dose rounded to
  4 d.p. (173 wells, 13 compounds). `dose_level` is for strata and eval
  grouping only. Do not copy RxRx's grid.

**HF tabular columns** (one row per retained well) — same names the
nuisance fitters already read, plus provenance:

```text
row_id, inst_id, gctx_col, cell_id, pert_id, pert_iname, pert_mfc_id,
pert_type, pert_dose, pert_time,
det_plate, det_well, plate_map, replicate, batch,
compound_idx, is_control, conc, log10_conc, dose_level, syn_c, cov_vec
```

`syn_c` is the step-C synthetic covariate (§3.8), assigned here so that every
consumer sees one fixed assignment. It is role `None` and unused outside
step C.

`plate_map`, `replicate`, `batch` are parsed from `det_plate`
(`LJP005_MCF7_24H_X1_B17` → `LJP005`, `X1`, `B17`; a few plates carry `X4`
or `X3.A2`).

Store $Y$ **out of band**: `data/expr.npy` `(N, 978)` **float32** (147 MB;
float16 would quantise log2 values at ~0.008), raw Level 3, aligned to table
row order, plus `gene_order.json`. This is the analogue of RxRx `latents.npy`,
not of PNG paths. **No normalisation statistics here**: they must come from
the training split, which does not exist yet at build time.

**Plate QC (part of decision 6).** In `build_dataset`, compute each plate's
DMSO spread: the per-gene robust SD (1.4826 · MAD over the plate's DMSO wells),
then the median over genes. Drop plates whose spread exceeds
**3× the median over plates**. Record the dropped plates and all spreads in
`population_qc.json`.

- The rule uses control outcomes only and is fixed before splits, so it is a
  population rule, not a fitted statistic.
- On MCF7 24 h, spreads run 0.14–0.53 (median 0.21), with one outlier at
  1.16. The rule drops only `REP.A002_MCF7_24H_X2_B29`: 367 wells; no arm
  loses all its wells, and 37 arms fall to 1 well.
- A tighter rule (median + 5 × MAD over plates) would drop 3 plates and leave
  333 arms with 1 well. It was not chosen.

**Plate centring (decision 6, accepted 2026-09-28).** After `splits`,
`expr_stats` writes `expr_meta.json`:

1. per-plate centre: per-gene median of that plate's **train** DMSO wells
   (18–28 DMSO per plate, ~14–22 after holdout);
2. per-gene **mean of the centred train DMSO wells** and per-gene **std of
   all centred train rows** (decision 5, amended 2026-09-29), plus each
   gene's fraction of values at the Level 3 cap of 15.0 (GAPDH: 41.5%; kept,
   and reported).

`dataset.py` applies `y = (x − centre[plate] − mean) / std`, then the step-C
injection if active (§3.8). Because the mean comes from vehicle wells,
z = 0 is the vehicle in every gene. A mean over all train rows would put
DMSO a median 0.09 SD (max 0.40) off zero, 34% of it driven by the
proteasome arms (measured).

Rationale: the plate explains 65% of DMSO variance, and each compound's wells
sit on its own plates. Uncentred, $\mathbb{E}[Y \mid a]$ carries the batch
offset of $a$'s plates. The generator (which does not see plate) learns that
offset, and so does the oracle. Under an additive plate effect, subtracting a
plate constant estimated from vehicle wells only removes the confounding
without touching treated outcomes, and $\tau$ keeps its meaning. It is the
Level-4 idea done on Level 3 with vehicle wells.

Measured evidence:

- **400 random arms:** the median naive $\|\hat\tau\|$ is 26.6, of which the
  plate offset is 24.3. After centring it is 13.0, against a null-arm noise
  floor of ≈10. The offset exceeds the effect in 93% of arms.
- **Held-out DMSO wells:** the plate's variance share goes from 0.67 to 0.20
  (chance 0.10). The remainder matches the noise of centres estimated from
  ~10 wells.
- **Additivity check** on bortezomib and MG-132 (84 plates each): the plate's
  share goes from 0.65 to 0.33 (chance 0.08).
  - The residual is not batch-level: batch explains 0.09 after centring.
  - The response direction is stable across plates: the median per-plate
    correlation with the overall response is 0.91.
  - Amplitude varies by about 5–20% between plates.

  So additivity holds roughly, with a modest plate × drug interaction for
  strong compounds. That interaction remains as noise.
- **Scaling variants do worse** on the same test: 0.39 with a per-plate
  scalar, 0.44 with a per-gene robust z-score against DMSO. Centring only.

Consequences to keep in mind:

- `Y` is expression relative to same-plate DMSO. Generated samples are on
  that scale.
- $\hat\mu(0) \approx 0$ by construction, so the generated-DMSO check tests
  only the noise around 0.
- Every function of the plate (batch, plate map, replicate) loses its effect
  on `Y`. That is why §3.8 uses a non-plate confounder.

Keep `plate_center="none"` as an ablation.

**`cov_vec` / context** — derive from metadata, not from $Y$:

- `cov_vec` carries one-hot columns for the two adjustable fields, `syn_c`
  and `cell_id` (§3.3). Both are role `None`, and v1 has $C = \varnothing$,
  so v1's $\alpha$ uses none of them and conditions on $A$ only. Steps C and
  A promote one of them to $C$ with `--adjustment_set` (§3.8).
- `plate` is carried in the context tensor (role E, invariance-only);
  `well_row`, `well_col`, `pert_time` are role `None`: present in the table,
  not in the generator.
- Persist `compound_vocab.json` and `covariate_encoder.json` as RxRx does.
  RxRx builds `context_encoder.json` **lazily** from the raw metadata CSV.
  LINCS must build it **eagerly in `build_dataset` from the filtered table**,
  and `ContextEncoder.load_or_build` must read the table, not `inst_info`.
  `inst_info` holds 346k wells over 98 cell lines and would give the wrong
  plate levels.
- Write `nuisance_meta.json` `{n_compounds, cov_dim}` here too (§3.5).

**Splits** — the split layer of `build_tiered_split` (P4, §3.8.1). v1 is
the instance with no scored compounds and no thinning; `load_splits` and the
stratification engine are copied from `splits.py`.

- The table is already filtered, so `population_filters` becomes a **cache
  key** `(cell_id, pert_time, pert_types, compounds-hash)` plus the table
  fingerprint from `nuisance_meta.json` (a rebuilt table invalidates the
  split), asserted against the table, **not** an equality mask. RxRx masks
  `str(ds[col]) == want`, which fails on a set-valued `pert_types` and on
  `"24.0" != "24"`: the population is silently empty.
- `population.compounds` reads `ds["treatment"]` → read `pert_id`.
- Stratum: `(compound_idx, dose_level)`, i.e. the arm. With ~3 wells per arm
  `round(0.2 · 3) = 1`, so each 3-well arm keeps 2 train wells and puts 1 in
  holdout. Measured on the built table, the **realised holdout is 28.5%, not
  20%** (and not ⅓): `round(0.2 · 2) = 0`, so the 692 two-well arms stay
  whole, and 725 arms have no holdout well. Log it (P3, §3.13). Stratifying
  on compound alone (18 wells → 4 held out) would leave ~0.6% of arms with
  no training well (measured: 65).
- DMSO wells are stratified **by plate** (P3): as one stratum they leave as
  few as 12 train DMSO wells on a plate, by plate 14. Those wells set the
  plate centres.

### 3.3 `spec.py` (rewrite declarations; copy machinery)

Copy the dataclasses and helpers (`DomainField`, `role_tag`,
`alpha_cov_fields`, `add_adjustment_set_cli`, `config_from_args`,
`CaseConfig.__post_init__` guards). Replace the **declarations**:

```text
A  compound   (cat, pert_id vocab)
   is_control (cat, card=2)
   dose       (cont, nullable, source=log10_conc)

E  plate      (det_plate, adjustable=False)   # invariance-only: V-REx, --include_env ablation
—  cell_id    role=None, adjustable=True       # constant on MCF7; C in step A (§3.8)
—  syn_c      role=None, adjustable=True,
              levels=("0", "1")                # synthetic; C in step C (§3.8)
—  well_row, well_col  (from det_well)  role=None   # aliased with the arm
—  pert_time                            role=None   # population constant
—  edge       not declared                           # 72 / 10,944 arms span edge and interior
```

`cell_id` and `syn_c` are role `None` by default, so no v1 arm uses them. They
are `adjustable=True` so that `--adjustment_set cell_id` or
`--adjustment_set syn_c` can promote them to $C$ for the confounding arms. That
is the same "roles are per-arm, columns are per-build" pattern RxRx uses for
`edge`. `syn_c`'s levels are declared because they are fixed by definition.

Why these roles (all measured):

- **`plate` must not be `adjustable`.** `ALPHA_ENV` / `alpha_cov_fields` feed
  $\alpha$ every field that is role E **and** adjustable. `plate(E,
  adjustable=True)` would therefore put 101 plate columns into $\alpha$ on
  every arm, v1 included. `fit_urr --target_support common` then needs arms
  present on every plate. There are 0, so it raises "target support is
  empty". The same holds for batch (9 levels) and plate map (33). **Plate-as-C
  is not an available ablation on this population.**
- **`well_row` / `well_col` must not be role E.** Within a plate map every
  arm sits at one fixed well (11,176 of 11,233 plate-map × arm pairs), so
  position is aliased with $A$. `invariance_env_fields` also keys V-REx on
  *all* role-E fields: plate|row|col is a single well, one row per env.

**v1 `adjustment_set = ()`.** The reason is not design balance: plate is a
strong confounder. It is out of $C$ because it has no overlap, and it is
removed from $Y$ by plate centring (§3.2). Consequences:

- $\alpha(x, a) = \nu(a) / P(a)$. `fit_urr` draws $\nu$ from **treated rows
  only**, so $\alpha \approx 1/(1-\pi_0) \approx 1.06$ on treated wells and
  $\to 0$ on DMSO. That clears the `learned` gate (beats the constant by
  ~0.06, std ~0.24) without reflecting any $X$. Read the gate as a pipeline
  check here.
- `export_urr_weights` scores treated rows only and gives vehicle rows
  $w = 1$ by policy, so the vehicle arm keeps its weight and the generator
  still learns $Y(0)$ (the old concern that raw $\alpha \to 0$ zeroes the
  vehicle arm does not arise). After the trainer's mean-1 normalisation the
  `net` weights are ≈1.004 (treated) and ≈0.942 (DMSO); `counts` weights
  with $\nu$ = the train pool are exactly 1 (§1).

Log `format_role_summary` at every job start. It should report
`E=['plate']` as INVARIANCE-ONLY.

Replace `ImageSpec` with **`OutcomeSpec`**:

```text
n_genes: 978
expr_path: data/expr.npy             # raw Level 3, float32
plate_qc_max_spread_ratio: 3.0       # drop plates with DMSO spread > 3× median — decision 6
plate_center: "dmso_median_train"    # | "none" (ablation) — decision 6
normalize_mean: "train_dmso"         # per-gene mean of centred TRAIN DMSO wells — decision 5 (amended)
normalize_std: "train_all"           # per-gene std of all centred TRAIN rows — decision 5
syn_effect: 0.0                      # step-C injected effect size; 0 = off (§3.8)
```

Samples live in z-space. Invert (`x = y·std + mean + centre[plate]`) only
where a raw-scale output is needed, and never clamp. No `latent_vae`, no
channel suffixes. Optional later: `n_pathway` if $Y$ is scores rather than
genes.

`Paths`: the four TSVs and the `.gctx` under
`PROJECT_ROOT / "lincs" / "GSE70138"`, `data/lincs_tabular`,
`data/nuisances`, `runs/`.

### 3.4 Models — MLP and 1D-DiT (both required)

**Shared interface** (so trainer, CFG, and eval do not branch). The trainer
and `generation.py` touch more than `forward`, so both backbones implement
all of this:

```text
forward(sample, timestep, cond, drop=None, return_dict=True)
  sample    MLP: (B, G)           1D-DiT: (B, G) internally reshaped
  timestep  int / float / (B,) — float under FM (t = tau · N)
  cond      {field: tensor} from CondSpec (copy CondEmbedder)
  drop      (B,) bool, nulls every role-A field; in train mode with
            drop=None and class_dropout_prob > 0 the model draws its own mask
  out       .sample, same shape as `sample`
model.cond_spec                        # generation reads it
model.class_dropout_prob               # generation checks it for CFG
model.calibrate_conditioning(probes)   # trainer calls once (dose probes)
model.enable_gradient_checkpointing()  # trainer flag; may be a no-op for MLP
```

Copy `RxRx19a/src/models/conditioning.py` unchanged (`Field`, `CondSpec`,
`CondEmbedder`, Fourier features, learned nulls). Copy `TimestepEmbedder`,
`DiTBlock` (adaLN-Zero) and `get_1d_sincos_pos_embed_from_grid` from
`dit.py`.

`arch.json` must record `arch ∈ {mlp, dit1d}` plus size kwargs and
`n_genes`. Eval rebuilds from this file only (`build_generator_from_ckpt`).
The RxRx version rejects `arch != "dit"` and requires `n_channels` /
`resolution`, so rewrite it to dispatch on `arch`. A 2D DiT checkpoint is
invalid here.

#### 3.4.1 MLP denoiser (`src/models/mlp.py`)

Vector-valued score/velocity network, adaLN from the **same** cond vector
the DiT uses:

- Input: concatenate or FiLM `sample` $(B, G)$ with
  $c = t\_emb + CondEmbedder(cond)$.
- $L$ residual blocks: `LayerNorm → Linear → SiLU → Linear`, with
  adaLN-Zero from $c$ (zero-init last linear, identity at start — same
  spirit as DiT).
- Output: linear to $G$.
- Sizes (start here, tune after a smoke run):

  | name | hidden | depth | ~params (order) |
  |---|---|---|---|
  | S | 512 | 4 | ~few M |
  | B | 1024 | 6 | ~tens of M (~35M with adaLN) |
  | L | 2048 | 8 | larger |

Default train arm: **MLP-B**. 978-d is small; an MLP is the honest
baseline and the cheaper DR replicate.

#### 3.4.2 1D-DiT (`src/models/dit1d.py`)

Same blocks as RxRx `DiT2DModel`, but:

- **Patch embed**: `Conv1d` / linear on patches of `patch_size` genes
  (`in_channels=1`, length $G$). `patch_size=6` divides 978 exactly
  (163 tokens, no pad). `patch_size=10` pads 978 → 980 (98 tokens). Persist
  pad + gene order in `arch.json`.
- **Positional embedding**: 1-d sin-cos (MAE 1-d, copied helper), frozen.
- **Unpatchify** → `(B, G)` (slice off pad).
- Sizes: reuse `DIT_SIZES`. **Default S/10** (~33M params, comparable to
  MLP-B), with B/10 (~130M) as a size ablation. With ~25k training wells,
  DiT-B has ~4× MLP-B's parameters for the same data. Do not start at L/XL.

Do **not** wrap 978 into `(B, 1, H, W)` to call `DiT2DModel`.

#### 3.4.3 Not in v1

- Image VAE (`ostris/vae-kl-f8-d16`), `--latent 1`, OpenPhenom, ResNet18.
- Optional later: a **gene autoencoder** (978 → $z$ → 978) with the MLP
  or 1D-DiT on $z$. That is a third arm, not a substitute for the
  MLP vs 1D-DiT ablation.

### 3.5 Nuisances (copy algorithm; strip RxRx population bits)

| Module | Action |
|---|---|
| `knn_dr.py` | **Copy** as a new file: `fit_urr` and `export_urr_weights` import `ess` / `tail_index` from it. `knn_dr_weights` (kNN AIPW) is not used. |
| `export_urr_weights.py` | **Copy.** Drop `disease_condition` / `inf`: score treated rows, vehicle rows keep $w = 1$. `--mode net` is the primary source in v1 (P1); `--mode counts` is the fallback there: the count ratio $n_\nu(a)/n_\text{train}(a)$; key actions on the `dose_level` arm, not RxRx's `round(log10_conc·1000)` (10,944 vs 10,479 keys: float variants split arms), and on (arm, C) once $C \neq \varnothing$ (P2). In a thinning instance (steps C / A) `counts` is the source, keyed on the design's positivity cell (P12), and `--mode net` is refused. `--mode net`: the cross-fitted AlphaNet from `fit_urr`. Writes `dr_weights_{counts,urr}.npz` `{row_id, w}`; checks against `dr_weights_design.npz` when it exists. Import torch lazily so `counts` runs as a CPU job without it. |
| `build_nu_rows.py` | **Rewrite.** $\nu$ = the unthinned train pool minus reserve wells, written as `nu_rows.npy` for `fit_urr --nu_rows` (v1), `export --mode counts` and `expr_stats`. `build_tiered_split` calls it. v1 has no thinning, so $\nu$ = the train rows. |
| `alpha_net.py` | **Rewrite input.** Drop `infected`. `in_dim = cov_idx + embed + log10_conc + is_control` (the `+3` becomes `+2`). Keep softplus head / `SP_SHIFT`. |
| `fit_urr.py` | **Copy URR loss** $L = \mathbb{E}[\alpha(X,A)^2] - 2\mathbb{E}[\alpha(X,A_t)]$. **Delete** the `infected==1` train restriction, the `disease_condition` array, `--cell_type {HRCE,VERO}` and the `experiment` / `cell_type` reads. Take control ids from `__control__` (RxRx compares vocab keys to the empty `control_token`, which never matches). Population mask = filters already applied at build (optionally `--population_compounds`). Cross-fit folds **stratified on the `dose_level` arm** (P5) instead of RxRx's random halves; `--nu_rows`; common support and $\nu$ keyed on the `dose_level` arm, not `round(log10_conc, 6)` (P2); ESS/tail gate; `nuisance_meta.json`. |
| `alpha_truth_check.py`, `fit_knn_dr.py` | **Not ported.** Removed from RxRx in `78d4284`; kNN AIPW retired 2026-09-29. |
| `precompute_cmean.py` | **Copy, optional** (P9). Means of normalised `y` (not VAE latents) over TRAIN rows, keyed on the `dose_level` arm (vehicles one key); `--min_n` configurable (RxRx default 8 keeps 34 of 10,479 arms; coverage logged); stores `train_idx`, which the trainer checks. Feeds the FM-only `--cmean_lambda` loss (default 0 = off). |

**Arms are tiny.** With ~2 train wells per arm, an (arm, C) cell holds ~1
train well, too few to estimate a thinned weight per arm or to cross-fit a
net at that key (P12): steps C / A estimate the weights at the design's
positivity cell instead. In v1, $\nu$ = the train pool and the ratio is
exactly 1. `net` is cross-fitted:
a row is scored by the net fit on the other fold. With random folds, 35% of
treated train rows have no well of their own arm in the other fold (their
compound is always there, measured); arm-stratified folds fix that (P5).
Arms are keyed on `dose_level` everywhere: `fit_urr` support, fold strata,
v1 `counts` keys, and the tiered split's scored arms (RxRx rounds log-dose to
3 d.p.). Float variants (0.37 vs 0.3704 µM) would otherwise split arms,
10,944 vs 10,479 (P2).

`--adjustment_set` / `role_tag` directories stay. The trainer reads
`n_compounds` from `nuisance_meta.json`; `build_dataset` now writes it
(Phase 0), so the `conditional` arm no longer depends on `fit_urr`. A
tiered-split dir (§3.8.1) copies `nuisance_meta.json`, `compound_vocab.json`
and `covariate_encoder.json` from the base dir, as RxRx's
`build_tiered_split` does.

### 3.6 Training (adapt one file)

Copy `train_diffusion.py` structure: Accelerate, EMA, `arch.json`, DR
weights aligned to `train_idx`, optional V-REx over plate.

Edits:

- Batch key `y` (or keep `"image"` as an alias — prefer `y`).
- `pred = model(noisy, t, cond).sample` with `sample.shape == y.shape`.
- Per-sample loss, in the training loop **and** `_validation_loss`:
  `F.mse_loss(..., reduction="none").flatten(1).mean(1)`
  (rank-agnostic; works for `(B, G)` and any future latent).
- Dataset: hold `y` in memory (37,707 × 978 float32 ≈ 147 MB, centred and
  normalised once). Do not go through per-row HF `ds[idx]` access.
- `--arch {mlp,dit1d}` (required), `--dit_size` / `--mlp_size`,
  `--patch_size` (1D-DiT only).
- Drop `--latent`, VAE loading, `n_channels` / `resolution` as image
  concepts. Record `n_genes`, `arch`, `patch_size`, `plate_center` in
  `arch.json`.
- V-REx: rewrite the env-key block. It re-encodes RxRx columns (`cell_type`,
  `experiment`, `well`, `site`, `disease_condition`). The env is
  `spec.invariance_env_fields`, i.e. `plate` (~375 wells per plate, ~250 in
  train), set with `--environment_set`; RxRx's free-form `--invariance_env`
  is dropped (Phase 2).
- Re-tune batch size / epochs: RxRx defaults (`--train_batch_size 8`, epochs
  over 305k sites) do not transfer to ~25k train rows.
- `--dr_mode {conditional, weighted}`, renamed from RxRx's `knn_dr` (P7), and
  `--dr_weights_file` (`dr_weights_urr.npz` from `net`, the default for
  `weighted`; `dr_weights_counts.npz`; or `dr_weights_design.npz`). The
  trainer checks `row_id == train_idx` and normalises $w$ to mean 1.
- `--cmean_lambda` (default 0 = off) and `--cmean_file`, copied (P9): the
  $\tau = 0$ conditional-mean auxiliary loss, **FM only**. Record both in
  `arch.json`.
- `--diffusion_method {ddpm,fm}` unchanged (`FlowMatching.add_noise`
  already handles `ndim`).
- CFG dropout default 0.1 on role $A$.

**v1 train matrix** (same data, same split, same seed):

| arm id | `--arch` | `--dr_mode` | notes |
|---|---|---|---|
| `mlp_conditional` | mlp | conditional | baseline |
| `mlp_dr` | mlp | weighted | ADIGen, `net` weights (≈ baseline on v1, §1; P1) |
| `dit1d_conditional` | dit1d | conditional | architecture ablation |
| `dit1d_dr` | dit1d | weighted | ADIGen × 1D-DiT |

Plus DDPM vs FM only after the four arms above train, and the `cmean`
ablation (P9: `--cmean_lambda` 0 vs > 0, FM arms only). Do not cross
`--latent` or OpenPhenom into this matrix.

The weighted arm is two or three jobs, not one command:

- v1, `net` (primary): `fit_urr --nu_rows …` → `export_urr_weights --mode net`
  → `train_diffusion --dr_mode weighted --dr_weights_file dr_weights_urr.npz`;
- v1, `counts` (fallback): `export_urr_weights --mode counts` (CPU) →
  `train_diffusion --dr_mode weighted --dr_weights_file dr_weights_counts.npz`;
- steps C / A (P12): `export_urr_weights --mode counts --adjustment_set C`
  (CPU, positivity-cell weights) → the same `train_diffusion` command with
  `dr_weights_counts.npz`. `fit_urr` refuses a thinning instance.

Document that in the LINCS README.
Write Python `-m` commands; SLURM wrappers later.

### 3.7 Evaluation (rewrite)

Replace the rescue-panel / FID-on-pixels suite. Everything below is on the
plate-centred $Y$ (§3.2).

**Estimand (gene space)**

- $\hat\mu(0)$: mean $Y$ of DMSO wells (≈0 after centring; report it as a
  check).
- $\hat\mu(c, d)$: mean $Y$ of wells with compound $c$ at `dose_level` $d$.
- $\hat\tau(c, d) = \hat\mu(c, d) - \hat\mu(0) \in \mathbb{R}^{978}$.

**Pool and sample size.** An arm has ~3 wells, and the holdout holds at most
one of them. So:

- **Per-arm oracle:** `--pool all` (also RxRx's default) with
  `--min_dose_n` ≤ 3. RxRx's default of 20 keeps 12 of ~10.4k arms.
  One arm's $\hat\tau$ is noisy, so read per-arm results **in aggregate**:
  gene-wise cosine / Pearson of $\hat\tau_\text{gen}$ vs
  $\hat\tau_\text{oracle}$ per arm, then the median over arms; Spearman of
  $\|\tau\|_2$ across arms; pooled MSE / bias.
- **Out-of-sample:** on `--pool holdout`, only aggregates over arms: per
  dose level ($\tau(d)$ averaged over compounds) and per compound (pooled
  over doses).
- **High-n anchors:** bortezomib and MG-132 (~980 wells each, 20 µM, 84
  plates) are the only arms where a single-arm oracle is tight. The next tier
  is ~20-well arms of PD-0325901, dasatinib, and GSK-1059615.

**Reference scale (E11, §3.14).** A per-arm number from ~3 wells cannot be read
on its own, so `--source real` measures two references from real wells and
records them in the oracle:

- the **noise floor** per arm size $n$ — the median
  $\|\hat\tau_\text{null}\|$ of pseudo-arms built from $n$ real DMSO wells
  against a disjoint DMSO subsample. This is the floor §3.8.1's responder rule
  (`> 1.5 ×`) refers to;
- the **split-half reliability** per $n$ — the median
  $\cos(\hat\tau_A, \hat\tau_B)$ over two disjoint halves of each arm's real
  wells — and its **Spearman–Brown lift**, $\sqrt{2r/(1+r)}$, which is the
  actual bound on $\cos(\hat\tau_\text{gen}, \hat\tau_\text{oracle})$.
  Accuracy is reported raw and as a fraction of that bound.

Amplitudes need the same care: $\mathbb{E}\|\hat\tau\|^2 = \|\tau\|^2 +
\mathbb{E}\|\text{noise}\|^2$, and at n = 3 the floor (14.5) is most of the
measured median (15.2), so the report also carries
$\|\hat\tau\|_\text{denoised} = \sqrt{\max(\|\hat\tau\|^2 - \text{floor}^2, 0)}$
and the generated-over-denoised ratio.

The per-arm threshold is per pool: `--min_dose_n` 2 on `all` (E4), but
`--min_dose_n_holdout` **1** on the holdout, where the P3 split leaves almost
every arm exactly one well.

**Where $\mu(0)$ comes from.** RxRx takes its anchors from real rows unless
`--gen_anchors`. Here the vehicle arm is part of the estimand, so report both:
$\hat\mu(0)$ from real DMSO (default, isolates treated-arm error) and from
generated DMSO (`--gen_anchors`, tests $Y(0)$).

**EFFECTS** — report $\|\hat\tau\|_2$ and cosine to a **curated compound
list** drawn from what the MCF7 24 h table holds (well counts across doses; do
**not** reuse the RxRx remdesivir panel):

- proteasome: bortezomib, MG-132 (~980 each)
- HSP90: geldanamycin 48, NVP-AUY922 23, alvespimycin 18
- HDAC: entinostat 36, belinostat 35, vorinostat 27
- mTOR: sirolimus 51, torin-1 42
- MEK: PD-0325901 117, selumetinib 42, trametinib 20
- ER (MCF7 is ER+): estradiol, tamoxifen, fulvestrant (18 each)

**ACCURACY (`--truth`)** — same JSON from `--source real` as oracle.
Compare generated $\hat\tau$ to oracle with the aggregate metrics above and
on a few pathway reductions if defined. This is the number that says the
generator recovered the causal quantity.

**QUALITY** — MMD on standardized $Y$ everywhere. A Fréchet distance in 978-d
needs $n \gg 978$ per group to estimate the covariance, so compute it on the
marginal, or on top-$k$ PCs fit on train real $Y$. Reuse the matrix-sqrt / RBF
code in `dist_metrics.py`, and delete Inception, the 5→3 channel hack, and the
module-level `torchvision` import. Optional TRTS: train a linear probe on
`dose_level` or compound id from real $Y$, test on generated (conditioning
check).

**Generation** — copy the CFG sampling loop in `generation.py`. Sample at
real (A, C, E) rows as RxRx does. `_generate_batch` returns `(B, 978)`,
never decodes a VAE, and **drops** `x.clamp(-1, 1)` and `channels_last`.
Rebuild MLP vs 1D-DiT from `arch.json` only.

`--encoder {domain,openphenom,inception}` is **out**.

### 3.8 Confounding (after v1: step C, then step A — decision 7)

**Why the RxRx lever is replaced (measured).** RxRx's tiered split selects
wells by position: a plate + row + column vehicle baseline. A LINCS arm sits
at one well position (only 72 of 10,944 arms span edge and interior), and
plate, batch, and plate map have no within-arm variation. Plate centring
(decision 6) removes their effect on `Y` anyway, and the replicate index
spans arms but carries ~2% of DMSO variance. The confounder therefore has to
be something other than the plate:

- **Step C** — a semi-synthetic covariate on MCF7, as a sanity check with a
  known effect.
- **Step A** — cell line, on a 5-line population, for real-data results.

Run C to completion before building A, and then C2 (§3.8.4): step C could
not tell `dr` from `conditional` (§5), and C2 is the known-truth case
designed so that it can. The vehicle is RxRx's current
**tiered split** (P6); `rarity.py` (MCAR drops, `_confound_cells`,
`_allocate_drops`, `min_cell`) is outdated and not ported.

#### 3.8.1 Shared mechanism: the tiered split

`src/data/build_tiered_split.py` rewrites RxRx's builder for LINCS. It keeps
the three independent layers and `_calibrate_pi`, and replaces the RxRx
panel, the outcome-derived position score, and the image arms. One instance
is one split dir, `<nuisance_dir>_tier_<C>_k<k>_g<γ>_s<seed>/` (RxRx:
`nuisances_tier_k<k>_pg<γ>_s<seed>`). The full data stays unconfounded, only
**training** wells of scored compounds are thinned, and the oracle is
computed on the full, unablated pool.

1. **Reserve** (`--k_reserve`, **default 0** on LINCS): k wells per scored
   arm, drawn before anything else, never trained on, as out-of-sample
   truth. RxRx needs ≥ k + 3 wells per scored arm. At k = 2 only 593 LINCS
   arms and 94 compounds (all six doses) qualify, so the default truth is the
   full-pool oracle (§3.7, `--pool all`). The layer stays available for
   high-n arms.
2. **Split:** the P3 split (arm strata, DMSO stratified by plate,
   `holdout_frac` 0.2, table fingerprint in the cache key). v1 is this
   builder with no scored compounds and no thinning (P4), so v1 and steps C /
   A share one split implementation and, at a given seed, one holdout. Each
   stratum draws from its own `(seed, stratum)` generator, so with
   `k_reserve` > 0 every unscored stratum (DMSO included) still keeps v1's
   holdout.
3. **Thinning:** for each scored compound, every train well $i$ gets
   retention $\pi_i \propto \exp(-\gamma z_i)$, calibrated so that
   $\sum_i \pi_i$ = `keep_frac` · $n$ and clipped to [`pmin`, 1] (RxRx
   `_calibrate_pi`; `pmin` default 0.05). Survivors are drawn, and redrawn
   until every positivity cell (below) keeps ≥ 1 train well. Kept wells of
   scored compounds get the design weight $P_c/\pi_i$, every other row 1. γ = 0
   gives constant $\pi$: the uniform (MCAR) control at the same `keep_frac`.
   - $P_c = 1 - \prod_{j \in c}(1 - \pi_j)$ for the well's positivity cell
     $c$. The redraw conditions on "$c$ keeps ≥ 1", so well $i$ is kept with
     probability $\pi_i / P_c$, not $\pi_i$ (the cells are independent).
   - $1/\pi_i$ (the pre-implementation text) ignored that conditioning. At
     γ = 1 it over-weights small low-$\pi$ cells up to ~4× (Phase 1 review,
     2026-09-29; checked by Monte Carlo on the limit build).

Outputs, as in RxRx:
- `splits.json`: thinned `train_idx`, `holdout_idx`, and a `tier` block with
  γ, `pmin`, `keep_frac`, the covariate, and the scored compounds;
- `dr_weights_design.npz`: the true weights $P_c/\pi$;
- `nu_rows.npy` (`build_nu_rows`): the unthinned train pool minus reserve;
- `tier_meta.json`: per-compound $\pi$, kept counts, and the expected naive
  vs IPW bias;
- copies of `nuisance_meta.json`, `compound_vocab.json`, and
  `covariate_encoder.json`.

`--plan` prints the expected bias per γ and writes nothing.

- **Scored (rare) compounds come from responders.** Only ~29% of MCF7 arms
  clear 1.5× the 3-well noise floor, and confounded selection cannot bias a
  compound that does not respond.
  - `responders.json`: compounds whose full-data oracle $\|\hat\tau\|$ (max
    over doses) exceeds 1.5× the noise floor. It is written once per
    population from the Phase 4 `--source real` oracle, before any thinning.
  - `--n_tier_compounds` (or a fraction) of the responders are scored.
  - Choosing *where* to test from outcomes is an experiment-design choice. The
    within-compound thinning depends on $(A, C)$ only; RxRx's position score
    was estimated from outcomes, and LINCS's is not.
- **Selection score:** $z_i$ = `standardize(z_dose · z_C)` within the compound,
  with `z_dose = ±1` for the high / low dose half (levels 4–6 vs 1–3 of the
  compound's 6) and `z_C = ±1` for the two levels or groups of C.
- **Positivity cells** differ by step (below). The covariate and its groups
  go into the dir tag and the `tier` block.
- **$\nu$ must see the unablated design.** $\nu$ = `nu_rows.npy`, read by
  `export_urr_weights --mode counts`. If $\nu$ came from the thinned pool,
  $\alpha$ would be blind to the thinning by construction.
- **DR weights (P12).** `export_urr_weights --mode counts` estimates
  $\alpha$ at the positivity cell: $w = n_\nu(\text{cell}) /
  n_\text{kept}(\text{cell})$ on treated train rows, 1 on vehicles. It is
  the post-stratified version of the design weight: the design weight is
  constant within a cell, every cell keeps ≥ 1 train well, and unthinned
  cells get exactly 1. It is not cross-fitted: it uses (A, C) only, never Y,
  at a few cells per compound. The cross-fitted net cannot be used, because in
  each fold most (arm, C) cells have no factual fit row (measured in the
  Phase 5 TODO note); `fit_urr` refuses a thinning instance.
- **Design truth.** `export_urr_weights` compares its weights with
  `dr_weights_design.npz` (correlation on the design-weighted rows).
- **Arms per step** (MLP only; same split and seed; γ > 0 plus the γ = 0
  control at the same `keep_frac`):

  | arm | `--adjustment_set` | `--dr_mode` / weights | role |
  |---|---|---|---|
  | `naive` | `''` | conditional | shows the bias the thinning creates |
  | `conditional` | C | conditional | g-formula baseline |
  | `dr` | C | weighted, `dr_weights_counts.npz` (positivity-cell counts, P12) | ADIGen |
  | `dr_design` (reference) | C | weighted, `dr_weights_design.npz` | true weights: separates weight-estimation error from the generator |

  Train at least 2 seeds per arm, so that "DR beats conditional" is judged
  against seed noise.
- **Eval:** `--pool all` on the unablated population. Generate at the real
  rows of that pool, whose C is balanced by design, so pairing A with the
  rows' C is the product measure. Report metrics separately for scored vs
  unscored compounds, and for the low vs high dose half.
- **Success:**
  1. `naive` shows bias on scored compounds at γ > 0 and none at γ = 0.
  2. `dr`'s error on scored compounds is below `conditional`'s by more than
     seed noise, and close to `dr_design`'s.
  3. Unscored compounds are unchanged.
  4. The `counts` weights vary with $C$ on scored compounds (unlike v1), are
     1 on unscored ones, and track `dr_weights_design.npz`.

#### 3.8.2 Step C — semi-synthetic covariate on MCF7

- **`syn_c`** is assigned in `build_dataset` (`src/data/synthetic.py`) with a
  fixed `syn_seed`. It is balanced at random within every
  (compound, `dose_level`) arm (a 3-well arm gets a 1/2 or 2/1 split) and
  within every plate's DMSO wells. The full data is therefore unconfounded,
  and `syn_c` is pre-treatment by construction.
- **Injected effect**, in `dataset.py` after centring and z-scoring:
  `y ← y + syn_c · β · v`, applied to every row with `syn_c = 1` (treated and
  DMSO alike).
  - `v` is a seeded random unit vector in 978-d z-space.
  - `β = --syn_effect × median responder ‖τ̂‖` (z-space), default 1.0.
  - The resolved `β`, `v`'s seed, and `syn_seed` are recorded in
    `arch.json`.
  - `--syn_effect 0` (the default) makes `syn_c` inert, which is what v1
    uses.
- **The oracle uses the same injection.** `evaluate --source real` takes
  `--syn_effect` / `--syn_seed` and records them, and `--truth` refuses a
  mismatch with the arm's `arch.json`.
- **Positivity cells are (compound, dose half, `syn_c`)**, about 3 train
  rows each. Per arm, the ~2 train wells split about 1 + 1 across `syn_c`,
  so arm-level cells would hold ~1 row and could not be thinned.
  - Keeping ≥ 1 row per cell bounds how far selection can go (at most 2 of 3
    rows per cell), so the default is `--keep_frac 0.4`. `tier_meta.json`
    reports the realised kept fraction.
- **Positivity is per compound-half, not per arm.** After thinning, some
  scored arms keep train wells in only one `syn_c` level.
  - The weights are therefore estimated at the (compound, dose half,
    `syn_c`) cell, which keeps ≥ 1 train well by construction (P12):
    `export_urr_weights --mode counts --adjustment_set syn_c`.
  - The generator still conditions on the arm and `syn_c`; the `syn_c`
    effect is the same additive shift for every arm, so it can share it
    across doses.
  - The design-weight comparison is the check.
- **The ground truth is known.** The bias lives along `v`. Report the signed
  projection $\langle \hat\tau_\text{gen}(a) - \hat\tau_\text{oracle}(a),
  v\rangle$ for scored arms by dose half, alongside the aggregate metrics of
  §3.7.
- **Sweep:** γ ∈ {0, 1} at `--syn_effect 1.0` first; use `--plan` to size γ.
  Add γ = 2 or `--syn_effect 0.5 / 2` only if the first result is ambiguous.

#### 3.8.3 Step A — cell line on a 5-line population

- **Gate before building.** Measure whether cell lines change responses after
  centring:
  - read the responders' wells in the 5 lines (a small h5py read, as in the
    2026-09-28 checks);
  - compare per-line centred $\hat\tau$ across lines against the 3-well noise
    floor.

  If lines respond alike, `cell_id` has no effect on centred `Y`, and step A
  collapses into another null check. Record the result in `population_qc.json`
  for the new population.
- **Population `core5_24h`:** 24 h, `trt_cp ∪ ctl_vehicle`, `cell_id ∈
  {MCF7, HT29, HA1E, A375, PC3}`, restricted to compounds present in all 5.
  - Overlap (measured): ≥99.8% of MCF7 arms appear in each of these lines.
  - HELA and YAPC are excluded: they miss ~10% of arms.
  - Size: ~172k treated + ~10k DMSO wells.
  - It is a **separate population**. `PopulationSpec` has a `name`
    (`mcf7_24h` default, `core5_24h`), selected with `--population`, which
    sets `Paths` to `data/<name>/…` and `runs/<name>/…`. Name it in every
    result.
  - v1's MCF7 decisions and results are untouched.
- **Outcome**, same rules per plate (plates are single-line):
  - plate QC with the 3× rule, using each line's median spread;
  - DMSO-median centring on train wells;
  - per-gene z-score pooled over lines (mean from train DMSO, std from all
    train rows; decision 5).
- **Roles:** `cell_id` is promoted with `--adjustment_set cell_id`. $\alpha$
  gets 5 one-hot columns, and the generator conditions on `cell_id` as C
  (never dropped by CFG).
- **Estimand:** $\tau(c, d)$ averaged over the 5 lines, with the line mix of
  the pool (near-uniform by design). Per-line $\tau$ is secondary.
- **Positivity cells are (compound, `dose_level`, `cell_id`)**, about 2 train
  rows each. Keeping ≥ 1 row per cell means every scored arm keeps ≥ 1 row in
  every line.
  - So **arm-level positivity survives the thinning**. The weights are
    `counts` at this cell, i.e. (arm, `cell_id`) (P12); with ~2 train rows
    per cell the net cannot be cross-fitted here either.
  - Selection can remove at most ~half of a scored compound's rows (1 of ~2
    per cell), so the default is `--keep_frac 0.6`, with 0.5 as the floor.
- **`z_C`:** a fixed split of the 5 lines into two groups, declared in
  `spec.py` before any step-A result is seen and recorded in the dir tag.
- **Bias readout:** for scored compounds, the error of the line-pooled
  $\hat\tau_\text{gen}$ vs the oracle, plus its correlation with the oracle's
  group contrast $\hat\tau_{G_2} - \hat\tau_{G_1}$. Confounding pulls
  estimates toward the over-kept group.
- **Scale:** ingest reads ~183k full rows (~9 GB) from the GCTX (CPU job). The
  in-memory `y` is ~0.7 GB. Nuisance and training time grow ~5×.
- **Plan of record, readout and success criteria (2026-10-05):** `STEP_A.md`.
  Its §2 holds the decisions (line groups, arms, cuts), its §3 the bias readout
  and criteria S0–S5, fixed before any step-A result, and its §5–§7 the TODO
  list and jobs. Where it differs from the bullets above, it is the later text:
  - the bias readout is the error projected on each arm's own line-group
    contrast, taken over holdout wells, as a difference in differences against
    γ = 0, with compound-clustered errors;
  - `keep_frac` is the realised kept fraction, 0.75 (approved 2026-10-05);
  - the arms are `naive` / `conditional` / `dr` / `dr_p2`, plus P1 as a
    baseline.

#### 3.8.4 Step C2 — compound-specific injection (decided 2026-10-01; before step A)

**Why.** Step C cannot separate `dr` from `conditional` (§5, "Phase 5 step-C
results", finding 3).
- Its injection is one shift shared by every row, and the conditional
  generator learns it exactly: the slope is 25.5 against beta = 25.48.
- So `conditional`'s g-formula is unbiased, and the weights have nothing to
  correct.

C2 keeps everything about step C except the injection, which becomes
compound-specific. `conditional` then has to learn a compound × `syn_c`
interaction from the few rows the thinning leaves. That is the misspecified
outcome model DR is meant to protect against, and the same structure step A
will have (cell line × compound), here with a known truth.

- **Injection.** y ← y + `syn_c` · beta · v_{k(i)}, where k(i) is the row's
  `compound_idx` (vehicles are k = 0).
  - v_k = normalise(√(1−ρ) · v + √ρ · u_k). Here v is step C's direction
    (`vec_seed` 0), and u_k is a unit vector seeded by (`vec_seed`, k), drawn
    independently of every other compound.
  - **ρ = 1 is the C2 setting.** ρ = 0 reproduces step C bit for bit, which is
    checked rather than assumed.
  - **`syn_c` now modifies the treatment effect.** In step C, the vehicles and
    every arm moved along the same v, so `syn_c` was purely prognostic and the
    unablated τ̂ did not move. In C2, a compound-k arm's `syn_c` = 1 wells move
    along v_k and the vehicles' along v_0.
    - So τ̂ on the unablated pool moves by exactly
      beta · (mix_a · v_k − mix_0 · v_0), where mix is the arm's (or the
      vehicles') `syn_c` mean. `check_phase5` asserts this to float precision.
    - That shift is part of C2's estimand: the oracle carries it, and a
      generator that learned the interaction reproduces it. It is the
      interaction step C lacked.
    - It cancels in the error projected on v_k, so the readout is unaffected.
    - Per-arm cosines against this oracle are inflated by the −beta · mix_0 · v_0
      term that every arm shares, so they are not used for C2.
  - beta is unchanged (`--syn_effect 1.0`, 25.483).
  - Reused unchanged, because they depend on (A, C) only: `syn_c`, both tier
    instances, the positivity cells, `dr_weights_counts.npz` and
    `dr_weights_design.npz`.
  - Resolved once into `<data>/nuisances/syn_meta_compound_r1.json`
    (`python -m src.data.synthetic --mode compound --rho 1`). The file records
    the mode, ρ, `vec_seed`, the number of directions, and a `v_sha1` that
    hashes the **whole direction matrix**. The matrix itself is stored once,
    next to the file, as `syn_meta_compound_r1_V.npy`. Regenerating it is not
    bit-portable across CPUs (§5, "Step C2 implementation and review"). Every existing provenance guard
    (arch.json, `check_arm_against_data`, `TRUTH_GUARD`) therefore tells C2
    from C with no new field.
  - Selected with `--syn_meta <file>` on the trainer, `evaluate` and
    `step_c_report`. The default, `syn_meta.json`, keeps step C. The choice
    lives on `CaseConfig`, not `OutcomeSpec`, because `OutcomeSpec` is
    embedded in `decisions_record` and a new field there would invalidate both
    builds.
- **Why `conditional` should fail and `dr` should not.** Take an arm a of
  compound k. Let p_kept be the `syn_c` mix of its kept train wells, p_a the
  mix of its pool wells, and λ the share of k's shift the generator learns.
  - Along v_k, `conditional`'s bias is (p_kept − p_a)(1 − λ) beta.
  - `dr`'s is (p_w − p_a)(1 − λ) beta. The weights balance p_w within each
    (compound, dose half) cell, so its high − low contrast is ~0 for any λ.
  - Step C had λ = 1. Here each scored compound's shift is learned from ~6 kept
    rows, through a conditioning vector that is a sum
    (`t + e_compound + e_dose + e_syn_c`). A shared shift is native to that
    sum; a compound-specific one is not.
- **Readout.** The same as step C: the DiD of the mean scored high − low
  contrast, on `--pool all`, over 3 training seeds. Each arm's error is now
  projected on **its own compound's** v_k.
- **New diagnostic: the learned share λ̂**, per arm, from the generated per-row
  means.
  - Step-C generators condition on compound, `is_control`, dose and `syn_c`
    only (checked in arch.json). So within one arm, generated rows differ only
    in `syn_c`.
  - λ̂ = ⟨mean gen(`syn_c` = 1 rows) − mean gen(`syn_c` = 0 rows), v_k⟩ / beta.
    Its sampling noise is ~0.01 per arm.
  - Expected: `naive` ≈ 0, and step C's adjusted arms ≈ 1.
  - `evaluate` reports it on the generation pool as `learned_syn_effect`, for
    step C and C2 alike.
- **Arms.** `naive` / `conditional` / `dr` / `dr_design` × γ ∈ {0, 1} × training
  seeds {0, 1, 2}, MLP-B DDPM 500 epochs, on the seed-42 tier instances,
  exactly as step C.
  - Run dirs are `stepc2_<arm>_g<γ>_s<seed>`.
  - Each is scored at the §3.14 settings on `--pool all` against
    `oracle_mcf7_24h_poolall_syn1-cmp1.json`.
- **Success, fixed before any C2 result:**
  1. `naive` is biased: |DiD| > 3 seed sd.
  2. **`dr` beats `conditional`**: `dr`'s |DiD| is below `conditional`'s by
     > 2σ (`step_c_report`'s `dr_vs_conditional_sigma`; §3.8.1 criterion 2).
  3. `dr` **agrees with** the design-weight arm:
     |DiD(`dr`) − DiD(`dr_design`)| < 0.25 · |DiD(`conditional`) − DiD(`dr_design`)|.
     §3.8.1 criterion 2 always said "close to `dr_design`'s"; the step-C report
     judged only its first half, and now judges both.
     - **Reworded 2026-10-05** (was "`dr` is close to the true weights"). The
       design weights are the true *inclusion probabilities*, which is not the
       same as the weights that balance the realised sample: they balance only
       in expectation over thinning draws. Measured (§5, "P1 results"): the
       `counts` weights reproduce the unthinned confounder mix of every group
       **exactly**, while the design weights leave a systematic 0.5% dose-half
       contrast that accounts for ~90% of the design arm's residual bias. So
       `design` is not a target to converge on; this criterion tests that the
       two weight sets **agree in sign and magnitude**, which is a check on the
       weight model, not a ranking.
  4. Unscored compounds unchanged (§3.8.1 criterion 3).
- **Diagnostics, not gates:**
  - λ̂ < 1 on `conditional`'s scored arms.
  - DiD(`conditional`) ≈ (1 − λ̂) · DiD(`naive`), within a factor of 2.
  - `naive`'s leak onto unscored compounds ≈ 0. With compound-specific
    directions, no shared pathway can carry it (tests finding 5's mechanism).
  - The weights' precision cost, as in step C's finding 4.
- **If criterion 2 fails because λ̂ ≈ 1**, the pre-declared escalation is one
  direction per (compound, dose half). That is still the positivity cell, so
  the weights still balance it. It costs 24 more runs. Any other change is a
  new decision.
- **Compute.**
  - The injected oracle and `check_phase5` are a numpy-only CPU job, so they
    follow the CPU rule: `ma` within its headroom, else `bindel`.
  - The GPU smoke and the 24 × (~9.4 min training + ~7.6 min scoring),
    ~7 GPU-h in all, run on `zabih` (approved 2026-10-01).

### 3.9 Reuse vs rewrite (checklist)

Every “copy” below is a **file copy** `RxRx19a/src/…` → `lincs/src/…`
(§3.1.1), then LINCS-only edits. No cross-package imports.

**Copy with light edits**

| Source (`RxRx19a/src/`) | LINCS use |
|---|---|
| `models/conditioning.py` | as-is |
| `processes/__init__.py`, `ddpm.py`, `flow_matching.py` | as-is (ndim-safe) |
| `nuisances/knn_dr.py` | as-is (only `ess` / `tail_index` are used) |
| `nuisances/export_urr_weights.py` | copy; no `infected`; `dose_level` action keys; controls $w = 1$ (§3.5) |
| `nuisances/precompute_cmean.py` | copy, optional (P9): `y` instead of latents, `dose_level` arm keys, configurable `--min_n` |
| `data/dataset.py` `build_cond_spec`, `cond_from_arrays`, `cond_from_batch`, `dose_probe` | copy (spec-driven, not image code) |
| `spec.py` helpers (`role_tag`, CLI, `CaseConfig` guards) | copy; new FIELDS / paths / OutcomeSpec |
| `train/train_diffusion.py` loop | copy; y-rank (train + val), `--arch`, drop VAE, V-REx key |
| `models/dit.py` `DiTBlock`, `TimestepEmbedder`, `DIT_SIZES`, adaLN init, `get_1d_sincos_pos_embed_from_grid` | copy into `dit1d.py` |
| `eval/dist_metrics.py` Frechet / MMD / KID on feature matrices | copy; drop Inception + `torchvision`; feed $Y$ |
| `data/splits.py` `load_splits` + stratification engine | copy; cache key = population + table fingerprint; arm strata; DMSO by plate |

**Rewrite**

| Concern | Why |
|---|---|
| `data/build_dataset.py` | GCTX + LINCS joins/filters; write `expr.npy` not PNG paths |
| `data/expr_stats.py` (new) | plate DMSO centres + per-gene stats need the split |
| `data/synthetic.py` (new) | step-C `syn_c` assignment + injected effect |
| `data/build_tiered_split.py`, `build_nu_rows.py` | keep the three layers and `_calibrate_pi`; replace the RxRx panel, outcome-derived position score, and image arms with responders-only scored compounds, dose-half × C selection, per-step positivity cells, `k_reserve = 0` default (§3.8.1) |
| `spec.py` `PopulationSpec.name` / `--population` | step A is a second population with its own paths |
| `spec.py` `FIELDS`, `Paths`, `PopulationSpec`, `ImageSpec` | different columns and $Y$ |
| `data/dataset.py` loader | in-memory vector loader; drop PIL / image VAE |
| `ContextEncoder` | `det_plate` / `det_well`; no disease/site; built from the filtered table |
| `models/mlp.py`, `models/dit1d.py` | new backbones |
| `models/__init__.py` `arch.json` | `arch=mlp\|dit1d`, `n_genes`; dispatch in `build_generator_from_ckpt` |
| `nuisances/alpha_net.py` + `fit_urr.py` + `export_urr_weights.py` scoring | no `infected` |
| `eval/evaluate.py`, `generation.py` | gene ATE, not rescue/FID/OpenPhenom; no clamp |
| OpenPhenom, `feature_extractor.py`, `encode_latents` | not applicable |

**Do not reuse**

- `DiT2DModel` / `PatchEmbed` 2-d / `unpatchify` to `(C,H,W)`
- `ostris/vae-kl-f8-d16` and `--latent 1`
- RxRx dose grid and empty-string `control_token`
- `PANEL_HITS` / Mock / untreated-infected axis
- `third_party/maes_microscopy`
- RxRx `scripts/*.slurm` (missing here anyway)
- RxRx `_confound_cells` edge × dose lever (§3.8)
- `fit_knn_dr.py`, `alpha_truth_check.py`, `knn_dr_weights` (kNN AIPW retired 2026-09-29)
- `data/rarity.py` (removed from RxRx in `78d4284`; superseded by the tiered split, P6)

### 3.10 Phased work

**Phase 0 — spec + ingest (blocks everything)**
`spec.py` + `build_dataset.py`. Smoke: `--limit` wells, print $n$,
compound count, `dose_level` counts, landmark shape `(n, 978)`, vehicle
fraction, DMSO wells per plate. The cell-line coverage table is done (§3.12).

**Phase 1 — table consumers**
`splits.py` + the split layer of `build_tiered_split` (v1 instance: no
reserve, no thinning), `build_nu_rows`, `expr_stats.py`, `dataset.py`,
`fit_urr` (no infected), `export_urr_weights` (`net`; `counts` as fallback). Gate: URR beats constant baseline,
`mean(alpha)≈1`, ESS/n not collapsed. On v1 this gate passes trivially
(§3.3), so also check the known answers: the net gives
$\alpha \approx 1/(1-\pi_0) = 1.066$ on treated rows and ≈0 on DMSO, and the
`counts` weights are exactly 1. Centring gate (P11), on **held-out** DMSO
wells: per-plate means ≈0, and plate's share of DMSO variance drops from
0.67 toward the ~0.20 measured in §3.2. On train DMSO wells the check is
trivial, because they set the centres.

**Phase 2 — MLP arms**
MLP + trainer + DDPM (FM optional). `mlp_conditional` then `mlp_dr`.
Port `precompute_cmean` and `--cmean_lambda` (off by default; P9).
Smoke on `--limit` / `--max_steps`.

**Phase 3 — 1D-DiT arms**
`dit1d.py`, same trainer flags. `dit1d_conditional` / `dit1d_dr`.
Confirm `arch.json` rebuild.

**Phase 4 — eval**
Real oracle JSON, then each of the four generator arms with `--truth`.
Gene-space MMD. No image metrics. Then the `cmean` ablation (P9) on FM arms.

**Phase 5 — step C: semi-synthetic confounder on MCF7 (§3.8.2)**
`synthetic.py` (the `syn_c` column already exists from Phase 0), responders
from the Phase 4 oracle, the step-C tiered-split instance (§3.8.1), $\nu$
from the unthinned pool, `export_urr_weights` checked against the design
weights. Train `naive` / `conditional` / `dr` (+ `dr_design`) at
γ ∈ {0, 1}, ≥2 seeds.
Score them against the injected oracle. Gate for Phase 6: the success
criteria of §3.8.1 hold, or the failure is understood and written up.

**Phase 6 — step A: cell line on `core5_24h` (§3.8.3)**
Effect-modification gate first. Then the `--population` switch, ingest of the
5-line population, its own splits / stats / Phase 4 oracle / responders, and
the same three arms with `--adjustment_set cell_id`.

**Phase 7 — (optional)**
V-REx over plate, pathway $Y$, FM vs DDPM, 1D-DiT on the confounding steps,
SLURM.

### 3.11 Dependencies

`lincs/requirements.txt` exists: the RxRx pins plus `h5py` (torch is
installed separately). Raw h5py is enough for the GCTX; cmapPy / tables are
optional. Drop the assumption that `timm` PatchEmbed-2d is required for every
arm; 1D-DiT may use `timm` Attention/Mlp only. Do not add OpenPhenom /
`maes_microscopy`.

This machine has no GPU: do **not** run training here. Ship a small
`src/tests/test_lincs_shapes.py` (synthetic `(B, 978)`, both arches, one
URR step) for a GPU box. Put it under `src/` because `.gitignore` drops
`lincs/scripts/`, and an untracked file cannot go through the review §4
requires.

### 3.12 Phase 0 decisions (resolved and frozen; 3 and 5 amended 2026-09-29; 7 amended 2026-09-30)

1. **Cell line: `MCF7`.** A metadata scan of the 24 h
   `trt_cp ∪ ctl_vehicle` slice found 35,623 chemical and 2,084 vehicle
   wells (37,707 total), the largest eligible cell-line population
   (next: HT29, 36,556; re-verified 2026-09-28). `PopulationSpec.cell_id` is
   frozen to `MCF7`.
2. **Compound universe: all retained `trt_cp`** (1,750 `pert_id`s).
   `PopulationSpec.compounds=None`; controls are always retained. This
   includes bortezomib and MG-132 at 20 µM, the per-plate proteasome
   controls: ~1,960 wells, 5.5% of treated, on 84 of 101 plates. They weigh
   heavily in the empirical action marginal $\nu$. Any later curated subset is
   a new population/estimand and must be named in every result rather than
   silently replacing this v1 universe.
3. **Dose: continuous log-dose.** Ingest converts dose to µM, and nuisances
   and the model use `log10_conc`. The 97 unique positive treated doses are
   **float variants, not a 97-point grid**. 99.5% of treated wells lie within
   0.05 log10 of 0.0412, 0.1235, 0.3704, 1.1111, 3.3333, 10 µM (some plates
   record 0.04, 0.12, 0.37, 1.11, 3.33), plus 20 µM for the proteasome
   controls. 1,741 of 1,750 compounds have exactly 6 dose groups. The
   **evaluation grid is therefore `dose_level`** (§3.2): the 7 nominal levels
   plus 173 off-grid wells from 13 compounds kept at their own dose.
   RxRx's `ActionSpec.continuous_grid` is declared but read by no code, so
   evaluation must use `dose_level` explicitly. The kNN bandwidth
   (`0.026920511435899325`) was frozen in `spec.py` for `fit_knn_dr`. That
   estimator is retired, so the field is **removed** from `ActionSpec`
   (P8, 2026-09-29); the value is kept here for the record. No
   `(compound, dose_level)` cell spans more than it (measured), so it never
   disagreed with `dose_level`.
4. **Headline backbone: MLP.** Both MLP and 1D-DiT remain required and are
   reported; MLP is the main DR result because no trusted 1-d gene topology
   justifies privileging 1D-DiT.
5. **Normalization: per-gene z-score after centring — amended 2026-09-29
   (P10).**
   - **Mean:** the per-gene mean of the centred **train DMSO** wells
     (`OutcomeSpec.normalize_mean="train_dmso"`), so that z = 0 is the
     vehicle. Before the amendment it was the mean of all centred train rows,
     which puts DMSO a median 0.09 SD (max 0.40) off zero; 34% of that mean
     came from the proteasome arms.
   - **Std:** unchanged, the per-gene std of **all** centred train rows
     (`normalize_std="train_all"`). It is a median 1.15× the DMSO noise,
     up to 4.7× for strong responders.
   - **Genes:** all 978 are kept, including GAPDH (at the 15.0 cap in 41.5%
     of wells; 9 other genes exceed 1%). `expr_meta.json` reports each gene's
     fraction at the cap.

   `expr_stats` (after splits) computes both from training rows only, after
   plate centring, and persists them in `expr_meta.json`. Samples are
   z-scores, so generation must not clamp.
6. **Outcome centring and plate QC — accepted 2026-09-28.**
   - Subtract the per-plate, per-gene median of that plate's **train** DMSO
     wells, with no scaling (`OutcomeSpec.plate_center="dmso_median_train"`).
   - Drop plates whose DMSO spread exceeds 3× the median plate's
     (`plate_qc_max_spread_ratio=3.0`; on MCF7 this drops one plate).
   - Evidence and consequences are in §3.2. `plate_center="none"` is the
     ablation.
7. **Confounding program — accepted 2026-09-28; weight estimation amended
   2026-09-30 (P12).** Two steps after v1, in
   order (§3.8):
   - **step C**, semi-synthetic `syn_c` on MCF7, as a sanity check with a
     known effect direction;
   - **step A**, `cell_id` as C on the separate `core5_24h` population
     (MCF7, HT29, HA1E, A375, PC3), for real-data results.

   Both use:
   - responders-only rare compounds;
   - dose-half × C selection by probabilistic thinning in the tiered split
     (§3.8.1; P6), with known design weights;
   - $\nu$ from the unthinned design pool, with the DR weights estimated
     by `export_urr_weights --mode counts` at the positivity cell (P12);
   - the `naive` / `conditional` / `dr` arms (+ `dr_design` reference) with
     a γ = 0 control.

   Step A starts only after step C's gate and step A's own
   effect-modification gate.

Do not change these decisions between the four v1 arms. A different compound
universe or cell line defines a separate population and result set. Step A's
`core5_24h` is such a population, and it does not replace decision 1.

### 3.13 Phase 1 decisions (resolved 2026-09-29; P12 and the P1 / P2 amendments 2026-09-30)

Numbers are measured on the built `mcf7_24h` table (37,340 wells) unless
marked analytic.

- **D1 — DR weights come from `export_urr_weights`.** `fit_knn_dr.py` (kNN
  AIPW) is outdated: RxRx removed it in `78d4284`, and LINCS follows. The
  weighted risk is $\sum_i w_i\,\ell_i$ with $w_i = \hat\alpha(X_i, A_i)$ on
  treated rows and $w = 1$ on vehicle rows, normalised to mean 1. It has no
  kNN plug-in / correction legs.
- **P1 — `net` weights are primary; `counts` is the fallback.** v1's `dr`
  arms train on `net` weights (≈1.004 treated / 0.942 DMSO after
  normalisation, analytic). That exercises the `fit_urr` → export → trainer
  path step C needs. `counts` (≡ 1 in v1) runs as a check and stays
  available if the net fails its gate. *Amended 2026-09-30 (P12): this holds
  for v1; steps C / A use `counts` at the positivity cell.*
- **P2 — Arms are keyed on `dose_level`** everywhere an arm is keyed:
  `fit_urr` common support and $\nu$, the fold strata (P5), the tiered
  split's scored arms, and the `counts` keys. Once C is non-empty, keys are
  (arm, C). This still matters under P1: RxRx's float keys give 10,944 arms
  instead of 10,479 (0.37 vs 0.3704 µM split), which shrinks common support
  in steps C / A. *Amended 2026-09-30 (P12): in a thinning instance the
  `counts` key is the design's positivity cell, not (arm, C).*
- **P3 — Split:** `holdout_frac` 0.2 with arm strata (realised 28.5%, train
  26.7k); DMSO stratified by plate (min 14 train DMSO per plate, vs 12 as one
  stratum); the table fingerprint in the cache key.
- **P4 — v1's split is the tiered builder's split layer** (no scored
  compounds, no thinning), with `load_splits` and the stratification engine
  copied from `splits.py`. One split implementation serves v1 and steps C / A.
- **P5 — Cross-fit folds are stratified on the arm.** With random halves,
  35% of treated train rows have no well of their own arm in the other fold.
- **P6 — The confounding vehicle is RxRx's current tiered split** (reserve /
  split / probabilistic thinning with design weights, `build_nu_rows`).
  `rarity.py` is outdated and not ported. `k_reserve` defaults to 0 (only 593
  arms / 94 compounds could support k = 2); truth is the full-pool oracle.
  §3.8 is rewritten accordingly.
- **P7 — The flag is renamed** `--dr_mode {conditional, weighted}` (RxRx:
  `knn_dr`); the weights file picks the source. Arm ids stay `*_dr`.
- **P8 — `ActionSpec.continuous_kernel_bandwidth` is removed** from
  `spec.py`; the value is recorded in §3.12.
- **P9 — `cmean` is ported as a configurable, optional loss** and ablated:
  `precompute_cmean.py` (normalised `y`, `dose_level` arm keys,
  configurable `--min_n`; the RxRx default of 8 covers 34 of 10,479 arms) and
  the trainer's `--cmean_lambda` (default 0 = off) / `--cmean_file`. It is FM
  only, so the ablation runs on FM arms.
- **P10 — Decision 5 amended:** the mean comes from centred train DMSO; the
  std stays over all centred train rows; GAPDH and the other capped genes are
  kept and reported (§3.12).
- **P11 — The centring gate is measured on held-out DMSO wells** (§3.10).
- **P12 — Steps C / A: DR weights are post-stratified counts at the design's
  positivity cell (decided 2026-09-30).**
  - Cells: (compound, dose half, `syn_c`) in step C; (compound, `dose_level`,
    `cell_id`) in step A (`splits.POSITIVITY_KEYS`).
  - $w = n_\nu(\text{cell}) / n_\text{kept}(\text{cell})$, 1 on vehicles
    and on every unthinned cell; `export_urr_weights --mode counts`
    (`--smooth_k` 0); no cross-fitting.
  - Why: an (arm, C) cell holds ~1 train well. In each cross-fit fold most
    (ν arm, C) pairs have no factual fit row, where the URR target
    α = ν/P is unbounded: 77–78% of arms per fold with X = `syn_c`, on
    mcf7_24h (Phase 5 TODO note). The design weight is constant within a
    positivity cell, which always keeps ≥ 1 well.
  - `fit_urr` and `export --mode net` refuse a thinning instance. A net
    given the coarse action (dose half) could be a secondary estimate later;
    it is not implemented, and it would still lack fit rows in 12–38% of cells
    per fold at 5 folds.
  - Dropping gap arms from each fold's ν was rejected: ν(a) = 0 sets α ≈ 0
    and zeroes the weights of the scored arms.

Compute: `fit_urr` imports torch but AlphaNet is tiny, so it runs as a CPU
job (`ma` under the headroom rule). `export --mode counts` needs no torch
once the import is lazy.

### 3.14 Phase 4 decisions (resolved 2026-10-01, before any Phase 4 code)

- **E1: Checkpoint.** Every arm is scored at `checkpoint-0499` with the EMA
  weights (`model_1.safetensors`). The best `val_loss_ema` epoch is within
  0.0002 of it in every DDPM arm, and one epoch keeps the arms comparable.
- **E2: Guidance.** The headline uses no guidance, w = 1.
  - Why: the estimand is a mean difference. Guidance (w > 1) pushes each arm
    away from the action-marginal, so it inflates ‖τ̂‖ and shrinks within-arm
    variance. Neither effect has anything to do with confounding.
  - The DR weights correct the denoising risk of the conditional model, which
    is the model sampled at w = 1.
  - `--guidance_scale` stays an option for a later ablation (e.g. 1, 1.5, 2,
    read as ‖τ̂_gen‖ / ‖τ̂_oracle‖).
- **E3: Samples.** 16 generated wells per real row, to start. Raise it if τ̂
  is too noisy; lower it if sampling is too slow.
- **E4: Arm filter.** `--min_dose_n 2`. It keeps the 692 two-well arms.
- **E5: Quality metrics.** Report both Fréchet variants, plus MMD everywhere.
  - Marginal (per-gene) Fréchet: usable on small groups (per dose level, per
    compound). It is blind to gene–gene correlations.
  - Top-k PC Fréchet: only on large pools (all treated, all DMSO), where the
    sample count is in the thousands. PCs are fit on real train Y; k is chosen
    by explained variance (e.g. 90%) and recorded.
- **E6: Scope.** One seed per arm. No pathway-score metrics for now.
- **E7: Compute.** Phase 4 eval and generation jobs run on `zabih` (approved
  2026-10-01). The CPU-only `--source real` oracle follows the CPU rule (`ma`
  under the headroom rule, else `bindel`).
- **E8: Tracking.** wandb entity `493302570`, project `lincs-adigen`
  (`--wandb_entity` default). The four DDPM v1 runs first logged to the
  login's default team (`pfn-diffusion`); copies were synced to `493302570`
  on 2026-10-01.
- **E9: cmean (P9), from the training results in §5.**
  - `--cmean_lambda 0` stays the default for both backbones, and the v1
    headline stays DDPM with λ = 0.
  - cmean is not used on the 1D-DiT: it doubles the training time for a
    slightly worse validation loss.
  - On the MLP it stays a candidate. It becomes a default only if Phase 4
    shows a τ gain on the holdout pool with no loss of sample spread (MMD /
    Fréchet).
  - **The fair pool for this ablation is `--pool holdout`.** μ̂(a) is the mean
    of each arm's ~2 train wells, and the `--pool all` oracle contains those
    wells, so a model that memorises them is flattered there.
  - All eight FM arms are scored in Phase 4 next to the four DDPM arms.
  - Follow-ups (a λ sweep, `--min_n 3`, more epochs) are run only if the MLP
    gain carries over to τ.
- **E10: Sampler — 100 steps** (resolved 2026-10-01, before any Phase 4 code).
  DDIM for the DDPM arms, Euler for the FM arms, 100 steps
  (`--num_inference_steps`). The schedule is not a choice: every arm records
  `zero_snr=true`, so `make_eval_scheduler_for_ckpt` returns
  `prediction_type="v_prediction"` with `timestep_spacing="trailing"`, and an FM
  arm returns `FlowMatching` instead.
  - Cost at `--pool all` (597k samples per arm): ~16 min per MLP-B arm, ~3.9 h
    per DiT-S/10 arm. 1,000 ancestral steps would be ~39 h per DiT arm.
  - τ̂ **amplitude** is the quantity being measured, so the discretisation error
    is checked rather than assumed: one MLP arm is scored at 50 / 100 / 250 /
    1000 steps (~2.3 h in total, because the MLP is ~14× faster than the DiT)
    before the DiT arms are launched.
- **E11: Reference scale — empirical floor + split-half ceiling.** §3.8.1 asks
  for "1.5× the noise floor" without defining it, and with ~3 wells per arm the
  oracle τ̂ is itself mostly noise, so a median per-arm cosine over all 10,479
  arms is near 0 even for a perfect generator. Both references are measured from
  **real wells only**, written into the oracle artifacts, and reused by `--truth`
  and by Phase 5:
  - **Noise floor, per arm size $n$:** `--n_floor_draws` (200) pseudo-arms of $n$
    real DMSO wells, each against an independent **disjoint** DMSO subsample on
    the $\hat\mu(0)$ side, so the two sides are independent exactly as a real
    arm's are. Median and p90 of $\|\hat\tau_\text{null}\|$ are recorded.
    Sizes are evaluated exactly up to 20 and on a geometric ladder above (90% of
    arms hold 2 or 3 wells), and other sizes read the nearest.
  - **Split-half reliability, per $n$:** for every arm with ≥2 wells,
    $\cos(\hat\tau_A, \hat\tau_B)$ from two disjoint halves of its real
    wells against the same $\hat\mu(0)$. Without it "median cosine 0.35" cannot
    be told from "at the noise limit".
  - **The ceiling is that value lifted by Spearman–Brown**, not the value
    itself (corrected 2026-10-01 against the first MLP results). The split-half
    cosine is the reliability of a *half-sized* estimate: with
    $s = \|\tau\|^2$ and $v = \sigma^2/n$,
    $r_\text{half} = s/(s+2v)$ while the oracle itself has
    $r_\text{full} = s/(s+v) = 2r_\text{half}/(1+r_\text{half})$, and a
    noiseless generator correlates with it at $\sqrt{r_\text{full}}$. On
    `mcf7_24h` the raw median is 0.451 but the attainable bound is **0.788**, so
    comparing a generated cosine with the raw value understates the model by
    ~1.75x. `evaluate.corrected_ceiling` does the lift; the raw value is still
    reported as `split_half_responder`.
  - **`responder`** (per arm and per compound) is
    $\max_d \|\hat\tau\| > 1.5 \times$ the floor — the §3.8.1 rule, now
    computable. Phase 5's `responders.json` reads it straight off the oracle.
- **E12: Generation pool per arm.** `--pool all` for the four DDPM headline
  arms; `--pool holdout` for the eight FM `cmean` arms, which exist only for the
  decision E9 already scopes to the holdout pool. Saves ~11 h of DiT GPU time.
  - A `--pool all` run also reports the `holdout` **sub-pool** from the same
    generation pass, so E9's headline is free on the DDPM arms.
  - The per-arm well threshold is per sub-pool: `--min_dose_n` 2 (E4) everywhere
    except holdout, where `--min_dose_n_holdout` defaults to **1**. Under the P3
    split a 3-well arm puts exactly **one** well in the holdout, so the E4
    threshold would empty that pool; §3.7 reads holdout in aggregate for the
    same reason.

---

## 4. Implementation TODO

Track progress here. **A box may be marked `[x]` only after a code review**
of the corresponding change (pull-request review, or an equivalent review
recorded on the PR). Author-done, a local smoke test, or “it runs on GPU”
is not enough to check the item. If review asks for follow-ups, leave the
box unchecked until those land and are re-reviewed.

Copied files still need review: confirm they were copied into `lincs/src`
(not imported from `RxRx19a`) and that LINCS edits did not silently change
the ADIGen contract.

### Phase 0 — spec + ingest

- [x] `lincs/src/spec.py`: LINCS `FIELDS` (plate E non-adjustable; well
      row/col role None; `cell_id` / `syn_c` role None, adjustable),
      `OutcomeSpec`, `Paths` (`lincs/lincs/GSE70138`), `PopulationSpec`
      (with `name`); helpers copied from RxRx19a then adapted
- [x] Phase 0 decisions frozen in `spec.py` (cell line, compound universe,
      dose handling, per-gene norm, plate centring + QC) — §3.12.
      Reopened and re-closed 2026-09-29 for the amended decisions 3 and 5:
      - `ActionSpec.continuous_kernel_bandwidth` and its `decisions_record`
        entry removed (P8);
      - `OutcomeSpec.normalize` replaced by `normalize_mean="train_dmso"` /
        `normalize_std="train_all"` (P10);
      - `mcf7_24h` and `mcf7_24h_limit1500` rebuilt. The table, `expr.npy`,
        and every encoder and vocab file are byte-identical to before; only
        `population_qc.json`'s decisions record changed;
      - `check_build` now also asserts that a build's recorded decisions equal
        the current `decisions_record` (35 checks, all pass on both builds).
- [x] `lincs/src/data/build_dataset.py`: h5py GCTX read joined on `inst_id`,
      filters, plate QC → `population_qc.json`, `-666` handling,
      `dose_level`, `syn_c` (balanced per arm / per plate DMSO), HF table +
      `expr.npy` / `gene_order.json` / vocabs / `context_encoder.json` (from
      the filtered table) / `nuisance_meta.json`
- [x] `lincs/requirements.txt` (RxRx pins + h5py; cmapPy optional)
- [x] Smoke on `--limit`: \(n\), compound count, `dose_level` counts,
      `(n, 978)`, vehicle fraction, DMSO per plate, plates dropped by QC

### Phase 1 — table consumers + nuisances

- [x] §3.13 decisions D1, P1–P12 chosen and recorded (2026-09-29; P12 and
      the P1 / P2 amendments 2026-09-30)
      - Reopened and re-checked 2026-09-30 (P12 added, P1 / P2 amended).
Checked 2026-09-29 after two independent code reviews (data layer;
nuisances + smoke test), a re-review of every follow-up, and smoke runs:
- local, numpy only, torch imports blocked: the `--limit 1500` build (v1,
  thinning instances k0 γ0/γ1 and k1 γ1);
- CPU jobs on `bindel`: the full `mcf7_24h` pipeline and the limit build
  (`scripts/phase1_cpu.sub`, `scripts/phase1_tier_smoke.sub`);
- a GPU job on `zabih`: `src/tests/smoke_phase1_torch.py`.

The read-back checks are `src/tests/check_phase1.py` (all pass on every
instance). What changed against the plan text is in §5, "Phase 1
implementation and review".

- [x] `splits.py`: copied `load_splits` + stratification engine; cache key =
      population key + table fingerprint; arm strata; DMSO stratified by
      plate; realised holdout fraction logged (P3)
      - Reopened and re-checked 2026-09-30 (P12): `dose_half`, `POSITIVITY_KEYS`,
        `positivity_cells` added. `dose_half` equals the old thinning rule on
        all 110 limit-build compounds (review).
      - mcf7_24h: train 26,690, holdout 10,650, realised 0.285 (treated
        0.290, DMSO 0.201); 725 / 10,479 arms without a holdout well; ≥ 14
        train DMSO per plate. Every stratum's holdout count is re-derived.
      - Each stratum has its own `(seed, stratum)` generator (not RxRx's
        single stream), so a reserve changes no other stratum's holdout.
- [x] `build_tiered_split.py`: reserve (`k_reserve` 0 default) / split /
      thinning layers with `_calibrate_pi`; v1 instance = no scored
      compounds, no thinning → `splits.json` + `nu_rows.npy` (P4, P6)
      - Reopened and re-checked 2026-09-30 (P12): the dose half comes from
        `splits.dose_half`, and `tier.positivity_key` is recorded. The three
        limit-build instances rebuild identical.
      - Thinning is generic over `--scored_compounds` and implements
        `--confounder syn_c`. `cell_id` raises until Phase 6 declares the
        line groups; `--plan` (it needs the oracle) is Phase 5.
      - Design weight $P_c/\pi$ (§3.8.1), recomputed from the table by
        `check_phase1`.
      - Unscored holdout = v1's at k = 0 and k = 1 (limit build).
- [x] `build_nu_rows.py`: \(\nu\) = unthinned train pool minus reserve
- [x] `expr_stats.py`: train-DMSO plate centres; per-gene mean from centred
      train DMSO, std from all centred train rows (decision 5, amended);
      per-gene fraction at the 15.0 cap → `expr_meta.json`
      - "Train" is `nu_rows.npy`, the unthinned train pool: the train rows in
        v1, and one z-scale for all γ instances of a seed.
      - GAPDH is at the cap in 41.4% of train wells; 9 genes > 1%.
- [x] Centring gate on **held-out** DMSO (P11): per-plate means ≈0; plate
      share of DMSO variance 0.67 → ~0.20
      - mcf7_24h, 414 held-out DMSO wells on 100 plates: per-plate mean
        |z| (against its sampling SE) median 0.38, vs 0.67 expected when
        centred and 1.36 uncentred.
      - Plate share: raw R² 0.728 → 0.310 at chance 0.240; chance-corrected
        ω² 0.642 → 0.092.
      - The raw R² stays above 0.20 because ~4 held-out wells per plate put
        chance at 0.24 (0.10 in the §3.2 measurement). The excess over
        chance (~0.07) matches §3.2's (~0.10).
      - The gate therefore passes on ω² ≤ 0.20 and ≤ 0.5 × before, plus
        |z| ≤ 1.0.
- [x] `dataset.py`: in-memory vector loader (no PIL / image VAE) + copied
      `build_cond_spec` / `cond_from_*` / `dose_probe`
      - The "level" dose encoding is dropped: dose is continuous (decision 3),
        and raw log-dose levels would split arms.
      - Needs `models/conditioning.py`, copied byte-identical here (Phase 2
        box below).
      - GPU smoke passed: z-space, rows re-derived, cond spec (v1 / env /
        C = `syn_c`), CondEmbedder with and without CFG drop.
- [x] `alpha_net.py`: copied; **no `infected` input**
- [x] `knn_dr.py`: copied as a new file (`ess` / `tail_index` for `fit_urr`
      and export). `knn_dr_weights` stripped (retired, D1).
- [x] `fit_urr.py`: copied URR loss; no `infected==1` restriction, no
      `--cell_type` / `experiment`; `--nu_rows`; support and \(\nu\) keyed on
      the `dose_level` arm (P2); arm-stratified cross-fit folds (P5)
      - Reopened and re-checked 2026-09-30 (P12): refuses a thinning instance
        up front, before it removes anything, pointing to `counts`. The limit
        job checks the message. v1 results are unchanged.
      - Folds cover the ν pool, stratified on (arm, X stratum, train or not).
        99.9% of train rows see their arm in the other fold.
      - Val rows are held out of both legs. The first full run diverged
        without this (§5).
      - A (ν arm × X stratum) product-support guard refuses fits it cannot
        support.
      - X = the shared adjustment set; `--nu_rows` defaults to
        `nu_rows.npy`; outputs share a `run_id`.
- [x] `export_urr_weights.py`: copied; no `inf`; vehicle rows \(w = 1\);
      `net` (primary, P1) and `counts` (fallback; `dose_level` arm keys,
      (arm, C) when C is non-empty); compares with `dr_weights_design.npz`
      when present; writes `dr_weights_{urr,counts}.npz` `{row_id, w}`
      - Reopened and re-checked 2026-09-30 (P12). On a thinning instance,
        `counts` keys on the positivity cell: w = n_ν / n_kept, `--smooth_k`
        0 (negative refused).
      - It refuses an adjustment set other than the tier's covariate, and
        `--mode net` exits before torch loads.
      - Limit γ = 1 instance: recomputed exactly from the table; unscored
        rows 1; kept scored rows mean 2.18 vs design 2.07, corr +0.83.
      - Fold f is scored by `fold{f}.pt`. RxRx's `nets[1 - f]` scored rows
        in-sample (§5).
      - Refuses unless the per-fold treated means equal `fit_urr`'s
        out-of-fold means.
- [x] URR gate: beats constant baseline, `mean(alpha)≈1`, ESS/n usable
      (trivial on v1 — §3.3); known answers hold: net
      \(\alpha \approx 1.066\) on treated / ≈0 on DMSO, `counts` \(w \equiv 1\)
      - Reopened and re-checked 2026-09-30 (P12): the v1 known answers
        re-verified on the final code (identical numbers; `counts` ≡ 1).
      - mcf7_24h: both folds USABLE. Beats constant by +0.070 / +0.061;
        mean 0.999 / 0.997; ESS 93.8% / 93.9%; tail k 0.001 / 0.005.
      - α treated 1.0649 / 1.0622 vs 1/(1−π₀) = 1.0652 / 1.0670; DMSO
        0.0001 / 0.0055.
      - `counts` w ≡ 1 exactly. Net weights 1.0636 treated, 1 vehicle;
        normalised 1.0037 / 0.9437 (plan: ~1.004 / ~0.942).
- [x] Trainer gets `n_compounds` without a hidden `fit_urr` dependency
      (`compound_vocab.json` or `build_dataset`'s `nuisance_meta.json`)
      - `nuisance_meta.json` (1751 compounds, `cov_dim` 3) comes from
        `build_dataset`. Tiered dirs copy it; `fit_urr` only checks it.

### Phase 2 — MLP arms

Checked 2026-09-30, after two independent code reviews, two re-reviews of
every follow-up, and these smoke runs:
- local, numpy only with torch blocked: `precompute_cmean` on the limit build;
- a CPU job on `bindel` (`scripts/phase2_cpu.sub`);
- GPU jobs on `zabih`:
  - `scripts/phase2_gpu.sub` on the limit build: 68 checks, a 12-run trainer
    matrix, 2 resumes, 9 refusals and 10 rebuilds. Final job 717146.
  - `scripts/phase2_sanity.sub` on `mcf7_24h`, job 715222.

What changed against the plan text and the RxRx copy is in §5, "Phase 2/3
implementation and review". Phases 2 and 3 were built and reviewed together.

- [x] `models/conditioning.py`: copied as a new file
      - Already copied in Phase 1 (byte-identical; `dataset.py` needs it).
        Re-verified with `cmp` against RxRx19a (Phase 2 review).
- [x] `processes/__init__.py` + `ddpm.py` + `flow_matching.py`: copied as new
      files
      - Byte-identical (`cmp`). They are rank-agnostic: FM broadcasts over
        `x0.ndim`, and diffusers' `add_noise` / `get_velocity` unsqueeze to the
        sample's rank.
- [x] `models/mlp.py`: vector denoiser, full shared interface (§3.4)
      - adaLN-Zero on vectors (`layers.vec_modulate`), and the output is
        exactly 0 at init.
      - MLP-B has 39.8M parameters at the v1 vocabulary.
      - The GPU test checks that CFG drop nulls role A and never C, that a
        mixed mask acts per row, that the timestep is handled per row, and
        that gradient checkpointing gives the same outputs and gradients.
      - It also checks that a fixed-batch overfit uses the conditioning:
        shuffled compounds score worse.
- [x] `models/__init__.py`: `arch=mlp` in `arch.json` (`n_genes`, no image
      VAE keys); `build_generator_from_ckpt` dispatches on `arch`
      - `arch_kwargs` holds the resolved sizes; `gene_pr_ids` and its sha1
        record the gene order.
      - The rebuild uses `arch.json` alone and cross-checks the geometry and
        gene count. It rejects RxRx's `arch="dit"`.
- [x] `train/train_diffusion.py`: copied loop; rank-agnostic loss in train
      and validation; `--arch mlp`; `--dr_mode {conditional, weighted}` +
      `--dr_weights_file` (P7); no `--latent`; V-REx key
      - Fixes (a)–(e) against RxRx (§5).
      - Rows are held on the device, and a resume replays epoch 1 bit for bit.
      - wandb logging (user request); `--mixed_precision` defaults to `no`.
      - Run-dir guards: identity on resume, a lock, refusals before any
        write.
      - V-REx: env = `invariance_env_fields` (plate).
- [x] `nuisances/precompute_cmean.py` + trainer `--cmean_lambda` (default 0)
      / `--cmean_file`, FM only; `--min_n` configurable, coverage logged (P9)
      - numpy only, with `dose_level` arm keys (P2) and `row_gid` for the
        trainer.
      - mcf7_24h, `--min_n 8`: 34 / 10,479 treated arms plus the vehicle key,
        13.6% of train rows (`cmean.npz`). `--min_n 2`: 10,446 arms, 99.9%
        (`cmean_min2.npz`).
      - The vehicle μ̂ has max |·| 3e-9, i.e. z = 0 by construction. The GPU
        test recomputes every group mean from `LincsDataset.y` (max |diff| 0).
- [x] `src/tests/test_lincs_shapes.py` (or equivalent) for a GPU box
      - Synthetic, real-data and `--ckpt_dir` sections.
      - `--ckpt_dir` rebuilds a run from `arch.json` alone. It reproduces the
        logged `val_loss_ema` (EMA) and `val_loss` (training weights) to rel
        0.0.
- [x] `mlp_conditional` arm trainable
      - mcf7_24h, 20 epochs (~2.1k steps, 82 it/s): val loss 0.864 → 0.784,
        EMA 0.952 → 0.811, still falling. wandb online worked.
- [x] `mlp_dr` arm trainable (after `fit_urr` + `export_urr_weights --mode
      net`)
      - `dr_weights_urr.npz` on mcf7_24h, after the mean-1 normalisation:
        treated 1.0037, vehicle 0.9437 (plan: ~1.004 / ~0.942), ESS/n 1.000.
      - `dr_weights_counts.npz` (≡ 1) reproduces the conditional loss exactly.
      - The weighted-design run on the step-C tier dir, with C = `syn_c`,
        trains.

### Phase 3 — 1D-DiT arms

- [x] `models/dit1d.py`: 1-d patch embed + copied `DiTBlock` /
      `TimestepEmbedder` / 1-d sin-cos (not `DiT2DModel`)
      - The patch embed is a Linear on the zero-padded `(B, T, p)` view.
        `DiTBlock` is copied without cross-attention (xattn not ported). The
        frozen 1-d sin-cos is a persistent buffer.
      - p = 10: 98 tokens, pad 2. p = 6: 163 tokens, no pad. The pad is sliced
        off.
      - DiT-S/10 has 33.3M parameters. It runs at 5.6 it/s in fp32/TF32 on an
        A6000 (~2.5 h per 50k steps).
- [x] `arch.json` rebuild for `arch=dit1d` (`patch_size`, pad, gene order)
      - `arch_kwargs.patch_size`, `patch_geometry` `{pad, n_tokens}`,
        `gene_pr_ids`.
      - `test_lincs_shapes --ckpt_dir` rebuilds the dit1d runs and reproduces
        their logged losses: `dit_cond_ddpm`, `dit_cond_fm`, `dit_urr` and
        the mcf7_24h sanity run.
- [x] `dit1d_conditional` arm trainable
      - mcf7_24h, 3 epochs: val loss 0.955 → 0.804.
- [x] `dit1d_dr` arm trainable
      - The smoke run `dit_urr` trains; the resume replays bit for bit.

Full-length v1 training (plan step F: DDPM, seed 0, 500 epochs ≈ 52k steps,
wandb project `lincs-adigen`) was launched 2026-09-30 on `zabih`. Jobs:

| arm | job |
|---|---|
| `mlp_conditional` | 717515 |
| `mlp_dr` | 717516 |
| `dit1d_conditional` | 717517 |
| `dit1d_dr` | 717518 |

Run dirs: `runs/mcf7_24h/{mlp-B,dit1d-S-p10}_{conditional,weighted-urr}_ddpm_s0`.
Phase 4 scores `checkpoint-0499` (E1, §3.14).

All four finished (exit 0, 500 epochs, five checkpoints each). Final
`val_loss_ema`, on the 2,000 fixed holdout rows:

| arm | `val_loss_ema` | wall time |
|---|---|---|
| `mlp_conditional` | 0.7027 | 11 min |
| `mlp_dr` | 0.7026 | 11 min |
| `dit1d_conditional` | 0.5545 | 2 h 37 |
| `dit1d_dr` | 0.5547 | 2 h 38 |

- `dr` ≈ `conditional` on both backbones (to ~2e-4): the v1 null check holds
  on the denoising loss.
- The MLP has plateaued with mild overfitting; the DiT was still improving
  slowly (~0.002 per 100 epochs).
- This is denoising loss, not τ accuracy. Phase 4 decides between backbones.

Eight FM arms were also trained on 2026-10-01 for the P9 `cmean` ablation
(same split, seed and budget; `cmean_min2.npz`, λ ∈ {0, 0.1}). Their results
are in §5, "cmean ablation: training results".

### Phase 4 — eval

§3.14 decisions E10–E12 chosen and recorded (2026-10-01, before any Phase 4
code). Written, smoke-tested and run 2026-10-01. What landed and what it changed
is in §5, "Phase 4 implementation and review".

Reviewed 2026-10-05, in the status review the user asked for (§5, "Phase 4 and
Phase 5 status review"):
- an independent code review of the Phase 4 code, which had none until then;
- every artifact and job of record read back from disk;
- the results table recomputed from the arm JSONs.

The review found no bug that changes a recorded number, so the two experiment
boxes and `dist_metrics.py` are checked. It asks for follow-ups on three code
items (R1–R8 in that entry), and those stay unchecked until the fixes land and
are re-reviewed. The quality block is a further open item (last box).

- [ ] `eval/generation.py`: CFG sampling → `(B, 978)`; no clamp, no
      `channels_last`; rebuild from `arch.json` only
      - Also `check_arm_against_data` (the gene-order / provenance guard
        `build_generator_from_ckpt` delegates to eval) and
        `generate_for_rows`, the streaming driver.
      - **`clip_sample=False`** had to be pinned in `processes/ddpm.py`: the
        diffusers default clips predicted \(x_0\) to \([-1, 1]\) every step,
        which truncated every DDPM sample (§5).
      - Reviewed 2026-10-05: the path every run used (DDIM or FM Euler, 100
        steps) is correct. Open:
        - R1: `--sampler ddpm` uses the wrong timestep spacing and is unseeded.
          No run used it.
        - R2: `check_arm_against_data` skips an identity field that is absent
          from `arch.json`.
        - R6: the reservoir cap (with `evaluate`).
- [x] `eval/dist_metrics.py`: Frechet (marginal / PCs) + MMD on \(Y\)
      (copied math; no Inception, no torchvision)
      - RxRx's `rbf_mmd2` had to be rewritten: its `(n, n, d)` temporary is
        ~125 GB at d = 978 (§5).
      - `_eigh_psd` and `assert_blas_ok` were added on 2026-10-02, after the
        Sapphire Rapids BLAS was found to be wrong (§5, "Step C2 scoring").
      - Reviewed 2026-10-05: the math is correct, and `_eigh_psd` fails closed.
        Four notes are not acted on (§5).
- [ ] `eval/evaluate.py`: gene-space \(\hat\tau(c,d)\) on `dose_level`;
      `--pool all` per-arm aggregates + holdout aggregates; real and
      generated \(\mu(0)\); `--truth` accuracy; no OpenPhenom / rescue panel
      - The `--source real` path imports no torch, so the oracle is a CPU job.
      - Plus the E11 reference scale: the DMSO noise floor per arm size and the
        split-half reliability ceiling.
      - No `--gen_anchors` run was made: the generated \(\hat\mu(0)\) is
        reported as a norm, and \(\hat\tau\) always uses the real vehicle.
      - Reviewed 2026-10-05: the τ̂ path, the arm filters, the row alignment and
        the floor are correct, and one arm's row was re-derived from its
        `_tau.npz`. Open:
        - R3: the oracle's file name omits `responder_mult` and three other
          settings, so a non-default oracle run overwrites the one Phase 5
          reads.
        - R4: the `--truth` and overwrite guards run after sampling.
        - R5: the `/ceil` column's arm set, and the unequal halves.
        - R6: pooled-treated quality compares two dose mixtures.
      - The responder rule itself passes noise often (§5, gap 3). That is a
        question about E11, not about the code.
- [ ] `src/tests/check_phase4.py` (numpy read-back) and
      `src/tests/smoke_phase4_torch.py` (GPU sampler), with
      `scripts/{eval_oracle,eval_arm,eval_steps,phase4_gpu}.sub` and
      `scripts/eval_all_arms.sh`
      - Reviewed 2026-10-05: the scripts are clean, and the τ / μ̂(0) recompute
        and the sampler's invariance checks do test what they claim. Open:
        - R7: `check_phase4`'s responder and floor checks cannot fail.
        - R8: `smoke_phase4_torch` does not test the schedule's start, and its
          docstring lists a check that has no code.
      - `phase4_gpu.sub` last ran on 2026-10-01 (job 792606), before the later
        changes to `evaluate.py`. `check_phase4` passed again on 2026-10-05
        (regression job 991754), and the later GPU smokes run the eval end to
        end. The sampler checks themselves have not been rerun since.
- [x] Oracle (`--source real`) + all four v1 generator arms scored
      - Settings are fixed in §3.14 (E1–E12): `checkpoint-0499` EMA, w = 1,
        16 samples per real row, `--min_dose_n 2`, 100 sampler steps, both
        Fréchet variants and MMD.
      - Oracle: job 793791 on `bindel`. It replaced job 791843's file with the
        same numbers plus the noise-corrected ‖τ̂‖ arrays.
      - Arms: jobs 793825–793828 on `zabih`, all exit 0. Results in §5,
        "Phase 4 results": every arm is at the reliability ceiling, and `dr`
        equals `conditional` on τ.
      - E10's step-count check ran as job 792670. 100 steps are enough for τ̂
        on the DDIM arms (§5, status review).
      - The quality numbers of these runs are not valid (last box).
- [x] `cmean` ablation (P9): FM `mlp_conditional` / `mlp_dr` with
      `--cmean_lambda` 0 vs > 0, scored with `--truth`
      - Training is done for both backbones (eight FM arms; §5).
      - All eight FM arms are scored on `--pool holdout` (E9, E12): jobs
        793829–793836, all exit 0.
      - Verdict (§5, "Phase 4 results", finding 4): no. The τ gain is mixed, so
        `--cmean_lambda 0` stays the default. The per-compound aggregate and the
        DMSO marginal Fréchet agree (§5, status review).
- [ ] **Open (status review, 2026-10-05): recompute the quality block** (PC
      Fréchet, MMD) of the 12 Phase 4 arms and the 24 step-C arms
      - They were scored on `zabih` before the BLAS fix, so those numbers are
        wrong and E5 is not met for these arms.
      - Phase 4's are visibly broken (PC Fréchet ~1e141). Step C's look
        plausible, and nothing in the JSON marks them.
      - Every `_gen.npz` holds its reservoirs, so this is a CPU recompute on
        `bindel`, not the GPU rescoring §5 costed. No script does it yet.
      - Fix R6 in the same pass: the pooled-treated numbers of every run,
        C2's included, compare two dose mixtures.
      - No Phase 4 or step-C conclusion uses these numbers.

### Phase 5 — step C: semi-synthetic confounder on MCF7 (§3.8.2)

Reviewed 2026-10-05, in the status review the user asked for (§5, "Phase 4 and
Phase 5 status review"):
- an independent code review of the step-C base code, which had none until
  then. C2, P1 and P2 had theirs when they were written (§5);
- every artifact and job of record read back from disk;
- the four verdict tables recomputed from the JSONs.

The review found no bug that changes a recorded number. It asks for follow-ups
on `step_c_report` and on the Phase 5 checks and launchers (R9–R14 in that
entry), so those two boxes and C2's parent stay unchecked. One analysis item is
also open (last box).

A checked box means the item is built, reviewed and run. It does not mean its
criterion passed: step C's criterion 2 is untestable, C2's fails, and P2's pass
rule fails. Each is recorded under its item.

- [x] `responders.json` from the Phase 4 real oracle (max-over-dose
      \(\|\hat\tau\|\) > 1.5× noise floor)
      - `src/data/responders.py`. On `mcf7_24h`: 651 responders, 641 scored
        after the thinnability screen (§5, "Phase 5 progress").
      - The rule passes noise often, so part of the scored set may have no real
        effect. Steps C and C2 do not depend on that; step A does (§5, status
        review, gap 3).
- [x] `synthetic.py`: seeded `v`, `β` resolution, injection in `dataset.py`
      after centring / z-scoring; `syn_effect` / `syn_seed` in `arch.json`
      - The resolved injection is `syn_meta.json`: β = 25.483, `v` sha1
        `492aa3d1`.
- [x] `evaluate.py`: `--syn_effect` / `--syn_seed` on the real oracle;
      `--truth` refuses mismatches; signed bias along `v` by dose half,
      rare vs non-rare
- [x] `build_tiered_split` step-C instance: responders-only scored
      compounds, \(z\) = dose half × `syn_c`, positivity cells (compound,
      dose half, `syn_c`), `keep_frac` 0.4, γ ∈ {0, 1}; `--plan` bias table;
      `dr_weights_design.npz`, `nu_rows.npy`, `tier_meta.json`
      - Both instances are on disk (`nuisances_tier_Csyn_c_k0_g{0,1}_s42`):
        21,293 / 21,600 train rows, and v1's 10,650 holdout rows in both.
      - `--plan` prints its table and writes nothing. `tier_meta.json`'s `bias`
        field still holds its Phase 1 placeholder text.
      - The control is not size-matched: the realised kept fraction is 0.468 at
        γ = 0 and 0.498 at γ = 1. Step A's fix for this (`STEP_A.md` D9) covers
        `cell_id` only.
- [x] Nuisances: `export_urr_weights --mode counts --adjustment_set syn_c`
      (positivity-cell weights, P12); weights vary in `syn_c` on scored
      compounds, are 1 on unscored ones, and track `dr_weights_design.npz`
      - Decided 2026-09-30 (P12). The cross-fitted net cannot run here. Share
        of ν actions per fold without a factual fit row, all / scored
        (mcf7_24h, replayed fold/val construction, 100 simulated scored
        compounds, γ = 1, keep_frac 0.4):

        | action key | v1, X = ∅ | v1, X = `syn_c` | thinned, X = `syn_c` | same, 5 folds |
        |---|---|---|---|---|
        | arm (compound, dose_level) | 0 / 0 | 77% / 57% | 78% / 69% | 62% / 80% |
        | compound × dose half | 0 / 0 | 32% / 31% | 34% / 67% | 12% / 38% |

      - On the limit build's instances the positivity-cell weights track the
        design weights (kept scored rows, γ = 1: mean 2.18 vs 2.07, corr
        +0.83), where (arm, `syn_c`) keys gave means of 1.1–1.3.
      - On `mcf7_24h`, kept scored rows (jobs 798892 and 817859): corr +0.27 at
        γ = 0 and +0.84 at γ = 1; means 2.16 vs 2.14 and 2.03 vs 2.00. The +0.73
        and +0.88 quoted in §5 and in the verdict files are over all train rows
        (§5, status review, gap 5).
- [x] `naive` / `conditional` / `dr` (+ `dr_design`) × γ ∈ {0, 1} × ≥2
      seeds trained and scored
      - All 24 runs trained and scored 2026-10-01 (3 training seeds, one
        thinning seed; jobs 828810–828878 on `zabih`, all exit 0). Results in
        §5, "Phase 5 step-C results".
- [x] Step-C verdict against the §3.8.1 success criteria written up (gate
      for Phase 6)
      - Written up in §5 (2026-10-01). Criteria 1, 3 and 4 pass. Criterion 2
        cannot be tested under step C's injection: the conditional arm is
        correctly specified, so `dr` has no bias left to remove.
- [ ] **Step C2** (§3.8.4): a step-C variant in which `dr` can beat
      `conditional`, via a compound-specific injection at ρ = 1 (user,
      2026-10-01: option B). **Runs before Phase 6.**
      - The experiment is finished and its verdict is written. The box stays
        unchecked for the two code sub-items below.
  - [x] `synthetic.py --mode compound --rho`: per-compound directions,
        `v_sha1` over the whole matrix, ρ = 0 bit-identical to step C;
        `--syn_meta` selects the file (trainer, eval, report, tests)
  - [x] `evaluate.py`: injection and vehicle offset per compound;
        `bias_along_v` projects each arm on its own v_k; `learned_syn_effect`
        (λ̂); oracle tag `_syn1-cmp1`
  - [ ] `step_c_report`: `--syn_meta` filter, λ̂ columns, criterion 2 judged
        in full (including "close to `dr_design`")
        - Reviewed 2026-10-05: the labelling, the `SCORING` filter, the
          duplicate refusal and the DiD arithmetic are correct, and both
          verdicts match §5. Open:
          - R9: criterion 1's γ = 0 half is not judged, criterion 4's
            correlations are constants, and no clustered SE is reported.
          - R10: admission does not pin the tier instance, the model class or
            the sampler.
  - [ ] `check_phase5` / `smoke_phase5_torch` / `phase5_gpu.sub` cover the
        compound mode; `phase5_arms.sh` takes the variant
        - Reviewed 2026-10-05. Open:
          - R11, R12: three checks cannot fail on what they name.
          - R13: the step-C `--truth` refusal is not exercised.
          - R14: `phase5_cpu.sub`'s tier-dir glob, and `phase5_arms.sh`'s
            dependency flag.
  - [x] Injected oracle (CPU) + GPU smoke green, then
        `naive` / `conditional` / `dr` / `dr_design` × γ ∈ {0, 1} × 3 seeds
        trained and scored on `zabih`
        - The code above is written, reviewed (automated `code-review`; 8 of 10
          findings fixed, §5) and smoke-tested.
        - Oracle and checks: CPU job 850257 on `bindel`, 81 checks, 0
          failures. GPU smoke: 850258 on `zabih`, 52 checks, 0 failures.
        - The 24 runs were submitted 2026-10-01 with
          `VARIANT=c2 bash scripts/phase5_arms.sh`: training 850418–850465
          (even IDs from 850418 to 850426, odd from 850429), each scored by the
          next ID.
        - The first scoring wave crashed on a wrong BLAS (§5) and was
          cancelled. It was rescored on the fixed scripts: 854859 and
          856859–856968, all 24 exit 0.
  - [x] Verdict against §3.8.4's success criteria written up (gate for
        Phase 6)
        - Written up in §5, "Step C2 results" (2026-10-02). Criteria 1, 3 and 4
          pass; **criterion 2 fails**: `dr` over-corrects (+0.58 against
          `conditional`'s −0.28), from arm-level Hájek bias of cell-level
          weights.
        - **That mechanism is not established** (§5, status review, gaps 1 and
          2). The whole gap is a difference in λ̂ between the dose halves, and
          `conditional`'s −0.28 is its γ = 0 control's +0.32. The DiD values and
          the FAIL do not change.
        - **Resolved by P1** (§5, "P1 results", 2026-10-04): the same weights
          spent on the estimand instead of the training risk remove
          essentially all of the bias, so the failure was *where* α was
          applied, not α itself.
        - Phase 6 proceeded on the user's instruction (2026-10-05; §5, "Step A
          launch").
- [x] **P1 — post-hoc DR targeting** (`understand.md` §3.3.1), the response to
      C2's criterion-2 failure: stop weighting the training risk, keep the
      unweighted generator as the outcome model, and spend α only on an AIPW
      correction to the estimand at the group where positivity holds —
      g = the positivity cell with the confounder dropped
      (`splits.target_groups`).
  - [x] `src/eval/dr_target.py` (the estimator, numpy only, post hoc over
        finished scoring runs), `src/tests/check_p1.py`,
        `scripts/p1_target_cpu.sub`, and `step_c_report`'s `--dr_arm` /
        `--design_arm` / `--baseline_arm` so a targeted arm can be judged by
        the same four criteria without moving the existing verdicts
        - Reviewed when written (automated `code-review`; five findings, all
          fixed, §5). Not reviewed again on 2026-10-05.
  - [x] Targeted arms: `conditional` **and `naive`** sources × {`counts`,
        `design`} = `p1_{cond,naive}_{counts,design}`, plus the `ones`
        α-sensitivity control. `naive` is the textbook double-robustness case
        (outcome model blind to the confounder, α correct) and fills the 2×2
        corner the retired "Option A" was to provide
        - 48 targeted documents on disk: 36 on C2 (2 sources × 3 weight sets ×
          6 runs) and 12 on step C (2 sources × `counts` × 6 runs).
  - [x] **Success**, judged by §3.8.4's criteria with `--dr_arm
        p1_cond_counts --design_arm p1_cond_design` (and the `naive` family
        against `--baseline_arm naive`), plus three controls that must all
        hold: the step-C NULL (where `conditional` is already unbiased, so the
        correction must not introduce a bias), the α-sensitivity control
        (`ones` must be far WORSE than `counts` — the evidence that P1 is not
        fitting the statistic with one free parameter per group), and the
        unscored negative control
        - All three controls are asserted by `check_p1` (job 964812, ALL CHECKS
          PASSED).
  - [x] Read on `--pool holdout` as well as `all`: δ is built from train rows
        and the `all` oracle shares those wells, so the holdout (three disjoint
        row sets) is the leakage diagnostic for the absence of cross-fitting
  - [x] Verdict written up in §5; if it passes, P1 goes into Phase 6 beside the
        weighted arm and step A compares the two
        - **PASSED** (2026-10-04), see §5 "P1 results". All four criteria in
          both families, on `--pool all` and `holdout`.
        - Those margins are in seed sd. The `naive` family is decisive on any
          error (−1.93 → −0.001). The `cond` family's paired gain over
          `conditional` has no clustered error yet (open item below).
- [x] **P2 — group-normalised, capped weights in the training risk**
      (`understand.md` §3.3.2; `STEP_A.md` §4, track B, where it is tracked as
      B1–B5): six `dr_p2` runs on step C2
      - Code: `src/nuisances/weight_norm.py`, `--dr_weight_norm` /
        `--dr_weight_clip`, `step_c_report`'s `dr_p2` arm and `p2_pass_rule`,
        `src/tests/check_p2.py`, `scripts/p2_gpu.sub`. Written and reviewed
        2026-10-05 (§5, "P2 implementation and review").
      - GPU smoke 990819; training 990856–990866 and scoring 990857–990867 on
        `zabih`, all exit 0.
      - **Pass rule: FAIL on the third part** (|DiD| 0.599 against ≤ 0.58).
        The other two parts pass: λ̂ on unthinned compounds 0.67, and MSE
        1.05× `conditional`'s (§5, "P2 results").
      - User decision 2026-10-05: override. `dr_p2` runs in step A next to
        `dr` (`STEP_A.md` D8).
- [ ] **Open (status review, 2026-10-05): put the C2 readout on firm ground**
      (§5, status review, gaps 1 and 2)
      - Report λ̂ by dose half and the γ = 0 contrasts next to every DiD, and
        split each DiD into its λ term and its remainder.
      - Replace the seed sd with a compound-clustered error
        (`src/eval/contrast_stats.py`), on both pools and for the paired
        `p1_cond_counts` − `conditional` difference.
      - Move the one-off analyses into the repo: the memoriser's DiD, the
        tables by kept train wells, the λ slope, the MSE orthogonal to v.
      - All of it is CPU work on the saved `_tau.npz` and `_gen.npz` files.

### Phase 6 — step A: cell line on `core5_24h` (§3.8.3)

- [ ] Effect-modification gate: responders' centred \(\hat\tau\) differs
      across the 5 lines beyond the 3-well noise floor
      - `src/data/line_gate.py` + `scripts/phase6_gate_cpu.sub`, written and
        launched 2026-10-02 (job 862040 on `bindel`; `ma` was at 8.6% CPU /
        8.7% memory free, failing the 20% rule). **PASSED** — see §5, "Phase 6
        effect-modification gate". Cell line modifies the response, so step A
        is not another null check.
      - It needs **no `core5_24h` build**: it selects the population from
        `inst_info`, reads the 641 MCF7 responders' wells in the 5 lines
        straight from the GCTX (76,587 wells, 286 MiB), and applies the same
        outcome rules the build would (3x plate QC on each line's median
        spread, DMSO-median centring, one pooled z-scale).
      - **Statistic.** Per arm and line pair, \(\cos(\hat\tau_{L_1},
        \hat\tau_{L_2})\) against the full-sample reliability
        \(r_\text{full} = 2 r_\text{half} / (1 + r_\text{half})\) from the
        within-line split-half (E11): two independent estimates of the *same*
        \(\tau\) correlate at \(r_\text{full}\), so
        \(r_\text{cross} < r_\text{full}\) is effect modification. Plus
        §3.8.3's literal reading, \(\|\hat\tau_{L_1} - \hat\tau_{L_2}\|\)
        over \(\sqrt{\text{floor}_1^2 + \text{floor}_2^2}\), which is ~1
        under one shared \(\tau\). Restricted to arms responding in at least
        one of the two lines; the threshold
        (`--max_ratio_alike`, default 0.8) is recorded in the artifact.
      - **MCF7 pairs are reported apart.** `responders.json` is MCF7's, so
        MCF7's \(\hat\tau\) is selected on being large. The headline is the
        median over the 6 pairs among the four unselected lines.
- [ ] `--population` switch; `core5_24h` paths; `PopulationSpec.name` in
      every artifact and result
      - **Written 2026-10-05** (`spec.POPULATIONS`, `--population`; §5, "Step A
        implementation and review"). A command needs only `--data_dir`: the
        build names its own population, and a contradicting `--population` is
        refused. Smoke-tested on a `--limit` build; no full build yet.
- [ ] Ingest, plate QC (per-line median), splits, `expr_stats`, Phase 4
      oracle and `responders.json` for `core5_24h`
      - Unblocked 2026-10-05: `scripts/phase6_cpu.sub` runs this and the two
        items below as one CPU job. Done on a `--limit` build only (2 whole
        plate maps, 11,904 wells); the full build (182,919 wells before QC) is
        not run yet. The ingest reads landmark genes only, so it needs far less
        than the ~9 GB §3.8.3 quotes.
- [ ] `build_tiered_split` step-A instance: positivity cells (compound,
      `dose_level`, `cell_id`), `keep_frac` 0.6; line groups declared in
      `spec.py`
      - Written 2026-10-05; line groups are `spec.LINE_GROUPS`
        (`STEP_A.md` D3). **`keep_frac` is the realised fraction here, default
        0.75** (`STEP_A.md` D9, approved 2026-10-05; §5).
- [ ] Nuisances: `export_urr_weights --mode counts --adjustment_set cell_id`
      (weights at (arm, `cell_id`), P12); weights track `dr_weights_design.npz`
- [ ] `naive` / `conditional` / `dr` (+ `dr_design`) × γ ∈ {0, γ>0} × ≥2
      seeds trained and scored; group-contrast bias readout
      - **Carry P1 too** (§5, "P1 results"): `src/eval/dr_target.py` needs no
        change for step A — the group is derived from `POSITIVITY_KEYS`, and
        for `cell_id` it is the arm itself. That is the resolution C2 showed
        the weighted risk cannot reach, so step A is where the two approaches
        should separate on real data. The group is the arm pooled over its
        lines: ~10 train wells before thinning, 7–8 after (n_eff 7–9 measured
        on the limit build), so δ_g is no noisier than in step C2. (An earlier
        note here said n_eff ≈ 2; that was the (arm, line) cell, not the group.)

### Phase 7 — optional

- [ ] V-REx over plate
- [ ] Pathway-score \(Y\) option
- [ ] DDPM vs FM comparison
- [ ] 1D-DiT on the confounding steps
- [ ] LINCS SLURM launchers (`python -m` remains source of truth)

---

## 5. Review log (2026-09-28)

Corrections made against the RxRx19a code and the files on disk. The
pre-review text is in `IMPLEMENT.md.orig` (untracked).

| Changed | Why (measured unless noted) |
|---|---|
| `rna_plate` / `rna_well` → `det_plate` / `det_well` | GSE70138 `inst_info` has no `rna_*` columns (those are GSE92742 names) |
| `plate` role E **adjustable** → E non-adjustable | `ALPHA_ENV` would put plate in $\alpha$ on every arm; 0 arms span all plates → `fit_urr` raises "target support is empty" |
| `well_row` / `well_col` E → role None; `edge` dropped | arms are position-fixed (72 / 10,944 span edge and interior); plate\|row\|col env = one well for V-REx |
| "design-balanced" rationale for `C = ∅` | plate explains 65% of DMSO variance, and compounds sit on their own plates; added plate centring (decision 6) |
| v1 DR expectation | $C = \varnothing$ and no α-reachable E ⇒ `knn_dr` ≈ `conditional`; gate passes trivially; `ipw` would zero the vehicle arm |
| No transpose on the h5py path; join on `inst_id` | matrix is stored (wells, genes); GCTX column order ≠ `inst_info` order |
| Data path `lincs/GSE70138` → `lincs/lincs/GSE70138`; `LINCS_ROOT = lincs/` | where `download.sh` put the files; mirrors `RxRx19a/RxRx19a/` |
| `-666` vehicle dose; `pert_time` `"24.0"`; vocab on `pert_id` | LINCS sentinels / string types; 17 inames map to >1 `pert_id` |
| Eval grid = 7 nominal `dose_level`s, not 97 doses | 99.5% of treated wells within 0.05 log10 of the 3-fold series + 20 µM |
| Eval pool / `--min_dose_n` | ~3 wells per arm; holdout ≤1 per arm; RxRx `--min_dose_n 20` keeps 12 of ~10.4k arms |
| Split strata = arm; realised holdout ≈⅓ | `round(0.2·3) = 1` |
| Splits `population_filters` as cache key | RxRx masks by string equality → empty population on `"24.0"` / set-valued keys |
| Normalisation stats move to `expr_stats` after splits | build runs before the split exists |
| `expr.npy` float32 | 147 MB; float16 quantises log2 values |
| `context_encoder.json` built eagerly from the filtered table | RxRx builds it lazily from the raw CSV, which here would be all 346k wells |
| Generation: no clamp, no `channels_last`; val loss rank-agnostic | code in `generation.py` / `_validation_loss` (code read) |
| Model interface adds `calibrate_conditioning`, `cond_spec`, `class_dropout_prob`, grad-ckpt; `build_generator_from_ckpt` dispatch | trainer / generation call these (code read) |
| 1D-DiT default B/10 → S/10; "978 tokens" → 98 | patch 10 gives 98 tokens; S ≈ MLP-B parameter count |
| RxRx confound lever marked non-transferable | (compound, dose, edge) cell = whole arm with ~2 train wells vs `min_cell=2` |
| Test script → `src/tests/` | `.gitignore` drops `lincs/scripts/` |
| Reference panel named | compounds and well counts present in MCF7 24 h |

### Decisions after the review (2026-09-28)

| Decision | Basis (measured) |
|---|---|
| 6 accepted: DMSO-median centring on train wells, no scaling | held-out DMSO plate share 0.67 → 0.20 (chance 0.10); poscon plate share 0.65 → 0.33, while scaling variants give 0.39 / 0.44; plate offset > effect in 93% of 400 arms |
| 6 includes plate QC: drop plates with DMSO spread > 3× median | spreads 0.14–0.53 plus one outlier at 1.16; drops 1 plate (367 wells), no arm emptied |
| 7 accepted: confounding in two steps, C then A | plate-level confounders are removed by centring; the replicate index carries 2% of variance; cell lines share ≥99.8% of MCF7 arms (4 lines) |
| Step C confounds at (compound, dose half) with `--target_support all` (weights superseded by P12, 2026-09-30) | ~2 train wells per arm leave (arm, `syn_c`) cells with ~1 row |
| Step A confounds at (compound, `dose_level`, `cell_id`) with `--target_support common` (weights superseded by P12, 2026-09-30) | ~2 train rows per cell keep arm-level positivity at `min_cell 1` |
| `syn_c` and `cell_id` declared adjustable, role `None` | promotable per arm; inert in v1 |

### Decisions 2026-09-29

| Decision | Basis |
|---|---|
| D1: DR weights from `export_urr_weights` (`counts` / `net`); `fit_knn_dr` (kNN AIPW) retired as outdated | RxRx `78d4284` replaced `fit_knn_dr` / `alpha_truth_check` with `export_urr_weights` (+ `precompute_cmean`); the LINCS plan follows. Vehicle rows keep $w = 1$ by policy, so the old "`ipw` zeroes the vehicle arm" concern is gone |
| Stale text updated: §1, §2.2, §2.3, §3.1, §3.3, §3.5, §3.6, §3.8.1, §3.9, §3.10, §3.12, §4 | kNN buckets, `alpha_raw`, `--nu_source full` / `alpha_urr_nufull` → `--nu_rows` + export |
| Realised holdout corrected: 28.5% at `holdout_frac` 0.2, not ⅓ | measured on the built table: `round(0.2·2) = 0` keeps 692 two-well arms whole |
| §3.8 mechanism flagged, not rewritten | RxRx `78d4284` also removed `rarity.py` in favour of `build_tiered_split`; open decision P6 |
| Phase 1 open decisions P1–P11 listed in §3.13 | to be chosen before Phase 1 code |

### Phase 1 decisions chosen (2026-09-29)

| Decision | Plan changes |
|---|---|
| P1 `net` primary, `counts` fallback | §1, §3.5, §3.6 jobs, TODO |
| P2 `dose_level` arm keys everywhere (still needed under P1) | §3.5, §3.13 |
| P3 holdout 0.2, arm strata, DMSO by plate, fingerprint in cache key | §3.2 splits |
| P4 v1 split = tiered builder's split layer | §3.2, §3.8.1, TODO |
| P5 arm-stratified folds | §3.5 `fit_urr` row |
| P6 RxRx tiered split replaces `rarity.py`; `k_reserve` 0 default; `dr_design` reference arm added | §3.1, §3.8 rewritten, §3.9, §3.10, §3.12 decision 7, TODO Phases 5–6 |
| P7 `--dr_mode {conditional, weighted}` | §1, §2.3, §3.6, §3.8 |
| P8 kNN bandwidth removed from `spec.py` | §3.12 decision 3; Phase 0 box reopened |
| P9 `cmean` ported, optional (`--cmean_lambda` 0 default), ablated on FM arms | §3.1, §3.5, §3.6, §3.9, §3.10, TODO Phases 2 and 4 |
| P10 decision 5 amended: DMSO mean, all-row std, all genes kept | §3.2, §3.3 `OutcomeSpec`, §3.12 decision 5, TODO Phase 1; Phase 0 box reopened |
| P11 centring gate on held-out DMSO | §3.10, TODO Phase 1 |
| Phase 0 fix for P8 / P10 applied in `spec.py`; builds regenerated; `check_build` checks the decisions record | TODO Phase 0 box re-closed |

### Phase 1 implementation and review (2026-09-29)

Two independent code reviews (data layer; nuisances and smoke test), then a
re-review of every follow-up. Numbers are from `mcf7_24h` unless marked.

| Change against the plan text or the RxRx copy | Why |
|---|---|
| `fit_urr`: val rows are held out of **both** URR legs | first full v1 run diverged (fold 0: L_val −1.01 → +15,327, max α 1,405 by step 1,500). With ~1 row per arm per fold, an arm whose only fit row went to val stayed in ν with P_fit = 0, where α = ν/P is unbounded. Fixed run: 0 gap cells |
| `fit_urr`: folds cover the ν pool, stratified on (arm, X stratum, train or not) | with folds over train only, thinned-away rows entered every fold's ν at full weight: the fold target was 1 + (1−π)/(qπ), not 1/π (analytic, ~4.5 vs 2.5 at π = 0.4). Now both legs are half-samples of the design (simulated ratio 1.00–1.03) |
| `fit_urr`: product-support guard `--max_nu_gap_cells` (default 0), X = the shared adjustment set (no `--cov_blocks`), `--nu_rows` default `nu_rows.npy`, outputs written together under one `run_id` | an unsupported (ν arm, X) cell diverges instead of failing; RxRx emptied X whenever `--nu_rows` was set, which would drop `syn_c` in step C; a crashed run could leave old and new nets mixed |
| `export_urr_weights --mode net`: fold f scored by `fold{f}.pt` | RxRx's export uses `nets[1 - f]`, the net **fit on** fold f: in-sample weights. It is invisible in v1 (constant target) but not from step C on. The export now checks that each fold's treated mean equals `fit_urr`'s out-of-fold mean (|diff| 1e-11). **`RxRx19a/src/nuisances/export_urr_weights.py:90` has the same bug; recorded as an aside in §3.1.1, not patched** |
| Tiered split: design weight $P_c/\pi$, not $1/\pi$ (§3.8.1) | the positivity redraw conditions on each cell keeping ≥ 1 well, so the inclusion probability is $\pi_i/P_c$. Monte Carlo on the limit build: z of empirical vs recorded inclusion, mean −0.06, SD 0.98 |
| Split: one generator per stratum, seeded by `(seed, stratum)` | with RxRx's single stream, a reserve (k > 0) reshuffled every later stratum, DMSO included, and so moved the plate centres (up to 2.07 log2 on a k = 1 limit build). Now unscored holdouts equal v1's for any k |
| `expr_stats` fits on `nu_rows.npy` (the unthinned train pool) | thinning changed the per-gene std by up to 7.5% between γ instances; now all γ instances of a seed share one z-scale. v1 is unchanged (ν = train) |
| Centring gate passes on ω² (and per-plate mean \|z\|); R² is reported | ~4 held-out DMSO wells per plate put chance R² at 0.24, so R² after centring (0.31) has a thin margin against "halved". Centring gives ω² 0.642 → 0.092 |
| `dataset.py` drops the "level" dose encoding | 97 raw float doses would split arms; dose is continuous (decision 3) |
| `--data_dir` also moves `runs/` | smoke runs on a `--limit` build would have written into `runs/mcf7_24h` |

Open for Phase 5 at the time: step C's `fit_urr --adjustment_set syn_c` fit
is refused by the product-support guard. Decided 2026-09-30 (P12, below).

### Decision 2026-09-30: positivity-cell weights for steps C / A (P12)

| Change | Where | Why |
|---|---|---|
| `counts` keys on the tier's positivity cell and is the count ratio $n_\nu / n_\text{train}$ (`--smooth_k` 0); it refuses an adjustment set other than the tier's covariate | `export_urr_weights.py` | the post-stratified design weight; exactly 1 on unthinned keys, so v1's `counts ≡ 1` is unchanged |
| `--mode net` refused on a thinning instance | `export_urr_weights.py` | the net cannot be cross-fitted there |
| thinning instances refused up front, pointing to `counts` | `fit_urr.py` | same; the product-support guard stays for v1-type fits |
| `dose_half`, `POSITIVITY_KEYS`, `positivity_cells`: one definition of the cell | `splits.py` | builder, export and check must agree on the cell |
| thinning takes the dose half from `dose_half`; `tier.positivity_key` recorded in `splits.json` | `build_tiered_split.py` | behaviour unchanged: the three limit-build instances rebuild identical (splits, `nu_rows`, design weights, `tier_meta`) |
| counts weights on a thinning instance recomputed from the table (pandas; dose half by dense rank) | `check_phase1.py` | independent check of the new key |

Phase 0 code is unaffected: the dose half derives from `dose_level`, and
`spec.py` freezes decisions 1, 2, 3, 5 and 6, none of which concerns weight
estimation. Decision 7 (not encoded in `spec.py`) named `fit_urr --nu_rows`
for ν; its wording is amended in §3.12. Reopened, then re-checked after review
and smoke tests: the five Phase 1 code boxes these files carry, and the §3.13
decisions box (P12 added, P1 / P2 amended).

### Phase 2/3 implementation and review (2026-09-30)

Built together: the MLP and 1D-DiT backbones share `models/layers.py` and one
trainer. Two independent code reviews covered (1) the models, layers and GPU
test and (2) the trainer, cmean, scripts and README. Two re-reviews then
covered every follow-up. Checks and runs:

- **Local**, numpy only, with torch blocked: `precompute_cmean` on the limit
  build, and an independent recompute of its group means.
- **CPU job on `bindel`:** `precompute_cmean` on `mcf7_24h`
  (`scripts/phase2_cpu.sub`).
- **GPU jobs on `zabih`** (approved for all Phase 2–3 GPU jobs, 2026-09-30):
  - `scripts/phase2_gpu.sub`: `test_lincs_shapes`, then a two-epoch trainer
    matrix on the limit build, with resume, refusals and rebuild checks;
  - `scripts/phase2_sanity.sub`: short full-data runs.

| Change against the plan text or the RxRx copy | Why |
|---|---|
| New `models/layers.py`: `TimestepEmbedder`, token `modulate` and the output dataclass are copied from `dit.py`; new are `vec_modulate` and the timestep / CFG-drop helper taken from `DiT2DModel.forward` | The MLP needs `TimestepEmbedder` in Phase 2, before `dit1d.py`. The token `modulate` broadcasts a `(B, H)` vector to `(B, B, H)`, and `mse_loss` would only warn |
| `arch.json` records the resolved `arch_kwargs` (hidden / depth / heads / patch), not a size letter. `patch_geometry` `{pad, n_tokens}` is kept beside it | Re-tuning `MLP_SIZES` / `DIT_SIZES` cannot orphan a checkpoint. `build_generator_from_ckpt` reads `arch.json` only, and cross-checks the geometry and gene count |
| Gene order is `gene_pr_ids` plus the sha1 of that list | The `gene_order.json` file embeds the table fingerprint, so its hash differs between the full and limit builds even though the genes do not |
| The trainer keeps the rows on the device (`_RowBatcher`, a shuffle seeded by (seed, epoch)) instead of a DataLoader, and refuses `num_processes > 1` | Per-row dicts plus collate would cost more than an MLP step. A resume replays the uninterrupted batches: epoch-1 losses match bit for bit |
| `--mixed_precision` defaults to `no` (fp32 with TF32); RxRx hard-codes fp16 | fp16 GradScaler with fused AdamW hides skipped steps, and AMP buys nothing for the launch-bound MLP. bf16 is opt-in for DiT-B |
| wandb through accelerate (`--wandb_mode`); `loss_history.jsonl` also carries `val_loss_ema` and grad-norm statistics. RxRx used tensorboard | tensorboard is absent from the `adi` env. wandb 0.30.0 is installed and pinned (user request, 2026-09-30) |
| (a) The V-REx batch sampler gets `set_epoch` every epoch | RxRx never called it, so every epoch replayed the same batches. accelerate's `set_epoch` does not reach a custom batch sampler |
| (b) A checkpoint is always saved at the last epoch | RxRx lost the tail when `num_epochs % checkpoint_every != 0` |
| (c) `--cmean_lambda > 0` under DDPM is refused | RxRx skipped it silently |
| (d) The cmean aux forward passes an explicit no-drop mask (only when p > 0) | RxRx's train-mode forward drew its own CFG mask, so about 10% of aux rows pulled the null branch toward μ̂(a) |
| (e) cmean with a non-empty C or `--include_env` is refused. μ̂(a) stays unweighted in the `weighted` arm (recorded, not changed) | μ̂ is keyed on the action only |
| `precompute_cmean` writes `row_gid` (each train row's group), `plate_center` and `split_fingerprint`. It is numpy only, via `expr_stats.normalize_expr` | The trainer does not re-derive the key (RxRx duplicated `group_key`), and the file cannot silently pair with another split or centring |
| Run-dir safety, all refused before anything is written: a fresh start into a dir with checkpoints (checked again under the lock), or with another run's `arch.json`; a second launch of a running run (`fcntl.lockf` on `.train.lock`, NFS-enforced); a resume from a missing or torn checkpoint (`random_states_0.pkl`, which accelerate writes last, is removed before a re-save) | Auto-named runs (e.g. the Phase 4 cmean ablation) could overwrite each other. A mistyped `--resume_epoch` truncated the history |
| A resume compares every `arch.json` field except `train_args`, against both the run dir's and the source checkpoint's run. `--resume_from_checkpoint` can fork into a new dir, never into a dir that holds checkpoints; the fork starts a fresh history | RxRx compared three fields. Identity includes `ema_decay`, `val_cap`, the sha1 of the DR weights and cmean files, split-dir-relative file paths and a realpath `nuisance_dir` |
| `--reset_lr` sets the group lr inside the existing scheduler, and is refused inside the warm-up. A changed `--lr_decay_every` on resume is reported as ignored | RxRx swapped in a fresh ExponentialLR, whose state a later resume could not load into its SequentialLR |
| V-REx uses `spec.invariance_env_fields` (`--environment_set`); RxRx's free-form `--invariance_env` is dropped | It bypassed spec.py's role guards and duplicated `--environment_set` |
| `precompute_cmean --min_n` keeps the RxRx default 8, which covers 34 of 10,479 treated arms plus the vehicle key (13.6% of train rows). `cmean_min2.npz` (`--min_n 2`) covers 99.9% | P9: Phase 4's ablation picks the value |
| The trainer's smoke runs use `--ema_decay 0.9`. `test_lincs_shapes --ckpt_dir` reproduces both the logged `val_loss_ema` (EMA) and `val_loss` (training weights) | At 0.999 over about 40 steps the EMA is still close to its initialisation, so a wrong `cond_spec` would still have passed |

Runs from before these changes (`runs/*/smoke_714963_*`, `runs/mcf7_24h/sanity_715222_*`)
have an older `arch.json` and are not resumable. They are throwaway.

`src/models/__init__.py` now imports both backbones, and with them timm and
diffusers. Every importer of `src.data.dataset` loads them too, as in RxRx. No
numpy-only path (`check_phase1`, `precompute_cmean`, `export --mode counts`)
imports `src.models`.

### Decisions 2026-10-01: Phase 4 settings (E1–E9)

Taken with the user before any Phase 4 code; recorded in §3.14. No Phase 4
box is checked.

### cmean ablation: training results (2026-10-01)

Eight FM arms on `mcf7_24h`: {MLP-B, DiT-S/10} × {`conditional`, `dr`} ×
λ ∈ {0, 0.1}. All use flow matching, 500 epochs, seed 0, the v1 split, and
`cmean_min2.npz` (10,447 arms, 99.9% of train rows). Jobs 756915–756918 (MLP)
and 759400, 759404–759406 (DiT), all on `zabih`, all exit 0. Run dirs:
`runs/mcf7_24h/{mlp-B,dit1d-S-p10}_{conditional,weighted-urr}_fm[_cm0.1-min2]_s0`.

Final values at epoch 499. FM losses start near 2.0 and are not comparable
with the DDPM arms.

| backbone | arm | λ | train loss | `val_loss` | `val_loss_ema` | wall time |
|---|---|---|---|---|---|---|
| MLP-B | conditional | 0 | 1.0698 | 1.1465 | 1.1428 | 11 min |
| MLP-B | dr | 0 | 1.0696 | 1.1466 | 1.1429 | 11 min |
| MLP-B | conditional | 0.1 | 1.0928 | 1.1291 | 1.1261 | 17 min |
| MLP-B | dr | 0.1 | 1.0925 | 1.1293 | 1.1263 | 17 min |
| DiT-S/10 | conditional | 0 | 0.8516 | 0.8789 | 0.8722 | 2 h 38 |
| DiT-S/10 | dr | 0 | 0.8523 | 0.8792 | 0.8723 | 2 h 38 |
| DiT-S/10 | conditional | 0.1 | 0.8774 | 0.8818 | 0.8758 | 5 h 12 |
| DiT-S/10 | dr | 0.1 | 0.8768 | 0.8826 | 0.8766 | 5 h 09 |

The cmean training loss includes λ · aux (≈0.023 at the end).

`val_loss_ema` gap, cmean minus plain (`conditional` arm):

| epoch | 9 | 49 | 99 | 199 | 299 | 399 | 499 |
|---|---|---|---|---|---|---|---|
| MLP-B | −0.004 | −0.039 | −0.030 | −0.017 | −0.015 | −0.015 | −0.017 |
| DiT-S/10 | −0.042 | −0.000 | +0.001 | +0.001 | +0.002 | +0.003 | +0.004 |

Findings:

- **MLP: cmean lowers the final validation loss by 0.017 (1.5%).**
  - The comparison is paired (same val rows, noise draws and seed), and the
    gap is ~100× the `conditional`–`dr` difference (1e-4).
  - Net of the aux term, the training denoising loss equals the plain run's
    (≈1.070). The train–val gap shrinks from ~0.073 to ~0.056, so cmean acts
    as a regulariser, not as a better fit.
  - The plain runs are flat from epoch 399; the cmean runs were still
    improving slowly at epoch 499.
- **DiT: cmean does not help.**
  - It is ahead only in the first epochs, level by epoch 49, and +0.004 (0.4%)
    worse at the end, in both arms. The gap grows steadily and is 5–40× the
    `conditional`–`dr` difference.
  - It costs 2× the training time (MLP: ~1.5×).
- **The aux loss ends at ≈0.23 on both backbones.** That looks like the noise
  floor of a two-well mean. A reading, not tested: the DiT already learns the
  per-arm mean, so cmean adds no information there and pulls toward noisy
  two-well targets.
- **The backbone matters far more than cmean.** Under FM the DiT reaches 0.872
  against the MLP's 1.143; the MLP's cmean gain is ~6% of that gap.
- **`dr` ≈ `conditional` in every setting** (null check).
- **Caveats.**
  - One seed.
  - None of the four FM DiT runs has plateaued: the best epoch is the last
    one, and the loss still falls ~0.002 per 50 epochs. The comparison is at a
    fixed budget.
  - FM DiT runs show larger pre-clip gradient spikes (max 5.7 plain, 7.6
    cmean; clip 4.0) than the DDPM runs, with no visible effect on the
    curves.
  - All of this is denoising loss, a proxy. τ error decides.

What follows from this is decision E9 (§3.14). One optional item is left
open: if Phase 4 prefers FM for the DiT, its two λ = 0 runs can be resumed
for a few hundred more epochs.

### Decisions 2026-10-01: Phase 4 sampler, reference scale and pool (E10–E12)

Taken with the user before the Phase 4 code, after costing the sampler from the
measured training throughput; recorded in §3.14.

### Phase 4 implementation and review (2026-10-01)

New: `src/eval/{__init__,dist_metrics,generation,evaluate}.py`,
`src/tests/{check_phase4,smoke_phase4_torch}.py`,
`scripts/{eval_oracle,eval_arm,phase4_gpu}.sub`. Checks and runs:

- **Local**, numpy only with torch execution blocked: the `--source real` oracle
  and `check_phase4` on the `--limit` build, plus a numeric self-test of
  `dist_metrics` on synthetic matrices.
- **CPU job on `bindel`** (`scripts/eval_oracle.sub`): the full `mcf7_24h`
  oracle and its checks. `ma` was at 96% of its memory allocated (1,979 of
  2,064 GB), so it failed the 20% headroom rule.
- **GPU job on `zabih`** (`scripts/phase4_gpu.sub`): the sampler checks, the
  eval end to end on the `--limit` build, and the refusals.

| Change against the plan text or the RxRx copy | Why |
|---|---|
| **`make_eval_scheduler` now pins `clip_sample=False`** (and `thresholding=False` on DPM). A Phase 2 file, changed because sampling is the first thing to call `step()` | `DDIMScheduler` and `DDPMScheduler` both **default to `clip_sample=True`**, which clips the predicted $x_0$ to $[-1, 1]$ at every step. That is an image-range assumption, and the outcome here is a z-scored gene vector: it silently truncates the sampled distribution and biases $\|\hat\tau\|$ downward -- the same hazard §2.2 records for the explicit clamp, which the plan removed while this one survived inside diffusers. Found by the GPU smoke: with the default, **0.0%** of sampled values fell outside $[-1, 1]$ on both DDPM arms (max $|x| = 1$ exactly), while the flow-matching arm -- whose Euler `step` does no clipping -- was unaffected. Training is untouched: it only calls `add_noise` / `get_velocity`, neither of which reads the flag, so no trained arm has to be redone |
| `rbf_mmd2` rewritten through `_pairwise_sq_dists` (the Gram trick RxRx already had for PRDC) | RxRx builds `((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)` in both the kernel and the median-heuristic block. At d = 978 and its own `max_samples=4000` that is a ~125 GB temporary, so the first quality metric would have died. Also: `seed` is a parameter instead of a hard-coded `default_rng(0)`, and a degenerate median now raises instead of silently becoming bandwidth 1.0 via `float(...) or 1.0` |
| Two Fréchet variants, `marginal_frechet` and `pc_frechet` + `fit_pca` (E5) | A full-covariance Fréchet in 978-d needs n >> 978 per side. The marginal form is the closed-form 1-d $W_2^2$ summed over genes, so it works on a 2-well group; checked against `frechet_distance` on diagonal data (0.5287 vs 0.5286). The PC form runs only where n > k |
| `generation.check_arm_against_data` is new | `build_generator_from_ckpt`'s docstring assigns the gene-order check to eval, and nothing else compared a checkpoint's provenance with the split and outcome in front of it. It checks the gene order and sha1, `n_genes`, population, both fingerprints, `plate_center`, both normalisation modes, `syn_effect` / `syn_seed`, `adjustment_set` and the compound cardinality. The GPU smoke doctors each field in turn and requires a refusal |
| `generate_for_rows` streams, and its sample subset is drawn **up front** | 597k samples x 978 floats is 2.3 GB per arm, so only per-row sums and sums of squares are kept. Drawing the retained subset from the known item count up front makes it invariant to `--gen_batch_size`; an online reservoir would depend on both chunk size and arrival order. The GPU smoke checks that invariance |
| `META_COLUMNS` includes `CONTEXT_SOURCE_COLUMNS` | Found by the GPU smoke: `generation.targets_from_rows` calls `context_for_rows`, which reads `cell_id` / `syn_c` / `det_well` / `pert_time` off the same frame. Without them every `--source generated` run died with `KeyError: 'cell_id'`. v1's `cond_spec` has no context field, but the `(N, F)` tensor must still be built |
| The smoke's batching-invariance checks compare across batch **sizes** with a tolerance (1e-4), and bit-exactly only under re-**ordering** | Reordering keeps the tensor shape, so the kernels are identical and the samples match to 0.0. A different batch size makes cuBLAS pick different kernels, so the same row's sample moves by ~1e-7. That is float nondeterminism, not a seeding bug; the reservoir's *selection* is still required to be identical, since it is drawn up front from the item count |
| `--min_dose_n_holdout`, default 1 (E12) | Under the P3 split a 3-well arm puts exactly **one** well in the holdout, so E4's `--min_dose_n 2` empties that sub-pool: measured 0 arms on the limit build, and it would drop 9,754 of 9,754 on `mcf7_24h`. §3.7 already reads holdout in aggregate for this reason |
| $\|\hat\mu(0)\|$ is reported against its own DMSO sampling scale, not against 0 | z = 0 is the mean of the centred **train** DMSO wells (decision 5 / P10), so a pool's own DMSO mean is 0 only up to the sampling noise of its $n_\text{DMSO}$ wells. The check is the ratio: measured 0.50x on `--pool all` and 1.09x on the holdout |
| The oracle path imports no torch, and that is enforced | `--source real` reaches `expr.npy` through `load_expr_meta` / `plate_codes` / `normalize_expr` and reuses `precompute_cmean.group_means`, so it runs as a CPU job. The local run asserts `torch` never entered `sys.modules` |
| Eval artifacts live under `runs/<build>[/<run>]/eval_artifacts/` | `Paths` has no eval field and refuses reassignment of `data_dir` / `population`; §3.1 already said `runs/` holds the eval JSON. The 978-vectors go to a `_tau.npz` sidecar (10,479 x 978 as JSON text would be ~400 MB), which is what `--truth` reads |

#### Oracle results (`mcf7_24h`, `--pool all`, job 791843, 6.9 s on `bindel`)

10,446 arms (33 one-well arms dropped by `--min_dose_n 2`), 2,064 DMSO wells.
Every check in `check_phase4` passes, including τ̂ and $\hat\mu(0)$ recomputed
from `expr.npy` with pandas (max diff 4.8e-07) and the empirical floor against
its analytic value $\sqrt{\sum_g \sigma_g^2 (1/n + 1/n_\text{ref})}$ at all 18
evaluated sizes.

- **The reliability ceiling is the headline, and it is low.** Median split-half
  $\cos(\hat\tau_A, \hat\tau_B)$ is **0.078** over all 10,446 arms: 0.051 at
  n = 2, 0.072 at n = 3 (8,835 arms), 0.311 at n = 4, 0.721 at n = 8. So on a
  typical 3-well arm **a perfect generator would score a median per-arm cosine
  of about 0.08**. Phase 4's per-arm cosines must be read against this, which is
  what E11 exists for; the holdout median (0.329) is higher only because just
  the 59 multi-well arms can be measured there.
- **Responders: 17.2% of arms** (1,795 / 10,446) and 651 compounds clear 1.5x
  the floor. §3.8.1 quotes "~29%" from the 2026-09-28 pre-build scan; the
  measured number under E11's floor definition is lower, and 17.2% is what
  Phase 5 should size its scored-compound set against.
- **The curated panel behaves** (all 16 compounds resolved). ‖τ̂‖ by family:
  HDAC 50–56 (belinostat 56.2, vorinostat 51.2, entinostat 50.3), HSP90 ~40,
  proteasome 41.0 / 40.7 at floor ratio ~35 with 968 / 971 wells, mTOR 17.8–34.6,
  MEK 17.6–27.7, ER 14.7–29.6. 15 of 16 are responders; **estradiol is the
  exception** (floor ratio 1.0), which is what an already-ER+ line at baseline
  should look like.
- $\|\hat\mu(0)\|$ 0.294 = 0.50x its DMSO sampling scale, i.e. z = 0 is the
  vehicle as decision 5 intends.


### Phase 5 progress: step-C data layer on `mcf7_24h` (2026-10-01)

Built while the Phase 4 generator arms were still running, since the only Phase 4
input step C needs is the `--source real` oracle, which was already finished.
CPU job `scripts/phase5_cpu.sub` on `bindel` (`ma` was at 89.6% of its memory
allocated, failing the 20% headroom rule).

New: `src/data/responders.py`, `scripts/phase5_cpu.sub`.

| Change against the plan text | Why |
|---|---|
| `responders.py` screens for **thinnability**, not just response: a scored compound must already have a train well in each of its four positivity cells, {low, high} x `syn_c`, and >= 2 distinct dose levels | `build_tiered_split.thin_compound` requires exactly that and refuses the **whole instance** over one bad compound. Measured on `mcf7_24h`: of 651 responders, 3 have < 2 dose levels and 7 have an empty cell -- `syn_c` is balanced within an *arm*, so after the holdout takes one of ~3 wells a compound can end up with every high-half train well at one `syn_c` level. 641 are scored |
| The responder subsample (`--n_compounds` / `--frac`) lives in `responders.py`, not as `--n_tier_compounds` on `build_tiered_split` | §3.8.1 named the flag but the builder never grew one, and keeping the selection in its own artifact makes it reviewable and re-runnable per population (Phase 6 needs one for `core5_24h`) |
| **The tier instances use one thinning seed (42), not two.** §3.8.1's "at least 2 seeds per arm" is read as TRAINING seeds (`train_diffusion --seed`) | The split layer runs *inside* the builder, so a second seed draws a different **holdout** -- and §3.8.1 itself requires that v1 and steps C / A "share one split implementation and, at a given seed, one holdout", which the Phase 4 oracle is built on. A seed-43 instance also failed the cell screen, because the screen can only use the base split's train rows |
| `beta`'s calibration input is recorded both raw and noise-corrected | §3.8.2 sets beta = `--syn_effect` x the median responder ||tau_hat||. Measured: raw 28.98, noise-corrected 25.48 -- only 12% apart, because responders are by definition above the floor. (The large distortion is in the median over *all* arms, 15.2 against a floor of 14.5.) The corrected value is the one to use, but the choice is minor |

Both instances build and pass `check_phase1 --adjustment_set syn_c`, which
recomputes pi, the positivity cells and the design weights from the table:

| instance | n_train | n_holdout | design w mean / max | counts w mean / max | corr(counts, design) |
|---|---|---|---|---|---|
| gamma = 0 (MCAR control) | 21,293 | 10,650 | 1.248 / 2.50 | 1.253 / 11.0 | +0.73 |
| gamma = 1 (confounded) | 21,600 | 10,650 | 1.228 / 9.62 | 1.236 / 22.0 | +0.88 |

- 641 scored compounds thin the train pool from 26,690 to ~21,300-21,600 wells
  (about 20%), and the **holdout is 10,650 in both, identical to v1's**, so the
  shared-holdout property holds.
- The counts weights track the design weights better at gamma = 1 (+0.88) than at
  gamma = 0 (+0.73), as they should: at gamma = 0 the design weights are nearly
  constant (max 2.5) so the correlation is dominated by estimation noise, while
  at gamma = 1 there is real variation to track. This is §3.8.1's success
  criterion 4, measured on the full population for the first time (the limit
  build gave +0.83 at gamma = 1).
- **Corrected 2026-10-05 (status review, §5).** +0.73 and +0.88 are
  correlations over *all* train rows, including the ~16.6k unthinned and vehicle
  rows where both weights are exactly 1. On the kept scored rows, where the
  weights vary, `check_phase1` and `export_urr_weights` print **+0.273**
  (gamma = 0, 4,662 rows) and **+0.840** (gamma = 1, 4,941 rows) in both jobs of
  record (798892 and 817859). The limit build's +0.83 is on kept scored rows, so
  the like-for-like comparison is +0.840 against +0.83. The reading stands, and
  gamma = 0's +0.27 fits it better than +0.73 did.


### Phase 5 code (B2–B7) implemented 2026-10-01

`synthetic.py` (injection + CLI), `spec.add_syn_cli`, `dataset.py`'s injection,
`evaluate.py`'s oracle-side injection and bias-along-v readout,
`build_tiered_split --plan`, `src/tests/check_phase5.py`,
`src/tests/smoke_phase5_torch.py`, `scripts/phase5_gpu.sub`.

| Change against the plan text | Why |
|---|---|
| The resolved injection lives in **`<nuisance_dir>/syn_meta.json`** (`syn_effect`, `syn_seed`, `vec_seed`, `scale`, `beta`, `v`, `v_sha1`), written once by `python -m src.data.synthetic` and read back by both `dataset.py` and `evaluate.py` | `beta` depends on a *measured* scale and `v` on its own seed, so resolving it twice would let the trainer and the oracle drift onto different ground truths -- the one error step C cannot detect, because it would look exactly like bias. Same pattern as `expr_meta.json`: one artifact, validated on every read |
| `v`'s seed is **not** an `OutcomeSpec` field | `spec.decisions_record` embeds `asdict(cfg.outcome)` verbatim and `check_build` compares it against every build's `population_qc.json`, so a new field would invalidate both builds on disk and force a re-ingest for a knob that is not a Phase 0 decision. It lives in `syn_meta.json` instead |
| An injected oracle is tagged `_syn<effect>[-v<seed>]` | It is a **different oracle**: without the tag it would overwrite the v1 oracle at `oracle_mcf7_24h_poolall.json`. `responders.py` therefore names the uninjected oracle exactly rather than globbing |
| `arch.json` records `syn_beta`, `syn_vec_seed`, `syn_v_sha1`, and both the eval identity guard and `--truth` check them | `syn_effect` alone does not pin the ground truth, since `beta = syn_effect x scale` and the scale is measured |
| A **tiered** arm's `split_fingerprint` is checked via its unthinned pool | Steps C / A train on a thinned split but are scored on the unablated pool (§3.8.1), so the arm's own fingerprint cannot match the eval's. `split_fingerprint(nu_rows, holdout_idx, reserve_idx)` recovers the base one exactly -- verified: both tier instances give v1's `7c480bd13be30dfe` -- which is a stronger check than skipping the field |
| `thin_compound`'s pi/cell computation is factored into `pi_and_cells`, shared with `--plan` | The planned bias and the realised thinning must not be computable from different pi |
| `check_phase4`'s independent recompute applies the injection, and its responder band is skipped on an injected oracle | Caught by running it: the recompute disagreed by beta, and the responder share legitimately drops (below) |

#### Measured, on `mcf7_24h` at `--syn_effect 1.0` (beta = 25.483)

- **`--plan` sizes gamma exactly as §3.8.2 intends.** Median over 641 scored
  compounds of `E[syn_c | kept] - E[syn_c]`, by dose half:

  | gamma | kept | dp low | dp high | bias low | bias high | high − low | IPW |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.512 | +0.0000 | +0.0000 | +0.00 | +0.00 | +0.00 | 0 |
  | 0.5 | 0.523 | +0.1122 | −0.1122 | +2.86 | −2.86 | −5.20 | 0 |
  | 1 | 0.545 | +0.1872 | −0.1859 | +4.77 | −4.74 | −8.32 | 0 |
  | 2 | 0.556 | +0.2121 | −0.2043 | +5.41 | −5.21 | −9.42 | 0 |

  gamma = 0 is exactly 0 in both halves (the MCAR control), gamma = 1 is
  equal-and-opposite (the dose-half x `syn_c` lever's signature), and gamma = 2
  adds little because `pmin` clips. **gamma = 1 is the right operating point**,
  as §3.8.2 guessed. IPW is 0 by construction.
- **The injection is exactly additive and leaves tau alone on the unablated
  pool.** `syn_c = 0` rows are bit-identical; `syn_c = 1` rows shift by exactly
  `beta * v` (max |diff| 9e-07 on a shift of norm 25.48); and
  `<tau_syn - tau_plain, v>` has median **+0.012**, i.e. < 5% of beta. So
  `syn_c` really is balanced within arms, and any step-C bias is attributable to
  the thinning alone.
- **Two known offsets, both now reported rather than mistaken for errors.**
  - $\hat\mu(0)$ moves by $\beta\,\mathbb{E}[\text{syn\_c}] \approx \beta/2$:
    measured 12.725 against 12.729 predicted. The vehicles are injected too
    (§3.8.2), and `evaluate` subtracts the known offset before reporting the
    centring ratio (0.45x and 0.99x, i.e. intact).
  - The per-arm residue is $\beta/6$: measured median 4.235 against
    $\beta/6 = 4.247$. A 3-well arm splits `syn_c` 2/1, so its mean is 2/3, not
    1/2. `check_phase5` asserts this bound explicitly.

#### Open decision before the step-C arms are trained: how the oracle handles the injection

The injection at `--syn_effect 1.0` **degrades the per-arm oracle**, because the
$\beta/6$ residue is comparable to the thinning bias it is meant to expose:

| | uninjected | injected, beta = 25.5 |
|---|---|---|
| split-half cos, `--pool all` | 0.0776 | **0.0218** |
| split-half cos, `--pool holdout` | 0.3291 | 0.2673 |
| responder arm fraction, `all` | 17.2% | 14.4% |

Two consequences:

1. **The Spearman–Brown lift (E11) is not valid on an injected oracle.** It
   assumes a half-sample's noise is 2x the full sample's, which holds for i.i.d.
   well noise but not for the injection: a 1-well half has `syn_c` mean 0 or 1,
   so its injected shift swings by the whole of beta, far worse than half-sample
   scaling predicts. The reported ceiling is therefore a *lower* bound on an
   injected oracle.
2. Lowering `--syn_effect` does **not** help: the thinning bias is
   `0.187 * beta` and the residue is `beta / 6`, so their ratio is 1.12
   regardless of beta.

The fix, if wanted, is to define the step-C oracle **`syn_c`-stratified** --
$\hat\tau$ = the mean over `syn_c` levels of (arm mean within the level −
vehicle mean within the same level) -- which cancels the injection exactly and
restores the uninjected oracle's precision. §3.8.2 does not specify this, so it
is left open rather than decided here.

**The aggregate readout is unaffected either way.** The residue is zero-mean
across arms, so the signed median by dose half over ~1,900 scored arms per half
has a standard error of roughly 0.12 against a bias of ±4.77 -- a margin of
~40 sigma. Step C's headline does not depend on this decision; the per-arm
cosine and accuracy metrics do.


### Phase 5 step-C smoke: green (2026-10-01, job 818325)

`scripts/phase5_gpu.sub` end to end on `mcf7_24h`: **38 checks, 0 failures**.
The sampler/injection checks (17), the step-C oracle, a tiny tiered `dr` arm
whose `arch.json` pins `syn_effect=1.0 beta=25.4828 v_sha1=492aa3d1 gamma=1.0
C=['syn_c']`, that arm scored against the injected oracle, the syn-mismatch
refusal (which names `syn_effect`, `syn_beta` **and** `syn_v_sha1`), and
`check_phase5`.

**The bias-along-v readout runs end to end.** Signed
`<tau_gen - tau_oracle, v>`, median by dose half, on a deliberately untrained
arm (1 epoch x 5 steps), scored on `--pool holdout`:

| arms | low | high | high − low |
|---|---|---|---|
| scored (3,623) | −12.89 | −22.09 | −9.20 |
| unscored (6,131) | −2.82 | −2.18 | +0.64 |

**Corrected 2026-10-01 (superseding the first reading of this table).** It was
first recorded here that −9.20 "reproduces" `--plan`'s predicted −8.32. It does
not: those are two unrelated numbers that happen to land close together, and an
untrained model cannot exhibit thinning bias. The cause is the pool. On
`--pool holdout` the P3 split leaves **one well per arm**, so
`<tau_oracle, v> = beta * (syn_c - 1/2)` is **bimodal at +/- beta/2 = +/-12.74**
-- measured on the injected oracle: |proj| median 12.73, and **100%** of arms
beyond beta/4. The *median* of a bimodal variable lands on a mode, so it reports
+10.03 for the oracle alone, while its mean is +0.32. On `--pool all` (3-well
arms, where the mix is 2/1 and the offset is beta/6) the same statistic is
**+0.22 median / +0.29 mean**.

Two consequences for the step-C readout, both adopted:

- **Score step C on `--pool all`**, which §3.8.1 already required; the `holdout`
  sub-pool's `bias_along_v` is not interpretable and the verdict ignores it.
- **Read the contrast as a mean, not a median.** The per-arm projection is
  discrete and bimodal by construction, which is exactly the case a median
  handles badly. `bias_along_v` reports both.

What the smoke does establish is that the path runs end to end and that the
scored/unscored split, the dose halves and the projection are all wired
correctly.

| Change against the plan text | Why |
|---|---|
| **Every `.sub` sets `PYTHONPYCACHEPREFIX` to a per-job, node-local dir** | `src/` is on NFS, and job 817949 started 15 s after an edit and silently ran bytecode compiled 9 minutes earlier (`.pyc` recorded mtime 17:45:24 / size 10479 against a source of 17:54:20 / 11513, and contained the old string). A private prefix makes stale reuse impossible and avoids races between concurrent jobs. Audited the earlier runs: only 817949 was affected |
| A tiered dir needs its own `expr_meta.json` | `LincsDataset` reads it from the split dir it is handed. `expr_stats` fits on `nu_rows.npy`, so every gamma instance shares v1's z-scale: `centre` / `mean` / `std` / `cap_frac` / `plates` come out identical, and only `split_fingerprint` differs (it records that dir's own split). `phase5_cpu.sub` asserts exactly that |
| `adjustment_set` **removed** from `TRUTH_GUARD` | The oracle is a mean of real wells and never builds a cond spec, so its tau is **bit-identical** with and without a C (verified, max abs diff 0.0). Keeping it would force a redundant oracle per C and would block step C by construction, whose arms condition on `syn_c` while the oracle marginalises over it. The C that matters -- the arm's own -- stays checked in `check_arm_against_data` |
| Three tests were passing for the wrong reason, and now cannot | `_refuses` requires the message to name the intended failure; the missing-`syn_meta` case pointed at a nonexistent dir and was really failing on `expr_meta`; and the syn-mismatch gate matched `checkpoint / data mismatch`, which the `adjustment_set` guard also emits |

### Phase 4 results: all twelve arms (2026-10-01)

Scored at `checkpoint-0499` EMA, w = 1, 16 samples/real row, 100 steps (E1–E12).
`cos` is the median per-arm cosine of tau_gen vs tau_oracle; `/ceil` is it as a
fraction of the attainable ceiling (E11's Spearman–Brown lift).

| arm | pool | cos all | cos resp | /ceil | Spearman ‖tau‖ | sec |
|---|---|---|---|---|---|---|
| mlp-B conditional ddpm | all | 0.480 | 0.792 | 1.01 | 0.800 | 428 |
| mlp-B weighted-urr ddpm | all | 0.480 | 0.792 | 1.01 | 0.800 | 423 |
| dit1d-S conditional ddpm | all | 0.457 | 0.778 | 1.00 | 0.803 | 21,147 |
| dit1d-S weighted-urr ddpm | all | 0.459 | 0.777 | 1.00 | 0.804 | 21,268 |
| mlp-B conditional ddpm | holdout | 0.208 | 0.451 | 0.96 | 0.607 | — |
| dit1d-S conditional ddpm | holdout | 0.210 | 0.441 | 0.96 | 0.627 | — |
| mlp-B conditional fm (λ=0 / 0.1) | holdout | 0.208 / 0.210 | 0.436 / 0.452 | 0.98 | 0.626 / 0.612 | 135 / 138 |
| dit1d-S conditional fm (λ=0 / 0.1) | holdout | 0.211 / 0.209 | 0.451 / 0.450 | 0.97 | 0.630 / 0.619 | 6,088 / 6,107 |

1. **Every arm sits at 96–101% of the attainable ceiling** on responder arms.
   The oracle's own reliability, not the generator, is the binding constraint,
   so tau accuracy cannot separate these models. Any future comparison needs
   either more wells per arm or the high-n anchors.
2. **`dr` = `conditional` on tau**, to three decimals on both backbones
   (0.792 / 0.792 MLP, 0.778 / 0.777 DiT; Spearman 0.800 / 0.800). §1's v1 null
   check holds on the causal quantity, not just on the denoising loss.
3. **The DiT's better denoising loss does not transfer.** It beats the MLP by
   21% on `val_loss_ema` (0.5545 vs 0.7027) yet is no better on tau --
   marginally worse on `--pool all` (0.778 vs 0.792) and on the holdout (0.441
   vs 0.451) -- for **49x the eval compute** (21,147 s vs 428 s). That supports
   decision 4's choice of the MLP as the headline backbone.
   - **Qualified 2026-10-05** (§5, status review). On §3.7's per-compound
     holdout aggregate the DiT is 0.013 ahead (0.509 against 0.496). With one
     seed and no error bar the backbones are tied on held-out τ; the compute
     argument is unchanged.
4. **cmean (E9): no.** On the holdout, λ = 0.1 moves the responder cosine
   +0.016 on the MLP and −0.001 on the DiT, while Spearman falls on both
   (0.626 → 0.612, 0.630 → 0.619). E9 required a tau gain with no loss of
   spread; this is mixed, so `--cmean_lambda 0` stays the default.

### Phase 5 step-C results: all 24 arms (2026-10-01)

MLP-B, DDPM, 500 epochs. The matrix is {`naive`, `conditional`, `dr`,
`dr_design`} × γ ∈ {0, 1} × training seeds {0, 1, 2}, on the seed-42 tier
instances. Every arm is scored on `--pool all` at the §3.14 settings, against
the injected oracle (beta = 25.483). Jobs 828810–828878 ran on `zabih`: training
averaged 9.4 min, scoring 7.6 min, and all exited 0. The verdict is in
`runs/mcf7_24h/eval_artifacts/step_c_verdict.json` (`src.eval.step_c_report`).

The statistic is the one `step_c_report` documents: a difference-in-differences
(DiD) of the **mean** high − low contrast of ⟨τ_gen − τ_oracle, v⟩ on scored
compounds, γ = 1 minus γ = 0, with the spread over training seeds as the noise.
The numbers below that are not in the verdict JSON were computed from the arms'
`_tau.npz` files and the tier splits, with one-off numpy that is not in the repo.

| Change against the plan text | Why |
|---|---|
| **`step_c_report` refuses duplicate cells, and admits only JSONs scored at the §3.14 settings** (`SCORING`: epoch 499, EMA, w = 1, 16 samples per row, 100 steps), at the JSON's own `syn_effect`, and against a single injection (beta, `v_sha1`). The verdict JSON records the source file of every cell | The default glob also matched `p5smoke_828242_dr`. That smoke arm is tiered with γ = 1, seed 0, C = `syn_c` and counts weights, but it was scored at epoch 0 with 1 sample and 4 steps. It sorted ahead of `stepc_dr_g1_s0`, and the report kept it, logging the real run as a "duplicate". `dr`'s DiD came out −0.11 ± 0.16 instead of −0.03 ± 0.02. The flags did not change, but the seed spread was inflated 7×. Checked after the fix: the default run skips both smoke JSONs and names the reason; a duplicated cell and a JSON with a different `v_sha1` are both refused |

| arm | DiD, scored (mean ± seed sd) | DiD, unscored | as a share of `naive` |
|---|---|---|---|
| `naive` | **−6.85 ± 0.31** | −0.53 ± 0.27 | — |
| `conditional` | +0.005 ± 0.020 | +0.00 | 0.07% |
| `dr` (counts weights) | −0.025 ± 0.025 | −0.01 | 0.4% |
| `dr_design` (true weights) | −0.052 ± 0.015 | −0.01 | 0.8% |

§3.8.1 criteria:

1. **Pass.** `naive`'s DiD is 22 seed-sd from 0. Its γ = 0 contrast is +0.15 to
   +0.27, and differencing removes it.
2. **Fails as written:** `dr` vs `conditional` is −1.1 σ. It **cannot be tested
   under this injection** (finding 3). `dr` is within 1.6 σ of `dr_design`.
3. **Pass**, at the 25% threshold. `naive` leaks onto unscored compounds
   (finding 5).
4. **Pass**, measured before training (corr(counts, design) +0.88 at γ = 1).

Findings:

1. **`naive` shows about 65% of the selection bias in its training data.**
   - A memoriser (each arm's mean over its kept train wells) has DiD **−10.56**
     on this thinning, and −9.00 when pooled to (compound, dose half). `naive`
     has −6.85.
   - The ratio is the same in both halves: +3.31 vs +5.28 (low), −3.55 vs −5.29
     (high). So the MLP shrinks across arms rather than missing one half.
   - `--plan`'s −8.32 is a median over compounds of a cell-level quantity. It is
     not the reference for a mean over arms; the memoriser is.
2. **Adjusting for `syn_c` removes ≥ 99% of that bias in all three arms**
   (last column of the table). `dr_design`'s −0.052 is consistent over seeds
   but is 0.2% of beta.
3. **Why `dr` cannot beat `conditional` here: the outcome model is correctly
   specified, and it is learned exactly.**
   - Slope of ⟨τ_gen(a), v⟩ on each arm's `syn_c` mix, relative to the
     vehicles (seed 0, both γ): **25.48–25.52** for all three adjusted arms,
     against beta = 25.483. For `naive` it is 3.5.
   - The injection is one additive shift shared by every row, learned from
     ~21k training rows. With E[Y | arm, `syn_c`] right, the g-formula is
     unbiased, and the weights have nothing to correct.
   - So this is a limit of the design, not a failure of DR. §3.8.1 anticipated
     it: "a null result is an admissible outcome".
4. **The weights cost precision on scored compounds, and none on unscored
   ones.**
   - Read on the holdout sub-pool, whose one-well oracle was never trained on.
     The `--pool all` oracle contains the train wells and rewards memorising
     them (E9).
   - Mean squared error orthogonal to v, relative to `conditional` at the same
     γ (3 seeds):

     | arm | γ = 1 scored | γ = 1 unscored | γ = 0 scored | γ = 0 unscored |
     |---|---|---|---|---|
     | `dr` | +3.5% | +0.1% | +2.0% | +0.0% |
     | `dr_design` | +2.6% | +0.1% | +0.2% | −0.0% |
     | `naive` | −0.3% | +0.0% | −0.4% | +0.1% |

   - These shares include the one-well oracle noise common to every arm, so the
     generator's own error rises by more.
   - Kish ESS / n over the train rows: counts 0.68, design 0.75 at γ = 1 (max
     weight 22.0 vs 9.6); counts 0.77, design 0.87 at γ = 0 (max 11.0 vs 2.5).
   - At γ = 0 the true weights are nearly flat, and the counts weights still
     cost 2.0%. That share is weight-estimation noise from ~3 rows per cell,
     not the reweighting itself.
   - On `--pool all` the same ranking shows as pooled gene MSE at γ = 1:
     0.2356 (`conditional`), 0.2465 (`dr_design`), 0.2509 (`dr`).
5. **`naive`'s bias leaks onto compounds that were never thinned.**
   - Unscored DiD: −0.32, −0.44, −0.84 over the three seeds. That is ~8% of the
     scored bias, in the same direction (high-dose half pulled along −v).
   - The memoriser's unscored DiD is exactly 0, because unscored compounds are
     identical in both instances. So the shift comes from parameter sharing in
     the network.
   - The likely carrier is the dose input every compound shares. That is not
     tested yet.
   - The adjusted arms show no leak (|DiD| ≤ 0.02). Criterion 3 passes at its
     25% threshold, but the leak is not zero, so later steps should keep
     reporting unscored compounds separately.
6. **Per-arm cosine cannot rank the arms.**
   - Every arm sits at 98–101% of the (lower-bound) ceiling.
   - `naive`'s lower cosine, 0.47 vs 0.53 for `conditional` and present even at
     γ = 0, comes from the oracle, not the generator. Each arm's oracle carries
     its own `syn_c`-mix offset, which a model blind to `syn_c` cannot
     reproduce. `naive`'s mean squared error along v is 21–23 per arm, against
     (beta/6)² = 18.0 from that offset alone; the adjusted arms' is 0.14–0.38.
   - The DiD along v is therefore the step-C readout. A `syn_c`-stratified
     oracle (the earlier open decision) is needed only if per-arm metrics are
     to be reported for step C.
7. **Arm-level positivity fails for 15–19% of scored arms.** 589 (γ = 1) and
   739 (γ = 0) of the 3,835 scored arms keep no train well after thinning. The
   generator fills them in from the compound's other doses, and they are
   included in the readout. Positivity holds at the (compound, dose half,
   `syn_c`) cell, by construction.

**Decision (user, 2026-10-01): Phase 6 waits.** First, a step-C variant in which
`dr` can beat `conditional`. That requires an outcome model that cannot learn
the confounder's effect exactly, which is where DR is supposed to help.

#### Decision: which variant (resolved 2026-10-01)

**Option B at ρ = 1, run on `zabih` (user, 2026-10-01).** Option A is not run.
ρ = 0.5 is added only if ρ = 1 shows the effect. The design, the success
criteria and the escalation are fixed in §3.8.4. The options as they were laid
out:

**Option A: IPW-only arms.**
- The generator does not see `syn_c` (C = ∅) but is trained with the counts or
  the design weights (`ipw`, `ipw_design`).
- This completes the 2×2 of {C in the generator} × {weights}, and tests the
  weights' half of double robustness on their own. Prediction: DiD ≈ 0, against
  `naive`'s −6.85.
- What it reuses: the injection, the oracle, the tier instances and the weights.
- What it needs: an opt-in past the trainer's guard
  (`train_diffusion.py`, "a weighted arm adjusts for it"), a report label, and a
  launcher case. 12 runs, ~3.4 GPU-h.
- Its limit: the arm it beats is `naive`, not `conditional`. The
  misspecification is omitting C entirely, which no ADIGen arm does.

**Option B: compound-specific injection (effect modification).**
- y ← y + `syn_c` · beta · v_k, with
  v_k = normalise(√(1−ρ) · v + √ρ · u_k) and u_k a seeded unit vector per
  compound (the vehicle is compound 0). ρ = 0 is the current step C; ρ = 1 is
  fully compound-specific.
- Thinning, positivity cells and weights depend on (A, C) only, so they are
  reused unchanged.
- Mechanism:
  - `conditional` must now learn each scored compound's `syn_c` shift from ~6
    kept rows. The MLP's conditioning is an additive sum, so a shared shift is
    native and a compound × `syn_c` interaction is not.
  - Any shrinkage λ < 1 of that interaction leaves (1 − λ) of `naive`'s bias in
    the g-formula.
  - The weights balance `syn_c` within every (compound, dose half) cell, so
    `dr`'s arm fits stay balanced whatever λ is.
- Predictions:
  - DiD(`conditional`) ≈ (1 − λ̂) · DiD(`naive`), with λ̂ read off the arm
    (finding 3's slope, per compound).
  - `dr` ≈ `dr_design` ≈ 0.
  - `naive` stays strongly biased, though possibly below −6.85: shrinking
    toward structure shared across compounds no longer carries the bias, since
    the directions differ by compound. The memoriser's −10.56 is the bound.
  - `naive`'s unscored leak should vanish for the same reason. That tests
    finding 5.
- Power: the DiD's seed noise is ~0.02, so even λ = 0.99 would put
  `conditional` ~3 σ off.
- Code: `synthetic.py` (directions per compound, ρ, a sha1 of the direction
  matrix), the two `inject` call sites (`dataset.py`, `evaluate.py`),
  `bias_along_v` (each arm projected on its own v_k), the vehicle offset, the
  provenance guards, `check_phase5` and the smoke.
- Compute: a new injected oracle (CPU, seconds) and 24 runs at ρ = 1, ~7 GPU-h
  or ~1.5 h of wall time on six `zabih` GPUs.
- Risk: the MLP learns every compound's interaction (λ̂ → 1), which would give a
  null again. The escalation is a direction per (compound, dose half), which is
  still the positivity cell, so the weights still balance it.

**Recommendation: B at ρ = 1.**
- It tests what ADIGen actually claims: DR protects against an outcome model
  that cannot learn the confounder's interaction with the treatment.
- It rehearses step A, where the cell line × compound interaction plays the same
  role, with a known truth.
- A is optional. It is worth running only if the write-up wants the full 2×2 of
  double robustness, and it can run on the existing injection while B is being
  coded.

### Step C2 implementation and review (2026-10-01)

Code for §3.8.4, written after the plan text above and before any C2 result.
- **Local checks** (numpy only, torch made unimportable):
  - unit checks of the directions, the injection, the readout and λ̂ on
    synthetic arrays;
  - the real `syn_meta_compound_r1.json` resolved and read back, including
    four tampering refusals;
  - `step_c_report` re-run on the 24 step-C arms. The deltas are identical, and
    all 12 weighted runs' weight hashes match their files.
- **Jobs:** a CPU job on `bindel` (`ma` was at 13.6% free memory, under the 20%
  rule) builds the injected oracle and runs `check_phase4` and `check_phase5`.
  The `zabih` GPU smoke depends on it. The first pair (850067 / 850069) ran
  before the matrix was stored (table below) and was discarded; 850257 / 850258
  are the runs of record.

The resolved injection: beta 25.4828, the same as step C, and 1,751 directions
(the vocab, vehicle included). The stored matrix's sha1 is `11c32b37`; step C's v
is `492aa3d1` and is C2's shared component. Median cos(v_k, v) is 0.0011, and the
median |cos| between compounds is 0.0216, the isotropic value 0.6745/√978. The
maximum is 0.158.

| Change | Why |
|---|---|
| `synthetic.py`: `--mode compound --rho`, `effect_directions`, a matrix form of `inject`, `inject_meta` / `directions_for` as the single call shape, and `default_meta_name` | Trainer, oracle and tests must apply one injection through one function. At ρ = 0 the compound path is **bit-identical** to the global one (asserted in the unit checks and in `check_phase5`), so C2 at ρ = 0 is step C |
| In compound mode, `v_sha1` hashes the whole direction matrix, and `v_shared_sha1` hashes v | Every existing guard (arch.json `syn_v_sha1`, `check_arm_against_data`, `TRUTH_GUARD`) then tells C2 from C with no new field |
| **The matrix is stored** in a sidecar, `syn_meta_compound_r1_V.npy` (13.7 MB). It is written atomically before the JSON that names it, and **written once**: rerunning `synthetic` on the same injection is a no-op. Every consumer loads the stored bytes and holds them to `v_sha1`; regenerating from the seeds is now only a construction check, to 1e-12 | Found by the first CPU job (850067). The first design regenerated the matrix on load and compared hashes. On `bindel` the same seeds gave sha1 `f42063ad`; the dev box gave `11c32b37`; the entries agree to ~1e-16. Each normalisation goes through BLAS, which rounds the last bits differently on different CPUs. The job had also rewritten the file with its own hash, because the parameters matched. Unfixed, a `zabih` trainer could have refused the file, or recorded a hash the oracle did not share. The stale oracle and smoke (850067 / 850069) were discarded and rerun on the stored matrix |
| `CaseConfig.syn_meta_name` and `--syn_meta` (trainer, evaluate), with `check_syn_args` refusing `--syn_meta` without a nonzero `--syn_effect` | Not on `OutcomeSpec`, which `decisions_record` embeds (a new field would invalidate both builds). The refusal comes from review: `syn_effect` alone switches the injection on, so `--syn_meta` without it would silently train or score uninjected data |
| `evaluate`: the per-compound injection and vehicle offset; `bias_along_v` projects each arm on its own v_k; oracle tag `_syn1-cmp1`; `learned_syn_effect` (λ̂) on the generation pool's kept arms | §3.8.4's readout and diagnostic. λ̂ is restricted to the arms the accuracy block keeps (`min_dose_n`), so both describe one arm set (review) |
| arch.json gains `syn_meta_name`, `syn_mode`, `syn_rho` | Readability only: `syn_v_sha1` already pins the injection. Uninjected runs record `None` for all three, so resuming a v1 arm is unaffected. A step-C run resumed now would show these three as differing; none is planned |
| `check_phase5`, compound mode: the matrix shape (one row per vocab entry), the rows **rebuilt independently** from the raw RNG streams and the §3.8.4 formula, unit norms, cos with v ≈ √(1−ρ), near-orthogonality at ρ = 1, the ρ = 0 bit-identity, and the shift of beta · v_k on each row | Review: the first version compared the matrix with `effect_directions` on the same arguments, which could never fail |
| `check_phase5`: **τ now moves, by exactly beta · (mix_a · v_k − mix_0 · v_0)**, asserted instead of "the injection leaves τ alone" | Found while writing the check. In C2 the vehicles move along v_0 and a compound's wells along v_k, so `syn_c` modifies the treatment effect (§3.8.4). Step C's check still runs in global mode |
| `check_phase4` and `smoke_phase5_torch` apply the injection the oracle or `--syn_meta` names | Their independent recomputes would otherwise disagree with a C2 oracle by the injection |
| `step_c_report`: `--syn_meta` selects JSONs by `v_sha1` (step C and C2 share a runs dir); the reference is loaded through `load_syn_meta`; the λ̂ columns and the (1 − λ̂) × `naive` diagnostic; criterion 2 judged in full; criterion 4 recomputes each weighted run's `dr_weights_sha1` from the file it names; the verdict file is named from the basename | §3.8.4. The first version read the reference raw (unvalidated), always passed criterion 4 with constants, and built the verdict path from a path-valued `--syn_meta` (review) |
| `phase5_arms.sh` `VARIANT=c2`, `phase5_gpu.sub` `SYN_META=...` (plus two cross-injection refusals: a C2 arm scored under step C's injection, and against step C's oracle), and a new `phase5_oracle_cpu.sub` | One launcher and one smoke for both variants; the oracle is a CPU job |

Review findings not acted on:
- Regenerating the matrix on every load costs ~1,751 small RNG draws per
  process, a fraction of a second. A cache would hand out a shared mutable
  array.
- In global mode, `directions_for` broadcasts v to (N, G), materialising ~16 MB
  per sub-pool. The 1-D branch of `bias_along_v` stays reachable for other
  callers.

### Step C2 scoring: numpy's matrix products are wrong on the Sapphire Rapids nodes (2026-10-02)

**What happened.** The first five C2 scoring jobs (850419, 850421, 850423,
850425, 850427, and 850430 before the cancel landed) all died in the quality
block with `LinAlgError: Eigenvalues did not converge`. Every PC-projected row,
real and generated alike, was non-finite: the PCA basis itself was NaN. The
other 18 scoring jobs were cancelled before they could repeat it. Training was
unaffected, and all 24 runs continued.

**Diagnosis** (`src/tests/check_eigh.py` and `src/tests/check_blas.py`, run as
short CPU-only jobs on `bindel` and on `zabih`, and locally):

| | `zabih-compute-01` (Xeon Gold 6426Y) | login node (Gold 6448Y) | `bindel` (E5-2620 v3) |
|---|---|---|---|
| dgemm, 300×300 and larger | **wrong** (rel. err ~1.5) at 1, 4 and all threads | **wrong** | correct |
| dgemm, 100×300 @ 300×100 | wrong at 4 threads, correct at 1 | the same | correct |
| `eigh`, `svd` (they are built on dgemm) | **wrong** (NaN or finite garbage) | **wrong** | correct |
| dgemv, ddot, `np.cov` (syrk), element-wise ops | correct | correct | correct |
| everything above with `OPENBLAS_CORETYPE=Haswell` | **correct** | **correct** | correct |

numpy 1.23.5's bundled OpenBLAS picks its Sapphire Rapids kernels on both Intel
Sapphire Rapids hosts, and their dgemm is wrong. On the C2 covariance the
result was NaN, which crashed the job. On every earlier eval it was **finite
garbage**, which crashed nothing.

**Scope: what was computed wrong.** Every MMD, PC Fréchet and full Fréchet (and
KID / PRDC where requested) from `evaluate` runs on `zabih`:
- all 12 Phase 4 arms, the E10 step check (`eval_steps`) and the Phase 4 smokes;
- all 24 step-C arms.

Each of those logs carries the "negative Frechet … clamping" warning. One
example had a cross term of 476,123 against covariance traces of ~7,000, which
is impossible.

**Not affected, so the Phase 4 and step-C conclusions stand:**
- τ̂, the per-arm cosine and Pearson, Spearman, pooled MSE, `bias_along_v` (a
  matrix-vector product), λ̂ (`einsum`), and the marginal Fréchet (closed form,
  element-wise);
- everything computed on `bindel`: all oracles, `check_phase1/4/5`, and the
  data layer;
- training, which runs in torch on the GPU, and torch ships its own BLAS.

The local step-C analysis in §5 was recomputed with `OPENBLAS_CORETYPE=Haswell`
and is identical to every printed digit. That includes the λ slope of
25.48–25.52, which had gone through `np.polyfit`.

The Phase 4 and step-C **quality numbers must not be quoted.** Rescoring them is
optional:
- step C: 24 × ~7.6 min on `zabih`;
- Phase 4: the 4 DDPM arms on `--pool all` (MLP ~16 min, DiT ~3.9 h each) and
  the 8 FM arms on the holdout.

| Change | Why |
|---|---|
| Every `scripts/*.sub` exports `OPENBLAS_CORETYPE=Haswell` | It must be set before numpy loads. On `bindel` it is a no-op |
| `evaluate` refuses a broken BLAS at start-up (`dist_metrics.assert_blas_ok`: a 300×300 dgemm against a loop reference, like the CUDA preflight) and logs the result | A wrong dgemm corrupts metrics silently. 300×300 failed 12/12 draws at every thread count tested; the 100×300 case slipped through at 1 thread |
| `dist_metrics._eigh_psd` replaces the three bare eigensolver calls (`fit_pca`, `_matrix_sqrt`, the Fréchet cross term). It accepts a result only if it is finite, orthonormal and reconstructs the matrix (1e-7), falls back to scipy's MRRR driver and then to an SVD, raises if none verifies, and records any fallback in the quality JSON (`eigh_fallbacks`) | A NaN check alone would have missed the finite garbage of every earlier run |
| Known gap, now covered: the GPU smoke runs eval with `--quality_n 0`, so the quality block never ran there | The BLAS guard runs in every eval regardless of `--quality_n` |

C2 scoring is resubmitted on the fixed scripts after one full scoring job
passes end to end (below). The 24 trainings are untouched.

### Step C2 results: all 24 arms (2026-10-02)

MLP-B, DDPM, 500 epochs, on the seed-42 tier instances, under the per-compound
injection (ρ = 1, beta 25.483, matrix sha1 `11c32b37`).
- Training: jobs 850418–850465, all exit 0.
- Scoring: 854859 and 856859–856968, all exit 0, on the fixed scripts (BLAS
  verified, no eigensolver fallbacks, no Fréchet warnings).
- Verdict: `runs/mcf7_24h/eval_artifacts/step_c_verdict_compound_r1.json`
  (`step_c_report --syn_meta syn_meta_compound_r1.json`). The split by
  kept-train status and the data-level memorisers below were computed locally
  (numpy, `OPENBLAS_CORETYPE=Haswell`) from the `_tau.npz` files, the tier splits
  and the weight files.

| arm | DiD, scored (mean ± seed sd) | DiD, unscored | λ̂ scored (γ = 1) | λ̂ unscored | pooled gene MSE |
|---|---|---|---|---|---|
| `naive` | −1.93 ± 0.03 | −0.29 ± 0.01 | 0.000 | 0.001 | 0.334–0.340 |
| `conditional` | **−0.28 ± 0.01** | +0.00 | 0.520 | 0.720 | 0.286–0.296 |
| `dr` (counts) | **+0.58 ± 0.05** | −0.05 ± 0.02 | 0.349 | 0.143 | 0.369–0.383 |
| `dr_design` | **+0.56 ± 0.03** | −0.05 ± 0.01 | 0.402 | 0.209 | 0.347–0.364 |

§3.8.4 criteria (fixed before any C2 result):

1. **Pass.** `naive` is biased, 56 seed-sd from 0.
2. **Fail.** `dr` does not beat `conditional`. Its |DiD| is 0.58 against 0.28,
   so `dr_vs_conditional_sigma` is −10.9: it is significantly *worse*, and of
   the opposite sign.
3. **Pass.** `dr` is close to the true weights: |0.58 − 0.56| = 0.01, against the
   bound 0.25 × |−0.28 − 0.56| = 0.21.
4. **Pass**, at the 25% threshold: the largest unscored |DiD| is 0.29, which is
   `naive`'s.

Diagnostics:
- λ̂ < 1 on `conditional`: yes, 0.52.
- (1 − λ̂) × DiD(`naive`) = −0.93 against the measured −0.28. That is off by
  3.3×, outside the factor of 2, so the linear-shrinkage picture is too simple.
- `naive`'s leak onto unscored compounds was predicted to vanish. It did not:
  −0.29, 15% of its scored DiD (step C: 8%).

**Read with the status review of 2026-10-05** (§5, "Phase 4 and Phase 5 status
review", gaps 1 and 2). The DiD values and the criteria above stand. The
mechanism in findings 2, 3 and 6 is not established: the whole `dr` −
`conditional` gap is a difference in λ̂ between the dose halves. Finding 1's
−0.28 is `conditional`'s γ = 0 control (+0.32) subtracted from a γ = 1 contrast
of +0.04.

Findings:

1. **C2 achieved its first aim: `conditional` is now biased.** Its DiD is
   −0.28 ± 0.01, 40 seed-sd from 0, in `naive`'s direction. In step C it was
   +0.005. The generator learns only about half of each scored compound's
   shift (λ̂ 0.52; 0.72 on unthinned compounds), against ~1.00 for the shared
   shift in step C.
2. **But the weighted risk over-corrects, with the true weights as much as the
   estimated ones.** `dr` and `dr_design` agree (+0.58 / +0.56), so this is
   not weight-estimation error. It is how the weights meet the outcome model.
3. **Mechanism: the weights balance `syn_c` per (compound, dose half) cell, but
   the generator fits arms, and the thinning leaves an arm 1–2 train wells.**
   - The data alone shows it. Take a memoriser of each arm's (weighted) mean of
     its kept train wells, with bias β · (its `syn_c` mix − the pool's), and
     arms classed by their own instance's kept wells. Its DiD of the scored
     high − low contrast:

     | | arms that kept both `syn_c` levels | arms that kept one level | all |
     |---|---|---|---|
     | unweighted | −1.63 | −14.05 | −10.56 |
     | counts / design weights | **+8.71 / +8.30** | −14.05 | −7.67 / −7.81 |

   - On an arm that kept both levels, the scarce level's few wells carry the
     whole cell's weight, and the arm's self-normalised (Hájek) mean
     **over-corrects**. On an arm that kept one level (2,345 of 3,835 scored arms
     at γ = 1; 589 kept none), reweighting cannot change the mix at all.
   - The weights are right for cell-level means, and wrong in both directions
     at the arm level, where the outcome model works.
   - The generator shows the same split, by the arm's kept wells at γ = 1
     (mean of 3 seeds):

     | arm | none | one level | both levels |
     |---|---|---|---|
     | `naive` | −0.43 | −2.10 | −2.44 |
     | `conditional` | +0.53 | −0.41 | −0.43 |
     | `dr` | −0.13 | +0.50 | **+1.23** |
     | `dr_design` | +0.39 | +0.37 | **+1.22** |

   - The over-correction is largest exactly where the arm kept both levels. So
     it is not an extrapolation artefact of arms with no train wells.
4. **The weighted risk also learns the interaction less.** λ̂ drops from 0.52
   to 0.35 / 0.40 on scored compounds, and from 0.72 to 0.14 / 0.21 on unthinned
   ones. At γ = 1 it also differs by dose half (low 0.31–0.38, high 0.37–0.44);
   `conditional`'s does not.
5. **The weights' precision cost is much larger than in step C:** pooled gene
   MSE is 25–30% above `conditional`'s (step C: a few %).
6. **What C2 says about ADIGen:** with cell-level weights (P12) and an outcome
   model that resolves finer than the cell, DR is not protective. The
   finite-sample ratio bias of the weights at the model's resolution can exceed
   the outcome model's own bias. DR needs the positivity cell to match the
   resolution at which the generator learns the confounder's interaction.
   - On MCF7 that cannot be arranged. An arm has ~2 train wells, about one per
     `syn_c` level, so arm-level positivity leaves nothing to thin (§3.8.2).
   - **Step A's design already matches it.** Its positivity cell is (compound,
     `dose_level`, `cell_id`), the arm × line level, and §3.8.3 notes that
     "arm-level positivity survives the thinning". That makes step A the regime
     where a DR advantage is possible.

**The pre-declared escalation does not apply.** It was for "criterion 2 fails
because λ̂ ≈ 1"; here λ̂ = 0.52 and the failure is the weights'
over-correction. Per §3.8.4, any further C2 run is a new decision.

### Phase 6 effect-modification gate: PASSED (2026-10-02)

§3.8.3 requires this before `core5_24h` is built: if the five lines respond
alike, `cell_id` has no effect on centred `Y` and step A collapses into another
null check, which would make the ~9 GB ingest pointless.

New: `src/data/line_gate.py`, `scripts/phase6_gate_cpu.sub`. Job 862040 on
`bindel` (`ma` was at 8.6% CPU / 8.7% memory free, failing the 20% rule),
3 min 21 s. It needs **no `core5_24h` build** and so no `--population` switch:
it selects the population from `inst_info`, reads the 641 MCF7 responders'
wells in the five lines straight from the GCTX (76,140 wells on 494 plates
after QC dropped 2), and applies the outcome rules the build would — 3× plate
QC on each line's median spread, DMSO-median plate centring, one pooled
z-scale. Result: `runs/core5_24h_line_gate.json`.

**The statistic.** Two independent estimates of the *same* τ correlate at the
full-sample reliability $r_\text{full} = 2r_\text{half}/(1+r_\text{half})$
(Spearman–Brown on the within-line split-half, E11). So per arm and line pair,
$\cos(\hat\tau_{L_1}, \hat\tau_{L_2})$ *below* $r_\text{full}$ is effect
modification. Restricted to arms responding in at least one of the two lines;
the 0.8 threshold was declared before the run and is recorded in the artifact.

| | pairs | $r_\text{cross}$ | $r_\text{full}$ | ratio | $\|\Delta\hat\tau\|$ / floor |
|---|---|---|---|---|---|
| the 4 unselected lines (headline) | 11,033 | 0.232 | 0.516 | **0.449** | 1.62 |
| pairs involving MCF7 | 7,913 | 0.248 | 0.516 | 0.481 | 1.57 |

- **Verdict: effect modification.** Cross-line agreement is 45% of what two
  estimates of one shared τ would reach, and every one of the 10 line pairs
  lands in 0.36–0.55 — this is not one odd line. The difference between two
  lines' τ̂ is 1.62× the noise floor, where one shared τ predicts ~1.
- **MCF7's pairs are reported apart** because `responders.json` is MCF7's, so
  its τ̂ is selected on being large. The headline is the median over the six
  pairs among the four lines that were never selected on; MCF7's pairs agree
  (0.481), so the selection does not drive the result.
- The lines are otherwise comparable: 3,825–3,845 arms each, median ‖τ̂‖
  16.8–18.3, responder arm fraction 35–43%.
- **It is compound-specific, which is what step A needs.** Per-compound median
  cross-line cosine over the unselected pairs runs from −0.10 (tetrindole,
  KU-60019, JW55) to +0.73 (triptolide, CGP-60474, WZ-3105). Broadly cytotoxic
  compounds act alike everywhere; others are line-specific. So `cell_id`
  genuinely modifies the compound response, and step A has a real effect to
  measure rather than another null.

### P1 implementation and review (2026-10-04)

Code for `understand.md` §3.3.1, written after C2's criterion-2 failure and
before any P1 result. New: `src/eval/dr_target.py`, `src/tests/check_p1.py`,
`scripts/p1_target_cpu.sub`, `splits.target_groups`; `step_c_report` extended.

**The estimator.** Over the kept train rows of each group,
δ_g = Σ w_i (Y_i − μ̂(X_i, A_i)) / Σ w_i, and τ̂_P1(a) = τ̂_gen(a) + δ_{g(a)}.
The generator is untouched; the vehicle side is uncorrected because μ̂(0) comes
from real DMSO wells, which are never thinned. It is post hoc and CPU only: it
reads `*_gen.npz` from a finished scoring run and never samples a model.

| Change | Why |
|---|---|
| **The group is derived, never written out again**: `splits.target_groups` returns the positivity cell with the confounder dropped. `syn_c` → (compound, dose half), 3,497 groups; `cell_id` → **the arm key itself**, verified equal to `arm_keys` on every treated row | It is the coarsest key at which positivity holds, which is what makes the AIPW correction well posed where C2's arm-level weighting was not. Deriving it from `positivity_cells` means the two cannot drift, and it makes P1 Phase-6-ready: under `cell_id` it runs at full arm resolution, the resolution C2 showed the weighted risk cannot reach |
| **Hájek, not Horvitz–Thompson** | Measured on both γ: `counts` satisfies Σ_{i∈g} w_i = n_ν(g) to 3e-08 relative, so the two coincide there — but the `design` weights are off by 7–9% on average and up to 20 rows, because P_c/π is HT and its sum is only *unbiased* for n_ν(g). `understand.md` §3.3.1's "the normaliser is exact" was wrong and is corrected. The module asserts the identity only for `counts`, and records the HT/Hájek gap otherwise |
| A standalone module rather than a flag in `evaluate.py` | `evaluate.main()`'s generated branch samples the model, so reusing it would mean threading a "load `_gen.npz` instead of generating" path through the code that produced the numbers already in this section. P1 must also run against runs scored weeks ago |
| The targeted document is built by **copy-and-overwrite**, then its `pools.<p>` and `.accuracy` key sets are asserted equal to the source's, recursively | A second module that rebuilds the schema would drift from `_pool_block`. This way a new real-side key is inherited silently and a renamed one raises |
| `learned_syn_effect`, `quality`, `reference` and `per_compound` are inherited and asserted byte-identical | λ̂ reads `row_mean`, which P1 does not touch, so it is identical by construction rather than merely invariant. `reference` holds a 200-draw RNG floor that must not be redrawn |
| **The baseline-reproduction guard.** Before applying δ, the module recomputes the source's own `cos_all`, `cos_responder`, `pooled_gene.mse` and `bias_along_v` on the *uncorrected* τ̂ and refuses unless they match within 1e-6 | It proves y_all, the oracle, the arm intersection and the whole metric path reproduce the original scoring before a single correction is applied. Measured 8.5e-10 |
| The frame is rebuilt from the **base** nuisance dir, the weights and train_idx from the **tier** dir | Found by the guard firing on the first run: a tiered arm trains on the thinned split but is *scored* on the unablated base split (§3.8.1), so its document carries the base fingerprint while its weights live on the tier's train_idx. The module also asserts the tier's `expr_meta` shares the base z-scale, or Y_i and μ̂ would be in different spaces |
| `step_c_report` gains `--dr_arm` / `--design_arm` / `--baseline_arm`; criteria 2–4 and the λ̂ diagnostic range over the ACTIVE set, the table shows every arm found | With the defaults the active set is the original four in the original order, so the recorded step-C and C2 verdicts do not move — asserted by a regression section in `check_p1` that re-runs the default invocation and diffs every criterion and every judged arm. `--baseline_arm` exists because the `p1_naive_*` family's comparator is `naive`, not `conditional` |
| `targeted_label(doc, src_label)` **derives** the arm from the source arch label plus the recorded weight mode, and only cross-checks `targeting.arm` | Keeps `arm_label`'s principle — the label comes from what was trained, never from a string the producer wrote — and rejects a hand-edited block |
| Criterion 4 gained a `targeting` branch | A targeted run's arch says `dr_mode: conditional`, so the existing weight-hash check never fired and a targeted arm could have passed with weights that no longer exist. It now recomputes the sha1 from the file the targeting block names, and also gates on that run's baseline reproduction and counts identity. `ones` is exempt (no file) and can never be a DR arm |
| **`ones` is an α-sensitivity control, not a negative control** | It was planned as a null expected to reproduce `conditional`. It is not: with flat weights the Hájek mean runs over the *thinned* mix, so δ_g ≈ β·p_kept(g)·v and the correction ADDS the thinning bias. Same one free parameter per group, wrong α, far worse — which is the evidence that P1 is not fitting the statistic it is judged on. `understand.md` is corrected |

**Review** (automated `code-review`, high): five findings, all fixed.

| Finding | Fix |
|---|---|
| **`gof` derived the dose half from the POOL's arm table.** `dose_half` ranks a compound's *distinct* dose levels, so a pool that dropped arms (`--min_dose_n`, or the holdout's one-well arms) shifts those compounds' halves. Measured: 12 arms in `all` and 150 in `holdout` got the other half's δ while `bias_along_v` still classified them by the table's halves — a direct bias on the statistic P1 is judged by | The arm → group map is now resolved on the whole table, exactly as `evaluate.py` builds `half_of_arm` and for the same reason. `check_p1` gained the check that would have caught it: the stored map must equal the whole-table derivation. Run against the pre-fix artifacts it reports exactly 12 and 150; after the fix, 0 |
| The §3.8.4 diagnostic indexed `per_arm["conditional"]`, which `--baseline_arm naive` would not populate (KeyError) | It follows `--baseline_arm` |
| `check_p1`'s α-sensitivity message formatted a `None` baseline, aborting the run instead of reporting | Guarded |
| `check_p1` assumed a full `_tau.npz`; a `--npz delta` artifact would KeyError | Missing arrays are skipped |
| The default `--out` suffix used only `--dr_arm`, so verdicts differing in `--baseline_arm` would overwrite each other | All three judged arms name the file |

**Checks before launch**: the unit section (closed form, Hájek scale invariance,
HT ≡ Hájek when Σw = n, gene-blocking invariance, within-group contrast
invariance) locally; then on `bindel` a two-run smoke in which δ re-derived
independently from `expr.npy` + `_gen.npz` matched to 5.5e-08, the arm → group
map matched the table, and every array P1 does not touch was identical to the
source's. Measured 2.7 GB peak and ~1 min per targeted output.

### P1 results: targeting passes every criterion (2026-10-04)

Jobs 959846 (48 targeted documents, 1 h) and 962295 (checks + verdicts) on
`bindel`. 959846 wrote all 48 and passed every artifact check, then died in
`check_p1`: the step-C source documents predate `learned_syn_effect` (it
arrived with the C2 code) and the inherited-block comparison assumed every
source had it. It now compares only the keys the source carries, and flags a
dropped or invented one. Re-running the checks and verdicts on the existing
outputs takes ~3 min (repeated as job 964812 on 2026-10-05, identical results). `check_p1`: ALL CHECKS PASSED, including that `step_c_report` with
its defaults still reproduces `step_c_verdict.json` and
`step_c_verdict_compound_r1.json` — every criterion and every judged arm — so
the recorded C2 verdict has not moved.

**DiD of the scored high − low contrast, step C2, `--pool all`**, mean ± sd over
the 3 training seeds. P1 arms are the SAME generators, re-estimated:

| arm | DiD | vs its baseline | criteria |
|---|---|---|---|
| `naive` | −1.930 ± 0.035 | — | |
| `conditional` | −0.279 ± 0.007 | — | |
| `dr` (C2's weighted risk) | +0.575 ± 0.046 | **worse** than `conditional` | **fails 2** |
| `dr_design` | +0.562 ± 0.032 | | |
| **`p1_cond_counts`** | **+0.047 ± 0.013** | 28.5σ better than `conditional` | **all 4 PASS** |
| `p1_cond_design` | +0.133 ± 0.010 | | |
| **`p1_naive_counts`** | **−0.001 ± 0.012** | 90.8σ better than `naive` | **all 4 PASS** |
| `p1_naive_design` | −0.147 ± 0.014 | | |
| `p1_cond_ones` (control) | −3.893 ± 0.098 | | |
| `p1_naive_ones` (control) | −9.000 ± 0.012 | | |

Findings:

1. **P1 removes the bias the weighted risk could not.** On the same generators
   C2 scored, targeting takes `conditional` from −0.279 to **+0.047** and
   `naive` from −1.930 to **−0.001**, where C2's weighted arm went the wrong
   way to +0.58. §3.8.4's criterion 2 passes in both families, on both halves
   (beats the baseline, and close to the true-weight arm).
2. **`p1_naive_counts` is the textbook double-robustness result.** Its outcome
   model is blind to `syn_c` (λ̂ = 0.000, measured), so the entire correction
   is carried by α — and the bias lands at −0.001 ± 0.012. That is the 2×2
   corner the retired "Option A" was to provide, obtained with no retraining.
3. **The weights are load-bearing: the α-sensitivity control is decisive.**
   With w ≡ 1 — the same one free parameter per group, the wrong α — the
   contrast goes to −3.89 and −9.00, far WORSE than the untargeted arms. So
   P1's near-zero result is not an artefact of having 3,497 free parameters for
   10,446 arms; it is the weights doing the work.
4. **It is not leakage from the missing cross-fitting.** On `--pool holdout`,
   where the generator's rows, the oracle's wells and δ's train rows are three
   disjoint sets, the result survives: `conditional` −0.234 → **+0.144**,
   `naive` −2.026 → **−0.009** (criteria pass on that pool too).
5. **Step C is the null, and it behaves.** Where `conditional` is already
   unbiased, targeting leaves it there (+0.005 → −0.062), and it still repairs
   `naive` (−6.855 → **+0.022**). Unscored compounds move by ≤ 0.004 everywhere.
6. **P1 trades variance for bias, and the trade is visible.** Pooled gene MSE on
   C2 *improves* (−3.8% `p1_cond_counts`, −12.5% `p1_naive_counts`), because
   there δ carries real signal. On step C, where there is nothing to correct,
   it *costs* +13.4% / +12.2%: δ_g is then an estimate of ~0 built from a
   handful of rows, i.e. pure added variance. Both are expected and only the
   first is asserted.
7. **`counts` beats `design`, consistently** (+0.047 vs +0.133; −0.001 vs
   −0.147), and the gap is measured, not just asserted. The design weights are
   the true *inclusion probabilities*: they balance the confounder in
   expectation over thinning draws, not in the draw we have. The counts
   weights are post-stratification on the positivity cell, so they balance the
   realised sample exactly. Over scored groups:

   | weights | realised confounder mix − unthinned target | median per-group error | high − low contrast, γ = 1 |
   |---|---|---|---|
   | `counts` | **exactly 0** | 0.00000 | 0.00000 |
   | `design` | systematic | 0.071 | +0.0052 |

   That leftover imbalance does not cancel: its DiD is −0.0052, and × β = 25.48
   predicts a residual bias of **−0.133** against the observed
   `p1_naive_design` −0.147 — ~90% of the gap. The mechanism is ratio bias: a
   design weight is constant within a cell, so the cell's total weight is
   proportional to a *random* kept count, and the cells the thinning shrinks
   (low π, large w) fluctuate most — and which cell is shrunk is set by the
   dose half, the very axis the readout measures. This is the classical result
   that IPW with estimated propensities beats IPW with the known true ones
   (Hirano, Imbens & Ridder 2003; post-stratification in survey sampling).
   **§3.8.4 criterion 3 was reworded on 2026-10-05 because of this**: `design`
   is not a gold standard to converge on, so that criterion tests agreement in
   sign and magnitude, not convergence.
8. **`p1_naive_*` is the most favourable misspecification, not an arbitrary
   one.** `naive` omits exactly the covariate the weights reweight, so its
   residual *is* the quantity the correction averages. `p1_naive_counts` ≈ 0
   shows AIPW's algebra works here; it does not show robustness to an
   arbitrary wrong outcome model, and no such arm exists on disk.

**The honest uncertainty is larger than the seed sd, and must be quoted with
it.** The three seeds share one `Y`, one weight set and one thinning draw, so
their spread measures generator noise alone, and the 28.5σ / 90.8σ figures are
correspondingly inflated. The compound-clustered SE of a single scored contrast
(jackknifing the 641 independently thinned compounds) is **0.088** for
`p1_cond_counts` and **0.160** for `p1_naive_counts`, against a seed sd of
~0.013. Read against that, P1's residual bias is **indistinguishable from
zero**, `conditional`'s −0.279 is ~2σ from zero, and `dr`'s +0.58 is clearly
non-zero. The ranking stands; the σ counts do not.

**What this says about ADIGen.** C2 showed the α-weighted *training risk* fails
when the positivity cell is coarser than the resolution the generator works at.
P1 shows the same α, spent on the *estimand* at the cell's group, removes the
bias — on a correctly specified outcome model and on one blind to the
confounder alike. The failure in C2 was the place α was spent, not α itself.

**For Phase 6.** `splits.target_groups` derives the group from the positivity
key, and under `cell_id` that group *is* the arm, so P1 runs at full arm
resolution in step A with no new code. It should be scored there beside the
weighted arm. Note the regime differs: with ~2 train rows per (arm, line) cell
the effective sample size per group falls from ~6 to ~2, which is where the
added-variance cost in finding 6 would bite hardest.

### P2 implementation and review (2026-10-05)

Code for `STEP_A.md` §4 (track B): the weighted risk with weights normalised
within each target group and capped. Written after the plan and its pass rule
were fixed, and before any P2 result.
- New: `src/nuisances/weight_norm.py`, `src/tests/check_p2.py`,
  `scripts/p2_gpu.sub`.
- Changed: `train_diffusion.py` (`--dr_weight_norm`, `--dr_weight_clip`),
  `step_c_report.py` (the `dr_p2` arm and the pass rule),
  `scripts/phase5_arms.sh` (the `dr_p2` arm; final checkpoint only).
- **Local checks** (numpy only, torch unimportable): `check_p2` on synthetic
  weights and on both tier instances' real weights; `step_c_report` with its
  defaults still reproduces `step_c_verdict.json` and
  `step_c_verdict_compound_r1.json` (criteria and judged arms identical).
- **GPU smoke** (job 990819, `zabih`, 2 min 30 s): 50 checks, 0 failures.

| Change | Why |
|---|---|
| `weight_norm.group_normalize`: every target group (`splits.target_groups`) sums to its row count; the cap is met **exactly** with the group total preserved, by capping and letting the group's other rows carry the rest until none exceeds it | `STEP_A.md` D6. A one-shot "cap, then renormalise" leaves weights above the cap. A row whose raw weight is 1 comes out at exactly 1.0, which is what "unthinned compounds are untouched" needs |
| The groups are derived on the **whole table** and indexed by the train rows (`weight_norm.train_groups`) | `dose_half` ranks a compound's distinct dose levels, so a subset that lost a level shifts its halves. This is the bug `dr_target` had ("P1 implementation and review") |
| The default path is untouched: `--dr_weight_norm global` with no cap runs the old line of code, and `arch.json` records `None` for every P2 field | Bit-identical weights for every existing arm, and old runs still resume and still read as `dr`. The smoke asserts both |
| `arch.json`: `dr_weight_norm`, `dr_weight_clip` and the integer counts (`dr_weight_groups`: groups, capped rows, rows at exactly 1) are resume identity; the float summaries (`dr_weight_stats`: ESS, max) are not | Review: the floats can differ in the last bits between CPU types (as the direction matrix did in step C2), but the counts are exact and would move if the grouping or the cap routine changed under a resumed run |
| An **infeasible cap is refused** | Review, confirmed: with zero-weight rows a group's positive rows can be too few to carry its row count at the cap (`[1, 0, 0, 0]`, cap 2). The first version returned the group short of its total without an error |
| `step_c_report.arm_label`: a weighted run is `dr` only with the global normalisation and no cap, `dr_p2` only with group normalisation and the cap `weight_norm.P2_CLIP`; any other combination is not an arm | Without it a P2 run would have been labelled `dr` and collided with the recorded cell. An ablation at another cap cannot fill either cell, and the skip message now says why |
| The P2 pass rule is judged in the report (`p2_pass_rule`), only on the C2 injection and `--pool all`, and it includes the `dr_p2` runs' weight-hash check | Review: the thresholds come from C2's numbers, so applying them to step C would be meaningless; and `dr_p2` is outside criterion 4's judged set, so its weight provenance was not being checked at all |
| The cap is written once, `weight_norm.P2_CLIP = 5.0`; the launcher, the smoke and the report read it | Review: three copies could drift, and a run at another cap would have vanished from the report without a message |
| `scripts/phase5_arms.sh`: `CKPT_EVERY` defaults to `$EPOCHS` | `STEP_A.md` D2: only the final checkpoint is ever scored. Pass `CKPT_EVERY=100` on the preemptible `gpu` partition |
| The smoke compares the trainer's recorded counts with an independent recompute (plain loops over (compound, dose half)), asserts the cap binds, and checks the trainer's exit status | Review: three of its first assertions could not fail, among them "weight exactly 1" checked on the script's own numbers. `arch.json` is written before the first step, so its presence alone did not show that training ran |

Review findings not acted on: the trainer's log line still uses its own ESS
helper next to `knn_dr.ess` (same formula; `weight_norm` now uses the shared
one).

**What the normalisation does to the real weights** (`check_p2`, counts, γ = 1;
the smoke's trainer log agrees to every digit):

| | global (`dr`) | group + cap 5 (`dr_p2`) |
|---|---|---|
| weight of an unthinned or vehicle row | 0.809 | exactly 1 |
| share of total weight on scored rows (23.4% of rows) | 38.0% | 23.4% |
| largest weight | 17.80 | 5.00 |
| ESS / n | 0.678 | 0.921 |
| `syn_c` mix of a scored group against the unthinned pool, mean error | 0.0000 | 0.0004 |

- **The cap hardly binds:** 4 rows in 4 groups at γ = 1, none at γ = 0 (the
  largest group-normalised weight is 7.49). P2 is in effect the group
  normalisation; the cap is a guard.
- **The confounder balance inside each group is kept**, so the weights still
  do the job they are for.
- **The reallocation it removes is a 19% down-weighting** of 16,542 unthinned
  and vehicle rows. That is smaller than the loss it is meant to explain (λ̂ on
  unthinned compounds 0.72 → 0.14), so the lower weight noise (ESS 0.68 →
  0.92) may matter as much. The six runs decide.
- The design weights do not balance `syn_c` within a group under either
  normalisation (mean error 0.078 at γ = 1), which is the P1 finding again.

**Launched 2026-10-05:** `VARIANT=c2 ARMS=dr_p2 bash scripts/phase5_arms.sh`,
training 990856–990866 (even IDs), each scored by the next ID, on `zabih`
(3 of 6 A6000s free at submission). Run dirs `stepc2_dr_p2_g<γ>_s<seed>`. The
verdict comes from `step_c_report --syn_meta syn_meta_compound_r1.json`.

### P2 results: the generator is repaired, the bias is not (2026-10-05)

Six `dr_p2` runs on step C2 (`stepc2_dr_p2_g{0,1}_s{0,1,2}`; training
990856–990866, scoring 990857–990867, all 12 exit 0 on `zabih`), scored at the
§3.14 settings on `--pool all`. Verdict in
`runs/mcf7_24h/eval_artifacts/step_c_verdict_compound_r1.json`
(`p2_pass_rule`). The clustered errors and the split by kept-train status below
were computed locally (numpy, `OPENBLAS_CORETYPE=Haswell`) from the `_tau.npz`
files.

**Pass rule (`STEP_A.md` §4, fixed before the runs): FAIL, on the third part.**

| Metric | `conditional` | `dr` | `dr_p2` | needed | |
|---|---|---|---|---|---|
| learned share λ̂ on unthinned compounds | 0.720 | 0.143 | **0.666** | ≥ 0.60 | pass |
| pooled gene MSE, relative to `conditional` | 1.000 | 1.288 | **1.054** | ≤ 1.10 | pass |
| \|DiD\|, scored | 0.279 | 0.575 | **0.599** | ≤ 0.58 | **fail** |

By `STEP_A.md` §4, a failed rule drops `dr_p2` from the step-A matrix. Whether
to override that is the user's decision; the facts for it are below.

**Read with the status review of 2026-10-05** (§5, gap 1). Finding 5's "arm-level
ratio bias" is not established. `dr_p2`'s learned share still differs between
the dose halves at γ = 1 (0.415 low, 0.472 high). That term alone is +0.71,
against a DiD of +0.60; the remainder is −0.11, as in `conditional`. The pass
rule's outcome and the user's override stand.

Findings:

1. **P2 repairs what the global normalisation broke.** On unthinned compounds
   the learned share recovers from 0.14 to 0.67 (0.70 at γ = 0, 0.63 at γ = 1),
   against `conditional`'s 0.72, and the MSE excess over `conditional` falls
   from +29% to +5%. So the degradation was the normalisation: the weights
   moving mass between groups, and the weight noise that came with it.
2. **The over-correction is unchanged.** DiD +0.599 ± 0.016 (seed sd), against
   `dr`'s +0.575 ± 0.046. Per seed: 0.583, 0.598, 0.615.
3. **The failed part is a miss by 0.019, well inside the noise.** With errors
   clustered on the 641 scored compounds (jackknife, seeds averaged):
   `dr_p2` +0.599 ± 0.054, `dr` +0.575 ± 0.072, and the paired difference is
   +0.023 ± 0.055, or 0.4 SE. `dr_p2` is not distinguishable from `dr` on bias.
   Against `conditional` it is clearly worse (|DiD| +0.32 ± 0.08 higher).
4. **The rule had no noise allowance, which was a flaw in the rule.** The
   threshold 0.58 was `dr`'s own value, rounded. "No worse than `dr`" should
   have been written as "not worse by more than 2 SE"; under that reading the
   third part passes. The rule is reported as declared all the same.
5. **The two C2 problems are now separated.** P2 removes the generator's
   degradation and leaves the bias where it was, so the over-correction is not
   a side effect of a damaged generator. It is the arm-level ratio bias of
   "Step C2 results", finding 3, and it shows the same signature. DiD by the
   arm's kept train wells at γ = 1 (mean of 3 seeds):

   | arm | none | one `syn_c` level | both levels | all |
   |---|---|---|---|---|
   | `conditional` | +0.53 | −0.41 | −0.43 | −0.28 |
   | `dr` | −0.13 | +0.50 | +1.23 | +0.58 |
   | `dr_p2` | −0.05 | +0.61 | **+1.05** | +0.60 |

6. **What this means for step A.** There the target group is the arm itself,
   so the within-group balance the weights provide is at the resolution the
   generator fits, and the ratio bias of finding 5 has nowhere to arise. The
   degradation of finding 1 does carry over: it comes from the global
   normalisation, which step A's `dr` arm would still use. On mechanism,
   `dr_p2` is therefore the better-founded weighted arm for step A, although
   it failed the rule as written.

**Decision (user, 2026-10-05): override.** `dr_p2` runs in step A next to
`dr` (`STEP_A.md` D8), on the grounds of findings 3, 4 and 6. The pass rule's
recorded outcome stays FAIL.

### Step A implementation and review (2026-10-05)

Code for `STEP_A.md` track A (A0 and A7), written after its plan and criteria,
and before any step-A result. Nothing has run on the full `core5_24h`
population: everything below is from a `--limit` build (2 whole plate maps,
11,904 wells on 32 plates, 110 compounds, 103 of them scored).

- **New:** `src/eval/{contrast_stats,step_a_power,step_a_report}.py`,
  `src/tests/{check_phase6,smoke_phase6_torch}.py`,
  `scripts/{phase6_cpu,phase6_gpu,phase6_p1_cpu,regress_mcf7_cpu}.sub`,
  `scripts/phase6_arms.sh`.
- **Changed:** `spec.py`, `build_dataset.py`, `splits.py`,
  `build_tiered_split.py`, `responders.py`, `evaluate.py`, `dr_target.py`,
  `step_c_report.py`, `check_build.py`, `check_phase1.py`, `check_phase4.py`,
  `scripts/train.sub`.
- **Jobs of record:**
  - 991752 (`bindel`, 64 s): the whole data layer on a fresh limit build.
    203 checks, 0 failures.
  - 991753 (`zabih`, 3 min 40 s): the GPU smoke. 29 checks, 0 failures.
  - 991754 (`bindel`, 3 min 24 s): the `mcf7_24h` regression. Every check
    passes.
- `ma` was at 4% of its memory unallocated, so every CPU job went to `bindel`.

| Change | Why |
|---|---|
| `spec.POPULATIONS` and `--population`; `apply_paths_args` adopts the population a `--data_dir` build records and refuses a contradicting flag; `splits.table_fingerprint` refuses a config whose population is not the build's | One flag cannot point a command at the wrong table, and every existing command line keeps working with `--data_dir` alone. No field was added to `PopulationSpec`, whose `asdict` is embedded in `decisions_record`: the `mcf7_24h` build's decisions still compare equal (regression job) |
| A population with more than one line keeps only compounds with a treated well in every line, after plate QC (`build_dataset.common_compound_rows`); its `--limit` keeps whole plate *maps* | §3.8.3. On the full population the rule drops nothing: all 1,750 compounds are in all five lines before QC (counted from `inst_info`). Whole maps keep each smoke-build compound in each line |
| `spec.LINE_GROUPS`: G1 = {MCF7, HT29, PC3}, G2 = {HA1E, A375}. Recorded in `population_qc.json`, the tier dir tag (`…_G2-A375-HA1E`), `splits.json` (tier and params), `arch.json` and the oracle | `STEP_A.md` D3, declared before any step-A result |
| `build_tiered_split`, `cell_id`: `z_C` from the line groups; cells are (dose level, line); the positivity redraw is **per cell** | With ~30 cells of ~2 wells per compound, redrawing the whole compound until every cell keeps a well would take ~10³–10⁵ draws. Cells are independent, so the per-cell redraw has the same distribution. Step C keeps its whole-compound loop, and an existing step-C instance still reads as "same parameters" (regression job) |
| **`--keep_frac` for `cell_id` is the expected *realised* kept fraction, default 0.75** (`_calibrate_pi_realised`: the scale of π is set by bisection so that Σ πᵢ / P_cell = keep_frac · n). `STEP_A.md` D9, approved by the user 2026-10-05 | Review. With 2-well cells the redraw moves the realised fraction far from the nominal one, and by a γ-dependent amount: at nominal 0.6 the γ = 0 control kept 68.0% and γ = 1 kept 73.9%, so the control was not size-matched. After the change: 74.9% and 74.8%. 0.75 is about what nominal 0.6 produced at γ = 1 and near the strongest lever a 2-well cell allows; a realised 0.6 would be matched too, with a lever about 5× weaker |
| `responders --confounder cell_id`: a compound is thinnable when every dose level has a train well in every line | The builder refuses a whole instance over one unthinnable compound (as in step C). Limit build: 103 of 108 responders |
| The oracle stores each arm's G2 − G1 contrast twice: over the pool's wells (`contrast`, descriptive) and over its **holdout wells only** (`contrast_dir`, the readout's direction) | See the next row. Per-line τ̂ is not stored: nothing reads it |
| **The readout's direction comes from holdout wells** (`evaluate.bias_along_contrast`) | Review, confirmed on the limit build. With a direction from the pool's own wells, an estimator that copies noise from its kept train wells projects on a direction built from those wells, and that term scales with the mix shift Δp. It is not removed by the γ = 0 control, because Δp is what γ switches on. The data-level bias read **−5.59** that way and **−1.69** with the holdout direction, so it was inflated ~3.3×, and S0 could have passed with no line effect at all. `STEP_A.md` §3 is corrected |
| `contrast_stats`: every error is a delete-one-compound jackknife, from per-compound sums | `naive` has one seed, and seeds share one thinning draw ("P1 results"). Checked against a brute-force jackknife (identical) and by Monte Carlo with compound-level effects (mean SE / true sd = 0.97; the arm-level SE understates it) |
| `step_a_power` (S0): the memoriser of each arm's kept train wells, with the report's statistic. Also `planned` (mix shift × the contrast over unthinned train wells) and the weighted memorisers | S0 must be read before any GPU job. `planned` shares no outcome noise with the memoriser, so their agreement is a cross-check |
| `step_a_report`: S1–S5, arms labelled from `arch.json` (`arm_label(arch, "cell_id")`), P1 documents as their own arms, admission by scoring settings, population, oracle and line groups; duplicates refused; S2 is a testability flag (`not_testable`), never a failed criterion | `STEP_A.md` §3. Review: the first version recorded an untestable S2 as FAIL |
| `dr_target` carries the step-A readout through the correction, and its reproduction guard covers it | P1's targeted documents must be judged by the same statistic. `STEP_A.md` had said "no code change expected" |
| `phase6_arms.sh` launches only if S0 passed **on the tier instances it is about to train on** (split fingerprints compared); `FORCE=1` overrides | Review: a rebuilt tier with a stale `step_a_power.json` would have been waved through, and the message named a `FORCE` that nothing read |
| `build_tiered_split.tier_dir` and `--print_out_dir` are the one definition of a tier dir's name | Review: the name was rebuilt by hand in five places |
| `scripts/train.sub`: `AUTO_RESUME` / `PRUNE_CKPT`, set by the launcher on a preemptible partition only | Review: the `gpu` fallback had no resume path, so a requeued run would have died on "already holds checkpoints" and blocked its scoring job. The shell logic is tested on dummy directories; a real preempt-and-resume cycle is not |
| `check_phase4`: the QC arm counts and the responder band are skipped on a multi-line population | Both are single-line quantities: QC counts (line, compound, dose) arms, while the estimand's arm pools the lines; the band is MCF7's ~29% at 3 wells per arm. First limit run: 2 false failures |
| `regress_mcf7_cpu.sub` | The population switch touches every stage, so the MCF7 results are re-derived after it: both oracles rebuilt into scratch (24 arrays each, bit-identical; every recorded JSON value reproduced), both step-C verdicts reproduced, `check_build` / `check_phase1` (base and both tiers) / `check_phase4` / `check_phase5` / `check_p1` / `check_p2` all pass |

Review findings not acted on: the (dose level, line) cell key is still written
in three formats (the builder, the responder screen, and `check_phase1`'s
independent recompute). `check_phase1` and `check_phase6` verify the outcome
(every cell keeps a well; weights = n_unthinned / n_kept per cell).

**Measured on the limit build** (103 scored compounds, 618 scored arms; these
size the experiment and are not results):

| | value |
|---|---|
| kept fraction of scored train wells, γ = 0 / γ = 1 | 0.749 / 0.748 |
| shift of the G2 share among kept wells at γ = 1, low / high dose half | +0.18 / −0.13 |
| scored arms with a holdout-well direction | 607 of 618 |
| corr(counts, design weights) on kept scored rows, γ = 0 / γ = 1 | +0.10 / +0.72 |
| line effect per unit of that shift, ⟨c_train, u⟩ | 5.45 |
| **S0: bias in the training data** (memoriser DiD) | **−1.69 ± 0.16 (10.8 SE)** |
| the same from the line mix alone (`planned`) | −1.63 ± 0.13 |
| memoriser, counts weights | +0.01 ± 0.09 |
| memoriser, design weights | −0.10 ± 0.10 |
| counts-weighted G2 share of each scored arm against its unthinned share | equal to 1e-08 |
| scoring, host memory (11.9k rows, quality block on) | 2.2 GB |

- **The weights can work at the arm level here.** The counts weights restore
  every scored arm's unthinned line mix exactly, and the counts-weighted
  memoriser has no bias left. In step C2 the same weights over-corrected at the
  arm level ("Step C2 results", finding 3); that mismatch is absent in step A
  by construction.
- **P2's group normalisation keeps that balance** and leaves every unthinned
  row at exactly 1 (`check_phase6`); the cap does not bind on the limit build (max weight 4.0; on the full build it binds on 2 rows, see "Step A launch").
- **Two code paths agree.** P1's flat-weight control on an untrained generator
  is a memoriser, and the GPU smoke's report gives it −1.690 ± 0.157 against
  `step_a_power`'s −1.688 ± 0.156. P1 with the counts weights gives
  +0.015 ± 0.090 against the counts-weighted memoriser's +0.011 ± 0.087.
- **P1's groups are larger than feared.** The group is the arm pooled over its
  lines: n_eff 7–9 here, not the ~2 that "P1 results" expected for step A.
- **Not yet known:** how much of the data-level bias a trained `naive`
  generator shows (65% in step C, 18% in C2), the scoring memory on the full
  build, and whether the full population's S0 passes. On the full population
  the SE falls with ~6× the compounds, but so may the effect (next entry).

### Track B re-read for step A; approvals (2026-10-05)

**Approved by the user:** the holdout-well direction of the readout
(`STEP_A.md` D10) and `keep_frac` as the realised fraction, 0.75 (D9).

Track B is the six `dr_p2` runs of "P2 results"; nothing else ran. Read again
for what it implies about step A:

| Adjustment | Why |
|---|---|
| **S4 has a noise allowance** (`STEP_A.md` D11; `step_a_report`): an arm fails it only if its unscored DiD is above a quarter of `naive`'s scored bias **and** more than 2 SE from zero | The P2 rule failed on a threshold with no noise allowance (finding 4). S4 had the same shape, and on the GPU smoke's untrained arms it failed on differences of 1e-5. Re-run on those artifacts: it passes, with every arm within 2 SE of zero. S1–S3 already carried SE margins |
| `step_a_report` prints, per arm, the MSE relative to `conditional` and the learned line contrast on scored **and unscored** compounds | Track B's damage to the generator showed on unthinned compounds and in the MSE, not in the bias column |

No change to the arms, the data layer or the launch sequence.

**The limit build overstates what the full population will show.** It is the
first two plate maps, LJP005 and LJP006 (the kinase-inhibitor library):

| | LJP compounds | REP compounds |
|---|---|---|
| share of the `core5_24h` population | 16% (273) | 84% (1,477) |
| responders on MCF7 at 3 wells per arm | 96.7% | 26.2% |

- The limit build's S0 (−1.69, 10.8 SE) comes from the most active sixth of
  the compounds, so it cannot be scaled to the full population by the number
  of compounds. The full build's own S0 decides.
- It also made the scored share look extreme: 98% of the limit build's
  compounds are responders. A rough projection from the MCF7 oracle (floor at
  15 wells ≈ 6.5; a compound responds if its true ‖τ‖ exceeds ≈ 7.2) puts one
  half to two thirds of the full population in the scored set. So step A
  should keep a sizeable unthinned group, which S4 and the unscored diagnostics
  need. That projection uses 3-well estimates and is rough.
- With the global normalisation, unthinned rows sit at 0.78 on the limit tiers
  (C2: 0.81), so the mechanism that damaged `dr` on C2 is present in step A,
  and `dr` against `dr_p2` remains an informative comparison.

#### Step A launch: full data layer, S0, GPU smoke (2026-10-05)

Launched on the user's instruction. Nothing in the code changed for the launch.

| Job | What | Partition | Result |
|---|---|---|---|
| 992724 | `phase6_cpu.sub`: build, base split, oracle, responders, tiers γ = 0 and 1, S0, `check_phase6` | `bindel` (15 GB; `ma` had 4% memory unallocated) | all checks pass; 6 min 30 s; peak 2.5 GB |
| 992778 | `phase6_gpu.sub data/core5_24h`: the end-to-end smoke on the full build | `zabih` (32 GB requested) | all checks pass; 24 min; scoring peak 9.0 GB |
| 993188–993215 | 14 × (train → score), `phase6_arms.sh` | `zabih`, 16 GB each | running |
| 993216 | P1 + `step_a_report`, after every scoring job | `bindel` | pending |

**The full population** (`data/core5_24h`, table fingerprint `95680ff4c5ad23f0`):

- 182,174 wells on 494 plates; 1,750 compounds, every one with a treated well in
  all five lines. Plate QC dropped 2 plates (745 wells).
- 891 responders at 1.5× the floor (the pooled oracle, 10,484 arms); 848 pass
  the thinnability screen and are the scored set. 902 compounds are not thinned.
- Tiers keep 0.747 (γ = 0) and 0.751 (γ = 1) of the scored compounds' train
  wells, so the control is size-matched. At γ = 1 the G2 share moves +0.145 in
  the low dose half and −0.136 in the high half.

**S0, the power gate: PASS.**

| Quantity | DiD ± clustered SE |
|---|---|
| memoriser (the bias in the training data) | −0.941 ± 0.052 (18.2 SE; needs ≥ 5) |
| planned (line-mix shift × the arm's own line contrast) | −0.917 ± 0.043 |
| memoriser, counts weights | −0.046 ± 0.031 |
| memoriser, design weights | −0.047 ± 0.035 |

- The effect is 56% of the limit build's (−1.69), as its LJP-only composition
  predicted, and the error is a third of it.
- The counts weights remove 95% of the data-level bias; what is left is 1.5 SE
  from zero. This is the arm-level property step C2 lacked.
- An arm that learned all of the planted bias would show ≈ −0.94. Step C2's
  `naive` learned 18% of its data-level bias and step C's 65%; here those
  fractions would be −0.17 and −0.61, which are 3 and 12 SE. So S1 is not
  guaranteed: it depends on how much a trained `naive` picks up.

**P2's cap binds on the full build.** At γ = 1, 2 rows reach the cap of 5.00,
and one arm's weighted G2 share is off by 0.058; every other arm is balanced
to 1e-8. The earlier note that the cap never binds in step A came from the
limit build. It is 2 of 114,222 training rows, so the readout is unaffected.

**The smoke on the full build** repeats the limit build's checks (roles, CFG
keeps the line, refusals, eight legs, scoring, P1, report, cross-population
refusal). On its untrained generators the flat-weight P1 control reproduces the
memoriser (−0.942 ± 0.052) and P1 with the counts weights gives −0.038 ± 0.032,
so the P1 pipeline and `step_a_power` agree on the full population. Its eight
run dirs `runs/core5_24h/p6smoke_992778_*` and
`eval_artifacts/step_a_verdict_smoke_992778.json` are not results.

**Other users' jobs are no longer visible** (`PrivateData` includes `jobs`), so
preemptibility on `zabih` cannot be read from `squeue`. `sbatch --test-only`
reports the start time and the jobs it would preempt, and is now the
availability check.

### Phase 4 and Phase 5 status review (2026-10-05)

Asked for by the user: where Phases 4 and 5 stand, which §4 boxes can be
checked, and what is missing. Nothing ran on a GPU and no code changed (the
step-A jobs 993188–993216 were running from this tree).

**What was checked.**
- Every artifact of record, read back from disk: the three oracles, the 12
  Phase 4 arm JSONs, the step-C data layer, all 54 step-C / C2 / P2 run dirs
  (`arch.json` and scoring JSON), the 48 P1 documents and the six verdict files.
- Every §5 table that comes from a JSON, recomputed from it: "Phase 4 results",
  "Phase 5 step-C results", "Step C2 results", "P1 results", "P2 results".
- The jobs of record in `sacct`, and their logs.
- Two independent code reviews of the code that had none:
  - Phase 4: `generation`, `dist_metrics`, `evaluate`, the scheduler pin, the
    checks and the scripts;
  - the step-C base path: `responders`, `synthetic`, the injection,
    `build_tiered_split`, the counts weights, `step_c_report`, the checks and
    the scripts.

  C2, P1 and P2 had theirs when they were written. The reviews' main findings
  were re-derived from the artifacts before being recorded here.

**Outcome.**
- **No bug changes a recorded number.** Both reviews traced the τ̂ path, the
  injection, π, the positivity cells, both weight sets, the dose halves and the
  DiD arithmetic, and found them correct. One reviewer re-derived the
  `mlp-B_conditional_ddpm_s0` row from its `_tau.npz`; the other re-derived the
  memoriser's DiD (−10.56) and the plan's medians from the table and the tiers.
- **Every experiment is finished**: Phase 4's twelve arms, step C, C2, P1 and
  P2. Their boxes are checked in §4.
- **Six code boxes stay unchecked**, because the reviews ask for follow-ups and
  §4's rule keeps a box open until those land:
  - Phase 4: `generation.py`, `evaluate.py`, and the checks;
  - Phase 5: `step_c_report`, the checks and launchers, and with them C2's
    parent box.

  The follow-ups are R1–R14 below.
- **Two findings change how C2 is read** (gaps 1 and 2). They do not move a
  verdict.

**Confirmed.**
- Phase 4: the twelve-arm table reproduces to every printed digit. The oracle on
  disk is job 793791's, not 791843's: the same numbers, plus the noise-corrected
  ‖τ̂‖ arrays that came with the Spearman–Brown correction. The arms of record
  are scoring jobs 793825–793836, all exit 0.
- Phase 5: all four verdict tables reproduce (step C, C2, P1 on both pools, P2's
  pass rule). The 54 runs are configured as their names say:
  - injection sha1 `492aa3d1` on the 24 step-C runs and `11c32b37` on the 30
    C2 / P2 runs;
  - γ and seed as labelled;
  - counts or design weights on the weighted arms;
  - group normalisation with cap 5 only on `dr_p2`.
- Jobs: 24 + 24 step-C, 24 + 24 C2 and 6 + 6 P2 training and scoring jobs
  COMPLETED. `check_p1` in job 964812 passes the three P1 controls (the step-C
  null, the α-sensitivity control, unscored compounds).
- C2 and P2 scoring ran on the fixed BLAS: no eigensolver fallback in any of the
  30 JSONs, PC Fréchet 42–56 on treated wells.

**Three results that were on disk but not in this log.**

1. **E10, the sampler-step check** (job 792670, `mlp-B_conditional_ddpm_s0`,
   `--pool holdout`). 100 steps are enough for τ̂:

   | steps | median ‖τ̂_gen‖ | cos, responder arms | Spearman ‖τ‖ | marginal Fréchet, DMSO |
   |---|---|---|---|---|
   | 50 | 9.37 | 0.449 | 0.606 | 33.7 |
   | 100 | 9.48 | 0.451 | 0.607 | 27.1 |
   | 250 | 9.54 | 0.452 | 0.608 | 23.5 |
   | 1000 | 9.56 | 0.453 | 0.608 | 21.9 |

   - The amplitude at 100 steps is 0.9% below 1,000 steps, and the cosine 0.002
     below. E10's choice stands for the DDIM arms.
   - The marginal Fréchet of the DMSO wells is not converged at 100 steps. So
     sample-quality numbers depend on the step count where τ̂ does not.
   - The check covers DDIM only. The eight FM arms' 100 Euler steps were not
     checked.
   - The DiT arms started at 12:00, after the 50 / 100 / 250 runs and
     13 minutes before the 1,000-step run finished.
   - Only the 1,000-step JSON is still on disk. Its `/ceil` ratio (1.58)
     predates the Spearman–Brown correction and should not be read.
2. **The curated panel on the generated arms** (§3.7's high-n anchors), from the
   four DDPM arms' `effects` blocks:
   - bortezomib and MG-132 (968 / 971 wells, one arm each): cosine with the
     oracle 0.999–1.000 and ‖τ̂_gen‖ / ‖τ̂_oracle‖ 1.01–1.02 on all four arms.
   - `dr` equals `conditional` on every panel compound, to 0.003 on the MLP
     and 0.02 on the DiT.
   - So the anchors do not separate the backbones or the arms either. The other
     14 compounds spread their wells over several dose arms, and their best-arm
     cosines (0.46–0.98) are bounded by the oracle.
   - The vehicle side: ‖μ̂_gen(0)‖ is 0.87–0.88× its DMSO sampling scale on the
     MLP and 1.24–1.33× on the DiT (`--pool all`). No `--gen_anchors` run was
     made, so τ̂ with a generated vehicle (§3.7) is not reported. It can be
     rebuilt from the `_tau.npz` files without sampling.
3. **§3.7's out-of-sample aggregate**: the per-compound cosine pooled over
   doses, on the holdout (median over 1,698 compounds; `conditional` / `dr`).

   | backbone | DDPM | FM, λ = 0 | FM, λ = 0.1 |
   |---|---|---|---|
   | MLP-B | 0.496 / 0.496 | 0.484 / 0.483 | 0.497 / 0.496 |
   | DiT-S/10 | 0.509 / 0.511 | 0.509 / 0.511 | 0.504 / 0.504 |

   - On this aggregate the DiT is 0.013 **ahead** of the MLP on the holdout.
     "Phase 4 results", finding 3, calls it marginally behind there, from the
     per-arm responder median (0.441 against 0.451). Both gaps come from one
     seed and have no error bar, so the backbones are tied on held-out τ.
     Finding 3's conclusion (no gain worth 49× the eval compute) stands; its
     "marginally worse on the holdout" does not.
   - This aggregate is the better holdout readout: the per-arm responder median
     there rests on 1-well arms whose flags are mostly noise (gap 3).
   - cmean: +0.013 on the MLP and −0.005 on the DiT. That is finding 4's mixed
     picture again.
   - On `--pool all` it is 0.825 (MLP) against 0.790 (DiT). That pool contains
     the train wells (E9).
   - Per dose level (τ(d) averaged over compounds, the six levels with ≥ 200
     arms): 0.936–0.947 on every arm. No separation there either.

**Gaps.** None changes a recorded verdict. In order of weight:

1. **C2's mechanism is not established: the `dr` − `conditional` gap is a
   difference in the learned share between the dose halves** (step-C code
   review; re-derived here from the 30 C2 / P2 JSONs).
   - With λ < 1, an arm's error along v_k is its intercept error minus
     (1 − λ_a) · β · mix_a. So the high − low contrast holds a term
     (β/2) · (λ̂_high − λ̂_low), taking mix = 1/2. §3.8.4's "~0 for any λ"
     leaves it out by assuming one λ for both halves.
   - `evaluate` writes λ̂ by half, and the report reads only the pooled mean.
     "Step C2 results", finding 4, notes the difference by half and does not
     connect it to the DiD.
   - Mean of 3 seeds:

     | arm | λ̂ low / high, γ = 0 | λ̂ low / high, γ = 1 | DiD of the λ term | recorded DiD | remainder |
     |---|---|---|---|---|---|
     | `naive` | 0.000 / 0.000 | 0.000 / 0.000 | −0.007 | −1.930 | −1.923 |
     | `conditional` | 0.495 / 0.500 | 0.523 / 0.518 | −0.131 | −0.279 | −0.148 |
     | `dr` | 0.400 / 0.406 | 0.317 / 0.380 | +0.720 | +0.575 | −0.144 |
     | `dr_design` | 0.469 / 0.476 | 0.371 / 0.433 | +0.698 | +0.562 | −0.136 |
     | `dr_p2` | 0.478 / 0.479 | 0.415 / 0.472 | +0.709 | +0.599 | −0.110 |

   - The remainder is the same −0.11 to −0.15 in all four adjusted arms. The
     whole +0.58 / +0.56 / +0.60 of the weighted arms is the λ term. Under the
     weighted risk at γ = 1 the generator learns less of the shift in the low
     dose half than in the high one; `conditional` learns the same share in
     both.
   - "Step C2 results", findings 2, 3 and 6, and "P2 results", finding 5,
     attribute the sign flip to the arm-level Hájek mean. That would show in
     the remainder, and it does not. Why the weights open a gap in λ̂ between
     the halves is not known.
   - `dr_p2` keeps that gap (0.057, against 0.063 for `dr`). So P2 repaired λ̂
     on unthinned compounds and left this untouched, which is why its DiD did
     not move.
   - The statistic is a small difference of large errors: each half's mean
     error is −4.8 to −8.7 along v_k at γ = 1, and the contrast is under 1.
   - Unchanged: every DiD, criterion 2's FAIL, and P1's pass. P1 removes each
     group's mean error whole, so it does not depend on which term carries the
     contrast.
   - Open: the mechanism, and with it the argument that step A's arm-level
     cells avoid the problem. Step A's own result will say.
2. **C2's γ = 0 control is not near zero, and nothing judges it.**
   - Raw scored high − low contrast, mean of 3 seeds:

     | arm | step C, γ = 0 / γ = 1 | C2, γ = 0 / γ = 1 |
     |---|---|---|
     | `naive` | +0.21 / −6.65 | +0.50 / −1.43 |
     | `conditional` | −0.02 / −0.01 | **+0.32 / +0.04** |
     | `dr` | −0.01 / −0.03 | +0.30 / +0.88 |
     | `dr_design` | −0.01 / −0.06 | +0.33 / +0.89 |
     | `dr_p2` | — | +0.26 / +0.86 |
     | `p1_cond_counts` | −0.01 / −0.07 | −0.04 / +0.01 |
     | `p1_naive_counts` | −0.17 / −0.14 | −0.06 / −0.07 |

   - **`conditional`'s −0.279 on C2 is +0.04 at γ = 1 minus +0.32 at γ = 0.** At
     γ = 1 its contrast is near zero. So "Step C2 results", finding 1
     ("`conditional` is now biased"), rests on the control's offset being common
     to both instances.
   - Criterion 1 is named "biased at γ = 1, not at γ = 0", but `step_c_report`
     tests only |DiD| / sd > 3. The γ = 0 contrasts are recorded and never
     judged. By the same 3-sd rule, C2's are 9 to 50 seed sd from zero in every
     untargeted arm.
   - The seed sd holds training noise only: the three seeds share the two
     seed-42 thinning draws. The reviewer redrew the thinning 400 times from
     the stored π and scored the memoriser. Its DiD has a draw sd of about
     0.30, and the realised γ = 0 draw sits +1.8 sd high. (Not re-derived
     here.)
   - A second draw with the same holdout cannot be built from the CLI, because
     the thinning RNG is tied to the split seed.
   - Each scoring JSON carries an arm-level SE of its contrast
     (`scored_high_minus_low_se`, 0.10–0.16 on C2). The report loads it and
     does not use it.
   - The analyses behind the findings are not in the repo. One-off numpy
     produced the memoriser's DiD, the tables by kept train wells, the λ slope,
     the MSE orthogonal to v and every compound-clustered SE.
   - Never computed: a clustered SE for step C, for the holdout pool, and for
     the paired `p1_cond_counts` − `conditional` difference. So "P1 beats its
     baseline" is established for the `naive` family (−1.93 → −0.001), and for
     the `cond` family only against seed noise.
   - `src/eval/contrast_stats.py`, step A's jackknife, can supply the errors.
3. **The responder rule passes noise often** (Phase 4 code review). The code
   follows E11, so this is a question about the rule.
   - "‖τ̂‖ > 1.5 × the median null norm" is passed by a no-effect arm about 4%
     of the time at 3 wells and 11% at 1 well (the reviewer's 20,000-draw
     re-simulation of the floor). The oracle shows the second directly: at
     n = 1 the null p90 is 35.96, above the threshold 1.5 × 23.25 = 34.9.
   - A compound is a responder if any dose arm is. With six arms a no-effect
     compound is flagged up to 22% of the time, and 37.2% of compounds are
     flagged. So a sizeable part of the 651 responders, and of the 641 scored
     compounds, may have no real effect.
   - Steps C and C2 are not affected: their bias is read along the injected
     direction, whatever the compound does.
   - Step A is affected, by dilution: its readout needs a real line effect in
     the scored compounds. Its pooled arms have ~15 wells, where the null is
     tighter (p90 / median 1.23 at n = 8, against 1.33 at n = 3).
   - On the holdout, 1,503 of the 1,546 responder arms have one well, so about
     two thirds of those flags are expected from noise. The holdout `cos resp`
     column of "Phase 4 results" (E9's cmean headline) is read on that set.
     The comparison is paired across arms and still valid; the per-compound
     aggregate above is the cleaner one.
4. **The quality block of 36 scoring runs is invalid and was never
   recomputed**: PC Fréchet and MMD for the 12 Phase 4 arms and the 24 step-C
   arms, all scored on `zabih` before the BLAS fix ("Step C2 scoring").
   - E5 asks for both Fréchet variants and MMD, so it is not met for these
     arms.
   - Phase 4's values are visibly broken (PC Fréchet ~1e141). Step C's look
     plausible (PC Fréchet 32, MMD 0.11, against 0.001 on C2). A pre-fix JSON
     can be told only by its missing `eigh_fallbacks` key.
   - The pooled "treated" numbers have a second problem, in every run including
     C2's (Phase 4 code review). The generated reservoir is capped per dose
     group over rows × 16 and the real side over rows, so the two sides are
     different dose mixtures. On `mlp-B_conditional_ddpm_s0` the 20 µM group is
     13.0% of the generated pool and 7.3% of the real one.
   - So the level of any pooled-treated metric, the marginal Fréchet included,
     is not interpretable. Differences between arms on one pool are. Per-group
     and DMSO numbers are not touched.
   - E9 asked for "no loss of sample spread", and that was never read off a
     valid metric. The valid ones agree with the decision. Marginal Fréchet on
     the holdout DMSO wells: 23.8 → 24.9 with cmean on the MLP, 23.8 → 27.0 on
     the DiT.
   - Every `_gen.npz` holds its reservoirs, so the fix is a CPU recompute on
     `bindel`, not the GPU rescoring "Step C2 scoring" costed. It should match
     the two mixtures. No script does it yet.
5. **Criterion 4's correlations are constants in `step_c_report.py`**
   (`corr_counts_design`: 0.73 / 0.88; `measured_before_training`: `True`),
   written into every verdict file. The criterion's `pass` checks only the
   weight-file hashes, so it cannot fail on what it is named for.
   - The constants are over all train rows. On kept scored rows, both
     data-layer jobs print +0.273 (γ = 0) and +0.840 (γ = 1). Recomputed here
     from the weight files, with the same result. The note under "Phase 5
     progress" is corrected.
   - The criterion still holds: at γ = 1 the weights track the design weights
     (+0.84, means 2.03 against 2.00), and at γ = 0 the design weights are
     nearly flat.
6. **Step C and C2's control is not size-matched.** The realised kept fraction
   of scored train wells is 0.468 at γ = 0 and 0.498 at γ = 1 (5,397 and 5,090
   wells thinned). Step A found the same effect and fixed it for `cell_id`
   (`STEP_A.md` D9). No run checks whether it moves the readout.
7. **The `/ceil` column rests on a different arm set than `cos resp`**
   (Phase 4 code review).
   - On `--pool all`, `cos resp` is over 1,795 arms and `/ceil` over the 1,707
     with a positive split-half. On those the cosine is 0.804 and the ceiling
     0.802. E11's 0.788 is a third basis, the lift of the 1,795-arm median.
   - Spearman–Brown assumes equal halves, and a 3-well arm splits 1 against 2.
     So the ceiling is about 1.6% low for 85% of arms.
   - Net: `/ceil` 1.01 → about 0.99–1.00 on `--pool all`. "At the ceiling"
     stands. The holdout's 0.96 is over the 43 responder arms that have a
     split-half there, not the 1,546.
8. **P2 had no entry in §4.** It was tracked only in `STEP_A.md` §4 (B1–B5).
   Added to Phase 5; B1–B5 are checked there.
9. **Smaller items.**
   - `tier_meta.json`'s `bias` field still holds its Phase 1 placeholder.
     `--plan` writes nothing, no log holds the `mcf7_24h` table of "Phase 5
     code", and `step_c_report --plan_contrast` defaults to the constant −8.32.
   - `phase4_gpu.sub` last ran before the reference-scale rewrite, the BLAS
     guard and the C2 / P1 / step-A changes to `evaluate.py`. Its sampler
     checks have not been rerun on the current tree (~2 min on `zabih`).
   - The P2 and step-A code, `STEP_A.md` and this file's changes are not
     committed.

**Code review: follow-ups.** All open; none affects a recorded number. Line
numbers are the reviewers'.

| # | Where | Finding | Asked |
|---|---|---|---|
| R1 | `processes/ddpm.py:37-41`, `generation.py:344` | `--sampler ddpm` returns before `timestep_spacing="trailing"` is set, so a zero-SNR arm never visits t = 999; its ancestral noise is also unseeded. Unused: every run logged `sampler=ddim` | Fix, or refuse the option |
| R2 | `generation.py:158-206`, `evaluate.py:1003-1005` | `check_arm_against_data` and `_load_truth` skip an identity field that is absent from `arch.json` or the oracle. Every artifact on disk carries them all | Refuse a missing field |
| R3 | `evaluate.py:633-648` | The oracle's file name omits `responder_mult`, `n_floor_draws`, `min_dose_n` and `seed`, so `--source real --responder_mult 2` overwrites the oracle Phase 5 reads. A generated run's name omits `--sampler` | Tag them, or refuse to overwrite a file with other settings |
| R4 | `evaluate.py:1183`, `:1262`, `:1312` | The `--truth` and overwrite guards run after sampling: a mistyped `--truth` on a DiT arm fails after 5.9 h | Check before sampling |
| R5 | `evaluate.py:256-295`, `:788-795` | Gap 7: the ceiling's arm set and the unequal halves | One arm set for `cos resp`, the ceiling and the ratio; the unequal-split form |
| R6 | `generation.py:348-369`, `evaluate.py:944-961` | Gap 4: pooled-treated quality compares two dose mixtures | Match the mixtures, in the recompute too |
| R7 | `tests/check_phase4.py` | The responder flag is checked for presence only. `FLOOR_RTOL = 0.25` cannot tell a floor at the wrong arm size (n = 2 and n = 4 both pass against n = 3). E4 / E12's thresholds are never asserted. No job runs `--arm` on the 12 production JSONs | Tighten |
| R8 | `tests/smoke_phase4_torch.py` | Docstring item 8 has no code. Nothing asserts that sampling starts at t = 999 with trailing spacing. Six identity fields are doctored, not "each in turn" as "Phase 4 implementation" says. `:145` is a tautology | Add the missing checks |
| R9 | `step_c_report.py:402-411`, `:474-485` | Gaps 2 and 5: criterion 1's γ = 0 half is not judged; criterion 4's constants; `scored_se` is unused | Judge γ = 0; compute the correlations; report a clustered SE and λ̂ by half |
| R10 | `step_c_report.py:229-266` | A cell is keyed on (arm, γ, training seed) only. Not checked: the tier's `keep_frac` / `pmin` / seed / scored set, the model's arch / size / epochs, the scoring `sampler` / `gen_anchors`. Criteria 1–3 do not require all seeds. Both verdicts on disk are clean (every cell's fields read) | Pin them at admission |
| R11 | `tests/smoke_phase5_torch.py:236-242` | "Weights vary with `syn_c`" compares marginal means at 1e-6. The lever is dose half × `syn_c`, so it passes at γ = 0 too (0.060). The real signal is within a half: 3.49 against 1.39 at γ = 1 | Test within the half |
| R12 | `tests/check_phase5.py:302-317` | The kept-fraction and dp checks pass with the two instances swapped. dp is checked against the plan only, never against the realised mix (right by hand: kept 0.29 / 0.70 / 0.29 / 0.72 by cell at γ = 1) | Compare the realised mix |
| R13 | `scripts/phase5_gpu.sub:191-205` | The step-C refusal omits `--syn_effect`, so `check_arm_against_data` fires first. `TRUTH_GUARD`'s syn entries are exercised only in C2 mode | Add the case |
| R14 | `scripts/phase5_cpu.sub:44`, `:61`; `phase5_arms.sh:72-83` | `responders.json` is rewritten on every run. `ls -d …g${G}*_s42 \| head -1` would also match a γ = 0.5 instance. A failed training job leaves its scoring job pending for ever | `--print_out_dir`; `--kill-on-invalid-dep` |

Review findings not acted on:
- The floor has 1–2% Monte-Carlo error at 200 draws (n = 3: 14.48 against
  14.32 at 20,000). It moves responder flags at the margin.
- `amplitude_ratio_est_over_denoised` is a median over the arms above the floor
  only, so one generator reads 0.997 on `--pool all` and 0.686 on the holdout.
  It is in no §5 table.
- The curated panel takes the first table row per `pert_iname`. Bortezomib has
  two entries (968 and 17 wells); the row order picked the right one.
- `dist_metrics`:
  - a negative Fréchet is clamped with a stderr warning and no JSON flag;
  - the MMD bandwidth is refit per call;
  - the PC Fréchet runs whenever n > k, where E5 says large pools only;
  - E5's per-compound quality is not implemented.
- `--plan` silently skips compounds it cannot plan and prints IPW as a literal
  0. Its "+0.0000" at γ = 0 is a median; the per-compound mean is +0.0015.
- `load_syn_meta` records `table_fingerprint` and does not compare it.
- The oracle JSON is written atomically and its `_tau.npz` is not.

#### Step A results (2026-10-05)

All 28 GPU jobs (993188–993215, `zabih`) and the P1 + verdict job (993216,
`bindel`) completed with no failure or requeue. Training took 47–52 min per
run and scoring 33–34 min. Verdicts:
`runs/core5_24h/eval_artifacts/step_a_verdict.json` (pool all, the readout
declared in `STEP_A.md` §3) and `step_a_verdict_poolholdout.json` (secondary).

**Verdict on the declared readout (pool all): S1–S5 all pass.**

| arm | seeds | DiD scored ± clustered SE | seed sd | DiD unscored | MSE / `conditional` | learned line contrast, scored / unscored |
|---|---|---|---|---|---|---|
| `naive` | 1 | −0.793 ± 0.042 | n/a | −0.050 ± 0.007 | 1.058 | 0.000 / 0.000 |
| `conditional` | 2 | −0.071 ± 0.015 | 0.002 | +0.000 ± 0.003 | 1.000 | 0.121 / 0.066 |
| `dr` | 2 | −0.000 ± 0.018 | 0.025 | +0.011 ± 0.003 | 1.016 | 0.123 / 0.064 |
| `dr_p2` | 2 | −0.005 ± 0.017 | 0.018 | +0.006 ± 0.003 | 1.013 | 0.122 / 0.065 |
| `p1_cond_counts` | 2 | −0.048 ± 0.031 | 0.000 | −0.001 ± 0.001 | 0.878 | (generator unchanged) |
| `p1_cond_ones` (control) | 2 | −0.004 ± 0.026 | 0.005 | −0.001 ± 0.001 | 0.746 | |
| `p1_naive_counts` | 1 | −0.051 ± 0.031 | n/a | −0.001 ± 0.001 | 0.886 | |
| `p1_naive_ones` (control) | 1 | −0.951 ± 0.052 | n/a | −0.001 ± 0.001 | 0.782 | |

| # | Criterion | Result |
|---|---|---|
| S0 | bias in the training data ≥ 5 SE | PASS: −0.941 ± 0.052 (18.2 SE) |
| S1 | `naive` biased, > 3 SE | PASS: 18.9 SE |
| S2 | `conditional` biased, > 2 SE (testability) | PASS: 4.7 SE |
| S3 | ADIGen arm beats `conditional` by > 2 paired SE | PASS: `dr` reduction +0.070 ± 0.019 (3.7 SE); `dr_p2` +0.066 ± 0.010 (6.9 SE) |
| S4 | unscored compounds unchanged | PASS for all four arms (bound 0.198) |
| S5 | weight hashes match | PASS, 14 runs |

Findings:

1. **`naive` learns 84% of the planted bias** (−0.793 of −0.941). Step C2's
   `naive` learned 18% and step C's 65%.
2. **`conditional` removes 91% of it but keeps a residual**, −0.071 (4.7 SE),
   reproduced by both seeds (−0.069, −0.072). This is the first experiment
   where "DR against `conditional`" is testable.
3. **Both ADIGen arms remove the residual.** `dr` is at −0.000 ± 0.018 and
   `dr_p2` at −0.005 ± 0.017. There is no over-correction: step C2's `dr`
   overshot to +0.58.
4. **`dr`'s 0.000 is a mean of two seeds of opposite sign** (−0.018, +0.018).
   Each seed alone is at 0.018, about a quarter of `conditional`'s residual, so
   the per-seed reduction is ≈ 0.053, still well above the paired SE.
5. **The generator is not damaged the way it was in C2.** MSE is 1.6% (`dr`)
   and 1.3% (`dr_p2`) above `conditional`, against +29% for `dr` in C2, and
   the learned line contrast is unchanged on scored and unscored compounds.
   The MSE ratio understates the cost, though: against the truth the weights
   add 17% (`dr`) and 11% (`dr_p2`) to the model's error on the thinned
   compounds ("MSE decomposition" below).
6. **`dr` and `dr_p2` are indistinguishable here.** P2's repair was not needed
   in step A; it does no harm.
7. **Unscored compounds move a little.** `naive` −0.050 ± 0.007 and `dr`
   +0.011 ± 0.003 are both more than 2 SE from zero. They pass S4 on the size
   bound (a quarter of `naive`'s scored bias), not on the noise allowance. So
   the thinning of 848 compounds leaks slightly into the 902 others through
   the shared generator.
8. **The AIPW baseline (P1) fixes `naive` but does not beat `conditional`.**
   `p1_naive_counts` cuts −0.793 to −0.051. `p1_cond_counts` is at
   −0.048 ± 0.031, a reduction of +0.023 ± 0.028 over `conditional` (0.8 SE).
   Both P1 arms equal the counts-weighted memoriser of the S0 table
   (−0.046 ± 0.031): with weights that are exact per cell, the correction
   replaces the generator's arm mean by the weighted mean of the kept wells,
   whatever the generator. Its error is that estimate's sampling noise, which
   is about twice the ADIGen arms' SE.
9. **P1's MSE advantage is in-sample.** On pool all its MSE is 12–25% below
   `conditional`, because the correction uses training wells that are also in
   the oracle. On the holdout pool it is 42–48% above `conditional`, while
   `dr` and `dr_p2` are within 1%.

**The flat-weight control on `conditional` did not come out worse**
(`p1_cond_ones` −0.004 against `conditional` −0.071), which §3 said it must.
On `naive` the control behaves as required (−0.951, the memoriser's −0.941).
My reading, from the algebra and not checked by a separate experiment:

- Write the generator's arm mean in line $c$ as the kept wells' mean plus a
  misfit $\delta_c$. `conditional`'s error is $\sum_c P(c)\,\delta_c$. The
  flat-weight correction leaves $\sum_c (P(c) - p_\text{kept}(c))\,\delta_c$,
  and exact weights leave 0.
- The flat correction therefore removes any misfit that is common to the
  lines. A generator that ignores the line has $\delta_c$ equal to minus the
  line effect, which is the fully line-specific case, and the control
  reproduces the memoriser. That is what the control was designed on.
- For `conditional` the control lands on zero, so its residual is mostly a
  line-common misfit that follows the kept mix: the part of the response the
  model shares across lines is pulled toward the lines that were kept.
- Consequence: the control's rule ("must be worse") is wrong for a generator
  that sees C and should be dropped for it. And an unweighted residual
  correction also removes `conditional`'s bias here, so step A shows that the
  weighted risk repairs the generator, not that the weights are the only way
  to repair the estimate.

**Secondary pool (holdout wells as truth): same direction, not significant.**

| arm | DiD scored ± SE | reduction over `conditional` |
|---|---|---|
| `naive` | −0.796 ± 0.043 | |
| `conditional` | −0.064 ± 0.016 (4.1 SE) | |
| `dr` | +0.017 ± 0.018 | +0.047 ± 0.032 (1.5 SE) |
| `dr_p2` | +0.013 ± 0.018 | +0.051 ± 0.032 (1.6 SE) |

- The report prints S3 as FAIL on this pool. It is not the declared readout:
  its truth is each arm's holdout wells only, and the paired SE is 1.7–3×
  larger. The point estimates agree with pool all.

What step A does and does not establish:

- It establishes, on a real confounder, the ordering `naive` ≫ `conditional`
  > ADIGen ≈ 0, with the ADIGen advantage at 3.7 SE (`dr`) on the declared
  readout and no cost in MSE.
- The advantage is small in absolute terms: `conditional` already removes 91%
  of the bias and ADIGen the remaining 9%.
- It does not establish that ADIGen beats the AIPW baseline on bias. The two
  are within noise of each other (0.000 ± 0.018 against −0.048 ± 0.031; the
  paired difference was not computed). ADIGen's advantage over P1 is variance
  out of sample (finding 9) and that it yields a generator.
- One tier seed (42) and one γ. The clustered SE covers the compounds, not a
  second thinning draw.

#### MSE decomposition: where the squared error comes from (2026-10-05)

Question (user): list every term that contributes to the MSE and show how
small the confounding-bias term is.

- **Code:** `src/eval/mse_decomposition.py` (new, numpy only),
  `scripts/mse_decomposition_cpu.sub`; jobs 157 and 185 on `bindel`, 3 min,
  4.4 GB.
- **Output:** `runs/core5_24h/eval_artifacts/mse_decomposition.json` (every
  number below, at γ = 0 and γ = 1, for scored, unscored and all arms).
- **Units:** squared error per arm, summed over the 978 genes. Divide by 978
  for the per-gene MSE that `evaluate` prints.

**Answer.**

- The bias the step-A readout measures is 0.004% of `conditional`'s squared
  error on the thinned arms and 0.3% of `naive`'s. MSE cannot see it.
- The reported MSE is roughly the oracle's own noise. The model's error
  against the truth is about a quarter of that size and is mostly hidden,
  because the generator copies noise from training wells the oracle shares.
- Against the truth, the weighted arms pay a variance cost on the thinned arms
  that the MSE ratio understated: +17% for `dr`, +11% for `dr_p2`.

**The terms.** For one arm, let τ_gen be the generator's estimate, τ_o the
all-wells oracle and τ\* the true effect, with e = τ_gen − τ\* and
ε = τ_o − τ\*. Then ‖τ_gen − τ_o‖² = ‖ε‖² − 2⟨e, ε⟩ + ‖e‖², and ‖e‖² splits
further:

| Row | What it is |
|---|---|
| total | ‖τ_gen − τ_o‖², the generator against the all-wells oracle; the reported MSE × 978 |
| oracle noise | ‖ε‖², the oracle's own sampling error (it averages ~15 wells); the same for every arm of the experiment |
| overlap credit | −2⟨e, ε⟩; negative because the generator reproduces noise from training wells that are also in the oracle, which makes it look closer to the oracle than it is to the truth |
| **model error** | ‖e‖², the generator against the truth |
| – generation noise | error from estimating each well's mean with 16 generated samples |
| – training variance | what changes when the same model is retrained with another seed |
| – seed-shared error | the rest: what both seeds have in common. It holds the model's misfit **and** the noise inherited from the particular training wells, which a new seed does not redraw |
| bias², as the DiD reads it | the shift of the estimate from γ = 0 to γ = 1 along the line-contrast direction, averaged over the arms of each dose half with its sign, then squared. One component of the seed-shared error |

**How the truth-dependent rows are measured.** Two references are built per
arm from disjoint wells: τ_h from its holdout wells and τ_tr from its unthinned
train wells. Each is standardised to the arm's line mix over all wells and
uses its own vehicle mean, so both are unbiased for τ\* with independent
noise. No generator trains on a holdout well. Then:

- ⟨τ_h, τ_tr⟩ estimates ‖τ\*‖² (the noise terms average to zero);
- oracle noise = ‖τ_o‖² − ⟨τ_h, τ_tr⟩;
- model error = ‖τ_gen‖² − 2⟨τ_gen, τ_h⟩ + ⟨τ_h, τ_tr⟩;
- overlap credit = total − oracle noise − model error;
- generation noise = Σ over the arm's rows of row_var/(m − 1)/n², m = 16,
  from the saved per-row variances;
- seed-shared error = model error − ½‖τ_gen(seed 0) − τ_gen(seed 1)‖².

No noise model is assumed. A single arm's estimate is very noisy; the means
are precise, with delete-one-compound jackknife errors. "Truth" is what the
arm's average converges to with unlimited replicate wells in this experiment.
The method needs a holdout and a train well in every one of the arm's lines:
7,466 of 10,484 arms (3,550 scored, 3,916 unscored). The all-wells reference
recomputed from rows matches the stored oracle to 3e-7.

**Scored (thinned) arms, γ = 1** (3,550 arms):

| term | `naive` | `conditional` | `dr` | `dr_p2` |
|---|---|---|---|---|
| **total** | 56.07 ± 1.05 | 52.46 ± 0.96 | 54.15 ± 0.98 | 54.04 ± 0.98 |
| oracle noise | 47.66 ± 0.86 | 47.66 | 47.66 | 47.66 |
| overlap credit | −11.11 | −12.22 | −13.38 | −12.46 |
| **model error** | 19.52 ± 1.05 | 17.02 ± 1.01 | 19.86 ± 1.06 | 18.84 ± 1.04 |
| – generation noise | 2.48 | 2.27 | 2.25 | 2.25 |
| – training variance | n/a (1 seed) | 0.8 to 3.0 | 1.0 to 3.2 | 1.0 to 3.2 |
| – seed-shared error | n/a | 11.7 to 14.0 | 14.4 to 16.6 | 13.4 to 15.6 |
| – – its change from γ = 0 to γ = 1 | −0.18 ± 0.46 (model error) | −0.63 ± 0.40 | +0.35 ± 0.52 | −0.17 ± 0.46 |
| **bias², as the DiD reads it** | 0.179 | 0.0019 | 0.00002 | 0.00009 |
| – the DiD itself on these arms | −0.780 | −0.076 | −0.005 | −0.018 |
| – share of the total | 0.32% | 0.004% | < 0.001% | < 0.001% |
| – share of the model error | 0.92% | 0.011% | < 0.001% | < 0.001% |
| model error minus `conditional`'s (paired) | +2.50 ± 0.40 | 0 | +2.84 ± 0.28 | +1.82 ± 0.22 |
| the same at γ = 0 | +2.09 ± 0.33 | 0 | +1.94 ± 0.18 | +1.37 ± 0.14 |

Oracle noise + overlap credit + model error = total, by construction. The two
ranges are explained under "Limits".

**Unscored arms and all arms, γ = 1:**

| term | `naive` | `conditional` | `dr` | `dr_p2` |
|---|---|---|---|---|
| *Unscored (3,916 arms)* | | | | |
| total | 32.65 | 32.11 | 32.45 | 32.19 |
| oracle noise | 34.96 ± 0.33 | 34.96 | 34.96 | 34.96 |
| overlap credit | −7.17 | −7.61 | −7.03 | −7.53 |
| model error | 4.86 ± 0.25 | 4.76 ± 0.23 | 4.51 ± 0.23 | 4.75 ± 0.23 |
| – generation noise | 1.81 | 1.76 | 1.79 | 1.76 |
| – training variance | n/a | 0 to 1.1 | 0 to 1.1 | 0 to 1.1 |
| – seed-shared error | n/a | 1.9 to 3.7 | 1.6 to 3.4 | 1.9 to 3.6 |
| bias², as the DiD reads it | 0.0005 | 0.00001 | 0.00002 | 0.00000 |
| model error minus `conditional`'s (paired) | +0.10 ± 0.10 | 0 | −0.25 ± 0.03 | −0.00 ± 0.02 |
| *All (7,466 arms)* | | | | |
| total | 43.78 | 41.79 | 42.77 | 42.58 |
| oracle noise | 41.00 ± 0.47 | 41.00 | 41.00 | 41.00 |
| overlap credit | −9.04 | −9.80 | −10.05 | −9.87 |
| model error | 11.83 ± 0.55 | 10.59 ± 0.52 | 11.81 ± 0.55 | 11.45 ± 0.54 |
| – generation noise | 2.13 | 2.00 | 2.01 | 2.00 |
| – training variance | n/a | 0.0 to 2.0 | 0.1 to 2.1 | 0.1 to 2.1 |
| – seed-shared error | n/a | 6.6 to 8.6 | 7.7 to 9.7 | 7.3 to 9.3 |
| bias², as the DiD reads it | 0.045 | 0.0005 | 0.00000 | 0.00002 |
| model error minus `conditional`'s (paired) | +1.24 ± 0.20 | 0 | +1.22 ± 0.14 | +0.86 ± 0.11 |

Findings:

1. **The bias term is negligible in the MSE.** No arm's squared error rises
   measurably from γ = 0 to γ = 1, `naive` included (all within 1.6 SE of
   zero). MSE cannot detect confounding of the size planted here; the signed
   readout of `STEP_A.md` §3 is the instrument for it.
2. **The MSE is roughly the oracle's noise.** Over all arms the oracle noise
   (41.0) is about the size of the total (41.8). The model error (10.6) is
   mostly cancelled by the overlap credit (−9.8).
3. **Most of the model error is common to both seeds and is there at γ = 0.**
   On thinned arms it is 12–17 of 17–20, and it does not change with γ. It is
   not confounding.
4. **The weights cost variance on the thinned arms.** `dr` is 2.84 ± 0.28
   above `conditional` (+17%) and `dr_p2` 1.82 ± 0.22 (+11%). Most of it is
   there at γ = 0, so it is the price of weighting, not of the correction. On
   unthinned arms there is no cost (`dr` is 0.25 ± 0.03 *below*
   `conditional`). Over all arms it is +12% and +8%.
5. **P2 appears to pay less than `dr`.** The two were not compared with their
   own paired SE.
6. **Thinned arms carry far more model error than unthinned ones** (17 against
   4.8). They are the responders, with a true effect five times larger
   (‖τ\*‖² 132 against 25), so there is more to get wrong.

**What the seed-shared row contains, and why `dr`'s is higher.** `dr` and
`conditional` share the architecture, the data and C; only the per-row loss
weights differ. Their misfit should be about equal, so `dr`'s excess must be
the other component: noise inherited from the training wells, which the
weights amplify. Evidence, on thinned arms:

| | `conditional` | `dr` | `dr_p2` |
|---|---|---|---|
| model error minus `conditional`'s, mean of γ = 0 and γ = 1 | 0 | +2.39 | +1.60 |
| sensitivity to a redraw of the kept wells (half of the shift ‖D‖² between the γ = 0 and γ = 1 instances) | 2.76 | 4.70 | 4.07 |
| – minus `conditional`'s | 0 | +1.94 | +1.31 |
| overlap with the oracle's noise, ⟨e, ε⟩ | 6.11 | 6.69 | 6.23 |

- The two instances keep different wells, so their difference shows how much
  an estimate depends on which wells it got. `dr`'s extra sensitivity accounts
  for about 80% of its excess error (1.94 of 2.39; 1.31 of 1.60 for `dr_p2`).
  Only the thinned-away quarter of the wells is redrawn, so this understates
  the full effect.
- The excess is on thinned arms only, where the weights are unequal.
- It grows with the spread of the weights: +1.94 at γ = 0, +2.84 at γ = 1.
- `dr` copies more of the training wells' noise (largest overlap).

Reading (mine, not separately tested): unequal weights inside an arm make its
fit lean on fewer wells, which `dr` and `dr_p2` share; and `dr`'s global
normalisation gives thinned arms more total weight than unthinned ones, so the
model fits their wells more closely, which would explain `dr` > `dr_p2`.

Limits:

- **Two rows are ranges.** Every run was scored with the same sampling seed
  (eval seed 0), so the generation noise is largely common to the two training
  seeds and cancels in their difference. The lower end of the training
  variance (upper end of the seed-shared error) assumes the sampling noise is
  not shared at all; the other end, that it is fully shared. On unscored arms
  the seed difference is smaller than the generation noise, so at least 38%
  of it is shared.
- **Misfit and inherited noise are not separated.** Of `conditional`'s
  11.7–14.0 on thinned arms, at least 2.8 is inherited noise (its own redraw
  sensitivity). Splitting the rest needs runs on a second split seed.
- `naive` has one seed, so only its model error and generation noise are
  separated.
- The redraw-sensitivity row mixes the redraw with the confounding shift. The
  confounding part is the bias² row, which is far smaller.

Corrections to what was said before this entry:

- The model error was first put at 3.5–5 per arm by subtracting a noise floor
  from the total. That assumed the model's error and the oracle's noise were
  independent, and they are not. It is 10.6 per arm over all arms.
- `naive`'s error is 12% above `conditional`'s over all arms, not 50–75%.
- The seed-shared row was first called "systematic error" and read as misfit.
  It also holds the noise inherited from the training wells.

#### Decision task on step A (2026-10-05)

Question (main author): the target is a debiased generator for decisions, not
the lowest MSE. Do the step-A generators choose a dose or a compound better
with the weighted risk? The plan, the effect definition and the criteria are
in `DECISION.md`; they were fixed before any decision number was read.

**Verdict: neither task is testable (D0 fails on both). The planted
confounding does not move either decision even in the raw data.** What the run
does show: the weights' extra error costs nothing in decisions (D4 passes), and
neither ADIGen arm chooses measurably better or worse than `conditional`.

- **Code:** `src/eval/decision_task.py` and `src/tests/check_decision.py` (new,
  numpy only), `scripts/decision_cpu.sub`, four constants in `spec.py`
  (`PROLIFERATION_GENES`, `DECISION_MIN_DOSES`, `DECISION_K`, `DECISION_K_CURVE`,
  `DECISION_K_CONTROL`). No training and no sampling: it reads the `row_mean`
  of the 14 step-A scoring runs.
- **Jobs** (`bindel`, 4 CPUs, 7 GB, no job preempted):
  - 4002, the smoke test (`MODE=check`): 48 checks, 0 failures, 1 min 11 s;
  - 4014, checks and analysis: exit 0, 4 min 40 s, peak 5.7 GB.
- **Output:** `runs/core5_24h/eval_artifacts/decision_task.json` (every number
  below, both truths) and `decision_task_values.npz` (per compound).

**Set-up as run.**

- Effect: the equal-weight mean over the five lines of [cell mean − that line's
  vehicle mean], for the truth and every estimator.
- Truth: holdout wells (primary); the same choices scored on all wells
  (secondary).
- 7,451 of 10,484 arms have a holdout well in every line. 632 thinned and 709
  unthinned compounds have at least 4 such doses and enter.
- Task A: the dose with the largest effect along the compound's own signature.
  Task B: the top-100 compounds (each at its best dose) on a fixed axis, minus
  the mean of 14 cell-cycle genes.
- Value: the mean true effect of what was chosen, in z units along a unit
  vector, reported as the gain over a random choice. The captured share is
  that gain as a fraction of the full-data reference's.
- Task-B axis check on the oracle, before any generator was read: the
  proteasome, HSP90 and HDAC compounds of the §3.7 panel sit at a median
  percentile of 0.96 (needed ≥ 0.8).
- The G2 share of the kept train wells moved by +0.154 (low dose half) and
  −0.134 (high half) at γ = 1, and by +0.003 / +0.001 at γ = 0.

**Primary truth (holdout wells).** Gain over a random choice, ± compound-level SE.

| policy | task A, γ = 0 | task A, γ = 1 | share, γ = 1 | task B top-100, γ = 0 | γ = 1 | share, γ = 1 |
|---|---|---|---|---|---|---|
| full-data reference | 3.911 ± 0.224 | (same) | 1 | 5.001 ± 0.215 | (same) | 1 |
| pooled real mean | 3.794 ± 0.227 | 3.751 ± 0.227 | 0.959 ± 0.018 | 4.883 ± 0.212 | 4.912 ± 0.211 | 0.982 ± 0.009 |
| stratified real mean | 3.787 ± 0.228 | 3.801 ± 0.227 | 0.972 ± 0.018 | 4.985 ± 0.209 | 4.993 ± 0.207 | 0.998 ± 0.005 |
| `naive` | 3.801 ± 0.223 | 3.971 ± 0.219 | 1.015 ± 0.032 | 4.912 ± 0.217 | 4.831 ± 0.228 | 0.966 ± 0.011 |
| `conditional` | 3.908 ± 0.218 | 3.924 ± 0.219 | 1.003 ± 0.033 | 4.937 ± 0.216 | 4.948 ± 0.221 | 0.989 ± 0.007 |
| `dr` | 3.982 ± 0.216 | 3.994 ± 0.218 | 1.021 ± 0.031 | 4.955 ± 0.218 | 4.935 ± 0.223 | 0.987 ± 0.008 |
| `dr_p2` | 3.969 ± 0.217 | 4.055 ± 0.217 | 1.037 ± 0.031 | 4.944 ± 0.215 | 4.940 ± 0.225 | 0.988 ± 0.008 |

Task A is over the 632 thinned compounds (jackknife); task B over all 1,341
candidates (bootstrap, 2,000 resamples). The SE of a single gain is dominated
by the spread between compounds, which is common to every policy; the paired
errors below are the ones to compare policies with.

| # | Criterion | Task A | Task B |
|---|---|---|---|
| D0 | pooled real mean loses value, ≥ 3 SE | **FAIL**: +0.043 ± 0.081 (0.5 SE) | **FAIL**: −0.029 ± 0.051 (−0.6 SE) |
| D1 | `naive` loses value, > 2 SE | not judged: −0.170 ± 0.092 (it gains) | not judged: +0.080 ± 0.053 |
| D2 | `conditional` loses value, > 2 SE | not judged: −0.016 ± 0.044 | not judged: −0.011 ± 0.022 |
| D3 | gain of `dr` / `dr_p2` over `conditional` | not judged: −0.004 ± 0.067 / +0.070 ± 0.074 | not judged: −0.031 ± 0.024 / −0.015 ± 0.021 |
| D4 | no decision cost of the weights at γ = 0 | PASS: +0.073 ± 0.036 / +0.061 ± 0.036 | PASS: +0.018 ± 0.018 / +0.007 ± 0.013 |
| net | `dr` / `dr_p2` minus `conditional` at γ = 1 | +0.070 ± 0.059 / +0.131 ± 0.069 | −0.013 ± 0.017 / −0.008 ± 0.019 |
| D5 | unthinned control unchanged | PASS (largest \|z\| 1.49) | PASS (largest \|z\| 1.14) |

Findings:

1. **D0 fails, so the comparison the task was built for cannot be made.** With
   the line ignored, the real kept wells choose as well at γ = 1 as at γ = 0.
   By `DECISION.md` §6 both tasks are recorded as not testable, and D1–D3 are
   descriptive.
2. **Why: the tilt is small against the dose curve.** The best dose beats a
   random one by 3.9 on task A, and every policy captures 96–104% of what the
   full data captures.
   - The pooled real mean changes its dose for 23% of the thinned compounds
     between the two instances, and the stratified mean for 25%. So the changes
     come from which wells were kept, not from the line mix.
   - The predicted direction is visible and tiny: the chosen dose moves by
     +0.08 ranks as predicted for the pooled real mean and +0.11 for `naive`,
     against −0.03 to 0.00 for the stratified mean and the three adjusted
     generators. No error was computed for these, and they cost no value.
3. **The weights' extra error costs nothing in decisions (D4).** At γ = 0,
   `dr` and `dr_p2` are not below `conditional` on either task. On task A they
   are slightly above it (+0.073 ± 0.036 and +0.061 ± 0.036). That is 2.0 and
   1.7 SE on one of several comparisons, so it is not a result. This is the
   main author's point about the MSE, and it holds here.
4. **Neither ADIGen arm chooses measurably better than `conditional` at
   γ = 1.** Every net difference is within 2 SE (largest: `dr_p2` on task A,
   1.9 SE).
   - `naive` on task A is not below `conditional` at γ = 1: +0.047 ± 0.087,
     paired, from the per-compound file.
   - `naive` on task B's top-100 has the lowest share at γ = 1 (0.966 ± 0.011
     against 0.989 ± 0.007 for `conditional`). That paired difference was not
     computed.
5. **On weak compounds the generators choose better than the real wells; on
   strong ones, as well.** (Paired differences are from the per-compound file,
   task A, γ = 1.)
   - Thinned compounds: the generators' shares are 1.00–1.04 against 0.96–0.97
     for the real means of the kept wells, and the difference is within noise
     (`conditional` minus the stratified real mean: +0.123 ± 0.129).
   - Unthinned compounds (mostly non-responders): the full-data real mean gains
     only +0.065 ± 0.067 over a random dose, and the four generators +0.33 to
     +0.36. Paired against that real mean: `conditional` +0.268 ± 0.056
     (4.8 SE), `dr` +0.281 ± 0.054, `naive` +0.277 ± 0.057. A generator pools
     information across doses and compounds, which a per-arm mean cannot.
   - This is a property of all four generators, not of the weighted risk.
6. **The secondary truth gives the same verdict and shows why it is
   secondary.** Against all wells, D0 fails on both tasks and D4 passes. The
   ranking of estimators reverses: the real means capture 0.97 on task A and
   the generators 0.90–0.92, because the real means' own wells are in that
   truth.
7. **The control behaves (D5).** On unthinned compounds no arm's value changes
   between the instances by more than 1.5 SE.

**Review** (automated `code-review`, high, before any job): ten findings, no
bug in the effect, design, jackknife or bootstrap code.

| Finding | Action |
|---|---|
| A failed D0 left D1–D3 with a PASS / FAIL | Fixed: they become "not judged" with the would-be result kept; D4 and D5 are still judged. Checked in `check_decision` |
| The job script passed `--verdict` to the analysis only | Fixed: `VERDICT=` goes to both steps; `--verdict` as an extra argument is refused |
| The secondary truth ran on the primary's compounds and doses, not "all six doses of all 1,750 compounds" as the memo said | Kept, and the memo changed: the same choices are scored on both truths, so only the truth differs |
| A `None` z-score crashed a print | Fixed |
| The SE of a captured share is unstable when the reference's gain is near zero | Fixed: no SE is given unless that gain is ≥ 3 SE above zero (the † in the report) |
| Seeds were paired across arms by list position | Fixed: the ADIGen arms must have `conditional`'s seeds, or the run stops |
| The npz name assumed `--out` ends in `.json` | Fixed |
| An ad-hoc hash for the bootstrap stream; a rank correlation through BLAS | Fixed: blake2b; an explicit sum |
| `check_decision` rebuilt the frame by copy | Fixed: `build_frame` and `read_verdict` are shared by the job and the checks |
| The loader duplicates `mse_decomposition`'s | **Not acted on.** That module's numbers are recorded in this log and it was left untouched. Open follow-up: one shared step-A frame loader |

The fixes were not reviewed again; the 33 synthetic checks and the 15 real-data
checks pass on the fixed code.

What this does and does not establish:

- It establishes that, in step A as built, these two decisions are insensitive
  to the planted confounding, that `conditional` and the ADIGen arms choose
  alike, and that the weights' MSE cost has no decision cost.
- It does not establish a decision advantage for ADIGen. The experiment cannot
  show one: the data-level gate fails, so there is nothing for `dr` to repair.
- It does not show that confounding never matters for decisions. A decision
  flips only when the tilt exceeds the gap between the best and the next-best
  option, and here it does not.
- One thinning draw, one γ, two seeds (one for `naive`), as in step A.

Options for a testable decision task (not decided; for the user and the main
author):

1. A stronger lever: a larger γ or a lower `keep_frac`. It needs new tier
   instances and 14 new runs (about 25 GPU-hours).
2. A decision that reads the direction the bias was planted on, for example
   which line group responds more to a compound (selectivity). The step-A
   readout found the bias there (−0.79 for `naive`). No new training.
3. The same tasks on compounds whose best and next-best doses are close.
   Choosing that subset after seeing this result would need its own
   pre-declared rule.

#### Policy learning on step A: implementation and review (2026-10-06)

Plan of record: `POLICY_LEARNING.md` (Algorithm 1 of the paper draft,
ADIGen-PO, with retargeting). Decisions P0 are the user's of 2026-10-06:
the signature utility as the headline, the disfavoured dose half keeps 1 of
6, halves by well, the reweighted risk as it stands.

**What was built** (all numpy except the rollout driver; nothing trained
yet when this entry was written):

| Piece | Where |
|---|---|
| Positivity at the dose half for `cell_id` (`cell_id@half` in `POSITIVITY_KEYS`; `tier_cell_key`) | `splits.py`, `build_tiered_split --positivity_key half` (dir tag `_pk-half`) |
| The weight export and the trainer's group normalisation read the tier's key; a `context` normalisation group for retargeted runs | `export_urr_weights.py`, `weight_norm.py`, `train_diffusion.py` |
| Two training halves per instance | `src/data/split_halves.py` (`<tier>_h1`, `_h2`) |
| The decision targets, shared by both sides | `src/policy/targets.py` |
| Rollouts: every (context, dose) × L, the generated vehicle per line | `src/policy/rollouts.py` (GPU) |
| Frame, logger, tilt, value, stages gate / round1 / pilot / round2 / report | `src/policy/learn.py`, `src/policy/values.py` |
| Checks: 21 synthetic, 28 on real data | `src/tests/check_policy.py` |
| Jobs: data layer, one chained launcher per instance, rollouts, CPU stages, an end-to-end smoke on the limit build | `scripts/policy_{cpu,smoke,rollouts,stage}.sub`, `scripts/policy_arms.sh` |

Measured on the limit build (103 scored compounds) before any GPU job: the
γ = 3 instance keeps 59% of the scored wells, 0.18 of the thin half against
1.00 of the favoured; weights up to 7.3, ESS/n 0.52; the γ = 0 control keeps
0.59 / 0.59 with near-flat weights. The counts logger's high-half share
tracks the design's at a correlation of 0.99.

**Review** (automated `code-review`, high): ten findings, all fixed before
the smoke; the fixes were not reviewed again.

| Finding | Fix |
|---|---|
| `oracle_h` chose its temperature on the estimated logger and was scored on the uniform one | One selector with an explicit logger |
| The (context, dose) half table was rebuilt through row presence; a target without a row would inherit its neighbour's half, a missing last one would crash | The half is the dose's rank among the compound's levels (as `dose_half` defines it), asserted against the table |
| Retargeting weights at λ = 1 could underflow to exactly 0 in float32 and stop the trainer's context normalisation | Floor of 1e-6 on the ratio; the count of floored rows is recorded |
| An unestimable error was FAIL in L1–L4 and PASS in L5–L6 | "not judged" everywhere |
| The report ignored `--prefix` | Passed through |
| The launcher did not forward the checkpoint epoch to the rollouts | `--gen_epoch EPOCHS−1` |
| The pilot stage recomputed round 1 | Reads the round-1 record; μ̂ from the two `dr` rollouts |
| A duplicated temperature selector; dead code; a check that could not fail | Removed |

**Smoke** (job 8510, `zabih`, 5 min 31 s): the whole chain on the limit
build, 26 tiny legs, rollouts, the five CPU stages and the verdict; green.
The full-build data layer (job 8523, `bindel`, 1 min 21 s): both instances,
their halves, 50 checks, 0 failures. The γ = 3 instance keeps 59.0% of the
scored wells, 0.19 of the thin half against 0.99 of the favoured; counts
weights up to 10.2, ESS/n 0.61; the halves hold 52k rows each. 6,773 of the
8,750 contexts have a holdout well at every dose: 3,337 thinned (1,320
validation, 2,017 test) and 3,436 control.

**Gate L0: FAIL on the full build** (utility A, the reference tilt at
β = 0.2). The reference puts 0.457 of its mass on the thin half (needs
≥ 1/3), but the kept-pooled policy does not lose value against it: the
difference is −0.53 ± 0.16, the wrong sign. Read with the paired numbers,
computed by hand on the same frame (values on the thinned contexts, gain over
a random dose in brackets):

| policy, β = 0.2 | value |
|---|---|
| random | 10.24 ± 0.19 |
| the logger itself (γ = 3) | 10.49 ± 0.21 (+0.25) |
| unthinned, per line (the declared reference) | 13.71 ± 0.33 (+3.47) |
| unthinned, pooled over lines | 14.64 ± 0.34 (+4.40) |
| kept, pooled over lines, γ = 3 | 14.24 ± 0.34 (+4.00) |
| kept, pooled over lines, γ = 0 | 14.19 ± 0.33 (+3.95) |
| kept, per line, γ = 3 | 12.94 ± 0.30 (+2.70) |
| kept, per line, γ = 0 | 13.23 ± 0.32 (+2.99) |
| the holdout truth tilted on itself (optimistic) | 18.78 ± 0.33 (+8.54) |

- **The confounding does not move a line-blind policy:** kept-pooled at
  γ = 0 minus γ = 3 is −0.05 ± 0.12 (paired by compound). The per-line
  memoriser does lose +0.29 ± 0.12 (2.4 SE), which is the thin half's single
  well, i.e. variance.
- **The logger's prior costs nothing:** the same policy with a uniform
  logger instead of π̂_b differs by −0.014 ± 0.011. At β = 0.2 the tilt is
  near the argmax and a dose gap of 3–4 overrides a prior of 1/6.
- **Pooling the lines beats the per-line reference by 0.9**, because the
  dose effect dominates the line effect (step A measured a learned line
  contrast of 12%), and five lines give a dose curve with five times the
  wells.
- So on LINCS the cell line is too weak a confounder, relative to the dose
  effect, for any line-based logger to bias a dose policy, even with a
  sixfold overlap gap. This is the decision task's lesson again, now with
  poor overlap added.

**Decision (user, 2026-10-06): launch the full chain anyway (`FORCE=1`),
both instances.** Recorded against the plan's rule, which says not to; the
likely outcome is L2 failing, in which case L3 and L4 are reported, not
judged.

#### Policy learning on step A: results (2026-10-06)

All 59 jobs of the two chains (8572–8632) completed with exit 0, no requeue:
26 training runs of about 21.5 min, 26 rollout jobs of about 5.5 min, the CPU
stages under 1 min each. The chain ran from 00:57 to 03:17 (2 h 20 min wall,
about 12 GPU-hours). Records: `runs/core5_24h/eval_artifacts/policy/`
(`round{1,2}_g{0,3}.json`, `pilot_*.json`, `values_*.npz`,
`policy_verdict.json`). The paired differences and the model error by dose half
below come from `src/policy/paired_followup.py` (job 11581, `bindel`), written
after the runs and not reviewed.

**Verdict: the experiment is not testable, and no arm chooses better than
another.** L0 failed before launch; L2 fails too (`conditional` loses
+0.009 ± 0.075 from γ = 0 to γ = 3), so L3 and L4 are reported, not judged.
The target, `dr` beating `conditional` under the policy-learning algorithm,
is not met: the differences are within noise and nominally the other way.

| # | Criterion | Quantity (test compounds) | Result |
|---|---|---|---|
| L0 | power gate on real wells | reference minus kept-pooled −0.53 ± 0.16 (needs ≥ +3 SE) | **FAIL** |
| L1 | `naive` loses value at γ = 3 | +0.039 ± 0.122 | not judged |
| L2 | `conditional` loses value (testability) | +0.009 ± 0.075 | not judged (would fail) |
| L3 | `dr` loses less than `conditional` | −0.043 ± 0.081 | not judged |
| L4 | retargeted beats `dr` at γ = 3 | +0.010 ± 0.060 | not judged |
| L5 | no value cost of the weights at γ = 0 | `dr` −0.031 ± 0.050; retargeted +0.003 ± 0.047 | PASS |
| L6 | control contexts unchanged | largest \|z\| 1.33 | PASS |
| L7 | λ and transfer factor | λ = 0.25 selected in all four cases; see finding 7 | FAIL as coded |

**Values** (utility A, holdout truth, 2,017 thinned test contexts; halves
averaged; the ± of a single value is the spread between compounds, common to
every arm, so compare arms with the paired table).

| policy | γ = 0 | γ = 3 | mass on the thin half, γ = 3 |
|---|---|---|---|
| random dose | 10.23 ± 0.25 | 10.23 | 0.500 |
| the logger | 10.23 | 10.43 ± 0.26 | 0.157 |
| real, unthinned, per line (reference) | 13.49 ± 0.42 | 13.49 | 0.454 |
| real, kept, per line | 13.09 ± 0.41 | 12.83 ± 0.40 | 0.322 |
| real, kept, pooled over lines | 13.85 ± 0.43 | 13.96 ± 0.43 | 0.437 |
| `naive` | 13.78 ± 0.41 | 13.74 ± 0.41 | 0.429 |
| `conditional` | 13.91 ± 0.41 | 13.90 ± 0.41 | 0.417 |
| `dr` | 13.88 ± 0.41 | 13.82 ± 0.41 | 0.445 |
| retargeted, λ = 0.25 (selected) | 13.91 ± 0.41 | 13.83 ± 0.41 | 0.426 |
| retargeted, λ = 0.5 | 13.98 | 13.88 | 0.423 |
| retargeted, λ = 0.75 | 14.00 | 13.84 | 0.425 |
| retargeted, λ = 1 | 13.69 | 13.57 | 0.430 |
| the holdout truth tilted on itself (optimistic) | 18.70 | 18.70 | 0.460 |

**Paired differences in value** (compound-clustered).

| difference | γ = 0 | γ = 3 | γ = 3, reference optimum in the thin half (913 contexts; post hoc) |
|---|---|---|---|
| `dr` − `conditional` | −0.031 ± 0.050 | −0.074 ± 0.066 | −0.156 ± 0.089 |
| retargeted (0.25) − `conditional` | +0.003 ± 0.047 | −0.064 ± 0.056 | −0.154 ± 0.086 |
| retargeted (0.25) − `dr` | +0.033 ± 0.053 | +0.010 ± 0.060 | +0.002 ± 0.092 |
| retargeted λ = 1 − λ = 0.25 | −0.224 ± 0.084 | −0.260 ± 0.089 | −0.147 ± 0.118 |
| `naive` − `conditional` | −0.131 ± 0.114 | −0.161 ± 0.116 | −0.181 ± 0.170 |
| `conditional` − real kept pooled | +0.059 ± 0.213 | −0.061 ± 0.222 | +0.035 ± 0.237 |
| `conditional` − real unthinned per line | +0.420 ± 0.152 | +0.410 ± 0.152 | +0.435 ± 0.244 |
| `conditional` − real kept per line | +0.814 ± 0.159 | +1.064 ± 0.167 | +2.103 ± 0.275 |

**Model error** (mean squared error of μ̂ against the holdout well, per dose;
its level, about 60, is mostly the holdout well's own noise, which cancels in
the differences).

| difference | thin-half doses, γ = 0 | thin, γ = 3 | favoured-half doses, γ = 0 | favoured, γ = 3 |
|---|---|---|---|---|
| `dr` − `conditional` | +0.13 ± 0.21 | **+3.85 ± 0.92** | −0.57 ± 0.48 | **+3.71 ± 0.53** |
| retargeted (0.25) − `conditional` | +0.79 ± 0.37 | +1.13 ± 0.38 | +1.28 ± 0.35 | +1.28 ± 0.33 |
| retargeted (0.25) − `dr` | +0.65 ± 0.39 | **−2.72 ± 1.00** | +1.85 ± 0.62 | **−2.43 ± 0.56** |
| retargeted λ = 1 − λ = 0.25 | +13.7 ± 1.4 | +19.4 ± 2.4 | +13.0 ± 1.3 | +12.8 ± 1.5 |
| `naive` − `conditional` | +15.2 ± 2.1 | +14.8 ± 2.3 | +12.3 ± 2.1 | +13.4 ± 1.9 |
| each arm, γ = 3 minus γ = 0: `conditional` | | +2.79 ± 0.90 | | −3.19 ± 0.94 |
| each arm, γ = 3 minus γ = 0: `dr` | | +6.51 ± 1.14 | | +1.09 ± 0.95 |

Findings:

1. **The planted logging does not move any policy's value.** Every arm's
   γ = 3 minus γ = 0 difference is within 1 SE (`naive` −0.04 ± 0.12,
   `conditional` −0.01 ± 0.08, `dr` −0.05 ± 0.08, retargeted −0.08 ± 0.08).
2. **`dr` does not beat `conditional`, and retargeting does not beat `dr`, on
   value.** All three are within 0.08 of each other on a gain over random of
   3.6, i.e. within 2%, at both γ. On the post hoc subset where the optimum
   sits in the thin half, the two weighted arms are nominally 0.15 below
   `conditional` (1.8 SE).
3. **The policies do go where the logger is thin.** At γ = 3 the tilts put
   0.42–0.45 of their mass on the thin half, where the logger has 0.157; the
   realised transfer factor (max over doses of π̂/π̂_b, averaged over
   contexts) is about 10, against 6 at γ = 0, with a maximum of 48–57. So
   the low-overlap condition of Theorems 3–4 is present; the generator's
   error there is not.
4. **The thinning costs `conditional` about 5% accuracy on the thin half and
   nothing in decisions.** Its error on thin-half doses rises by 2.8 ± 0.9
   (on a level of 58) and falls by 3.2 ± 0.9 on the favoured half. The dose
   gaps the policy reads are far larger than that.
5. **The counts weights cost `dr` accuracy at γ = 3, on both halves.** Its
   error is 3.9 ± 0.9 above `conditional`'s on the thin half and 3.7 ± 0.5 on
   the favoured half (4.2 and 6.9 SE); at γ = 0 the two are equal. `dr` also
   reads the thin half 0.40 ± 0.13 too high relative to the favoured half,
   where `conditional` (+0.09 ± 0.13) and `naive` (−0.01 ± 0.15) show no
   tilt. So here the weights add noise and a small optimistic tilt on the
   doses they up-weight, and remove no bias, because there is none to
   remove. This is the weight cost of the theory (1 + χ²) with no transfer
   gain to pay for it.
6. **Retargeting at λ = 0.25 recovers most of that cost.** Against `dr` at
   γ = 3 its error is 2.7 ± 1.0 lower on the thin half and 2.4 ± 0.6 lower on
   the favoured half; as a difference from γ = 0, −3.4 ± 0.9 and −4.3 ± 0.9
   (3.8 and 5.0 SE). It stays 1.1–1.3 above `conditional`. This is the one
   place the paper's mechanism shows: the mixture law is a cheaper training
   target than the balanced interventional law. It does not reach the level
   of a decision.
7. **λ = 1 hurts, and validation picks the smallest λ every time.** At
   β = 0.05 the pilot policy is almost a point mass, so at λ = 1 about half
   of the treated rows sit at the weight floor (ESS/n 0.15, largest weight
   48): the error rises by 13–19 and the value falls by 0.26 ± 0.09. L7 as
   coded compared the policy's transfer factor against the logger, which
   retargeting cannot change (10.4 against 10.1); the theorem's quantity is
   the ratio against the training law π_λ, bounded by 1/λ. The criterion was
   mis-specified and its FAIL carries no information.
8. **The temperature sits at the grid's lower edge** (0.05 or 0.1 for every
   generator), and the validation value is flat from 0.05 to 0.2. The
   policies are close to an argmax, so the KL regulariser plays no part.
9. **The line input matters for accuracy, not for the decision.** `naive`'s
   error is 13–15 above `conditional`'s at both γ, and its value is 0.13–0.16
   lower (1.2–1.4 SE).
10. **A generator chooses as well as the pooled real wells and better than the
    per-line real wells.** `conditional` minus the unthinned per-line mean is
    +0.41 ± 0.15 on the test contexts and +0.70 ± 0.07 on the control
    contexts; against the kept per-line mean, +1.06 ± 0.17. Against the kept
    wells pooled over lines it is −0.06 ± 0.22. The gain is the pooling over
    lines, which the dose effect allows.
11. **The generators are far less optimistic about their own policy than the
    raw means.** On utility B, whose axis shares no noise with the truth, the
    self-evaluation error (model value minus true value) is +0.06 to +0.13
    for `conditional`, `dr` and the retargeted arm, +0.19 for `naive`, +0.31
    for the kept pooled mean and +0.51 for the unthinned per-line mean. On
    utility A it is about −3 for every generator, which mixes the generator's
    shrinkage with the positive offset the holdout-built axis gives the
    truth (`DECISION.md` §4), so it is not read.
12. **Utility B gives the same picture** (descriptive): every generator
    captures 0.98–1.00 of the reference's gain at both γ, `naive` 0.93–0.95,
    λ = 1 about 0.94.

What this does and does not establish:

- It establishes that on the step-A data, with a sixfold overlap gap planted
  on the cell line, policy learning over doses is insensitive to the logging
  and to the choice among `conditional`, `dr` and the retargeted generator.
- It establishes two things about the generators themselves, at 4–7 SE: the
  balanced-law weights cost accuracy when overlap is poor, and the mixture
  law at a small λ recovers most of that cost.
- It does not establish a decision advantage for ADIGen or for retargeting.
  The experiment cannot show one: the cell line barely changes which dose is
  best, so a logger that depends on the line is nearly ignorable for this
  decision, whatever its overlap.
- One thinning draw, one training seed per half, one γ > 0. The model-error
  differences are 4–7 SE on one draw; the value differences are null.

Options (not decided; for the user and the main author):

1. **A semi-synthetic effect modifier on LINCS**, as step C2 did for `syn_c`:
   inject a line-group × dose interaction of known size into the utility, so
   that the best dose differs by line group and the logger hides it. This is
   the paper's own design (logs confounded through a factor that changes the
   reward) and needs the PL tier's machinery unchanged plus one injection;
   about 12 GPU-hours.
2. **Report LINCS as the robustness result it is**: the weights' cost and the
   retargeting recovery in model error (findings 5–6), with the value table
   as the null, next to the simulation and video experiments that carry the
   regret claim.
3. **A different action on LINCS**, the compound instead of the dose, with a
   logger that depends on the line. The line does change which compounds
   work (the step-A readout found the bias there), but it needs a
   parametric policy and a new rollout layout.

#### Policy learning with the semi-synthetic effect modifier: results (2026-10-06)

Plan: `POLICY_LEARNING.md` §14 (decided by the user after the null of the
previous entry; design, β rule, gate and criteria fixed before any number was
read). Build: `data/core5_24h_inj-m2`, β = 2 σ_w = 9.286 (the 1.5 σ_w build
failed gate (c); ladder table in `POLICY_LEARNING.md` §14). Runs:
`runs/core5_24h_inj-m2/pl_*` (γ = 3: the full chain; γ = 0: round 1 only);
records under `runs/core5_24h_inj-m2/eval_artifacts/policy/`. Jobs
12052–12103, 42 jobs, all exit 0, 11:06–12:42 (1 h 36 min wall, 8.3
GPU-hours). The paired differences below are from
`src/policy/paired_followup.py` (job 12920), written after the runs and not
reviewed.

**Verdict: the target is met. With the modifier, the task is testable, and
the ADIGen generator's policy beats the conditional generator's by
0.68 ± 0.10 (7 SE), while the two are identical when overlap is fine.
Retargeting does not add to it here (L4 fails).**

| # | Criterion | Quantity (2,017 thinned test contexts) | Result |
|---|---|---|---|
| L0′ | power gate | (a) 0.500; (b) +1.50 ± 0.17; (c) +0.96 ± 0.18 | **PASS** |
| L1 | `naive` loses value at γ = 3 | +0.693 ± 0.168 (4.1 SE) | **PASS** |
| L2 | `conditional` loses value (testability) | +0.668 ± 0.101 (6.6 SE) | **PASS** |
| L3 | `dr` loses less than `conditional` | **+0.678 ± 0.098 (6.9 SE)** | **PASS** |
| L4 | retargeted beats `dr` at the validation λ (0.5) | −0.607 ± 0.088 (−6.9 SE) | **FAIL** |
| L5 | no value cost of the weights at γ = 0 | `dr` +0.004 ± 0.051; retargeted not run at γ = 0 | PASS for `dr`; not judged for `rt` |
| L6 | control contexts unchanged | `conditional` +0.126 ± 0.035 (3.5 SE), `dr` −0.097 ± 0.039 (2.5 SE) | FAIL (see finding 7) |
| L7 | λ and transfer factor | λ = 0.5 both halves; TF against the training law 6.7–7.7 < 11 | PASS |

**Values** (utility A, holdout truth; halves averaged).

| policy | γ = 0, test | γ = 3, test | planted share, thin half, γ = 0 / γ = 3 | leak to the favoured half, γ = 3 |
|---|---|---|---|---|
| random dose | 10.48 | 10.48 | | |
| real, kept, pooled over lines | 15.42 | 14.88 | | |
| real, kept, per line | 16.14 | 15.97 | | |
| real, unthinned, per line (reference) | 16.47 | 16.47 | | |
| `naive` | 14.45 | 13.76 | 0.25 / 0.07 | 0.04 |
| `conditional` | 15.00 | 14.34 | 0.39 / **0.13** | 0.06 |
| `dr` | 15.01 | **15.02** | 0.39 / **0.33** | 0.14 |
| retargeted, λ = 0.25 | — | 14.48 | — / 0.15–0.16 | 0.07 |
| retargeted, λ = 0.5 (selected) | — | 14.41 | — / 0.14–0.15 | 0.07 |
| retargeted, λ = 1 | — | 13.85 | — / 0.12–0.13 | 0.08 |
| the holdout truth tilted on itself (optimistic) | 20.92 | 20.92 | | |

The planted share is the mean over the thin-half cells of
$m\,(\hat\mu_\text{inj} - \hat\mu_\text{uninj}) / \beta$ against the
uninjected run of the same name (same rows, seeds, noise seeds); on the
control compounds, where every cell has three wells, every generator learns
0.31–0.34 at both γ.

**Paired differences in value** (compound-clustered).

| difference | γ = 0 | γ = 3 | γ = 3, reference optimum in the thin half (1,009) | in the favoured half (1,008) | control (3,436), γ = 3 |
|---|---|---|---|---|---|
| `dr` − `conditional` | +0.004 ± 0.051 | **+0.682 ± 0.084** | +0.416 ± 0.136 | +0.773 ± 0.157 | +0.223 ± 0.044 |
| retargeted (0.25) − `conditional` | — | +0.147 ± 0.065 | +0.233 ± 0.119 | +0.148 ± 0.117 | +0.278 ± 0.043 |
| retargeted (0.25) − `dr` | — | −0.535 ± 0.078 | −0.183 ± 0.132 | −0.625 ± 0.126 | +0.056 ± 0.036 |
| retargeted λ = 1 − λ = 0.25 | — | −0.636 ± 0.123 | | | −1.485 ± 0.070 |
| `naive` − `conditional` | −0.553 ± 0.122 | −0.578 ± 0.144 | −1.123 ± 0.239 | +0.067 ± 0.201 | −1.185 ± 0.083 |
| `conditional` − real kept per line | −1.133 ± 0.199 | −1.637 ± 0.203 | | | |

**Model error** (squared error of μ̂ against the holdout well, per dose).

| difference | thin-half doses, γ = 0 | thin, γ = 3 | favoured-half doses, γ = 0 | favoured, γ = 3 |
|---|---|---|---|---|
| `dr` − `conditional` | −0.10 ± 0.31 | **−16.6 ± 1.4** | +0.42 ± 0.23 | **+6.8 ± 0.7** |
| retargeted (0.25) − `dr` | — | +14.3 ± 1.2 | — | −5.5 ± 0.7 |
| each arm, γ = 3 minus γ = 0: `conditional` | | +32.9 ± 1.6 | | −6.5 ± 1.1 |
| each arm, γ = 3 minus γ = 0: `dr` | | +16.4 ± 1.6 | | −0.1 ± 0.9 |

Findings:

1. **The modifier makes the decision depend on what the logger hides.** The
   line-blind real policy loses 1.50 ± 0.17 against the per-line reference
   (gate b), `naive` loses 0.69 and `conditional` 0.67 from γ = 0 to γ = 3.
   The previous entry's null is gone.
2. **`dr` keeps its value under the thinning and `conditional` does not.**
   `dr` moves by +0.01 ± 0.10 from γ = 0 to γ = 3, `conditional` by −0.67 ±
   0.10; at γ = 3 the paired gap is 0.68 ± 0.08, 15% of the gain over a
   random dose. At γ = 0 the two generators are the same in every number.
3. **The mechanism is the one the theory names.** On the thin half,
   `conditional` learns 0.13 of the planted effect and `dr` 0.33, the share
   both learn on the control compounds (0.31–0.34), where every cell has
   three wells. `dr`'s thin-half model error is 16.6 ± 1.4 below
   `conditional`'s (12 SE). The counts weights restore the thin cells to
   the learning rate they have with overlap.
4. **The gain is largest where the planted effect is a hidden harm.** On
   contexts whose reference optimum is in the favoured half, `dr` −
   `conditional` is +0.77 ± 0.16: `conditional` never learns that the thin
   dose is bad (its μ̂ there stays at the pooled level) and picks it. On
   contexts with the optimum in the thin half the gap is +0.42 ± 0.14.
5. **The weights' cost is a leak, not noise, this time.** `dr` spreads 0.14
   of the planted effect onto the favoured half, where nothing was planted
   (`conditional` 0.06), and its favoured-half error is 6.8 ± 0.7 above
   `conditional`'s. The thin-half gain outweighs it: the error at the policy
   is −1.8 ± 2.1.
6. **Retargeting loses what the balanced weights gained (L4).** At every λ
   the retargeted generator's planted share is 0.12–0.16, barely above
   `conditional`'s, and its value sits between the two (λ = 0.25: +0.15 ±
   0.07 over `conditional`, −0.54 ± 0.08 under `dr`); validation picks
   λ = 0.5, and λ = 1 is worse than `conditional`. Reading (mine, from the
   weights): the retargeting law $\lambda\tilde\pi + (1-\lambda)\pi_b$ has no
   component that balances the design. Its weight on a thin well is
   $\lambda\,\tilde\pi/\pi_b + (1-\lambda)$: large only where the pilot policy
   already goes, and about $1-\lambda$ where the pilot avoids a dose, which
   is exactly where the planted effect is a harm. `dr`'s counts weight is 6
   on every thin well, whatever its sign. So the retargeted generator
   re-hides half of the modifier. As λ grows the effective sample also
   collapses (ESS/n 0.76 → 0.17, weights up to 48).
7. **Spillover onto the control compounds (L6).** The unthinned compounds'
   values move between the instances although their rows do not: `dr`
   gains 0.10 ± 0.04 at γ = 3 and `conditional` loses 0.13 ± 0.04. Shared
   parameters carry the thinned compounds' fit to the others; the sign
   differs by arm. Small against the 0.68 effect, but it is a departure from
   the criterion, and the control is not a clean null here.
8. **The generators capture a third of the planted effect; the real wells
   capture it whole, noisily.** The per-line kept memoriser (15.97) beats
   every generator (`dr` 15.02) by about 1, and the unthinned reference
   (16.47) by 1.5. With β ≈ 2 σ_w a single well is informative, and the
   generators shrink it.
9. **Temperatures:** `dr` chose β = 0.2 on both halves (0.05–0.1 before the
   modifier), `conditional` 0.05–0.1; the retargeted runs run at the pilot's
   0.2.
10. **The transfer-factor criterion is trivial as built** (L7): the ratio
    against the training law is bounded by $1/\lambda$ by construction, and
    at λ = 1 it is astronomically large where the pilot put no mass. It
    carries no information beyond λ.

What this does and does not establish:

- With a line-group × dose-half effect modifier of 2 σ_w planted where the
  logger is thin, the ADIGen (counts-weighted) generator's dose policy is
  worth 0.68 ± 0.10 more than the conditional generator's, with no cost
  when overlap is fine. That is the user's target, on one thinning draw and
  one seed per half.
- The effect is semi-synthetic and placed, by design, on exactly the half the
  logger hides, with compound-specific signs; the write-up must say so.
- Retargeting, as Algorithm 1 specifies it and as implemented here (the
  mixture of the pilot policy and the logger as the training law), does not
  improve on the balanced weights on this data; it sits between
  `conditional` and `dr`.
- Two further criteria did not behave as declared: the control compounds
  move (L6), and L5's retargeted half was not run.
- One thinning draw, one seed per half, one γ > 0, one β.

Options (not decided):

1. **A retargeting law that keeps the balance:** mix the pilot policy with
   the balanced design instead of the logger,
   $\nu_\lambda = \lambda\tilde\pi + (1-\lambda)\,\nu_\text{balanced}$, so
   the weights are $\lambda\tilde\pi/\pi_b + (1-\lambda)\,\alpha_\text{counts}$.
   It costs 8 runs (3.5 GPU-hours) and reuses everything.
2. A second thinning seed and a second training seed, to put errors on the
   draw as well as the compounds (about 8 GPU-hours).
3. The write-up: the planted-share table (finding 3) is the mechanism
   figure; the value table the headline; the leak (finding 5) and the
   spillover (finding 7) the caveats.
