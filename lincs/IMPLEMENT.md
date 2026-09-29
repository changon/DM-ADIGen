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
  requirements.txt        # exists: RxRx pins + h5py (cmapPy / tables optional)
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
      mlp.py              # NEW vector denoiser
      dit1d.py            # NEW 1-d DiT
      __init__.py         # build_generator / build_generator_from_ckpt dispatch on arch
    processes/            # copy __init__.py + ddpm.py + flow_matching.py
    train/
      train_diffusion.py  # adapt: y rank, --arch {mlp,dit1d}
    eval/
      evaluate.py         # NEW gene-space ATE
      generation.py       # sample vectors, not images; no clamp
      dist_metrics.py     # MMD / Frechet on Y (no Inception, no torchvision)
    tests/
      test_lincs_shapes.py  # GPU-box smoke test; under src/ so git tracks it
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
| `export_urr_weights.py` | **Copy.** Drop `disease_condition` / `inf`: score treated rows, vehicle rows keep $w = 1$. `--mode net` is the primary source (P1); `--mode counts` is the fallback: closed-form $\nu(a)/f_\text{train}(a)$ with add-k smoothing; key actions on the `dose_level` arm, not RxRx's `round(log10_conc·1000)` (10,944 vs 10,479 keys: float variants split arms), and on (arm, C) once $C \neq \varnothing$ (P2). `--mode net`: the cross-fitted AlphaNet from `fit_urr`. Writes `dr_weights_{counts,urr}.npz` `{row_id, w}`; checks against `dr_weights_design.npz` when it exists. Import torch lazily so `counts` runs as a CPU job without it. |
| `build_nu_rows.py` | **Rewrite.** $\nu$ = the unthinned train pool minus reserve wells, written as `nu_rows.npy` for `fit_urr --nu_rows` and `export --mode counts`. `build_tiered_split` calls it. v1 has no thinning, so $\nu$ = the train rows. |
| `alpha_net.py` | **Rewrite input.** Drop `infected`. `in_dim = cov_idx + embed + log10_conc + is_control` (the `+3` becomes `+2`). Keep softplus head / `SP_SHIFT`. |
| `fit_urr.py` | **Copy URR loss** $L = \mathbb{E}[\alpha(X,A)^2] - 2\mathbb{E}[\alpha(X,A_t)]$. **Delete** the `infected==1` train restriction, the `disease_condition` array, `--cell_type {HRCE,VERO}` and the `experiment` / `cell_type` reads. Take control ids from `__control__` (RxRx compares vocab keys to the empty `control_token`, which never matches). Population mask = filters already applied at build (optionally `--population_compounds`). Cross-fit folds **stratified on the `dose_level` arm** (P5) instead of RxRx's random halves; `--nu_rows`; common support and $\nu$ keyed on the `dose_level` arm, not `round(log10_conc, 6)` (P2); ESS/tail gate; `nuisance_meta.json`. |
| `alpha_truth_check.py`, `fit_knn_dr.py` | **Not ported.** Removed from RxRx in `78d4284`; kNN AIPW retired 2026-09-29. |
| `precompute_cmean.py` | **Copy, optional** (P9). Means of normalised `y` (not VAE latents) over TRAIN rows, keyed on the `dose_level` arm (vehicles one key); `--min_n` configurable (RxRx default 8 keeps 34 of 10,479 arms; coverage logged); stores `train_idx`, which the trainer checks. Feeds the FM-only `--cmean_lambda` loss (default 0 = off). |

**Arms are tiny.** With ~2 train wells per arm, `counts` estimates each
arm's $\nu(a)/f_\text{train}(a)$ from counts of ~2, so smoothing
(`--smooth_k`, 0.5) matters once the train pool is thinned (step C / A). In
v1, $\nu$ = the train pool and the ratio is exactly 1. `net` is cross-fitted:
a row is scored by the net fit on the other fold. With random folds, 35% of
treated train rows have no well of their own arm in the other fold (their
compound is always there, measured); arm-stratified folds fix that (P5).
Arms are keyed on `dose_level` everywhere: `fit_urr` support, fold strata,
`counts` keys, and the tiered split's scored arms (RxRx rounds log-dose to
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
  `experiment`, `well`, `site`, `disease_condition`). `--invariance_env`
  choices become `plate` (~375 wells per plate, ~250 in train).
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

- `net` (primary): `fit_urr --nu_rows …` → `export_urr_weights --mode net`
  → `train_diffusion --dr_mode weighted --dr_weights_file dr_weights_urr.npz`;
- `counts` (fallback): `export_urr_weights --mode counts` (CPU) →
  `train_diffusion --dr_mode weighted --dr_weights_file dr_weights_counts.npz`.

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

Run C to completion before building A. The vehicle is RxRx's current
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
   A share one split implementation and, at a given seed, one holdout.
3. **Thinning:** for each scored compound, every train well $i$ gets
   retention $\pi_i \propto \exp(-\gamma z_i)$, calibrated so that
   $\sum_i \pi_i$ = `keep_frac` · $n$ and clipped to [`pmin`, 1] (RxRx
   `_calibrate_pi`; `pmin` default 0.05). Survivors are drawn, and redrawn
   until every positivity cell (below) keeps ≥ 1 train well. Kept wells of
   scored compounds get the design weight $1/\pi_i$, every other row 1. γ = 0
   gives constant $\pi$: the uniform (MCAR) control at the same `keep_frac`.

Outputs, as in RxRx:
- `splits.json`: thinned `train_idx`, `holdout_idx`, and a `tier` block with
  γ, `pmin`, `keep_frac`, the covariate, and the scored compounds;
- `dr_weights_design.npz`: the true weights $1/\pi$;
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
- **$\nu$ must see the unablated design.** $\nu$ = `nu_rows.npy`, passed to
  `fit_urr --nu_rows` and read by `export_urr_weights --mode counts`. If
  $\nu$ came from the thinned pool, $\alpha$ would be blind to the thinning
  by construction.
- **Design truth.** `export_urr_weights` compares its weights with
  `dr_weights_design.npz` (correlation on the design-weighted rows).
- **Arms per step** (MLP only; same split and seed; γ > 0 plus the γ = 0
  control at the same `keep_frac`):

  | arm | `--adjustment_set` | `--dr_mode` / weights | role |
  |---|---|---|---|
  | `naive` | `''` | conditional | shows the bias the thinning creates |
  | `conditional` | C | conditional | g-formula baseline |
  | `dr` | C | weighted, `dr_weights_urr.npz` (`counts` fallback) | ADIGen |
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
  4. The URR gate passes with $\alpha$ genuinely varying in $C$ (unlike v1),
     and the exported weights track `dr_weights_design.npz`.

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
  - `fit_urr --target_support common` would drop those arms from $\nu$, and
    they are exactly the arms the test is about. **Use `--target_support all`
    in step C.**
  - This is safe here: every (compound, half, `syn_c`) cell keeps ≥ 1 row,
    and the `syn_c` effect is the same additive shift for every arm, so both
    the $\alpha$ net and the generator can share it across doses.
  - `counts` (fallback) keys on (arm, `syn_c`) (P2).
  - The ESS / tail gate and the design-weight comparison remain the checks.
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
  - So **arm-level positivity survives the thinning**, and
    `--target_support common` works as in RxRx.
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

### 3.12 Phase 0 decisions (resolved and frozen; 3 and 5 amended 2026-09-29)

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
7. **Confounding program — accepted 2026-09-28.** Two steps after v1, in
   order (§3.8):
   - **step C**, semi-synthetic `syn_c` on MCF7, as a sanity check with a
     known effect direction;
   - **step A**, `cell_id` as C on the separate `core5_24h` population
     (MCF7, HT29, HA1E, A375, PC3), for real-data results.

   Both use:
   - responders-only rare compounds;
   - dose-half × C selection by probabilistic thinning in the tiered split
     (§3.8.1; P6), with known design weights;
   - $\nu$ from the unthinned design pool (`fit_urr --nu_rows`,
     `export_urr_weights`);
   - the `naive` / `conditional` / `dr` arms (+ `dr_design` reference) with
     a γ = 0 control.

   Step A starts only after step C's gate and step A's own
   effect-modification gate.

Do not change these decisions between the four v1 arms. A different compound
universe or cell line defines a separate population and result set. Step A's
`core5_24h` is such a population, and it does not replace decision 1.

### 3.13 Phase 1 decisions (resolved 2026-09-29)

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
  available if the net fails its gate.
- **P2 — Arms are keyed on `dose_level`** everywhere an arm is keyed:
  `fit_urr` common support and $\nu$, the fold strata (P5), the tiered
  split's scored arms, and the `counts` keys. Once C is non-empty, keys are
  (arm, C). This still matters under P1: RxRx's float keys give 10,944 arms
  instead of 10,479 (0.37 vs 0.3704 µM split), which shrinks common support
  in steps C / A.
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

Compute: `fit_urr` imports torch but AlphaNet is tiny, so it runs as a CPU
job (`ma` under the headroom rule). `export --mode counts` needs no torch
once the import is lazy.

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

- [x] §3.13 decisions D1, P1–P11 chosen and recorded (2026-09-29)
- [ ] `splits.py`: copied `load_splits` + stratification engine; cache key =
      population key + table fingerprint; arm strata; DMSO stratified by
      plate; realised holdout fraction logged (P3)
- [ ] `build_tiered_split.py`: reserve (`k_reserve` 0 default) / split /
      thinning layers with `_calibrate_pi`; v1 instance = no scored
      compounds, no thinning → `splits.json` + `nu_rows.npy` (P4, P6)
- [ ] `build_nu_rows.py`: \(\nu\) = unthinned train pool minus reserve
- [ ] `expr_stats.py`: train-DMSO plate centres; per-gene mean from centred
      train DMSO, std from all centred train rows (decision 5, amended);
      per-gene fraction at the 15.0 cap → `expr_meta.json`
- [ ] Centring gate on **held-out** DMSO (P11): per-plate means ≈0; plate
      share of DMSO variance 0.67 → ~0.20
- [ ] `dataset.py`: in-memory vector loader (no PIL / image VAE) + copied
      `build_cond_spec` / `cond_from_*` / `dose_probe`
- [ ] `alpha_net.py`: copied; **no `infected` input**
- [ ] `knn_dr.py`: copied as a new file (`ess` / `tail_index` for `fit_urr`
      and export)
- [ ] `fit_urr.py`: copied URR loss; no `infected==1` restriction, no
      `--cell_type` / `experiment`; `--nu_rows`; support and \(\nu\) keyed on
      the `dose_level` arm (P2); arm-stratified cross-fit folds (P5)
- [ ] `export_urr_weights.py`: copied; no `inf`; vehicle rows \(w = 1\);
      `net` (primary, P1) and `counts` (fallback; `dose_level` arm keys,
      (arm, C) when C is non-empty); compares with `dr_weights_design.npz`
      when present; writes `dr_weights_{urr,counts}.npz` `{row_id, w}`
- [ ] URR gate: beats constant baseline, `mean(alpha)≈1`, ESS/n usable
      (trivial on v1 — §3.3); known answers hold: net
      \(\alpha \approx 1.066\) on treated / ≈0 on DMSO, `counts` \(w \equiv 1\)
- [ ] Trainer gets `n_compounds` without a hidden `fit_urr` dependency
      (`compound_vocab.json` or `build_dataset`'s `nuisance_meta.json`)

### Phase 2 — MLP arms

- [ ] `models/conditioning.py`: copied as a new file
- [ ] `processes/__init__.py` + `ddpm.py` + `flow_matching.py`: copied as new
      files
- [ ] `models/mlp.py`: vector denoiser, full shared interface (§3.4)
- [ ] `models/__init__.py`: `arch=mlp` in `arch.json` (`n_genes`, no image
      VAE keys); `build_generator_from_ckpt` dispatches on `arch`
- [ ] `train/train_diffusion.py`: copied loop; rank-agnostic loss in train
      and validation; `--arch mlp`; `--dr_mode {conditional, weighted}` +
      `--dr_weights_file` (P7); no `--latent`; V-REx key
- [ ] `nuisances/precompute_cmean.py` + trainer `--cmean_lambda` (default 0)
      / `--cmean_file`, FM only; `--min_n` configurable, coverage logged (P9)
- [ ] `src/tests/test_lincs_shapes.py` (or equivalent) for a GPU box
- [ ] `mlp_conditional` arm trainable
- [ ] `mlp_dr` arm trainable (after `fit_urr` + `export_urr_weights --mode
      net`)

### Phase 3 — 1D-DiT arms

- [ ] `models/dit1d.py`: 1-d patch embed + copied `DiTBlock` /
      `TimestepEmbedder` / 1-d sin-cos (not `DiT2DModel`)
- [ ] `arch.json` rebuild for `arch=dit1d` (`patch_size`, pad, gene order)
- [ ] `dit1d_conditional` arm trainable
- [ ] `dit1d_dr` arm trainable

### Phase 4 — eval

- [ ] `eval/generation.py`: CFG sampling → `(B, 978)`; no clamp, no
      `channels_last`; rebuild from `arch.json` only
- [ ] `eval/dist_metrics.py`: Frechet (marginal / PCs) + MMD on \(Y\)
      (copied math; no Inception, no torchvision)
- [ ] `eval/evaluate.py`: gene-space \(\hat\tau(c,d)\) on `dose_level`;
      `--pool all` per-arm aggregates + holdout aggregates; real and
      generated \(\mu(0)\); `--truth` accuracy; no OpenPhenom / rescue panel
- [ ] Oracle (`--source real`) + all four v1 generator arms scored
- [ ] `cmean` ablation (P9): FM `mlp_conditional` / `mlp_dr` with
      `--cmean_lambda` 0 vs > 0, scored with `--truth`

### Phase 5 — step C: semi-synthetic confounder on MCF7 (§3.8.2)

- [ ] `responders.json` from the Phase 4 real oracle (max-over-dose
      \(\|\hat\tau\|\) > 1.5× noise floor)
- [ ] `synthetic.py`: seeded `v`, `β` resolution, injection in `dataset.py`
      after centring / z-scoring; `syn_effect` / `syn_seed` in `arch.json`
- [ ] `evaluate.py`: `--syn_effect` / `--syn_seed` on the real oracle;
      `--truth` refuses mismatches; signed bias along `v` by dose half,
      rare vs non-rare
- [ ] `build_tiered_split` step-C instance: responders-only scored
      compounds, \(z\) = dose half × `syn_c`, positivity cells (compound,
      dose half, `syn_c`), `keep_frac` 0.4, γ ∈ {0, 1}; `--plan` bias table;
      `dr_weights_design.npz`, `nu_rows.npy`, `tier_meta.json`
- [ ] Nuisances: `fit_urr --adjustment_set syn_c --target_support all
      --nu_rows …` + `export_urr_weights --mode net` (`counts` fallback keyed
      on (arm, `syn_c`));
      gate passes with \(\alpha\) varying in `syn_c`; weights track
      `dr_weights_design.npz` where \(\pi\) is known
- [ ] `naive` / `conditional` / `dr` (+ `dr_design`) × γ ∈ {0, 1} × ≥2
      seeds trained and scored
- [ ] Step-C verdict against the §3.8.1 success criteria written up (gate
      for Phase 6)

### Phase 6 — step A: cell line on `core5_24h` (§3.8.3)

- [ ] Effect-modification gate: responders' centred \(\hat\tau\) differs
      across the 5 lines beyond the 3-well noise floor
- [ ] `--population` switch; `core5_24h` paths; `PopulationSpec.name` in
      every artifact and result
- [ ] Ingest, plate QC (per-line median), splits, `expr_stats`, Phase 4
      oracle and `responders.json` for `core5_24h`
- [ ] `build_tiered_split` step-A instance: positivity cells (compound,
      `dose_level`, `cell_id`), `keep_frac` 0.6; line groups declared in
      `spec.py`
- [ ] Nuisances: `fit_urr --adjustment_set cell_id --target_support common
      --nu_rows …` + `export_urr_weights --mode net`; gate passes; weights
      track `dr_weights_design.npz`
- [ ] `naive` / `conditional` / `dr` (+ `dr_design`) × γ ∈ {0, γ>0} × ≥2
      seeds trained and scored; group-contrast bias readout

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
| Step C confounds at (compound, dose half) with `--target_support all` | ~2 train wells per arm leave (arm, `syn_c`) cells with ~1 row |
| Step A confounds at (compound, `dose_level`, `cell_id`) with `--target_support common` | ~2 train rows per cell keep arm-level positivity at `min_cell 1` |
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
