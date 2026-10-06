# Step A (Phase 6) and the P2 test — plan of record

*Written 2026-10-05, after the user's decisions of that day. Nothing in this
plan has been implemented or submitted yet. Every box is unchecked and stays
unchecked until its code has been reviewed, as in `IMPLEMENT.md` §4.*

*Background lives in `IMPLEMENT.md`: step A's design in §3.8.3, the shared
tiered-split mechanism in §3.8.1, the scoring settings in §3.14 (E1–E12), and
in §5 the "Step C2 results", "Phase 6 effect-modification gate" and "P1
results" entries. P1 and P2 are defined in `understand.md` §3.3.1 and §3.3.2.*

---

## 1. What this plan does

Two tracks run side by side.

- **Track A — step A on `core5_24h` (critical path).** The paper's only
  real-confounder experiment: the confounder is the cell line, thinned at
  (compound, `dose_level`, `cell_id`). It compares ADIGen's weighted generator
  (`dr`) against two baselines, `conditional` and P1 (AIPW on the estimand).
  The effect-modification gate has passed, so this is not another null check.
- **Track B — P2 on step C2 (small, day 1).** A 6-run test of whether
  within-group weight normalisation plus a cap stops the weights from
  degrading the generator. Its only purpose is to decide whether a `dr_p2`
  arm joins step A.

Budget: about 25 GPU-hours, plus under 2 hours of small CPU jobs. The code,
not the compute, sets the schedule (§8).

## 2. Decisions (user, 2026-10-05)

| # | Decision |
|---|---|
| D1 | Run both tracks; Phase 6 is the critical path |
| D2 | **Disk:** new runs save only the final checkpoint (`--checkpoint_every 500`). Old intermediate checkpoints are listed for the user to delete (§7); nobody else deletes them |
| D3 | **Line groups** for `z_C`: G1 = {MCF7, HT29, PC3} (carcinomas), G2 = {HA1E, A375}. Declared here, before any step-A result, and goes into `spec.py` and the dir tag |
| D4 | **Scoring memory:** up to 32 GB per scoring job on `zabih` if the measured need exceeds 16 GB and the memory is available |
| D5 | **P2 pass rule** (§4), fixed before the P2 runs |
| D6 | **P2 normalisation unit:** the positivity cell with the confounder dropped (`splits.target_groups`): (compound, dose half) on C2, the arm in step A. This replaces "within compound" in `understand.md` §3.3.2 |
| D7 | **Cuts:** no `dr_design` arm; `naive` at seed 0 only |
| D9 | **Approved (user, 2026-10-05):** `--keep_frac` for `cell_id` is the *realised* kept fraction, default **0.75**, instead of the nominal 0.6 of A6 (§5, A6 note) |
| D10 | **Approved (user, 2026-10-05):** the readout's direction $u_a$ comes from holdout wells only (§3) |
| D11 | **Applied 2026-10-05 after re-reading track B; reported to the user:** S4 gets a noise allowance. An arm fails it only if its unscored DiD is both above a quarter of `naive`'s scored bias and more than 2 SE from zero |
| D8 | **`dr_p2` joins step A** (2026-10-05, after track B). Track B failed its pass rule on the third part by 0.019, inside the noise; the user overrode the rule's outcome (§4, "Result") |

Unchanged from earlier decisions: MLP-B, DDPM, 500 epochs, and the §3.14
scoring settings (`checkpoint-0499` EMA, w = 1, 16 samples per row, 100 steps,
`--pool all`).

## 3. Step-A readout and success criteria

*Proposed here and to be copied into `IMPLEMENT.md` §3.8.3 in A0, before any
step-A result exists. They mirror §3.8.4, adapted to a real confounder and to
the cuts in D7.*

**Readout.** For an arm $a$ of a scored compound:
- $c_a = \hat\tau_{G_2}(a) - \hat\tau_{G_1}(a)$ is the group contrast of the
  arm **over its holdout wells only**, and $u_a = c_a / \|c_a\|$ its
  direction. (Each group's $\hat\tau$ is taken against that group's own
  vehicle wells.)
- $e_a = \hat\tau_\text{gen}(a) - \hat\tau_\text{oracle}(a)$ is the error of
  the line-pooled estimate against the `--pool all` oracle, and
  $b_a = \langle e_a, u_a \rangle$ its signed projection.
- Thinning that over-keeps $G_2$ by $\Delta p_a$ predicts
  $b_a \approx \Delta p_a \langle c_a^\text{true}, u_a \rangle$ for `naive`.
  The selection score is $z_\text{dose} \cdot z_C$ (§3.8.1), so $\Delta p$
  flips sign between dose halves.
- **Statistic:** the mean of $b_a$ over scored arms, high half minus low half,
  at γ > 0 minus γ = 0 (the DiD of steps C and C2, with $v$ replaced by $u_a$).
- **Why the direction comes from holdout wells (corrected 2026-10-05, in
  review).** The first version took $u_a$ from all of the arm's wells. An
  estimator that copies noise from its kept training wells then projects on a
  direction built from those same wells, and that term scales with
  $\Delta p$: it reads as "bias" with no line effect at all, and the γ = 0
  control cannot remove it, because $\Delta p$ is exactly what γ switches on.
  On the limit build it inflated the data-level bias about 2.4×. No estimator
  trains on a holdout well, so with a holdout direction the statistic is 0 in
  expectation when the lines respond alike. The cost is a noisier direction
  (one holdout well per arm and line), which attenuates the signal but does not
  bias it.
- The oracle's own noise is in both $\hat\tau_\text{oracle}$ and $u_a$. That
  term does not involve $\Delta p$ and is the same at both γ, so it cancels in
  the DiD; the γ = 0 contrast is reported as the check.

**Uncertainty.** `naive` has one seed, and the seed spread understates the
error anyway ("P1 results", last finding). Every criterion is judged against
the **compound-clustered SE** (jackknife over the scored compounds); the seed
spread is reported next to it.

| # | Criterion |
|---|---|
| S0 | **Power gate, before any GPU job:** the bias the realised thinning puts into the training data is ≥ 5 clustered SE. It is read from a memoriser (each arm's mean over its kept train wells), with the same statistic (`src.eval.step_a_power`). If not, stop and re-decide γ, `keep_frac` or the scored set |
| S1 | `naive` is biased: \|DiD\| > 3 SE |
| S2 | **Testability:** `conditional`'s \|DiD\| > 2 SE. If not, "DR vs `conditional`" is recorded as not testable in step A (as in step C), not as a failure |
| S3 | **The ADIGen arm beats `conditional`:** its \|DiD\| is smaller, by more than 2 SE of the paired difference. Judged for `dr`, and for `dr_p2` if it runs |
| S4 | Unscored compounds unchanged: per arm, their \|DiD\| < 0.25 × `naive`'s scored \|DiD\|, **or within 2 SE of zero** (D11) |
| S5 | Weights: each weighted or targeted run's weight hash matches its file. That the counts weights equal n_unthinned / n_kept per (arm, line) cell is `check_phase1`'s exact recompute; their correlation with the design weights is reported, not gated |

P1 arms are judged by S3 as a **baseline**, not as ADIGen. Their flat-weight
control must come out worse than the untargeted arm.

Secondary, reported but not gated: pooled gene MSE against `conditional`, and
the learned share of the line contrast (the step-A analogue of λ̂).

## 4. Track B — P2 on step C2

**The change.** In `train_diffusion.py`'s weighted path
(`--dr_weight_norm group --dr_weight_clip 5`; `src/nuisances/weight_norm.py`):
- normalise the weights within each group of D6, so a group's total weight
  equals its unweighted row count (untouched compounds keep weight exactly 1);
- cap each weight at **5** with the group total preserved: capped rows sit at
  the cap and the group's other rows carry the rest, repeated until no row
  exceeds it. So the cap holds exactly. It is defined once, as
  `weight_norm.P2_CLIP`; ESS and the capped rows are recorded in `arch.json`.

Both settings are recorded in `arch.json`, and the default stays the current
global normalisation, so every existing run is unaffected and still resumes.

**Measured on the real weights before any run** (`src/tests/check_p2.py`,
γ = 1, counts):

| | global (today's `dr`) | group + cap 5 (`dr_p2`) |
|---|---|---|
| weight of an unthinned or vehicle row | 0.81 | exactly 1 |
| share of total weight on scored rows (23.4% of rows) | 38.0% | 23.4% |
| largest weight | 17.8 | 5.0 |
| ESS / n | 0.68 | 0.92 |
| `syn_c` mix of a scored group vs the unthinned pool, mean error | 0.0000 | 0.0004 |

- The cap binds on only 4 rows in 4 groups (largest uncapped weight 7.5), so
  P2 is essentially the group normalisation; the cap is a guard.
- The balance of `syn_c` inside each group is untouched, which is the point:
  the weights still do their job, and stop moving mass between groups.
- The reallocation it removes is a 19% down-weighting of the unthinned rows.
  Whether that alone explains their loss of learned share (0.72 → 0.14) is
  what the runs test; the lower weight noise (ESS 0.68 → 0.92) may matter as
  much.

**Pass rule (D5), all three on C2, `--pool all`, mean over 3 seeds:**

| Metric | `conditional` | `dr` today | `dr_p2` must reach |
|---|---|---|---|
| learned share λ̂ on unthinned compounds | 0.72 | 0.14 | ≥ 0.60 |
| pooled gene MSE, relative to `conditional` | 1.00 | 1.25–1.30 | ≤ 1.10 |
| \|DiD\|, scored | 0.28 | 0.58 | ≤ 0.58 |

Pass: `dr_p2` joins the step-A matrix. Fail: it is dropped and the result is
recorded. P2 is not expected to beat `conditional` on C2; the arm-level ratio
bias remains there ("Step C2 results", finding 3).

Checked 2026-10-05, in the status review of Phases 4 and 5 (`IMPLEMENT.md` §5):
the code was reviewed when written, the six runs and their scoring are on
disk, and the pass rule's numbers reproduce from the verdict file. That review
also found that `dr_p2`'s unchanged DiD is carried by a difference in the
learned share between the dose halves. The arm-level ratio bias named in this
section is not established as its cause (`IMPLEMENT.md` §5, status review,
gap 1).

- [x] **B1** (code) `--dr_weight_norm {global, group}` and `--dr_weight_clip`;
      `arch.json` fields; `step_c_report` learns the `dr_p2` label and the pass
      rule; `src/tests/check_p2.py`. Written and reviewed 2026-10-05
      (`IMPLEMENT.md` §5, "P2 implementation and review")
- [x] **B2** (GPU) smoke: group weight sums equal the unweighted counts,
      untouched compounds keep weight 1, the cap binds where expected
      - Job 990819 on `zabih`, 2026-10-05: 50 checks, 0 failures.
- [x] **B3** (GPU) train `dr_p2` (counts weights) × γ ∈ {0, 1} × seeds {0, 1, 2}
      under the C2 injection
      - Jobs 990856–990866 (even IDs), all exit 0.
- [x] **B4** (GPU) score each, chained after its training
      - Jobs 990857–990867 (odd IDs), all exit 0.
- [x] **B5** (CPU) report against the pass rule; record the verdict in
      `IMPLEMENT.md` §5
      - Run locally 2026-10-05 (numpy only). **Verdict: FAIL on the third
        part** (`IMPLEMENT.md` §5, "P2 results").

**Result (2026-10-05).**

| Metric | `dr` | `dr_p2` | needed | |
|---|---|---|---|---|
| λ̂ on unthinned compounds | 0.14 | 0.67 | ≥ 0.60 | pass |
| pooled gene MSE, relative to `conditional` | 1.29 | 1.05 | ≤ 1.10 | pass |
| \|DiD\|, scored | 0.575 | 0.599 | ≤ 0.58 | **fail** |

- P2 repairs the generator and leaves the bias unchanged: the paired
  difference from `dr` is +0.023 ± 0.055 (compound-clustered), 0.4 SE.
- By the rule as written, `dr_p2` would not be in the step-A matrix. The
  threshold had no noise allowance, and on mechanism `dr_p2` is the
  better-founded weighted arm for step A (`IMPLEMENT.md` §5, "P2 results",
  findings 4 and 6).
- **Decision D8 (user, 2026-10-05): override.** `dr_p2` runs in step A, so the
  matrix of §6 is 14 runs.

### What track B means for step A (re-read 2026-10-05)

No change to the arms, the data or the launch sequence. Three adjustments, the
first of them to a criterion:

- **S4 gets a noise allowance (D11).** The P2 pass rule failed on a threshold
  with no allowance for noise (`IMPLEMENT.md` §5, "P2 results", finding 4). S4
  had the same shape: a quarter of `naive`'s scored bias, compared with a
  number that has its own sampling error. If `naive` shows little bias, that
  bound is below the noise and S4 fails for no reason. On the GPU smoke's
  untrained arms it did exactly that. S1–S3 already carry SE margins.
- **The report shows the two diagnostics track B found decisive.** A weighted
  risk can damage the generator on compounds it never thinned, and on C2 that
  showed in the learned share on *unscored* compounds (0.72 → 0.14) and in the
  MSE (+29%), not in the bias. `step_a_report` now prints both next to the bias
  for every arm, so `dr` and `dr_p2` can be compared on them.
- **`dr` against `dr_p2` stays informative in step A.** On the limit build the
  global normalisation puts unthinned rows at 0.78 (C2: 0.81), so the mechanism
  track B measured is present here too. The full build should leave a sizeable
  unthinned group: a rough projection from the MCF7 oracle puts one half to two
  thirds of compounds in the scored set, against 98% on the limit build.

Unchanged by track B: the cap, both weighted arms (D8), and the seeds (see §10
for an optional cut). On the limit build the cap did not bind (largest
normalised weight 4.0); on the full build it binds on 2 rows at γ = 1 (§5,
"Full build").

## 5. Track A — step A on `core5_24h`

*Status 2026-10-05: all code is written, reviewed and smoke-tested end to end on
a `--limit` build (2 whole plate maps, 11,904 wells, 110 compounds). The full
data layer is built, S0 passes (job 992724; "Full build" below), and **the 14
runs are trained, scored and judged: S1–S5 all pass** (A12). Details
in `IMPLEMENT.md` §5, "Step A implementation and review" and "Step A launch".*

- [ ] **A0** (code) `--population` switch through build, splits, expression
      stats, nuisances, trainer, `evaluate`, `dr_target` and the reports;
      `core5_24h` in `spec.POPULATIONS` (five lines, compounds present in all
      five); D3's line groups in `spec.LINE_GROUPS`; §3 above referenced from
      `IMPLEMENT.md` §3.8.3
      - Written. A command needs only `--data_dir`: the build names its own
        population, and a contradicting `--population` is refused.
- [ ] **A1** (CPU) `--limit` build of `core5_24h` and `check_build`
      - Done on the limit build (jobs 991752 / 991753).
- [ ] **A2** (CPU) full ingest (~183k wells, landmark genes only), plate QC on
      each line's median spread, `check_build`
- [ ] **A3** (CPU) base split, `expr_stats` (one pooled z-scale), vocabulary
      and encoders
- [ ] **A4** (CPU) oracle on `--pool all` with each arm's group contrast
      $c_a$, over the pool's wells and over its holdout wells (the readout's
      direction, §3); `check_phase4`
      - Per-line $\hat\tau$ is not stored: nothing reads it, and it would be
        200 MB per oracle.
- [ ] **A5** (CPU) `responders.json` for `core5_24h` from the pooled oracle,
      with the thinnability screen (a train well in every (`dose_level`,
      `cell_id`) cell)
- [ ] **A6** (CPU) tiered splits at γ = 0 and γ = 1; `--plan`; counts weights
      with `--adjustment_set cell_id`; `check_phase1`; the S0 power gate
      (`step_a_power`); `check_phase6`. **Gate S0 is read here.**
      - **`keep_frac` changed meaning (D9, approved).** With ~2 train
        wells per (arm, line) cell, the "keep ≥ 1 per cell" redraw moved the
        realised fraction far from the nominal 0.6, and by a γ-dependent
        amount: the γ = 0 control kept 68% and γ = 1 kept 74% on the limit
        build, so the control was not size-matched. For `cell_id`, `keep_frac`
        is now the expected *realised* fraction, default **0.75**: about what
        nominal 0.6 gave at γ = 1, and near the strongest lever a 2-well cell
        allows. Measured after the change: 0.749 and 0.748. A realised 0.6
        would be size-matched too, but its lever is about 5× weaker.
      - The positivity redraw is per cell here (the same distribution as step
        C's whole-compound redraw, which would need ~10³–10⁵ draws per compound
        with 30 small cells). Step C's code path is unchanged.
- [ ] **A7** (code) the §3 readout in `evaluate`; `step_a_report` with the
      clustered SE; `check_phase6`; the smoke script
      - Written: `bias_along_contrast`, `learned_line_contrast`,
        `src/eval/{contrast_stats,step_a_power,step_a_report}.py`,
        `src/tests/{check_phase6,smoke_phase6_torch}.py`.
- [ ] **A8** (GPU) smoke on the limit build, then on the full build
      - Limit build: green (job 991753, `zabih`, 3 min 40 s).
      - Full build: green (job 992778, `zabih`, 24 min). Scoring with the
        quality block on peaked at 9.0 GB, so the scoring jobs keep the default
        16 GB (D4's 32 GB is not needed). On these untrained generators the
        flat-weight P1 control gives −0.942 ± 0.052, the memoriser's −0.941, and
        P1 with the counts weights gives −0.038 ± 0.032.
- [ ] **A9** (GPU) train the arm matrix (§6)
      - **Launched 2026-10-05 13:53** on `zabih`: training jobs 993188–993214
        (even ids), each chained to its scoring job (odd ids, 993189–993215),
        then the P1 + verdict job 993216 on `bindel`. Run dirs
        `runs/core5_24h/stepa_{arm}_g{0,1}_s{seed}`.
      - Done 2026-10-05: 14 runs, 47–52 min each, no failure.
- [ ] **A10** (GPU) score each run, chained after its training
      - Done: 33–34 min each at 16 GB.
- [ ] **A11** (CPU) P1 (`dr_target`) on the `naive` and `conditional` runs,
      with the counts weights and the flat-weight control
      - `dr_target` needed a small change after all: it now carries the §3
        readout through the correction.
- [ ] **A12** (CPU) `step_a_report`; verdict against S1–S5 written into
      `IMPLEMENT.md` §5
      - Done (job 993216, with A11): **S1–S5 all pass on the declared
        readout.** `naive` −0.793 ± 0.042, `conditional` −0.071 ± 0.015,
        `dr` −0.000 ± 0.018, `dr_p2` −0.005 ± 0.017; `dr` beats `conditional`
        by +0.070 ± 0.019 (3.7 SE). Full table, the holdout-pool check and the
        caveats are in `IMPLEMENT.md` §5, "Step A results".
      - The secondary MSE is decomposed term by term in `IMPLEMENT.md` §5,
        "MSE decomposition": the bias the readout measures is at most 0.3% of
        any arm's squared error, and against the truth the weights cost +17%
        (`dr`) and +11% (`dr_p2`) on the thinned arms.

**How to run the rest** (from `lincs/`):

| Step | Command | Type |
|---|---|---|
| A2–A6 | `sbatch scripts/phase6_cpu.sub` | CPU, `bindel` (16 GB) |
| A8, full build | `sbatch --mem=32G scripts/phase6_gpu.sub data/core5_24h` | GPU, `zabih` |
| A9–A12 | `bash scripts/phase6_arms.sh` | 14 × (train + score) on `zabih`, then one CPU job for P1 and the verdict |

`phase6_arms.sh` refuses to launch unless S0 passed on the tier instances it is
about to train on (`FORCE=1` overrides).

**What the limit build already shows** (103 scored compounds; not a result, but
it sizes the experiment):

| Quantity | Value |
|---|---|
| bias in the training data (memoriser DiD, S0) | −1.69 ± 0.16, 10.8 SE |
| the same from the line mix alone (`planned`) | −1.63 ± 0.13 |
| memoriser with the counts weights | +0.01 ± 0.09 |
| memoriser with the design weights | −0.10 ± 0.10 |

- The two independent estimates of the data-level bias agree.
- The counts weights remove it at the arm level. This is the property step C2
  lacked, and the reason step A can show a DR advantage.
- **These numbers overstate the full population (noted 2026-10-05).** The limit
  build is the first two plate maps, LJP005 and LJP006, from the kinase-inhibitor
  library. On MCF7, 97% of LJP compounds are responders against 26% of the REP
  compounds, which are 84% of the population. So the full build has more
  compounds (smaller errors) but weaker ones (a smaller effect), and the limit
  build's 10.8 SE cannot be scaled up. S0 on the full build is the number that
  counts.

**Full build** (job 992724, `bindel`, 6 min 30 s, peak memory 2.5 GB; every
check passes):

| Quantity | Value |
|---|---|
| population | 182,174 wells, 494 plates, 1,750 compounds (all present in all five lines); 2 plates dropped by QC |
| scored (thinned) compounds | 848 of 1,750 (891 responders, 43 not thinnable); 5,052 scored arms carry a direction |
| train wells of scored compounds kept | γ = 0: 0.747; γ = 1: 0.751 (size-matched) |
| shift of the G2 share at γ = 1 | low dose half +0.145, high −0.136 |
| **bias in the training data (memoriser DiD, S0)** | **−0.941 ± 0.052, 18.2 SE: PASS** (needs ≥ 5) |
| the same from the line mix alone (`planned`) | −0.917 ± 0.043 |
| memoriser with the counts weights | −0.046 ± 0.031 |
| memoriser with the design weights | −0.047 ± 0.035 |

- The effect is a little over half the limit build's, as expected from weaker
  compounds, and the error is a third of it: the gate passes with more room.
- 902 compounds are not thinned, so S4 and the unscored diagnostics have a
  large control group.
- The counts weights remove 95% of the data-level bias; the remainder is within
  1.5 SE of zero.
- **P2's cap binds on 2 rows at γ = 1** (weight capped at 5.00), which leaves
  one arm's line balance off by 0.058 in G2 share. Every other arm is balanced
  to 1e-8. Negligible for the readout (2 of 114,222 training rows), but it
  corrects the earlier note that the cap never binds in step A.

## 6. Arm matrix for A9

Each cell is trained at γ = 0 and γ = 1 on the same tier instances.

| arm | generator sees `cell_id` | risk | seeds | runs |
|---|---|---|---|---|
| `naive` | no | plain | 0 | 2 |
| `conditional` | yes | plain | 0, 1 | 4 |
| `dr` (ADIGen) | yes | α-weighted, counts, global normalisation | 0, 1 | 4 |
| `dr_p2` (D8) | yes | α-weighted, counts, group normalisation + cap | 0, 1 | 4 |

- 14 runs.
- P1 adds no training: `p1_naive` and `p1_cond` are re-estimates of the
  `naive` and `conditional` runs (A11).
- Submission order: seed 0 of every arm first, then seed 1.

## 7. Jobs, partitions and disk

Requests follow `.claude/rules/slurm_partition_priority.md`. Availability is
rechecked before every submission; the suggestions below are the default.

| Job | Type | Partition | Request | Count × time (estimate) |
|---|---|---|---|---|
| B2 | GPU | `zabih` | 1 GPU, 4 CPU, 16 GB | 1 × 5 min |
| B3 | GPU | `zabih` | same | 6 × 10 min |
| B4 | GPU | `zabih` | same | 6 × 8 min |
| B5 | CPU | `bindel` | 4 CPU, 12 GB | 1 × 3 min |
| A1 | CPU | `bindel` | 4 CPU, 8 GB | 1 × 1 min (measured, whole chain A1–A6 on the limit build) |
| A2 | CPU | `bindel` | 4 CPU, 8 GB | 1 × 10 min |
| A3 | CPU | `bindel` | 4 CPU, 8 GB | 1 × 5 min |
| A4 | CPU | `bindel` | 4 CPU, 12 GB | 1 × 2 min |
| A5 | CPU | `bindel` | 4 CPU, 8 GB | 1 × 2 min |
| A6 | CPU | `bindel` | 4 CPU, 8 GB | 1 × 10 min |
| A8 | GPU | `zabih` | 1 GPU, 4 CPU, 16 GB | 1 × 10 min |
| A9 | GPU | `zabih`, up to 6 at once | 1 GPU, 4 CPU, 16 GB | 14 × 60 min |
| A10 | GPU | `zabih` | 1 GPU, 4 CPU, 16 GB (up to 32 GB, D4) | 14 × 40 min |
| A11 | CPU | `bindel` | 4 CPU, 16–24 GB | 12 outputs, ~1 h in total |
| A12 | CPU | `bindel` | 4 CPU, 12 GB | 1 × 5 min |

- **Totals:** about 25 GPU-hours (23 for step A, 2 for P2). Step A is about
  4 hours of wall time on six A6000s, or 8 on three.
- **Estimates:** step-A times are MCF7's measured times scaled by row count
  (training 10 min at ~21k rows, scoring 8 min at 37k). They are not measured.
- **CPU jobs:** `ma` is used instead of `bindel` only if it passes the 20%
  headroom check at submission. Every CPU job fits on one `bindel` node.
- **`zabih` today (2026-10-05):** 31 of 32 CPUs were allocated. Other users'
  jobs are hidden, so whether they are preemptible is only seen by submitting
  and reading the pending reason.
- **Fallback to `gpu`** when `zabih` has no room:
  - send **training** there, and keep **scoring** on `zabih`: training can
    resume from a checkpoint, a 40-minute scoring job cannot;
  - GPU order: `tesla_v100-sxm3-32gb-h`, then `nvidia_titan_rtx`, then
    `nvidia_geforce_rtx_2080_ti` (type strings as listed on 2026-10-05);
    memory and precision settings are confirmed before the first fallback job;
  - fallback training runs keep a checkpoint every 100 epochs so a preempted
    job resumes, and the job removes its own intermediates once epoch 499 is
    saved. The CUDA preflight is already in `train.sub`.

**Disk.** `/share/zabih` had 111 GB free (95% full) on 2026-10-05.

| New output | Estimate |
|---|---|
| step A, per run: final checkpoint 0.6 GB + per-row means ~2.1 GB + τ ~0.7 GB | ~3.5 GB × 14 = ~50 GB |
| P2, 6 runs | ~7 GB |
| P1 outputs, 12 | ~8 GB |
| **Total** | **~65 GB** |

**Old intermediate checkpoints, for the user to delete (D2).** The list is in
`runs/intermediate_checkpoints.txt`: 240 directories, **139.9 GB**, every
`checkpoint-0099/0199/0299/0399` of the 60 full runs. Each run's
`checkpoint-0499`, the only one ever scored or resumed from, is not on the
list.

| Family | Directories | Size |
|---|---|---|
| step C (24 runs) | 96 | 56.9 GB |
| step C2 (24 runs) | 96 | 56.9 GB |
| Phase 2–4 MLP (6 runs) | 24 | 14.2 GB |
| Phase 2–4 DiT (6 runs) | 24 | 11.9 GB |

`runs/smoke_run_dirs.txt` lists 27 smoke and sanity run directories (9.6 GB)
that no result depends on; they are a second, optional candidate.

## 8. Schedule

| Day | Work |
|---|---|
| 1 | A0; track B end to end (B1–B5); A1–A6 and gate S0 |
| 2 | A7, A8; launch A9 by evening; A9–A10 run overnight |
| 3 | A11, A12, write-up |

Track B must report before A9 is launched, because it decides the matrix.

## 9. Risks

- **A0 is the long item.** The population switch touches every stage, and a
  mistake there mixes populations. `PopulationSpec.name` is checked in every
  artifact, and A1 runs the whole chain on a limit build first.
- **S0 can stop the run.** If the real line contrast gives too small a
  predicted bias, the GPU jobs are not launched until γ, `keep_frac` or the
  scored set is re-decided.
- **Scoring memory** may exceed 16 GB (MCF7 used 3.1 GB at a fifth of the
  rows; the limit build used 2.2 GB at a fifteenth). The full-build A8 run
  measures it; D4 covers up to 32 GB.
- **The generator may show much less than the data holds.** The memoriser
  bounds what `naive` can show; in steps C and C2 the generator showed 65% and
  18% of it. How many SE that leaves depends on the full build's S0, which the
  limit build cannot predict (it holds only the most active compounds; §5).
- **P1's groups are larger than feared.** The group is the arm, pooled over
  lines: about 10 train wells before thinning and 7–8 after (measured
  n_eff 7–9), not the ~2 quoted in `IMPLEMENT.md` §5, "P1 results".
- **The `gpu` fallback** resumes a requeued run from its latest checkpoint and
  prunes its own intermediate checkpoints. The shell logic is tested; a real
  preempt-and-resume cycle is not.
- **Two seeds** give a weak seed spread, which is why S1–S4 use the clustered
  SE.
- **Numerics:** every job script already sets `OPENBLAS_CORETYPE=Haswell`, and
  `evaluate` refuses a broken BLAS ("Step C2 scoring" in `IMPLEMENT.md` §5).
  New scripts must copy both.

## 10. Further cuts, if time runs out

1. Drop the `dr_p2` arm from step A even if track B passes (4 runs).
2. Train 200 epochs instead of 500: still more than twice MCF7's optimiser
   steps, about 8 GPU-hours saved, but a departure from the MCF7 protocol.
3. Drop seed 1 of `conditional`, `dr` and `dr_p2`, leaving one seed per arm and
   the clustered SE as the only uncertainty (6 runs, ~10 GPU-hours). Track B
   suggests this costs little: on C2 the seed-to-seed sd (0.02–0.05) was well
   below the compound-clustered SE (0.05–0.07), so a second seed removes only a
   small part of the error.
