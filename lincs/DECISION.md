# Decision task on step A — plan of record

*Written 2026-10-05, after the step-A results and the main author's comment on
them. A box is checked once its code has been reviewed, as in `IMPLEMENT.md`
§4.*

*Status 2026-10-05: implemented, reviewed, smoke-tested and run. **Neither task
is testable: the power gate D0 fails on both** (§11). The verdict and the
findings are in `IMPLEMENT.md` §5, "Decision task on step A".*

*Background: the step-A design and results are in `STEP_A.md` and in
`IMPLEMENT.md` §5 ("Step A results", "MSE decomposition"). The experiment this
plan mirrors is RxRx19a's `rescue_best` (`RxRx19a/src/eval/evaluate.py`).*

---

## 1. What this plan does

**Question.** Step A showed that `dr` removes `conditional`'s residual
confounding bias (−0.071 → 0.000) at a cost of 11–17% more model error on the
thinned arms. The project's target is a debiased generator for decision
making, not the lowest MSE. So: when the step-A generators are used to choose
a dose or a compound, does `dr` choose better than `conditional`?

**How.** Each trained generator is turned into a scalar treatment effect per
(compound, dose), an argmax is taken, and the choice is scored against real
wells that no generator trained on. This is `rescue_best` with the infection
axis replaced by two LINCS axes (§4).

**Cost.** No training and no sampling. The 14 step-A scoring jobs saved the
generated mean of every one of the 182,174 rows (`*_gen.npz`, `row_mean`), and
the table gives each row's line. The whole experiment is one CPU job of a few
minutes (§8).

## 2. Decisions

| # | Decision | By |
|---|---|---|
| X1 | Run both tasks: dose selection (A) and compound selection (B) | user, 2026-10-05 |
| X2 | The target population is the five lines with **equal weight** | user, 2026-10-05 |
| X3 | The population's well-share mix is an optional later run (§10) | user, 2026-10-05 |
| X4 | Task B's headline is **k = 100**, with k = 25, 50 and 200 reported next to it (§5) | delegated to Claude |
| X5 | The truth is each arm's holdout wells; the all-wells oracle is secondary (§3). Proposed by Claude and run as proposed: the user asked for the implementation without changing it | Claude |
| X6 | Task B's axis is a fixed gene set with equal weights, not a direction estimated from data (§4). As X5 | Claude |
| X7 | The criteria D0–D5 of §6, fixed before any decision number was read. As X5 | Claude |

Unchanged from step A: the 14 runs, their scoring settings (`checkpoint-0499`
EMA, w = 1, 16 samples per row, 100 steps), the scored set (848 thinned
compounds, 902 not thinned) and the tier seed.

## 3. One treatment effect for every decision

An argmax compares numbers. They must be the same causal contrast on the same
population, or the argmax compares effects on different populations. The
definition below is used for the truth and for every estimator.

- **Population.** The five lines of `core5_24h`, each with weight
  $P(\ell) = 1/5$ (X2).
- **Vector effect of an arm** (compound $c$ at dose $d$):

  $$\tau(c, d) = \sum_{\ell} P(\ell)\,\bigl[\mu(c, d, \ell) - \mu(0, \ell)\bigr]$$

  where $\mu(c, d, \ell)$ is the mean expression (978 genes, z units) in line
  $\ell$ and $\mu(0, \ell)$ is that line's vehicle mean.
- **Scalar effect.** $\mathrm{ATE}_s(c, d) = \langle \tau(c, d), s \rangle$ for
  a unit vector $s$ fixed before any generated number is read (§4).
- **For a generator**, $\mu(c, d, \ell)$ is the mean of `row_mean` over the
  generated rows of that (arm, line) cell, and $\mu(0, \ell)$ the same over the
  line's generated vehicle rows.
  - `conditional`, `dr` and `dr_p2` see the line, so their cells differ and the
    sum is a g-formula.
  - `naive` does not see the line, so its cells are equal and it returns
    whatever its training mix taught it. That is the bias being tested.
- **For the truth**, the same formula on real wells.

This differs from the stored `tau` in two ways. The stored oracle and
`tau_gen` use each arm's realised line mix, which is close to uniform but not
equal across doses. And they are per arm, with no per-line cells stored, so the
cells are rebuilt from rows.

**Which real wells are the truth (X5).**

| | Holdout wells (primary) | All wells (secondary) |
|---|---|---|
| Wells per (arm, line) | 1 (median) | 3 (median) |
| Trained on by a generator | never | 2 of the 3 |
| Arms with a well in every line | 7,451 (71%) | 10,432 (99.5%) |

- The all-wells oracle shares two thirds of its wells with the training set. A
  generator that copies noise from its training wells then picks a dose partly
  because of that noise and is scored on wells that contain it. "MSE
  decomposition" measured this copying: over all arms it cancels most of the
  model error in the reported MSE (−9.8 against 10.6). For a decision it would
  reward memorising.
- The holdout truth is noisy per arm (5 wells) but independent of every
  generator, so the mean over compounds of the chosen arm's holdout effect is
  an unbiased measure of what the choice is worth. Step A's readout made the
  same choice for its direction (`STEP_A.md` D10).
- **Eligible doses.** An arm needs a holdout well in every line. Which arms
  lack one is the base split's random draw, so this drops arms at random. A
  compound enters if it has at least 4 eligible doses, and every estimator
  chooses among those same doses:

  | | 6 doses | 5 | 4 | 3 or fewer (dropped) | enters |
  |---|---|---|---|---|---|
  | thinned compounds (848) | 306 | 223 | 103 | 216 | **632** |
  | not thinned (902) | 276 | 284 | 149 | 193 | **709** |

  Four eligible doses always include both dose halves, which is where the
  thinning acts (§7).
- **The secondary truth scores the same choices against the all-wells effect**
  (changed 2026-10-05, at implementation): the same compounds and the same
  eligible doses, so only the truth differs and the two can be compared line
  by line. The first draft said "all six doses of all 1,750 compounds", which
  would have changed the truth and the choice set at once. It carries the
  caveat above.

**The vehicle.** A constant offset per line, such as a generated vehicle that
sits off the real one, adds the same amount to every arm's effect. It cancels
in every argmax and ranking, so RxRx19a's `real` / `own` vehicle-anchor choice
does not change any decision here. The offset is recorded.

## 4. The two tasks

**Task A — choose the dose, along the compound's own signature.**

- Axis: $s_c$ is the direction of the compound's effect averaged over its
  eligible doses, from holdout wells only.
- $\mathrm{ATE}_A(c, d) = \langle \tau(c, d), s_c \rangle$: how far dose $d$
  moves expression along what the compound does.
- Decision: $\hat d_c = \arg\max_d \mathrm{ATE}_A(c, d)$, the most potent dose.
- RxRx19a analogue: `best_dose_log10_conc` and `rescue_best`.
- The axis and the truth share holdout wells, which adds a positive offset to
  the measured effect. In expectation it is the same for every dose of a
  compound, so it cancels between policies. This is why §5 reports values only
  as differences from a reference.

**Task B — choose compounds, along one shared axis.**

- Axis (X6): proliferation. $s$ puts weight $-1/\sqrt{14}$ on 14 cell-cycle
  landmark genes and 0 elsewhere: `TOP2A`, `CCNB1`, `CDK1`, `PCNA`, `AURKA`,
  `PLK1`, `BIRC5`, `CCNA2`, `CDC20`, `KIF20A`, `CCNE2`, `MCM3`, `E2F2`,
  `CDC25A`. All 14 are among the 978 landmarks.
- $\mathrm{ATE}_B(c, d) = \langle \tau(c, d), s \rangle$: larger means these
  genes go down more, read as more anti-proliferative.
- Decision: rank compounds by $\max_d \mathrm{ATE}_B(c, d)$ and take the top
  $k$, each at its chosen dose. The candidates are all 1,341 compounds that
  enter (632 thinned, 709 not), as in a real screen.
- RxRx19a analogue: the mock → infected axis, and hits against negatives.
- Why a fixed gene set and not a direction fitted to the data: it needs no
  wells, so it shares noise with nothing, and a reader can check it. It
  replaces the fitted direction sketched before this memo. A sanity check runs
  before any generator is read: on the oracle, the cytotoxic compounds of the
  curated panel (`IMPLEMENT.md` §3.7) that are in this population should rank
  near the top.

## 5. Policies, value and metrics

A **policy** maps each compound to a chosen dose (task A), or picks $k$
(compound, dose) pairs (task B). Its **value** $V$ is the mean holdout effect
of what it chose. Higher is better.

| Policy | Chooses from |
|---|---|
| `naive`, `conditional`, `dr`, `dr_p2` | the generator's $\mathrm{ATE}$, at γ = 0 and γ = 1 |
| pooled real mean | each arm's mean over its kept train wells, lines ignored (what `naive` imitates) |
| stratified real mean | per-(arm, line) means of the kept train wells, equal line weights (what the adjusted arms imitate) |
| full-data reference | the stratified real mean over the *unthinned* train wells |
| random | a uniformly random dose (task A) or random compounds (task B) |

Metrics:

- **Captured share** $= \dfrac{V(\text{policy}) - V(\text{random})}{V(\text{full-data reference}) - V(\text{random})}$.
  1 means as good as choosing with all the real training data, 0 means no
  better than chance. A generator can exceed 1, because it pools information
  across doses and compounds.
- **Net difference at γ = 1**, $V_\text{dr} - V_\text{conditional}$ with a
  paired error. This is what a user of the generators would experience. It
  splits exactly into two parts:

  $$\underbrace{V_\text{dr}^{\gamma=1} - V_\text{cond}^{\gamma=1}}_{\text{net}} =
  \underbrace{V_\text{dr}^{\gamma=0} - V_\text{cond}^{\gamma=0}}_{\text{noise cost of the weights}} +
  \underbrace{\bigl(V_\text{dr}^{\gamma=1} - V_\text{dr}^{\gamma=0}\bigr) - \bigl(V_\text{cond}^{\gamma=1} - V_\text{cond}^{\gamma=0}\bigr)}_{\text{gain from removing the confounding}}$$

  The first part answers "does the extra MSE hurt decisions"; the second,
  "does the lower bias help them".
- **Task B, top-k.** $V$ at k = 100 is the headline (X4); k = 25, 50 and 200
  are reported. The choice of k costs no running time (it is one sort). It
  trades resolution against selectivity: at k = 25 the error of $V$ is about
  twice that at k = 100, and at k = 200 the selection reaches 15% of the
  candidates.
  100 of 1,341 is a 7.5% hit rate.
- **Descriptive, not judged:** how often two policies choose the same dose; the
  overlap of two top-k sets; the Spearman correlation of the best-dose effect
  with the full-data reference's; the mean shift of the chosen dose, signed by
  the direction the thinning predicts (§7).

**Errors.** Task A: delete-one-compound jackknife (`contrast_stats.py`). Task
B: a bootstrap over compounds, 2,000 resamples, because a top-k mean is not
smooth in the data. Seeds are scored separately and their values averaged; the
seed spread is reported next to the clustered error. `naive` has one seed.

## 6. Criteria

Fixed here, before any decision number exists. Each is judged per task, with
the holdout truth. Task A is judged on the 632 thinned compounds that enter,
and task B on the screen over all 1,341 candidates.

| # | Criterion |
|---|---|
| D0 | **Power gate, before any generator is read.** The pooled real mean loses value from γ = 0 to γ = 1 by ≥ 3 SE. If not, the planted confounding does not move this decision even in the raw data, and the task is recorded as not testable |
| D1 | `naive` loses value from γ = 0 to γ = 1, by > 2 SE |
| D2 | **Testability.** `conditional` loses value from γ = 0 to γ = 1, by > 2 SE. If not, "`dr` against `conditional`" is recorded as not testable on this decision, not as a failure |
| D3 | **The gain.** The ADIGen arm's loss is smaller than `conditional`'s by > 2 paired SE (the second part of the split in §5). Judged for `dr` and `dr_p2` |
| D4 | **No decision cost from the weights.** At γ = 0, $V_\text{dr} - V_\text{conditional}$ is within 2 paired SE of zero or above it (the first part of the split) |
| D5 | **Control.** On the compounds that were not thinned, no arm's value changes from γ = 0 to γ = 1 by more than 2 SE. Task A: the 709 that enter. Task B: a screen over those 709 only, at k = 50 (the same 7% rate) |

The net difference at γ = 1 is reported with its error and is not a criterion:
it is the sum of D3's and D4's quantities.

## 7. What to expect

**Why the confounding should move a decision.** The thinning raised the G2
share by 0.145 in the low dose half and lowered it by 0.136 in the high half.
An estimator that ignores the line then has its low doses pulled toward G2's
response and its high doses toward G1's. Along any axis where the two groups
respond with different strength, that tilts the whole dose curve one way, so
the chosen dose moves in a predictable direction. In task B the same tilt
inflates the best-dose effect of thinned compounds whose lines disagree, which
promotes them in the ranking.

**The likely outcome, stated before the run.**

- `naive` learned 84% of the planted bias, so D1 should pass if D0 does.
- `conditional` kept 9% of it. A tilt that small may not change an argmax
  often enough to measure, so **D2 may well fail**, and the comparison the main
  author asked for would then be "not testable here".
- The holdout truth is noisier than the all-wells oracle. In step A's own
  readout it gave a similar estimate (+0.047 against +0.070) with 1.7–3 times
  the paired error, and the `dr` advantage fell from 3.7 SE to 1.5 SE.
- If D2 fails, the result still has content: D4 says whether the weights' extra
  error costs anything in decisions, and D1 shows that ignoring the confounder
  does. That is the main author's claim about MSE, tested directly.
- The oracle's own noise bounds everything. Many compounds have a flat or
  saturating dose curve, where the best and second-best dose are nearly equal
  and no estimator can be right or wrong by much. Value handles this (a wrong
  choice between equal doses costs nothing); agreement rates would not, which
  is why they are only descriptive.

## 8. Work items, job and disk

- [x] **N1** (code) `spec.py`: `PROLIFERATION_GENES` and `DECISION_K`, declared
      once.
      - Also `DECISION_MIN_DOSES`, `DECISION_K_CURVE` and `DECISION_K_CONTROL`.
- [ ] **N2** (code) `src/eval/decision_task.py`, numpy only, no PyTorch. It
      takes its run list from `step_a_verdict.json`, as `mse_decomposition`
      does, and reuses that module's per-(arm, line) cell means.
      1. Real cell means from `expr.npy` for holdout, kept-train (γ = 0 and
         γ = 1), unthinned-train and all wells.
      2. Generated cell means from each run's `row_mean`.
      3. The equal-weight effects, both axes, every policy of §5.
      4. Values, the split of §5, the criteria, the errors. Output:
         `runs/core5_24h/eval_artifacts/decision_task.json`.
      - Written and reviewed 2026-10-05. The box stays open for one review
        follow-up: its loader duplicates `mse_decomposition`'s (`IMPLEMENT.md`
        §5, "Decision task on step A", review table).
- [x] **N3** (code) `src/tests/check_decision.py`:
      - with each arm's realised line mix in place of the equal weights, the
        rebuilt real effect equals the stored oracle (as `mse_decomposition`
        checks, to 1e-3);
      - adding a constant per line to a generator's cells changes no decision;
      - on synthetic cells with a planted best dose, every policy recovers it,
        and a planted line-dependent thinning moves the pooled mean's choice
        and not the stratified mean's;
      - the split of §5 adds up to the net difference to float precision.
      - 33 synthetic checks (they run anywhere, no data) and 15 on the real
        data; the latter read no decision number of any generator.
- [x] **N4** (code) `scripts/decision_cpu.sub`, copied from
      `mse_decomposition_cpu.sub` (it sets `OPENBLAS_CORETYPE=Haswell`).
      - `MODE=check` runs the checks only; `VERDICT=` points both steps at
        another verdict.
- [x] **N5** (CPU) run N3, then N2 with D0 read first.
      - Smoke test: job 4002, 48 checks, 0 failures. Full run: job 4014, exit
        0, 4 min 40 s, peak 5.7 GB. Both on `bindel` at 7 GB, which fitted
        without preempting a job.
- [x] **N6** verdict against D0–D5 written into `IMPLEMENT.md` §5.

| Job | Type | Partition | Request | Time (estimate) |
|---|---|---|---|---|
| N5 | CPU | `ma` if it keeps 20% headroom, else `bindel` | 4 CPU, 15 GB | under 10 min |

- The estimate is `mse_decomposition`'s measured 3 min and 4.4 GB, which read
  the same files. This job reads `row_mean` (713 MB per run, one run at a
  time) where that one read `row_var`. Measured: 4 min 40 s and 5.7 GB.
- It is not run on the login node: it loads about 10 GB of arrays in total.
- Disk: one JSON, under 10 MB. Nothing is deleted or overwritten.

## 9. Limits

- **One thinning draw and one γ**, as in step A. The errors cover the
  compounds, not a second draw.
- **Two seeds** per adjusted arm and one for `naive`.
- **Task A weighs strong compounds more**, because the value is in z units
  along a unit vector. The captured share is a ratio of sums and has no scale,
  but a few strong compounds can carry it. The per-compound distribution is
  saved.
- **Task B's axis is a proxy.** Lower expression of 14 cell-cycle genes at 24 h
  is not a viability measurement. The task tests whether the generators agree
  with real wells on a fixed readout, not whether the readout is the right
  biology.
- **The equal-weight population needs every line.** About 50 arms have no
  well in some line and are dropped for every estimator.
- **P1 is not in this plan.** In step A it corrects each arm pooled over its
  lines (its group is the arm), so it has no per-line cells and does not fit
  the definition of §3. The stratified real mean of §5 is the non-generative
  baseline in its place.

## 10. Optional later

1. The same analysis with $P(\ell)$ set to each line's share of wells (X3).
2. A data-driven task-B axis (the first principal direction of the holdout
   effects) as a robustness check of X6.
3. A second tier seed, which would turn the errors into ones that cover the
   thinning draw.
4. P1 on per-(arm, line) cells, to put the AIPW baseline on the same footing.

## 11. Result (2026-10-05)

Full tables, findings and the review are in `IMPLEMENT.md` §5, "Decision task
on step A". On the primary truth:

| # | Task A (dose) | Task B (top-100) |
|---|---|---|
| D0, power gate | **FAIL**: +0.043 ± 0.081 (0.5 SE; needs 3) | **FAIL**: −0.029 ± 0.051 |
| D1–D3 | not judged | not judged |
| D4, no decision cost of the weights | PASS (`dr` +0.073 ± 0.036, `dr_p2` +0.061 ± 0.036) | PASS (+0.018 ± 0.018, +0.007 ± 0.013) |
| net at γ = 1, against `conditional` | `dr` +0.070 ± 0.059, `dr_p2` +0.131 ± 0.069 | −0.013 ± 0.017, −0.008 ± 0.019 |
| D5, control | PASS | PASS |

- **Neither task is testable.** The planted confounding does not move either
  decision even in the raw data, so there is nothing for `dr` to repair. §7
  expected D2 to be the weak point; the gate one level below it failed.
- **The weights' extra error costs nothing in decisions**, which is the main
  author's point about the MSE.
- **Neither ADIGen arm chooses measurably better or worse than
  `conditional`.** Every policy captures 96–104% of what the full data
  captures.
- The secondary truth gives the same verdict.
- Options for a decision task that can be tested are listed at the end of the
  `IMPLEMENT.md` entry. None is decided.
