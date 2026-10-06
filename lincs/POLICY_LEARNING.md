# Policy learning on LINCS (ADIGen-PO, Algorithm 1) — implementation plan

*Written 2026-10-05, after the decision task (`DECISION.md`) found both
table-argmax tasks untestable, and after reading the paper draft
(`ADIGen.pdf`, "Are Good Generators Good Decision-Makers?", Algorithm 1
ADIGen-PO, Algorithm 2 ADIGen, Theorems 2–4). Nothing in this plan is
implemented. Every box is unchecked and stays unchecked until its code has
been reviewed, as in `IMPLEMENT.md` §4.*

*Status 2026-10-06: implemented, reviewed, smoke-tested and run on both
instances. **Not testable: the gate L0 and the testability criterion L2 both
fail, and no arm chooses better than another** (§12). Tables, findings and
options are in `IMPLEMENT.md` §5, "Policy learning on step A: results".*

*Background: step A's design and results are in `STEP_A.md` and `IMPLEMENT.md`
§5; the decision-task machinery this plan reuses is in `DECISION.md` and
`src/eval/decision_task.py`.*

---

## 1. Verdict in one paragraph

**Algorithm 1 can be implemented on the step-A data, including the
retargeting round, and almost everything it needs exists: the generators, the
weighted risk, the per-row weight files, a sampler that takes arbitrary
(compound, dose, line) targets, and the value machinery of the decision
task.** What is missing is small and CPU-side (the tilt, the value estimate,
a weight-export mode for the retargeting law, a two-half split builder) plus
one GPU driver for rollouts. **What the existing step-A runs cannot provide is
a reason for `dr` to win.** Theorems 3–4 locate the gain in the transfer
factor, the ratio between where the learned policy acts and where the
logger put its data. Step A's thinning keeps at least one of the ~2 train
wells in every (arm, line) cell, so the logger never differs from the
balanced design by more than a factor of 2 anywhere, and the decision task
already showed that nothing moves. A policy-learning
experiment that can show the paper's effect needs **a new thinning with poor
overlap on the doses the policy wants** (§5), about 14 new training runs
(§8), and a utility whose optimum is not always the top dose (§4). With that
in place the experiment is a faithful instance of the paper's setting; without
it, it is the decision task again with a softmax.

## 2. The algorithm, in the paper's terms

Per cluster $Z = (X, A, Y, E)$: context, intervention, outcome, environment;
utility $g(Y)$; mean model $\mu_0(x, a) = E[g(Y(a)) \mid x, a]$; logging
policy $\pi_b(a \mid x)$. The objective is the KL-regularised value
$J_\beta(\pi) = V(\pi) - \beta\, E_X \mathrm{KL}(\pi \| \pi_b)$, whose
maximiser is the exponential tilt
$\pi^*_\beta(a \mid x) \propto \pi_b(a \mid x)\, e^{\mu_0(x, a) / \beta}$.

Algorithm 1 (ADIGen-PO):

1. **Round 1.** Split the data into halves $D_1, D_2$. On each half fit the
   logger $\hat\pi_b^{(j)}$ and a generator $\theta_j$ (ADIGen at the target
   law $\pi$). For each context draw $K$ candidate interventions from
   $\hat\pi_b^{(j)}$, roll out $L$ outcomes each, average $g$ into
   $\hat\mu_{ik}$, form the tilt weights $w^\beta_{ik}$, fit a policy by
   weighted maximum likelihood, and pick $\beta$ on a validation value
   estimate. The result is the pilot policy $\tilde\pi_j$.
2. **Round 2 (retargeting).** For each half $j$ and each $\lambda \in \Lambda$,
   retrain the generator at the mixture law
   $\pi_\lambda = \lambda \tilde\pi_{3-j} + (1 - \lambda)\hat\pi_b^{(3-j)}$,
   whose Riesz representer is
   $\alpha_\lambda = \lambda\, \tilde\pi_{3-j} / \hat\pi_b + (1 - \lambda)$;
   roll out again and refit the policy at the temperature chosen in round 1.
3. Pick $\lambda$ on the validation value and return the policy.

Why it should help (Theorems 2–4): the regret of a tilt built on $\hat\mu$ is
$\lesssim \frac{1}{\beta}\,\rho_\nu\,\varepsilon_\text{gen}(\nu) + \beta\,
\varepsilon_\text{reg}$, where $\nu$ is the law the generator was trained at,
$\varepsilon_\text{gen}(\nu)$ its error under that law (carrying a weight
cost $1 + \chi^2(\nu \| \pi_b)$), and the transfer factor
$\rho_\nu = \|\hat\pi_\beta / \nu\|_\infty$ inflates it by how much the
policy concentrates where $\nu$ is thin. The conditional generator has
$\nu = \pi_b$ and pays the full $\rho$; a generator at a fixed interventional
law pays the weight cost and still the worst-case $\rho$; the mixture
$\pi_\lambda$ caps the transfer factor at $1/\lambda$ for a weight cost
$\lambda^2 \chi^2$, and Theorem 4 says the gain over the conditional
generator is at least $\tfrac{\sqrt\rho}{2}\sqrt{1 + a/d}$: large when
overlap is poor and when the generator is capacity-limited.

## 3. Mapping to the step-A data

| Paper | LINCS (`core5_24h`, step A) |
|---|---|
| cluster, $m$ | one well, $m = 1$ (no interference) |
| context $X$ | (compound, cell line): 1,750 × 5 = 8,750 contexts |
| intervention $A$ | the dose: 6 levels per compound (`dose_level`; the generator sees `log10_conc` as a continuous scalar) |
| outcome $Y$ | 978-gene z-scored expression |
| environment $E$ | plate (`det_plate`); used only by the invariance penalty, which no LINCS run has used ($\lambda_\text{inv} = 0$) |
| utility $g$ | a scalar readout of $Y$ (§4) |
| $\mu_0(x, a)$ | the mean of $g$ over the real wells of (compound, line, dose), minus the line's vehicle (a per-context constant that cancels in every tilt) |
| logger $\pi_b(a \mid x)$ | the thinning: which doses of a (compound, line) keep their train wells. Known exactly from the design (`tier_meta.json` holds every well's $\pi$), and estimable from counts |
| target law of `dr` | the balanced design: every dose equally likely in every context (the counts weights $n_\text{unthinned}/n_\text{kept}$) |
| `conditional` | the generator at $\nu = \pi_b$ (unweighted) |
| halves $D_1, D_2$ | a split of the kept train wells, stratified on (compound, line, dose) (§6) |
| $K$ candidates | all 6 doses, enumerated: the expectation over $\hat\pi_b$ is exact |
| $L$ rollouts | 8 per (context, dose), as in the paper |
| policy class $\Pi$ | tabular: one distribution over 6 doses per context, so the weighted-MLE step of line 11 returns the tilt itself (§7) |
| validation value $\hat V_\text{val}$ | the holdout wells of the validation compounds (never thinned, never trained on) |
| test regret | the holdout wells of the test compounds, reported as value against the full-data reference and a random dose, as in `DECISION.md` §5 |

Two things in this mapping differ from the paper and should be said in the
write-up:

- **The codebase's `dr` is the reweighted risk, not the full doubly-robust
  loss.** `train_diffusion --dr_mode weighted` minimises
  $\alpha(A, X)\,\ell(\theta; Y, A)$ with the counts weights. Eq. 5's pilot
  generator $\psi$ and its plug-in term are not implemented, and neither is the
  invariance penalty. The retargeting mechanism (train the generator where the
  policy acts) does not depend on either, but the rate statements of Theorem 1
  do.
  - **Cost of the full loss** (estimated, not measured): the pilots are free,
    because round 1's `conditional` generator of the other half is the
    cross-fitted pilot of Algorithm 2; pilot sampling is one 10-minute pass
    and ~1.8 GB per instance; the trainer needs a loader for each row's pilot
    samples, a three-term loss and the target-dose draw (about 200–300 lines,
    a GPU smoke test and a review, about 3 working days); every weighted run
    then costs 2.5–3× the training time (about 25 GPU-hours per instance
    instead of 8).
  - **Not needed for this plan.** The counts weights are exact post-stratified
    ratios, so the reweighted risk is already unbiased for the target risk
    (step A's `dr` hit zero bias), and retargeting is a choice of weights,
    which the trainer takes as a file. Where the full loss could matter is
    variance: the weights would multiply only the residual against the pilot,
    which may reduce the +17% noise cost step A measured. Decision: run with
    the reweighted risk; add the full loss for the retargeted runs only if L5
    fails or the main author wants the LINCS arm to match Eq. 5 for the
    paper.
- **The logger acts through the thinning of a balanced design**, so its
  support is the design's: a dose the thinning removed entirely from a context
  has $\pi_b = 0$ there, and an exponential tilt of $\pi_b$ can never choose
  it. "Low support" must therefore mean "few wells", not "no wells" (§5).

## 4. The utility

$g$ must be a scalar per well, bounded, and defined for a generated well. Two
candidates, both already in `decision_task.py`:

| | task A: the compound's own signature | task B: proliferation |
|---|---|---|
| $g(y)$ | $\langle y, s_c \rangle$, $s_c$ the unit direction of compound $c$'s holdout effect (per context) | $\langle y, s \rangle$, $s$ = $-1/\sqrt{14}$ on `spec.PROLIFERATION_GENES` |
| meaning | the most potent dose | the most anti-proliferative dose |
| top dose is the best dose (all-wells oracle, pooled over lines) | 45% of compounds (62% of responders) | 50% |

A utility whose optimum is always the top dose makes the policy problem
trivial and makes "low support on the top dose" the whole experiment. Half of
the compounds have an interior optimum under either utility, which is enough,
but a utility with a cost is the paper's own setting (its simulation uses a
quadratic reward) and is worth having as a third option:
$g(y) = \langle y, s_c \rangle - \kappa \|y - \mu(0, \text{line})\|$, efficacy
minus a magnitude penalty read as toxicity; at $\kappa = 0.5$ the top dose is
best for 35% of compounds. **Open decision for the main author (Q1):** which
utility is the headline. Proposed: task A, the most potent dose, with no
extra parameter.

Only one utility is judged. The thinning, the halves and the round-1 runs do
not depend on it, and if the rollout driver saves the generated gene vectors
(about 0.8 GB per generator) every utility is read off them on CPU, so round
1 is reported on all three descriptively. The pilot policy does depend on it,
so each utility needs its own round-2 runs (8 per instance, about 3.5
GPU-hours). Round 2 runs for the headline only; a second utility is added
only if the paper should show the result on two readouts.

## 5. The logger: why a new thinning is needed, and which

**The structural limit.** 93% of (arm, line) cells hold exactly 2 train
wells, 99.4% hold 2 or more, and 6% hold 3 or more. Step A's positivity rule
keeps at least one well in every cell, so a 2-well cell keeps 1 or 2: the
logger can never give a dose less than half its balanced share within a
context. $\chi^2(\text{balanced} \| \pi_b)$ is therefore tiny, and the
transfer factor of any policy against the conditional generator exceeds the
one against the balanced law by at most 1.5× (a point-mass policy has
$\rho = 6$ against a uniform logger and at most 9 against step A's). This is
why `conditional` removed 91% of the planted bias and why the decision
task's D0 failed. No choice of $\gamma$ or `keep_frac` changes it.

**The proposed "PL tier": positivity at the dose half.** Move the positivity
cell from (compound, `dose_level`, line) to (compound, `dose_half`, line), as
step C's key already is for `syn_c`. A context's disfavoured half (6 wells
over 3 doses) may then keep a single well, and its favoured half all 6:

| | step A (`keep_frac` 0.75, $\gamma = 1$) | PL tier (proposed) |
|---|---|---|
| positivity cell | (compound, dose, line) | (compound, dose half, line) |
| kept share, favoured : disfavoured half | about 1 : 0.5 (bounded by 2 : 1) | 1 : 1/6, calibrated (`keep_frac` ≈ 0.6 realised) |
| shift of the G2 share by dose half | +0.15 / −0.13 | about +0.4 / −0.4 |
| $\pi_b$ of a disfavoured dose | ≥ 1/9 | about 1/20 (the half keeps 1 of the context's 7 kept wells, spread over 3 doses) |
| transfer factor of a point-mass policy on it, against $\pi_b$ | ≤ 9 | up to about 20 |
| counts weights | per (arm, line) | per (compound, half, line); finite by construction |

- The selection score is unchanged: $z = z_\text{dose} \cdot z_C$, so G2 lines
  keep the low half and G1 lines the high half (or the reverse; the sign is
  the design's). The policy's optimal dose is the top dose for about half of
  the compounds, so in the three G1 lines about half of the contexts have
  their optimum in the thin half. That is the "held-out, low-overlap region"
  the paper evaluates on.
- The logger is known exactly: $\pi_b(\text{dose} \mid x) = p_\text{half}(x) / 3$,
  with $p_\text{half}$ the half's share of the context's kept wells.
  Both the oracle logger (as the paper's simulation uses) and the
  counts-estimated one are available; the plan uses the counts estimate for
  the method and reports the oracle as a check.
- The counts weights at the half key equal the design weights up to the
  realised draw, as in step C ($+0.84$ correlation there).
- `build_tiered_split` needs a new positivity key for `cell_id` (a
  `--positivity_key half` switch, not a change of the existing key, so step A's
  instances stay reproducible) and a calibration that targets the half-level
  shares. The $\gamma = 0$ control instance is kept: it is what separates the
  weights' noise cost from the confounding, as D4 did.
- **Gate L0, before any GPU job**, read from the real train wells exactly as
  S0 and D0 were: the tilt of the *unthinned* real means at a mid-grid $\beta$
  must put at least a third of its mass on the thin half over thinned
  contexts, and the tilt of the *kept* real means (lines ignored) must lose
  value against it by ≥ 3 SE. If the first fails, the optimum is not where
  the data are thin and the experiment cannot show a transfer effect; if the
  second fails, the thinning does not bite even at the data level.

**Open decision (Q2):** the kept share of the disfavoured half (1/6 as
proposed, or 2/6), and whether unthinned compounds stay as the no-confounding
control (proposed: yes, as in step A; 848 thinned, 902 untouched).

## 6. Cross-fitting halves

Algorithm 1 trains one generator per half and retargets each half's generator
toward the pilot policy of the other half. On LINCS:

- Split the kept train wells of the PL tier into two halves, stratified on
  (compound, line, dose) so both halves cover every context. A cell that keeps
  one well goes to one half only; its context is still covered in the other
  half through its other doses, and the policy only needs $\hat\mu$ at every
  (context, dose), which the generator provides by conditioning.
- Each half's run dir is a split dir of its own (`splits.json` with the
  halved `train_idx`, the base z-scale copied under the new fingerprint, the
  weight files recomputed on the half). The trainer verifies all of this
  already.
- Cost: two runs at half the rows per arm, about the same GPU time as one
  full run (training time scales with rows; step A's full runs took 47–52
  min).
- A cheaper variant with one generator and no cross-fitting (the pilot policy
  and the retargeting law from the same generator) is the fallback if time
  runs out; it loses the guard against retargeting toward the generator's own
  noise.

**Open decision (Q3):** halves by well (proposed) or by plate. By plate is
closer to "independent clusters" if plate effects are a worry, but every
(compound, line) sits on only 2–3 plates, so a by-plate split would leave
many contexts in one half only.

## 7. Round 1 on LINCS, step by step

1. **Logger.** $\hat\pi_b^{(j)}(\text{dose} \mid x)$ from the half's kept-well
   counts at the half level, uniform within the half (the thinning's own
   form). Contexts of unthinned compounds: uniform.
2. **Generators.** Per half: `conditional` and `dr` (counts weights at the
   half key, P2 group normalisation within the context, cap 5). `naive`
   (blind to the line) on one half only, as the bias anchor.
3. **Rollouts.** For every context and every dose: $L = 8$ samples from the
   half's generator, at the context's own codes (`SampleTarget` takes any
   `(compound_idx, log10_conc, is_control, context)`; the plate code is taken
   from one of the context's real rows and is ignored by the step-A
   conditioning, which sees `cell_id` only). $\hat\mu(x, a)$ = the mean of
   $g$ over the 8 samples, minus the generated vehicle mean of the line. Size:
   8,750 × 6 × 8 = 420k samples per generator, about 5 min on an A6000
   (step A's scoring drew 2.9M in 33 min).
4. **Tilt.** $\hat\pi_\beta(a \mid x) \propto \hat\pi_b(a \mid x)\, e^{\hat\mu(x, a) / \beta}$,
   $\beta \in B$. With a tabular class this *is* the weighted-MLE solution of
   line 11, so no policy network is fitted. $B$ spans the scale of $\hat\mu$:
   in z units along a unit vector the dose gaps are 0.1–3, so
   $B = \{0.05, 0.1, 0.2, 0.5, 1, 2\}$ (the paper's selected values are
   0.05–0.2 on a reward of order 1).
5. **Validation.** $\hat V_\text{val}(\pi) = $ mean over validation contexts of
   $\sum_a \pi(a \mid x)\, \hat\mu_\text{holdout}(x, a)$, with
   $\hat\mu_\text{holdout}$ the holdout well of (context, dose) — one well per
   cell, unbiased, never thinned, never trained on. Contexts are split by
   compound: a validation set of compounds for choosing $\beta$ and $\lambda$,
   a test set for reporting; proposed 40 / 60 of the thinned compounds, drawn
   once by seed and recorded. Unthinned compounds form the control, as in D5.
6. **Pilot policy** $\tilde\pi_j$ at the selected $\hat\beta_j$.

The $\gamma = 0$ instance runs the same steps, which gives every number its
no-confounding counterpart.

## 8. Round 2: retargeting on LINCS

1. **Retargeting law.** For half $j$, $\pi_\lambda = \lambda \tilde\pi_{3-j} + (1 - \lambda)\hat\pi_b^{(3-j)}$,
   $\Lambda = \{0.25, 0.5, 0.75, 1\}$ (the paper's grid).
2. **Weights.** On each kept treated row of half $j$,
   $w_i = \lambda\, \tilde\pi_{3-j}(a_i \mid x_i) / \hat\pi_b^{(3-j)}(a_i \mid x_i) + (1 - \lambda)$;
   vehicles keep weight 1. A new `export_urr_weights --mode retarget` writes
   it from the pilot policy's table and the logger's counts; the trainer's
   group normalisation needs a `context` group so every (compound, line)
   keeps its mass (the weights average to 1 under the logger within a context,
   so this is a small correction, not a change of target).
3. **Training.** `--dr_mode weighted --dr_weights_file dr_weights_retarget_l<λ>.npz`
   on half $j$: 2 halves × 4 values of $\lambda$ = 8 runs. Algorithm 2 trains
   from scratch; fine-tuning from the round-1 checkpoint is a cheaper
   deviation to keep in reserve.
4. **Rollouts and policy** as in round 1, at $\hat\beta_{3-j}$.
5. **Select $\lambda$** on the validation value; report on the test
   compounds.

**Arms compared**, as in the paper's experiments:

| arm | generator | law $\nu$ |
|---|---|---|
| Policy (Conditional) | `conditional` | $\pi_b$ |
| Policy (ADIGen) | `dr` | balanced design |
| Policy (Retargeted ADIGen) | `dr` retargeted | $\pi_{\hat\lambda}$ |
| Policy (Naive) | `naive` | $\pi_b$, blind to the line |
| real-data tilts | kept wells (pooled / stratified) and unthinned wells | the non-generative anchors, as in `DECISION.md` §5 |

A "retargeted conditional" (the mixture weights applied with no balanced
target, i.e. $\lambda$-mixing toward the conditional's own pilot) is a cheap
extra arm that isolates retargeting from debiasing; it is not in the paper
and is optional.

**Budget.** PL tier and halves: CPU, minutes. Round 1: 2 × 2 + 1 runs at half
rows, about 2.5 GPU-hours; rollouts 5 × 5 min. Round 2: 8 runs, about 3.5
GPU-hours; rollouts 8 × 5 min. With the $\gamma = 0$ control: double. **Total
about 13 GPU-hours on `zabih`**, one working day on six A6000s with the
chains, plus under 1 hour of CPU. A second thinning seed, which a positive
result needs before it is reported, doubles it again.

## 9. Metrics and criteria

The paper reports regret, model RMSE at $\pi^*$ and globally, signed
self-evaluation error, and the paired improvement from retargeting. On LINCS,
with the holdout truth $\hat\mu_h$:

- **Value** $\hat V_\text{test}(\pi)$ and its gain over a random dose, as a
  share of the full-data reference's gain (`DECISION.md` §5). Regret against
  the reference is the reference's value minus the policy's; the true
  $\pi^*_\beta$ is unknown and the same for every arm, so paired differences
  between arms are differences in regret.
- **Model error where the policy acts:** $\sum_a \hat\pi_\beta(a \mid x)(\hat\mu - \hat\mu_h)^2$,
  and globally over all doses; both minus the holdout noise term, which the
  split-half machinery of the MSE decomposition supplies.
- **Self-evaluation error:** $\hat V_\text{model}(\hat\pi) - \hat V_\text{test}(\hat\pi)$,
  the generator's optimism about its own policy. The paper's third panel.
- **Support of the chosen doses:** the mass $\hat\pi_\beta$ puts on the thin
  half, and $\hat\rho = \max_x \max_a \hat\pi_\beta / \hat\pi_b$ — the realised
  transfer factor, which the whole argument rests on.
- Errors: delete-one-compound jackknife for means; the two training halves
  and the two thinning instances are reported separately; no seed averaging
  hides a disagreement between halves.

Criteria, fixed before any number is read (the pattern of S0–S5 and D0–D5):

| # | Criterion |
|---|---|
| L0 | Power gate on the real train wells (§5): the optimum is in the thin half for ≥ 1/3 of the thinned contexts' tilt mass, and the kept-well tilt loses ≥ 3 SE of value against the unthinned one |
| L1 | Policy (Naive) loses value from $\gamma = 0$ to the PL instance, > 2 SE |
| L2 | Testability: Policy (Conditional) loses value, > 2 SE. If not, "ADIGen vs conditional" is not testable on this decision |
| L3 | Policy (ADIGen) loses less than Policy (Conditional), > 2 paired SE |
| L4 | Policy (Retargeted ADIGen) loses less than Policy (ADIGen), > 2 paired SE, at the validation-selected $\lambda$ |
| L5 | No value cost of the weights at $\gamma = 0$ (within 2 SE), for `dr` and the retargeted arm |
| L6 | Control: unthinned contexts unchanged between instances, every arm |
| L7 | The selected $\lambda$ is interior or 1, not forced; the realised $\hat\rho$ of the conditional policy exceeds that of the retargeted one |

L3 is the user's target; L4 is the paper's. Both are judged only if L2
passes, as D3 was.

## 10. What to expect, honestly

- **The gain has to come from the thin half.** Theorem 4's lower bound is
  $\tfrac{\sqrt\rho}{2}\sqrt{1 + a/d}$ and `conditional` only loses where
  $\hat\pi_\beta$ concentrates on doses it rarely saw. With the PL tier,
  $\rho$ is up to about 20 against $\pi_b$ and 6 against the balanced law;
  in the paper's simulation $\log_{10}\rho$ runs from 6 to 8 at $m = 4$.
  LINCS cannot approach that, so the effect, if present, will be modest and
  needs the paired errors over about 1,900 thinned test contexts (60% of 632
  compounds × 5 lines) to show.
- **Step A says the generator is capacity-limited on exactly the right
  quantity.** It learned 12% of the line contrast, so its per-line dose
  curves are mostly the pooled curve. Retargeting reallocates that limited
  capacity toward the (context, dose) region the pilot policy favours, which
  is the mechanism Theorem 4 rewards. Whether an MLP-B with 500 epochs
  reallocates enough is the empirical question.
- **The exponential tilt is conservative by construction.** It multiplies
  the logger, so where $\hat\pi_b$ is 1/20 the policy needs $\hat\mu$ to be
  about $\beta \log 20 \approx 3\beta$ higher than at a favoured dose to
  prefer it. At the paper's $\beta$ (0.05–0.2 on rewards of order 1) that is
  reachable; at $\beta \geq 1$ the policy barely moves from the logger and
  nothing is tested. The validation choice of $\beta$ is therefore itself a
  result to report.
- **A likely failure mode is L2, again.** If the conditional generator's
  dose curves in the thin half are already close to the truth (the compound
  embedding plus a continuous dose makes extrapolation easy), its policy
  does not suffer and there is nothing to repair. The decision task's lesson
  applies: build the gate (L0) from the raw data first, and read L2 before
  investing in round 2.
- **Noise.** The holdout truth is one well per (context, dose). Value is a
  mean over contexts, so the error is driven by the spread between compounds
  (0.2 on a gain of 4 in the decision task); the paired errors between arms
  were 0.02–0.07. A retargeting gain smaller than about 0.05 in value will
  not be seen.
- **Scope that is not in this plan:** a parametric policy class (needed only
  if contexts were held out, which they are not), the invariance penalty, the
  pilot-generator terms of Eq. 5, and a compound-selection variant of the
  action (the logger is uniform over compounds, so overlap is not the issue
  there).

## 11. Work items

- [x] **P0** (decisions, user, 2026-10-06) Q1: the signature utility is the
      headline; the other two are read descriptively in round 1 and left for
      a later round 2. Q2: the disfavoured half keeps 1 of 6; unthinned
      compounds stay the control. Q3: halves by well. The weighted risk as it
      stands; the full loss of Eq. 5 is an optional follow-up or ablation.
      Grids $B = \{0.05, 0.1, 0.2, 0.5, 1, 2\}$, $\Lambda = \{0.25, 0.5, 0.75, 1\}$;
      validation / test = 40 / 60 of the thinned compounds by seed. Target:
      the whole experiment within about 12 hours (soft).
- [x] **P1** (code, CPU) `build_tiered_split --positivity_key half` for
      `cell_id`; the half-level calibration; `tier_meta` records the key.
      - Written 2026-10-06. The key is `cell_id@half` in `splits.POSITIVITY_KEYS`;
        `tier_cell_key(tier)` resolves it for the weight export and the trainer's
        group normalisation, so step A's instances and names are unchanged. The
        dir tag is `_pk-half`. Checks are in `check_policy` (P10).
- [x] **P2** (code, CPU) L0 from the real wells: `src.policy.learn --stage gate`
      (the gate shares the frame with the other stages). **Gate L0 is read
      here; nothing below runs if it fails.**
- [x] **P3** (code, CPU) `src/data/split_halves.py`: two half split dirs per
      instance (train halves, base z-scale, counts weights at the half key
      recomputed per half, fingerprints).
      - The weights are the parent's, subset to the half's rows (the halving is
        stratified by cell, so the cell ratios are preserved; the group
        normalisation re-balances exactly in any case).
- [x] **P4** (GPU) round-1 training: `conditional`, `dr` per half, `naive`
      on one; both instances. Chained `.sub` as `phase6_arms.sh` does.
      - Jobs 8572–8581 and 8601–8610, about 21.5 min per run and 5.5 min per
        rollout job.
- [x] **P5** (code, GPU) `src/policy/rollouts.py`: every (context, dose) × $L$
      from a checkpoint, writing the mean over $L$ of the 978-gene sample (so
      any linear utility is read later), the per-sample vehicle-centred norm,
      and the generated vehicle per line. Reuses `generation.generate_batch`,
      `SampleTarget`, `load_arm`, `check_arm_against_data`. The targets' layout
      is `src/policy/targets.py`, shared with the numpy side.
- [x] **P6** (code, CPU) `src/policy/learn.py`: the frame (real wells, the
      utilities A and B, the logger from counts and the design's own), the
      tilt over $B$, $\hat V_\text{val}$ / $\hat V_\text{test}$ from holdout
      wells, the realised transfer factor, self-evaluation error, model error
      where the policy acts; stages gate / round1 / pilot / round2 / report.
      Reuses `decision_task.Est` and the jackknife. The toxicity utility is
      not implemented (deferred with the other round-2 utilities).
- [x] **P7** (code, CPU) the retargeting weights are written by the pilot stage
      (`dr_weights_retarget_l<λ>.npz` in the other half's dir, mode
      `retarget`), not by `export_urr_weights`; `weight_norm` gained the
      `context` group and the trainer `--dr_weight_norm context`.
- [x] **P8** (GPU) round-2 training, 8 runs per instance, chained to P5
      rollouts.
      - Jobs 8584–8599 and 8613–8628. The whole of both chains took 2 h 20 min
        wall and about 12 GPU-hours.
- [ ] **P9** (code, CPU) `src.policy.learn --stage report`: L0–L7, the value
      table per instance; verdict into `IMPLEMENT.md` §5.
      - Written; the verdict is recorded (job 8632). The box stays open for
        one follow-up: L7 compares the policy's transfer factor against the
        logger, where the theorem's quantity is the ratio against the training
        law π_λ; and the paired differences live in an unreviewed follow-up
        script (`src/policy/paired_followup.py`).
- [x] **P10** (CPU) `src/tests/check_policy.py`: synthetic checks of the tilt,
      the logger, the value, the retargeting weights and the halving (21), and
      real-data checks of the tier, the halves and the frame (28 on the limit
      build).
- [ ] **P11** (GPU) a second thinning seed for a positive result.

Order: P0 → P1 → P2 (gate) → P3 → P4 → P5/P6 (round-1 verdict, L1–L2 read
here) → P7 → P8 → P5/P6 again → P9. Round 2 is launched only if L2 passes on
round 1; otherwise the result is recorded as not testable and the thinning or
the utility is re-decided before any further GPU time.

**As implemented (2026-10-06), under the 12-hour target:** the whole chain of
one instance is submitted at once by `scripts/policy_arms.sh` with job
dependencies (round-1 training → rollouts → the round-1 and pilot stages →
round-2 training → rollouts → the round-2 stage), so round 2 does not wait for
a human reading of L2. The verdict applies L2 afterwards: if it fails, L3 and
L4 are reported, not judged. The $\gamma = 0$ control runs the same chain
(`FORCE=1`, since its own gate is not meant to pass). `scripts/policy_cpu.sub`
builds the tiers, the halves, runs the checks and the gate;
`scripts/policy_smoke.sub` runs the chain end to end on the limit build with
tiny legs.

**Review** (automated `code-review`, high, 2026-10-06, before any job): ten
findings, all fixed before the smoke: `oracle_h` chose its temperature on the
wrong logger; the (context, dose) half table assumed every target had a table
row; retargeting weights could underflow to zero at $\lambda = 1$ and stop the
trainer's normalisation (now floored at $10^{-6}$); an undefined error was
scored FAIL by L1–L4 and PASS by L5–L6 (now "not judged" everywhere); the
report ignored `--prefix`; the launcher did not pass the checkpoint epoch to
the rollouts; the pilot stage recomputed round 1 instead of reading its
record; a duplicated temperature selector; dead code; a check that could not
fail. The fixes were not reviewed a second time.

## 12. Result of the gate and the launch (2026-10-06)

**Gate L0: FAIL on the full build.** The reference tilt puts 0.457 of its
mass on the thin half (the optimum is often there), but the kept-pooled
policy does not lose value against it (−0.53 ± 0.16, the wrong sign), and
the paired difference of the same policy between the instances is −0.05 ±
0.12. The logger's prior costs nothing (−0.014 ± 0.011). Pooling the five
lines beats the per-line reference by 0.9. Full table in `IMPLEMENT.md` §5,
"Policy learning on step A". The reading: on LINCS the cell line is too weak
a confounder, relative to the dose effect, for a line-based logger to bias a
dose policy, even at a sixfold overlap gap; §10 named this as the likely
failure mode.

**Decision (user): launch the full chain anyway, both instances
(`FORCE=1`).** Jobs 8572–8629 on `zabih` (training and rollouts) and
`bindel` (stages), the report queued behind both round-2 stages. The
verdict applies L2 afterwards; if `conditional` does not lose value, L3 and
L4 are reported, not judged.

**Result (2026-10-06).** All 59 jobs completed. On the 2,017 thinned test
contexts:

| | γ = 0 | γ = 3 |
|---|---|---|
| value of `conditional` | 13.91 ± 0.41 | 13.90 ± 0.41 |
| `dr` − `conditional` (paired) | −0.031 ± 0.050 | −0.074 ± 0.066 |
| retargeted (λ = 0.25) − `dr` | +0.033 ± 0.053 | +0.010 ± 0.060 |
| model error on thin-half doses, `dr` − `conditional` | +0.13 ± 0.21 | +3.85 ± 0.92 |
| the same, retargeted − `dr` | +0.65 ± 0.39 | −2.72 ± 1.00 |

- **L2 fails** (`conditional` loses +0.009 ± 0.075), so L3 and L4 are not
  judged; L5 and L6 pass.
- **No arm chooses better than another**: the generators are within 2% of
  each other in value at both γ.
- **The mechanism shows in model error only**: the balanced-law weights cost
  `dr` about 6% accuracy at γ = 3, and retargeting at λ = 0.25 recovers most
  of it. Validation picks λ = 0.25 every time; λ = 1 hurts.
- **Why**: the cell line barely changes which dose is best, so a logger that
  depends on the line is nearly ignorable for a dose decision.
- Options are at the end of the `IMPLEMENT.md` entry; none is decided.

## 13. Jobs, partitions and disk

Requests follow `.claude/rules/slurm_partition_priority.md`. Availability is
rechecked before every submission.

| Job | Type | Partition | Request | Count × time (estimate) |
|---|---|---|---|---|
| P1–P3, P6, P7, P9, P10 | CPU | `ma` under the 20% headroom rule, else `bindel` | 4 CPU, 8–15 GB | each under 10 min |
| P4 | GPU | `zabih` | 1 GPU, 4 CPU, 16 GB | 5 runs × 25 min per instance |
| P5 | GPU | `zabih` | same | 1 × 5 min per generator |
| P8 | GPU | `zabih` | same | 8 runs × 25 min per instance |

- Disk: each half-run saves its final checkpoint (0.3 GB at half rows) and its
  $\hat\mu$ tables (8,750 × 6 × a few utilities, negligible). About 26 runs ×
  0.3 GB ≈ 8 GB per thinning seed. `/share/zabih` had 111 GB free on
  2026-10-05 before step A's 50 GB; the intermediate-checkpoint list of
  `STEP_A.md` §7 (140 GB) is still the user's to delete.
- Timing is scaled from step A's measured 47–52 min per full run; it is not
  measured for half runs.

## 14. The semi-synthetic effect modifier (2026-10-06)

*Decided by the user after §12: plant a line-group × dose-half effect of
known size so that the best dose depends on the line group in a way the
logger hides, then rerun the chain. Designed and fixed before any number
was read.*

**Why it can work where the real data could not.** At γ = 3 the thin half
of each context holds one well per line. An effect that lives only there,
with a compound-specific sign, can only be learned from those wells.
`conditional` fits them with the weight of one well in six and shrinks toward
the pooled dose curve; `dr` weights them ×6 and the retargeting law
concentrates on the doses the pilot policy wants. The old experiment had the
overlap gap but no effect to find.

**Injection** (`src/data/inject_modifier.py`, an injected copy of the build):
for compound $c$, line group $g$ and an independent sign
$m[c, g] \in \{-1, +1\}$ (seeded), every treated well of $c$ in a line of
group $g$ whose dose half is the group's thin half (G2: high, G1: low) gets

$$z \leftarrow z + \beta\, m[c, g]\, s_c,$$

with $s_c$ the compound's **uninjected** holdout signature
(`learn.compound_signatures`), i.e. utility A's own axis, so the utility of
an injected well moves by exactly $\pm\beta$. Vehicles untouched; every
compound injected (the unthinned ones, with ~3 wells per cell, are the
control: the same effect with overlap). Independent signs per (compound,
group): nothing is shared across compounds (the step-C lesson), and the
favoured half carries no information about the thin half. The shift is
applied in raw units under the base z-scale, so it is exact in every copied
split dir; the table, splits, tiers, halves and weights are byte copies.

**β rule.** $\beta = 1.5\,\sigma_w$, with $\sigma_w$ the per-well sd of
utility A within (compound, line, dose) cells on the real build: a single
thin well then gets the sign right about 93% of the time, with a noisy
magnitude, so shrinkage matters. Ladder $\{1.5, 2, 3\} \times \sigma_w$,
stepping up only if the gate fails; each step is a new build dir.

**Gate L0′** (real wells, before any GPU job; thinned contexts, β = 0.2
tilt): (a) the reference tilt puts ≥ 1/3 of its mass on the thin half;
(b) it beats the kept-pooled (line-blind) tilt by ≥ 3 SE; (c) the kept
per-line memoriser beats the kept-pooled one by ≥ 3 SE (the thin wells carry
usable signal).

**Arms and budget.** γ = 3: the full chain (13 runs). γ = 0: round 1 only
(5 runs; L1–L3 and L5's `dr` part; its retargeted part is not judged).
About 7.5 GPU-hours, ~2 h wall. One thinning draw, one seed per half.

**Criteria.** L0′ and L1–L7 of §9; L3 is the target, L4 the paper's, both
judged only if L2 passes. L7 is re-specified: the retargeted policy's transfer
factor against its *training law* $\pi_\lambda$ (bounded by $1/\lambda$)
must be below the conditional policy's against its training law, the logger.

**The planted share.** For each injected generator, paired with the
uninjected run of the same name (same rows, seeds and noise seeds):
$\hat\lambda_\text{bump}$ = mean over thin-half cells of
$m[c, g]\,(\hat\mu_\text{inj} - \hat\mu_\text{uninj}) / \beta$, and the
same over the favoured half (expected 0). This is the step-C2 λ̂ for the
modifier: it says directly whether `dr` learns more of the planted effect
than `conditional`, whatever the policies then do with it.

**Checks:** `src/tests/check_injection.py` (the copy, the switch, the exact
z-shift, the frame's truth moving by exactly ±β on thin cells) and
`check_policy` on the copy; the chain's smoke on an injected limit build.

**Ladder and launch (2026-10-06).** On the full build σ_w = 4.643 (68,944
degrees of freedom); 84,573 of the 172,058 treated rows are injected.

| step | β | (a) thin-half mass | (b) reference − kept-pooled | (c) kept per-line − kept-pooled | gate |
|---|---|---|---|---|---|
| 1.5 σ_w (`data/core5_24h_inj`) | 6.96 | 0.495 | +0.87 ± 0.16 (5.5 SE) | +0.25 ± 0.17 (1.5 SE) | FAIL on (c) |
| 2 σ_w (`data/core5_24h_inj-m2`) | 9.29 | 0.500 | +1.50 ± 0.17 (9.0 SE) | +0.96 ± 0.18 (5.3 SE) | **PASS** |

The chain was launched on the 2 σ_w build: γ = 3 in full, γ = 0 round 1
only, the report behind both. The injection checks (14) and the policy
checks (47 on the copy) pass on both builds; the chain's smoke on an injected
limit build is green (job 11958).

**Result (2026-10-06, jobs 12052–12103, all exit 0, 1 h 36 min wall).** Full
tables and ten findings in `IMPLEMENT.md` §5, "Policy learning with the
semi-synthetic effect modifier: results".

| # | quantity (thinned test contexts) | result |
|---|---|---|
| L1 | `naive` loses +0.69 ± 0.17 from γ = 0 to γ = 3 | PASS |
| L2 | `conditional` loses +0.67 ± 0.10 | PASS |
| **L3** | **`dr` loses less by +0.68 ± 0.10 (6.9 SE)** | **PASS** |
| L4 | retargeted (λ = 0.5) − `dr` = −0.61 ± 0.09 | FAIL |
| L5 | `dr` − `conditional` at γ = 0: +0.004 ± 0.051 | PASS (`rt` not run at γ = 0) |
| L6 | control contexts move by 0.10–0.13 (2.5–3.5 SE) | FAIL |

- **The target is met:** with the modifier, `dr`'s policy is worth 0.68 more
  than `conditional`'s at γ = 3 (15.02 against 14.34; 15% of the gain over a
  random dose), and the two are identical at γ = 0.
- **Mechanism:** on the thin half `conditional` learns 0.13 of the planted
  effect, `dr` 0.33, the share both learn where overlap is fine. `dr`'s
  thin-half model error is 16.6 ± 1.4 below `conditional`'s.
- **Retargeting does not help here:** its training law has no balancing
  component, so it re-hides the thin cells the pilot avoids (the planted
  harms); it lands between `conditional` and `dr` at every λ.
- One draw, one seed per half; the effect is semi-synthetic and planted,
  by design, exactly where the logger is thin.
