"""Case-agnostic specs for action/treatment + covariates.

A "case" is a (action_spec, covariate_spec, image_spec, paths) bundle. The same
nuisance + training code consumes the spec without branching on dataset name.

**FIELDS, at the top of this file, is the single declaration.** Everything the
model conditions on is named exactly once, there:

    spec.FIELDS
      |-- ACTION_FIELDS   -> the A block of the generator's CondSpec
      |-- CONTEXT_FIELDS  -> re-exported by data/context.py; COLUMN ORDER of the
      |                      (N, F) context tensor, so reordering forces a rebuild
      |-- ROLE_OF         -> A / C / E, read by domains/rxrx19a.py
      |-- CONTEXT_COL     -> column index of each context field
      |-- DECLARED_LEVELS -> fields whose levels are fixed, not scanned
      \-- ADJUSTABLE      -> CovariateSpec.categorical_cols, i.e. which fields get
                             a cov_vec column for alpha

and `CaseConfig.adjustment_set` names which of those are C for a given arm --
one declaration read by BOTH DR legs (`build_cond_spec` and `fit_urr`), because
if they disagree double robustness is silently gone.

Before 2026-08-26 the field list was written out three times and had drifted
twice: `plate` sorted as int in one place and string in another, and `edge`
present in the covariate list but absent from the context tensor, so alpha could
adjust for a field the generator could not see.

CARDINALITIES ARE COUNTED, NOT TYPED. `levels=None` / `cardinality=None` means
"scan the data": `ContextEncoder` reads the dataset, `compound` comes from
`compound_vocab.json`, and the resolved numbers are pinned into each checkpoint's
`arch.json`. A hand-typed cardinality is a number that can drift from what is on
disk. The exceptions are fields whose level set is fixed by DEFINITION rather
than observed -- `disease` and `edge` -- and there the order is load-bearing, so
declaring it is what PREVENTS drift rather than causing it. See DomainField.levels.

WHAT IS NOT HERE: the derivations. `spec.FIELDS` can say a field named `well_row`
exists; it cannot say "parse the letter prefix out of the `well` column". Those
live in `data/context.py`, which owns encoding, and both its `build` and `encode`
now raise if FIELDS declares something they cannot compute.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ActionSpec:
    # Categorical treatment column (string in metadata). None if action has no
    # categorical part. Cardinality is inferred from data at vocab-build time.
    categorical_col: str | None = "treatment"
    # Sentinel value in `categorical_col` that should be mapped to the "no-compound" / control token. Empty string in RxRx19a.
    control_token: str = ""

    # Continuous treatment column. None if action has no continuous part.
    continuous_col: str | None = "treatment_conc"
    # Continuous values that should be coerced to control (mapped to 0.0). Empty strings in the CSV map to control.
    continuous_control_tokens: Sequence[str] = field(default_factory=lambda: ("",))
    # Bandwidth on the log10(conc) scale for the Riesz kernel.
    continuous_kernel_bandwidth: float = 0.3
    # Grid of continuous levels (in original units) used for nuisance eval and counterfactual sampling. (the log dose progression)
    continuous_grid: Sequence[float] = field(
        default_factory=lambda: (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    )

    # Whether to log10-transform the continuous column before feeding the
    # alpha network (recommended for dose, since support spans 3 decades).
    continuous_log10: bool = True


@dataclass(frozen=True)
class DomainField:
    """One thing the generator conditions on, declared once for every consumer.

    `role` -- what the generator does with it:
        "A"   action: the intervention, the estimand. DROPPED to a learned null
              by classifier-free guidance; that is what makes guidance a
              treatment contrast rather than an unconditional sample.
        "C"   confounder: the adjustment set. Never dropped, and MUST match what
              the alpha leg adjusts for or double robustness is gone.
        "E"   environment: plate position, site, batch. Never dropped, but exempt
              from the equality check against alpha -- variance, not bias.
        None  excluded from the generator entirely.

    `adjustable` -- a SEPARATE axis: may alpha condition on it, i.e. does it get
    a `cov_vec` column? Separate because `edge` is E by default yet must be
    promotable to C for the confounding ablation, so alpha needs its column while
    the field is still E. Roles are per-arm; columns are per-build.

    `levels` -- None means SCAN THE DATA for them (the default and the rule).
    Declare them only when the set is fixed by definition rather than observed,
    and then the ORDER is load-bearing: it is the index a checkpoint was trained
    against. A scanned `disease` would renumber Mock from 2 to 1 in any build
    with no UV-inactivated wells, and every trained model would misread its own
    conditioning. Declaring pins it.

    `cardinality` -- derived from `levels` when those are declared; otherwise
    None means count it from the data. An int only for a field with no level
    names at all.

    `source` -- the metadata column or derivation this field comes from, when the
    name differs. Documentation for the encoder, not used for dispatch.
    """
    name: str
    role: str | None
    adjustable: bool = False
    kind: str = "cat"                     # "cat" | "cont"
    levels: tuple[str, ...] | None = None  # None -> scan the data
    cardinality: int | None = None         # None -> len(levels), else from data
    nullable: bool = False                 # a continuous field that can be N/A
    n_freqs: int = 0                       # continuous only: Fourier bands (0 = raw scalar)
    source: str = ""
    note: str = ""

    def __post_init__(self):
        if self.levels is not None and self.cardinality is None:
            object.__setattr__(self, "cardinality", len(self.levels))


# ===========================================================================
# THE SINGLE FIELD DECLARATION.  Edit here; everything below and the rest of
# the pipeline derives from it.
#
# Adding/removing/reordering a non-action name changes the context tensor, so it
# requires `build_dataset` + a retrain. `ContextEncoder.load` fails loudly on a
# stale `context_encoder.json` rather than silently re-indexing.
# ===========================================================================
FIELDS: tuple[DomainField, ...] = (
    # --- A: the intervention. CFG drops these to a learned null. -----------
    DomainField("compound",   "A", kind="cat",  source="treatment",
                note="cardinality from compound_vocab.json"),
    DomainField("is_control", "A", kind="cat",  cardinality=2,
                note="fixed by definition: a well either is a vehicle or is not"),
    DomainField("dose",       "A", kind="cont", nullable=True, n_freqs=8,
                source="log10_conc",
                note="loc/scale from the TRAIN split, treated rows only; a vehicle "
                     "well has no dose -> NaN -> learned null, never 0.0 (which is "
                     "a valid 1 uM dose). n_freqs>0 widens the one scalar the "
                     "estimand rides on into a band code the adaLN vector can "
                     "resolve; 0 feeds the raw standardised value"),

    # --- C / E: the context tensor, in COLUMN ORDER ------------------------
    DomainField("cell_type",  None,                  note="constant on this population (HRCE)"),
    DomainField("experiment", "E", adjustable=True,  note="batch; assignment is logistical, not outcome-driven"),
    DomainField("plate",      None, adjustable=True, note="f(a|x)=0 on 94.6% of the product measure"),
    DomainField("well_row",   "E",                   source="well", note="plate row -- edge effects"),
    DomainField("well_col",   "E",                   source="well", note="plate column -- edge effects"),
    DomainField("site",       "E",                   note="field of view within the well"),
    DomainField("disease",    None,                  source="disease_condition",
                levels=("Active SARS-CoV-2", "UV Inactivated SARS-CoV-2",
                        "Mock", "Unlabeled"),
                note="constant on this population (Active only); levels DECLARED "
                     "because a build with no UV wells would renumber Mock 2 -> 1"),
    DomainField("edge",       "E", adjustable=True,  source="well",
                levels=("edge", "interior"),
                note="outer edge_margin rows/cols vs interior; the confounding lever -- "
                     "the only covariate with the overlap to support an ablation "
                     "(400/400 compounds span both bins)"),
)

BY_NAME: dict[str, DomainField] = {f.name: f for f in FIELDS}
ACTION_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.role == "A")
# Column order of the (N, F) context tensor. Action fields travel separately.
CONTEXT_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.role != "A")
ROLE_OF: dict[str, str | None] = {f.name: f.role for f in FIELDS}
ADJUSTABLE: tuple[str, ...] = tuple(f.name for f in FIELDS if f.adjustable)
# The E fields alpha conditions on. ADIGen's Riesz representer is alpha(A, X, E):
# it models the ASSIGNMENT mechanism, so it takes everything that moves treatment,
# environment included -- while the outcome leg takes C only ("E is inputted into
# the URR, but not the outcome model"; see models/conditioning.py). Membership is
# derived, not typed: a field qualifies iff it is role E (so the outcome leg
# excludes it) AND adjustable (so cov_vec actually carries a column for it).
# `plate` drops out on the first test -- it is role=None, which is what keeps the
# f(a|x)=0-on-94.6% block out of alpha without a special case.
ALPHA_ENV: tuple[str, ...] = tuple(
    f.name for f in FIELDS if f.role == "E" and f.adjustable)


def invariance_env_fields(cfg) -> tuple[str, ...]:
    """The environments the V-REx penalty is taken over -- DERIVED from the roles.

    ADIGen's Causal Invariance is a statement across E:
        P_E(Y(a) | S(X), T(a)) = P_E'(Y(a) | S(X), T(a))
    so the environment index IS the role-E block, and declaring it separately
    (as a free-text --invariance_env did) is a second source of truth that can
    disagree with spec.FIELDS -- its old default, "plate", is role=None here and
    therefore not an environment at all under this declaration.

    Fields promoted to C are EXCLUDED: a promoted field is a confounder in that
    arm, and demanding invariance across a confounder would ask the model to
    ignore variation it is supposed to adjust for.
    """
    if cfg.environment_set is not None:
        return tuple(cfg.environment_set)
    adj = set(cfg.adjustment_set)
    return tuple(f.name for f in FIELDS if f.role == "E" and f.name not in adj)


def role_tag(cfg) -> str:
    """Path suffix identifying the (C, E) an artifact was produced under.

    Nuisance artifacts were previously keyed by RARITY alone, so two arms that
    differed only in the adjustment set wrote alpha nets and DR weights to the
    same directory and silently clobbered each other. That already happened: a
    dir held alpha fit under C=0 next to DR weights computed under C={edge}.

    Empty when both are default, so existing paths are unchanged.
    """
    parts = []
    if cfg.adjustment_set:
        parts.append("C" + "-".join(sorted(cfg.adjustment_set)))
    if cfg.environment_set is not None:
        parts.append("E" + ("-".join(sorted(cfg.environment_set)) or "none"))
    return ("_" + "_".join(parts)) if parts else ""


def role_summary(cfg) -> dict:
    """Exactly where every declared role lands downstream. Log this at startup.

    Makes the E split visible: which E fields reach alpha (cov_vec carries them)
    and which are invariance-only. Silent divergence between what an arm declares
    and what its models consume is the failure this whole spec exists to prevent.
    """
    have = set(cfg.covariates.categorical_cols)
    C = tuple(cfg.adjustment_set)
    E = invariance_env_fields(cfg)
    e_alpha = tuple(f for f in E if f in have)
    e_inv_only = tuple(f for f in E if f not in have)
    return {
        "C": C,
        "E": E,
        "alpha_X": alpha_cov_fields(cfg),      # C U (E n cov_vec)
        "psi_X": C,                            # C only
        "generator_C": C,
        "invariance_env": E,                   # all of E
        "E_reaching_alpha": e_alpha,
        "E_invariance_only": e_inv_only,
    }


def format_role_summary(cfg) -> str:
    r = role_summary(cfg)
    L = lambda t: list(t) if t else "[]"
    s = (f"C={L(r['C'])}  E={L(r['E'])}\n"
         f"    alpha(A,C,E) X={L(r['alpha_X'])}   psi(C,A) X={L(r['psi_X'])}   "
         f"generator C={L(r['generator_C'])}   invariance env={L(r['invariance_env'])}")
    if r["E_invariance_only"]:
        s += (f"\n    NOTE E={L(r['E_invariance_only'])} is INVARIANCE-ONLY "
              f"(no cov_vec column, so alpha cannot condition on it)")
    return s


def alpha_cov_fields(cfg) -> tuple[str, ...]:
    """What alpha conditions on: C UNION E, restricted to fields cov_vec carries.

    ADIGen's representer is alpha(A, X, E). Both halves are GIVEN by the arm --
    no `adjustable`-gated guessing, which is what silently pulled `experiment`
    (1.8% common support) into alpha and collapsed nu to 30 compounds.
    """
    have = set(cfg.covariates.categorical_cols)
    env = invariance_env_fields(cfg)
    # alpha takes C plus the E fields cov_vec can actually deliver. C is
    # guaranteed present by __post_init__; E may be partly invariance-only, which
    # `role_summary` reports rather than hiding.
    return tuple(sorted({f for f in tuple(cfg.adjustment_set) + env if f in have}))
# Column index of each context field, by name. The (N, F) tensor's layout.
CONTEXT_COL: dict[str, int] = {f: i for i, f in enumerate(CONTEXT_FIELDS)}
DECLARED_LEVELS: dict[str, tuple[str, ...]] = {
    f.name: f.levels for f in FIELDS if f.levels is not None}

if any(f.adjustable and f.role == "A" for f in FIELDS):
    raise ValueError("an action field cannot be in alpha's covariate set: alpha "
                     "conditions on X to model A, so that would be circular.")


@dataclass
class CovariateSpec:
    # Categorical covariate columns -> one-hot into cov_vec. This is which
    # columns EXIST for alpha to use; `CaseConfig.adjustment_set` picks which of
    # them a given run actually adjusts for.
    categorical_cols: Sequence[str] = field(default_factory=lambda: ADJUSTABLE)
    # Continuous covariate columns (kept as float). Empty by default.
    continuous_cols: Sequence[str] = field(default_factory=tuple)
    # `edge` is derived from `well`: outer `edge_margin` rows/cols of the plate vs the interior.
    edge_margin: int = 2


@dataclass
class ImageSpec:
    # Number of microscopy channels stacked into the diffusion input.
    n_channels: int = 5
    # Channel filename suffixes in the order they should be stacked.
    channel_suffixes: Sequence[str] = field(
        default_factory=lambda: ("w1", "w2", "w3", "w4", "w5")
    )
    # Target square resolution after downsample.
    resolution: int = 128
    # Per-channel mean/std for normalization. None = compute on the fly
    normalize_mean: Sequence[float] | None = None
    normalize_std: Sequence[float] | None = None

    # --- latent representation ------------------------------------------
    # The DiT denoises VAE latents, not pixels. This is the only latent CHOICE;
    # everything else in LatentSpec (z_per_channel, latent resolution,
    # scaling_factor, per-channel stats) is MEASURED from the VAE's own output
    # at encode time and persisted next to the latents, so it cannot drift from
    # what is on disk. See data/latents.py.
    latent_vae: str = "ostris/vae-kl-f8-d16"   # 16-channel, f8; picked by the rescue-metric gate


@dataclass
class PopulationSpec:
    """Rows the causal question is defined on.

    Both filters are population restrictions, not adjustment variables: every
    non-Active well is a control (zero compounds, so f(a|x)=0 structurally), and
    VERO ran 32 of 1,669 compounds as a shortlist counter-screen, which is
    selection on the outcome with no overlap. Excluded rows stay on disk -- Mock
    is still the real anchor rescue_panel measures against.
    """
    disease_condition: str | None = "Active SARS-CoV-2"
    cell_type: str | None = "HRCE"

    # Restrict the ACTION SPACE to these compounds (control/vehicle rows are
    # always kept -- they are the reference arm, not a treatment). None = all.
    #
    # This is the only lever on |A|, and |A| is what alpha's denominator scales
    # with: f(a|x) ~ 1/|A|, so 1,669 compounds x ~6 doses = 10,078 arms drives
    # the representer to either divergence or ~9% ESS. It is a POPULATION
    # restriction, not an adjustment -- it redefines which effects are being
    # estimated, so state it in any result that uses it.
    compounds: Sequence[str] | None = None


# Repo root (the directory holding `src/`), resolved from this file's location so
# the tree can be renamed or cloned anywhere. `RXRX19A_ROOT` overrides it for a
# checkout whose data lives on a different volume.
PROJECT_ROOT = Path(os.environ.get("RXRX19A_ROOT", Path(__file__).resolve().parent.parent))


@dataclass
class Paths:
    """Filesystem layout, all relative to PROJECT_ROOT.

    Nothing here is absolute: hardcoded /home paths were the single reason this
    tree only ran under one account.
    """
    # The RxRx19a download: metadata.csv + images/ as distributed by Recursion.
    metadata_csv: str = str(PROJECT_ROOT / "RxRx19a" / "metadata.csv")
    images_root: str = str(PROJECT_ROOT / "RxRx19a" / "images")
    # The tabular HF dataset build_dataset writes (one row per site, with paths).
    tabular_dataset_dir: str = str(PROJECT_ROOT / "data" / "rxrx19a_tabular")
    # Nuisance artifacts: compound vocab, covariate encoder, splits, alpha nets.
    nuisance_dir: str = str(PROJECT_ROOT / "data" / "nuisances")
    # Diffusion training output: one subdir per arm, each with its own arch.json.
    train_output_dir: str = str(PROJECT_ROOT / "runs")


@dataclass
class CaseConfig:
    name: str = "rxrx19a_compound_conc"
    action: ActionSpec = field(default_factory=ActionSpec)
    covariates: CovariateSpec = field(default_factory=CovariateSpec)
    image: ImageSpec = field(default_factory=ImageSpec)
    population: PopulationSpec = field(default_factory=PopulationSpec)
    paths: Paths = field(default_factory=Paths)
    seed: int = 42

    # THE ADJUSTMENT SET -- the C fields, promoted from their default role.
    #
    # This is a contract between two separate programs: `train_diffusion` builds
    # the generator's C block from it and `fit_urr` fits alpha over it. If they
    # disagree, double robustness is gone, so it is declared ONCE here rather
    # than as a CLI default in each. `--cov_blocks` survives on both as a
    # deliberate ablation override, and `CondSpec.validate_against` still asserts
    # equality where the two legs combine.
    #
    # Empty is CORRECT on the base population: the screen is balanced by
    # construction (98% of compounds appear in exactly one experiment, so
    # `experiment` has no common support to adjust over), and the measured alpha
    # is 1.000 with ESS 100%. The confounded arm sets ("well_row", "well_col")
    # or ("edge",) -- position is E in the base arm and C in the ablated one,
    # which is not a contradiction: C-vs-E is a property of the DGP, and the
    # ablation changes the DGP.
    adjustment_set: Sequence[str] = field(default_factory=tuple)

    # THE ENVIRONMENT SET -- the E fields, GIVEN, not inferred.
    #
    # Symmetric with adjustment_set: an arm is defined by the PAIR (C, E), and
    # both are stated. Empty means "take the role=E fields from spec.FIELDS",
    # which is a default, not a derivation -- E is never computed by subtracting
    # C from something, because that would make E implicit and force you to
    # arrange roles until the subtraction happened to yield what you wanted.
    #
    # Consumers: alpha conditions on C UNION E; psi and the outcome model on C
    # alone; the V-REx penalty is taken over E.
    # None = NOT GIVEN (derive from role=E). () = GIVEN AND EMPTY (no E at all).
    # These must stay distinguishable: an empty tuple is falsy, so testing it with
    # `if cfg.environment_set:` silently treats "explicitly empty" as "unset" and
    # reinstates the default -- which invalidated a run before this was caught.
    environment_set: Sequence[str] | None = None

    def __post_init__(self):
        unknown = [c for c in self.adjustment_set
                   if c not in self.covariates.categorical_cols]
        if unknown:
            raise ValueError(
                f"adjustment_set names {unknown}, which have no cov_vec column "
                f"(categorical_cols={list(self.covariates.categorical_cols)}), so "
                f"alpha cannot condition on them. Mark the field adjustable=True "
                f"in spec.FIELDS and re-run build_dataset.")
        not_carried = [c for c in self.adjustment_set if c not in CONTEXT_FIELDS]
        if not_carried:
            raise ValueError(
                f"adjustment_set names {not_carried}, which the context tensor "
                f"does not carry, so the GENERATOR cannot condition on them "
                f"while alpha can -- exactly the mismatch that breaks double "
                f"robustness. Add them to spec.FIELDS and re-run build_dataset.")
        # E has TWO consumers and a field may legitimately serve one:
        #   alpha            -- needs a cov_vec column (adjustable=True)
        #   V-REx penalty    -- needs only a context column
        # So an E field without a cov_vec column is INVARIANCE-ONLY, which is a
        # valid arm, not an error. What is NOT acceptable is that split being
        # silent, so `role_summary()` reports it and callers log it. C is
        # different: it must reach BOTH legs or double robustness is gone, and
        # that stays a hard error above.
        _eset = self.environment_set or ()
        e_unknown = [c for c in _eset if c not in CONTEXT_FIELDS]
        if e_unknown:
            raise ValueError(
                f"environment_set names {e_unknown}, which the context tensor "
                f"does not carry. Add them to spec.FIELDS and re-run build_dataset.")
        both = sorted(set(self.adjustment_set) & set(_eset))
        if both:
            raise ValueError(
                f"{both} named in BOTH adjustment_set and environment_set. A field "
                f"is a confounder or an environment in a given arm, not both: C is "
                f"adjusted for by alpha AND the outcome leg, while E enters alpha "
                f"and the invariance penalty only.")


def default_config() -> CaseConfig:
    return CaseConfig()


def add_adjustment_set_cli(parser) -> None:
    """Register `--adjustment_set` -- the per-arm C override.

    `CaseConfig.adjustment_set` is declared once, but it is a property of the
    ARM, not of the dataset: position is E in the base DGP and C once the
    ablation makes treatment depend on it. Without this flag the declaration is
    global, so a confounded arm and the base conditional arm cannot coexist.
    """
    parser.add_argument(
        "--population_compounds", default=None,
        help="Restrict the action space to these compounds: a comma-separated "
             "list, or a path to a newline-delimited file. Controls are always "
             "kept. Redefines the estimand -- report it alongside any result.")
    parser.add_argument(
        "--environment_set", default=None,
        help="Comma-separated E fields for THIS arm. '' = explicitly empty; omit "
             "to take the role=E fields from spec.FIELDS. alpha conditions on "
             "C UNION E; the V-REx penalty is taken over E. Given, not inferred.")
    parser.add_argument(
        "--adjustment_set", default=None,
        help="Comma-separated C fields for THIS arm, overriding "
             "CaseConfig.adjustment_set. '' = explicitly empty; omit to take the "
             "declared default. The generator and BOTH DR legs must be given the "
             "same value, or the two legs adjust for different confounders and "
             "double robustness is silently gone.")


def config_from_args(args) -> CaseConfig:
    """`default_config()` with a per-arm adjustment_set applied AND re-validated.

    The `__post_init__` re-call is load-bearing: dataclass validation fires only
    at construction, so assigning the override afterwards would sail past both
    guards (has a cov_vec column / is carried by the context tensor) and produce
    exactly the generator-vs-alpha mismatch they exist to catch.
    """
    cfg = default_config()
    if not hasattr(args, "adjustment_set"):
        raise AttributeError(
            "config_from_args got an args object with no `adjustment_set` "
            "attribute. If this is a dataclass carrier (e.g. TrainArgs) rather "
            "than an argparse Namespace, ADD THE FIELD -- silently defaulting "
            "here would make the generator's C disagree with the DR legs' C "
            "with no error, which is the exact failure this flag exists to stop.")
    raw = args.adjustment_set
    if raw is not None:
        cfg.adjustment_set = tuple(c.strip() for c in raw.split(",") if c.strip())
    raw_p = getattr(args, "population_compounds", None)
    if raw_p is not None:
        if os.path.isfile(raw_p):
            with open(raw_p) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
        else:
            names = [c.strip() for c in raw_p.split(",") if c.strip()]
        cfg.population.compounds = tuple(names) or None
    raw_e = getattr(args, "environment_set", None)
    if raw_e is not None:
        cfg.environment_set = tuple(c.strip() for c in raw_e.split(",") if c.strip())
    if raw is not None or raw_e is not None:
        cfg.__post_init__()
    return cfg
