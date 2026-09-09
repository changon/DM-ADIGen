"""Case-agnostic specs for action/treatment + covariates.

    spec.FIELDS
      |-- ACTION_FIELDS   -> A, intervention
      |-- CONTEXT_FIELDS  -> all but A, COLUMN ORDER of the (N, F) context tensor, used in data/context.py
      |-- ROLE_OF         -> A / C / E, read by domains/rxrx19a.py
      |-- CONTEXT_COL     -> column index of each context field, dict mapping names to col indices like ctx[:, CONTEXT_COL["edge"]]
      |-- DECLARED_LEVELS -> fields whose levels are fixed, not scanned. category to integer code mappings, so Mock = 2, etc.
      \-- ADJUSTABLE      -> CovariateSpec.categorical_cols, which fields should go into a dataset that models can adjust on, i.e. a cov_vec column for alpha
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
    continuous_grid: Sequence[float] = field( default_factory=lambda: (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0) )

    # Whether to log10-transform the continuous column before feeding the alpha network (recommended for dose, since support spans 3 decades).
    continuous_log10: bool = True


@dataclass(frozen=True)
class DomainField:
    """generator conditions on the `role` 
        "A"   action: the intervention, the estimand.
        "C"   confounder: the adjustment set.
        "E"   environment: plate position, site, batch... Never dropped
        None  excluded from the generator entirely.

    `adjustable` -- alpha may condition on it, giving it a `cov_vec` column

    `levels` -- None means derive from data. O.w. declare them, then the ORDER matters

    `cardinality` -- derived from `levels` when those are declared; otherwise None means count it from the data.

    `source` -- the metadata column or derivation this field comes from, just documentation
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
# FIELD DECLARATION.
# ===========================================================================
FIELDS: tuple[DomainField, ...] = (
    # --- A: the intervention. CFG drops these to a learned null. -----------
    DomainField("compound",   "A", kind="cat",  source="treatment",  note="cardinality from compound_vocab.json"),
    DomainField("is_control", "A", kind="cat",  cardinality=2, note="fixed by definition: a well either is a vehicle or is not"),
    DomainField("dose",       "A", kind="cont", nullable=True, n_freqs=0, source="log10_conc", note="loc/scale from the TRAIN split, treated rows only; a vehicle well has no dose -> NaN -> learned null, never 0.0 (which is a valid 1 uM dose)."),
    # --- C / E: the context tensor, in COLUMN ORDER ------------------------
    DomainField("cell_type",  None,                  note="constant on this population (HRCE)"),
    DomainField("experiment", "E", adjustable=True,  note="batch; assignment is logistical, not outcome-driven"),
    DomainField("plate",      None, adjustable=True, note="f(a|x)=0 on 94.6% of the product measure"),
    DomainField("well_row",   "E",                   source="well", note="plate row -- edge effects"),
    DomainField("well_col",   "E",                   source="well", note="plate column -- edge effects"),
    DomainField("site",       "E",                   note="field of view within the well"),
    DomainField("disease",    None,                  source="disease_condition", levels=("Active SARS-CoV-2", "UV Inactivated SARS-CoV-2", "Mock", "Unlabeled"), note="constant on this population (Active only)"),
    DomainField("edge",       "E", adjustable=True,  source="well", levels=("edge", "interior"), note="outer edge_margin rows/cols vs interior"),
)

# set objects
BY_NAME: dict[str, DomainField] = {f.name: f for f in FIELDS}
ACTION_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.role == "A")
# Column order of the (N, F) context tensor. Action fields travel separately.
CONTEXT_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.role != "A")
ROLE_OF: dict[str, str | None] = {f.name: f.role for f in FIELDS}
ADJUSTABLE: tuple[str, ...] = tuple(f.name for f in FIELDS if f.adjustable)
# The E fields alpha conditions on. ADIGen's Riesz representer is alpha(A, X, E)
ALPHA_ENV: tuple[str, ...] = tuple( f.name for f in FIELDS if f.role == "E" and f.adjustable)

def invariance_env_fields(cfg) -> tuple[str, ...]:
    """The environments the V-REx penalty is taken over -- DERIVED from the roles.
    ADIGen's Causal Invariance is a statement across E:
        P_E(Y(a) | S(X), T(a)) = P_E'(Y(a) | S(X), T(a))
    """
    if cfg.environment_set is not None:
        return tuple(cfg.environment_set)
    adj = set(cfg.adjustment_set)
    return tuple(f.name for f in FIELDS if f.role == "E" and f.name not in adj)

def role_tag(cfg) -> str:
    """Path string identifying the (C, E) it was produced under, for bookkeeping.
    Empty when both are default, so existing paths are unchanged.
    """
    parts = []
    if cfg.adjustment_set:
        parts.append("C" + "-".join(sorted(cfg.adjustment_set)))
    if cfg.environment_set is not None:
        parts.append("E" + ("-".join(sorted(cfg.environment_set)) or "none"))
    return ("_" + "_".join(parts)) if parts else ""


def role_summary(cfg) -> dict:
    """Exactly where every declared role is. Log this at startup.
    Indicates which E fields reach alpha (cov_vec carries them) and which are invariance-only. 
    guards against (uncaught) divergence between what an arm declares and what its models use
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
         f"    alpha(A,C,E) X={L(r['alpha_X'])}   psi(C,A) X={L(r['psi_X'])}   generator C={L(r['generator_C'])}   invariance env={L(r['invariance_env'])}")
    if r["E_invariance_only"]:
        s += (f"\n    NOTE E={L(r['E_invariance_only'])} is INVARIANCE-ONLY (no cov_vec column, so alpha cannot condition on it)")
    return s


def alpha_cov_fields(cfg) -> tuple[str, ...]:
    """What alpha conditions on: C UNION E, restricted to fields cov_vec carries.
    ADIGen's representer is alpha(A, X, E).
    """
    have = set(cfg.covariates.categorical_cols)
    env = invariance_env_fields(cfg)
    return tuple(sorted({f for f in tuple(cfg.adjustment_set) + env if f in have}))

# Column index of each context field, by name. The (N, F) tensor's layout.
CONTEXT_COL: dict[str, int] = {f: i for i, f in enumerate(CONTEXT_FIELDS)}
DECLARED_LEVELS: dict[str, tuple[str, ...]] = { f.name: f.levels for f in FIELDS if f.levels is not None}

if any(f.adjustable and f.role == "A" for f in FIELDS):
    raise ValueError("an action field cannot be in alpha's covariate set: alpha conditions on X to model A, so that would be circular.")


@dataclass
class CovariateSpec:
    # Categorical covariate columns -> one-hot into cov_vec. This is which columns EXIST for alpha to use; `CaseConfig.adjustment_set' specifies further
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
    channel_suffixes: Sequence[str] = field(default_factory=lambda: ("w1", "w2", "w3", "w4", "w5"))
    # Target square resolution after downsample.
    resolution: int = 128
    # Per-channel mean/std for normalization. None = compute on the fly
    normalize_mean: Sequence[float] | None = None
    normalize_std: Sequence[float] | None = None

    # --- latent representation ------------------------------------------
    # The DiT denoises VAE latents, not pixels.
    latent_vae: str = "ostris/vae-kl-f8-d16"   # 16-channel, f8; picked by the rescue-metric gate


@dataclass
class PopulationSpec:
    """Rows the causal question is defined on. HRCE here, and infected bc that's what has positivity guarantees and causal testing.
    """
    disease_condition: str | None = "Active SARS-CoV-2"
    cell_type: str | None = "HRCE"
    compounds: Sequence[str] | None = None


# Repo root holding src
PROJECT_ROOT = Path(os.environ.get("RXRX19A_ROOT", Path(__file__).resolve().parent.parent))

@dataclass
class Paths:
    """Filesystem layout, all relative to PROJECT_ROOT.
    """
    # The RxRx19a download: metadata.csv + images/ as distributed by Recursion.
    metadata_csv: str = str(PROJECT_ROOT / "RxRx19a" / "metadata.csv")
    images_root: str = str(PROJECT_ROOT / "RxRx19a" / "images")
    # The tabular HF dataset build_dataset writes (one row per site, with paths).
    tabular_dataset_dir: str = str(PROJECT_ROOT / "data" / "rxrx19a_tabular")
    # Nuisance details: compound vocab, covariate encoder, splits, alpha nets.
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

    # THE ADJUSTMENT SET -- the C fields vs E
    adjustment_set: Sequence[str] = field(default_factory=tuple)

    # THE ENVIRONMENT SET -- the E fields, GIVEN, not inferred.
    environment_set: Sequence[str] | None = None

    def __post_init__(self):
        unknown = [c for c in self.adjustment_set if c not in self.covariates.categorical_cols]
        if unknown:
            raise ValueError("problem building due to uknown covs")
        not_carried = [c for c in self.adjustment_set if c not in CONTEXT_FIELDS]
        if not_carried:
            raise ValueError("generator cannot condition on it, not used/declared as adjustment set member")
        # E can go into
        #   alpha            -- needs a cov_vec column (adjustable=True)
        #   V-REx penalty    -- needs only a context column
        _eset = self.environment_set or ()
        e_unknown = [c for c in _eset if c not in CONTEXT_FIELDS]
        if e_unknown:
            raise ValueError("env set name not known")
        both = sorted(set(self.adjustment_set) & set(_eset))
        if both:
            raise ValueError("role of C vs. E is distinct here. You have specified something as both, conflicting with the causal structure.")


def default_config() -> CaseConfig:
    return CaseConfig()


def add_adjustment_set_cli(parser) -> None:
    """Register `--adjustment_set` """
    parser.add_argument("--population_compounds", default=None, help="Comma-separated compounds or a file path; restricts the action space (controls always kept). Redefines the estimand.")
    parser.add_argument("--environment_set", default=None,  help="Comma-separated E fields for THIS arm; '' = explicitly empty, omit = role=E defaults. alpha sees C UNION E; V-REx runs over E.")
    parser.add_argument("--adjustment_set", default=None, help="Comma-separated C fields for THIS arm; '' = explicitly empty, omit = declared default. Generator and BOTH DR legs must match.")


def config_from_args(args) -> CaseConfig:
    """`default_config()` with an adjustment_set. """
    cfg = default_config()
    if not hasattr(args, "adjustment_set"):
        raise AttributeError("args has no `adjustment_set` attribute; add the field to the carrier -- defaulting silently would let the generator's C diverge from the DR legs'.")
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
