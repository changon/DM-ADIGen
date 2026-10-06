"""LINCS L1000 (GSE70138) specs for action/treatment, covariates, outcome, population.

Copied from RxRx19a/src/spec.py, then adapted (IMPLEMENT.md §3.3). The helpers
(`DomainField`, roles, `role_tag`, CLI, `CaseConfig` guards) are the RxRx ones;
the DECLARATIONS (`FIELDS`, `ActionSpec`, `OutcomeSpec`, `PopulationSpec`,
`Paths`) are LINCS-only.

    spec.FIELDS
      |-- ACTION_FIELDS   -> A, intervention
      |-- CONTEXT_FIELDS  -> all but A, COLUMN ORDER of the (N, F) context tensor (build_dataset.ContextEncoder)
      |-- ROLE_OF         -> A / C / E
      |-- CONTEXT_COL     -> column index of each context field, e.g. ctx[:, CONTEXT_COL["plate"]]
      |-- DECLARED_LEVELS -> fields whose levels are fixed, not scanned (syn_c)
      \-- ADJUSTABLE      -> CovariateSpec.categorical_cols: the one-hot blocks of cov_vec

Phase 0 decisions (IMPLEMENT.md §3.12; 3 and 5 amended 2026-09-29), frozen here:
    1. cell line MCF7                    -> PopulationSpec.cell_ids
    2. compound universe = all trt_cp    -> PopulationSpec.compounds = None
    3. continuous log10 dose; eval grid = dose_level -> ActionSpec
    5. per-gene z-score: mean of centred train DMSO, std of all centred train rows
                                         -> OutcomeSpec.normalize_mean / normalize_std
    6. DMSO-median plate centring + 3x plate QC -> OutcomeSpec.plate_center / plate_qc_max_spread_ratio
A different cell line is a separate population (its own `PopulationSpec.name`
and paths), never an edit of these defaults: `POPULATIONS` is the registry, and
`--population` (or the build dir's own population_qc.json) selects from it.
Step A's `core5_24h` and its line groups are declared there (STEP_A.md). A compound subset
(`--population_compounds`) is applied downstream on the same build and paths;
it redefines the estimand, is part of the splits cache key, and must be named
in every result.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ActionSpec:
    """A = (compound, log10 dose, is_control). Decision 3."""
    # Categorical treatment column. Vocab is built on pert_id, not pert_iname:
    # 17 inames map to more than one pert_id. pert_iname is a display label only.
    categorical_col: str = "pert_id"
    label_col: str = "pert_iname"
    # A well is a control iff pert_type is a vehicle (never "empty pert_iname").
    control_col: str = "pert_type"
    control_values: tuple[str, ...] = ("ctl_vehicle",)
    # Every vehicle in the population must be one of these; compound_idx=0 merges them.
    vehicle_pert_ids: tuple[str, ...] = ("DMSO",)

    # Continuous treatment column (dose) and its unit. Treated doses are asserted
    # to be in `dose_unit`, not converted.
    continuous_col: str = "pert_dose"
    dose_unit_col: str = "pert_dose_unit"
    dose_unit: str = "um"
    # LINCS writes -666 for "not applicable" (vehicle dose and unit). It must never
    # reach conc or log10_conc.
    na_sentinel: str = "-666"

    continuous_log10: bool = True
    # log10_conc of a control row. Controls feed the generator a NaN dose (learned
    # null), never 0.0 and never this sentinel.
    control_log10_sentinel: float = -10.0

    # Evaluation / strata grid (NOT a model input): the 3-fold series plus the
    # 20 uM proteasome plate controls. A treated dose within
    # `dose_level_tol_log10` of a level snaps to it; otherwise it keeps its own
    # dose rounded to `dose_level_offgrid_decimals` d.p. (173 wells, 13 compounds).
    dose_levels_um: tuple[float, ...] = (0.0412, 0.1235, 0.3704, 1.1111, 3.3333, 10.0, 20.0)
    dose_level_tol_log10: float = 0.05
    dose_level_offgrid_decimals: int = 4


@dataclass(frozen=True)
class DomainField:
    """generator conditions on the `role`
        "A"   action: the intervention, the estimand.
        "C"   confounder: the adjustment set.
        "E"   environment: plate, batch... Never dropped
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
# FIELD DECLARATION (IMPLEMENT.md §3.3).
# ===========================================================================
FIELDS: tuple[DomainField, ...] = (
    # --- A: the intervention. CFG drops these to a learned null. -----------
    DomainField("compound",   "A", kind="cat",  source="pert_id", note="cardinality from compound_vocab.json; __control__ = 0"),
    DomainField("is_control", "A", kind="cat",  cardinality=2, source="pert_type", note="1 iff pert_type == ctl_vehicle"),
    DomainField("dose",       "A", kind="cont", nullable=True, n_freqs=0, source="log10_conc", note="loc/scale from the TRAIN split, treated rows only; a vehicle has no dose -> NaN -> learned null, never 0.0 and never -666."),
    # --- C / E / None: the context tensor, in COLUMN ORDER ----------------
    DomainField("plate",      "E", adjustable=False, source="det_plate", note="invariance-only (V-REx, --include_env). NOT adjustable: 0 arms span every plate, so alpha on plate has empty common support. Its effect on Y is removed by DMSO plate centring."),
    DomainField("cell_id",    None, adjustable=True, source="cell_id", note="constant on mcf7_24h; promoted to C with --adjustment_set cell_id in step A"),
    DomainField("syn_c",      None, adjustable=True, source="syn_c", levels=("0", "1"), note="synthetic, balanced per arm and per plate-DMSO in build_dataset; promoted to C in step C"),
    DomainField("well_row",   None, source="det_well", note="aliased with the arm: within a plate map every arm sits at one well"),
    DomainField("well_col",   None, source="det_well", note="aliased with the arm"),
    DomainField("pert_time",  None, source="pert_time", note="population constant (24 h)"),
    # `edge` is deliberately not declared: 72 / 10,944 arms span edge and interior.
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
# Measured LINCS constraints (IMPLEMENT.md §3.3, review log §5). Guarded so a
# FIELDS edit cannot silently reintroduce them.
if BY_NAME["plate"].adjustable or ALPHA_ENV:
    raise ValueError(
        f"E fields {list(ALPHA_ENV) or ['plate']} are adjustable, so ALPHA_ENV feeds them to alpha on "
        f"every arm. No arm spans every plate (or batch / plate map), so "
        f"fit_urr --target_support common would find an empty target support.")
if any(BY_NAME[f].role == "E" for f in ("well_row", "well_col")):
    raise ValueError(
        "well_row / well_col cannot be role E: every arm sits at one fixed well "
        "within its plate map (aliased with A), and V-REx over plate|row|col "
        "would make each well its own environment.")


@dataclass
class CovariateSpec:
    # Categorical covariate columns -> one-hot into cov_vec. This is which columns EXIST for alpha to use; `CaseConfig.adjustment_set' specifies further
    categorical_cols: Sequence[str] = field(default_factory=lambda: ADJUSTABLE)
    # Continuous covariate columns (kept as float). Empty by default.
    continuous_cols: Sequence[str] = field(default_factory=tuple)


PLATE_CENTER_MODES = ("dmso_median_train", "none")
NORMALIZE_MEAN_MODES = ("train_dmso",)
NORMALIZE_STD_MODES = ("train_all",)


@dataclass(frozen=True)
class OutcomeSpec:
    """Y = 978 landmark genes of Level 3, replacing RxRx's ImageSpec.

    Samples live in z-space: y = (x - centre[plate] - mean) / std, with the
    centres and stats fitted on TRAIN rows after the split (src.data.expr_stats).
    Invert only where a raw-scale output is needed; never clamp.
    """
    n_genes: int = 978
    # Decision 6: drop plates whose DMSO spread (median over genes of
    # 1.4826 * MAD over the plate's DMSO wells) exceeds this multiple of the
    # median spread over plates of the same cell line. Population rule, fixed
    # before splits; uses control outcomes only.
    plate_qc_max_spread_ratio: float = 3.0
    # Decision 6: per-plate, per-gene median of the plate's TRAIN DMSO wells,
    # subtracted without scaling. "none" is the ablation.
    plate_center: str = "dmso_median_train"
    # Decision 5 (amended 2026-09-29): per-gene mean of the centred TRAIN DMSO
    # wells, so z = 0 is the vehicle, and per-gene std of ALL centred TRAIN rows.
    # All 978 genes are kept, including those at the Level 3 cap of 15.0.
    normalize_mean: str = "train_dmso"
    normalize_std: str = "train_all"
    # Step C (IMPLEMENT.md §3.8.2). `syn_seed` is the ASSIGNMENT seed: build_dataset
    # draws syn_c with it once, and population_qc.json["syn_c"]["seed"] records the
    # seed the table on disk was drawn with. Consumers read it back from there and
    # refuse a mismatch; the seed of the injected direction v is a separate knob
    # (Phase 5). `syn_effect` scales the injected shift (0 = inert, as in v1).
    syn_effect: float = 0.0
    syn_seed: int = 0

    def __post_init__(self):
        if self.plate_center not in PLATE_CENTER_MODES:
            raise ValueError(f"plate_center={self.plate_center!r}; expected one of {PLATE_CENTER_MODES}")
        if self.normalize_mean not in NORMALIZE_MEAN_MODES:
            raise ValueError(f"normalize_mean={self.normalize_mean!r}; expected one of {NORMALIZE_MEAN_MODES}")
        if self.normalize_std not in NORMALIZE_STD_MODES:
            raise ValueError(f"normalize_std={self.normalize_std!r}; expected one of {NORMALIZE_STD_MODES}")
        if not self.plate_qc_max_spread_ratio > 1:
            raise ValueError("plate_qc_max_spread_ratio must be > 1 (a multiple of the median plate's spread)")


DEFAULT_POPULATION = "mcf7_24h"


@dataclass(frozen=True)
class PopulationSpec:
    """Rows the causal question is defined on. Applied in build_dataset; the
    table on disk is already filtered, so downstream stages compare `key()`
    (a cache key), never re-mask by string equality.
    """
    name: str = DEFAULT_POPULATION
    # Decision 1: MCF7, the largest eligible 24 h trt_cp + vehicle population.
    cell_ids: tuple[str, ...] = ("MCF7",)
    # Compared numerically: the column is the string "24.0".
    pert_time_h: float = 24.0
    pert_types: tuple[str, ...] = ("trt_cp", "ctl_vehicle")
    # Decision 2: all retained trt_cp. A subset (--population_compounds, pert_ids)
    # redefines the estimand, is applied downstream, and must be reported.
    compounds: tuple[str, ...] | None = None

    def key(self) -> dict:
        """Cache key of the population, for splits and every artifact."""
        out = {"name": self.name,
               "cell_ids": ",".join(sorted(self.cell_ids)),
               "pert_time_h": f"{float(self.pert_time_h):g}",
               "pert_types": ",".join(sorted(self.pert_types))}
        if self.compounds:
            out["compounds"] = f"{len(self.compounds)}:" + hashlib.sha1(
                "|".join(sorted(self.compounds)).encode()).hexdigest()[:12]
        return out


# Step A (IMPLEMENT.md §3.8.3; STEP_A.md): the five lines with >= 99.8% of
# MCF7's arms. A population with more than one cell line keeps only compounds
# with a treated well in EVERY line (applied in build_dataset after plate QC).
# That rule is a property of "more than one line", not a PopulationSpec field:
# `asdict(PopulationSpec)` is embedded in decisions_record, and a new field
# would invalidate the mcf7_24h build.
CORE5_LINES: tuple[str, ...] = ("MCF7", "HT29", "HA1E", "A375", "PC3")
POPULATIONS: dict[str, PopulationSpec] = {
    DEFAULT_POPULATION: PopulationSpec(),
    "core5_24h": PopulationSpec(name="core5_24h", cell_ids=CORE5_LINES),
}

# The two line groups step A's selection score contrasts (z_C = -1 / +1), per
# population. Declared 2026-10-05 (STEP_A.md D3), BEFORE any step-A result:
# G1 = the three carcinoma lines, G2 = HA1E (immortalised kidney) and A375
# (melanoma). The tier dir tag and every step-A artifact record them.
LINE_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "core5_24h": {"G1": ("MCF7", "HT29", "PC3"), "G2": ("HA1E", "A375")},
}
for _pop, _g in LINE_GROUPS.items():
    if sorted(_g["G1"] + _g["G2"]) != sorted(POPULATIONS[_pop].cell_ids) or set(_g["G1"]) & set(_g["G2"]):
        raise ValueError(f"LINE_GROUPS[{_pop!r}] must partition the population's cell_ids")


def line_groups(population: "PopulationSpec | str") -> dict[str, tuple[str, ...]] | None:
    """{"G1": lines, "G2": lines} for a population, or None if it declares none
    (a single-line population has no line contrast)."""
    name = population if isinstance(population, str) else population.name
    return LINE_GROUPS.get(name)


def line_group_sign(cell_id, population: "PopulationSpec | str"):
    """(N,) float: -1 for G1 lines, +1 for G2 lines -- step A's z_C. Raises on a
    line in neither group, and when the population declares no groups."""
    import numpy as np
    g = line_groups(population)
    if g is None:
        raise ValueError(f"population {population if isinstance(population, str) else population.name!r} "
                         f"declares no line groups (spec.LINE_GROUPS); step A needs them")
    c = np.asarray(cell_id).astype(str)
    out = np.where(np.isin(c, g["G2"]), 1.0, np.where(np.isin(c, g["G1"]), -1.0, np.nan))
    if np.isnan(out).any():
        raise ValueError(f"cell_ids {sorted(set(c[np.isnan(out)]))[:5]} are in neither line group {g}")
    return out


def line_group_tag(population: "PopulationSpec | str") -> str:
    """Dir-tag form of G2 (G1 is the rest), e.g. "G2-A375-HA1E"."""
    return "G2-" + "-".join(sorted(line_groups(population)["G2"]))


# The decision task on step A (DECISION.md §4-5). Declared 2026-10-05, BEFORE any
# decision number was read. Task B's axis is fixed here, not fitted: equal
# weights on these cell-cycle landmark genes, signed so that a larger effect
# means LOWER expression of the set (read as more anti-proliferative).
PROLIFERATION_GENES: tuple[str, ...] = (
    "TOP2A", "CCNB1", "CDK1", "PCNA", "AURKA", "PLK1", "BIRC5",
    "CCNA2", "CDC20", "KIF20A", "CCNE2", "MCM3", "E2F2", "CDC25A")
DECISION_MIN_DOSES = 4                  # a compound enters with >= this many eligible doses
DECISION_K = 100                        # task B's headline top-k (X4)
DECISION_K_CURVE: tuple[int, ...] = (25, 50, 100, 200)
DECISION_K_CONTROL = 50                 # D5's screen over the unthinned compounds only


def population_of_build(data_dir: str) -> str | None:
    """The population a build dir was made for (its population_qc.json), or None
    when the dir holds no build yet."""
    import json
    path = os.path.join(data_dir, "population_qc.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return str(json.load(fh)["population"])


# LINCS_ROOT: the directory holding src/ (i.e. lincs/), as RXRX19A_ROOT is RxRx19a/.
PROJECT_ROOT = Path(os.environ.get("LINCS_ROOT", Path(__file__).resolve().parent.parent))

# The GSE70138 download (scripts/download.sh, run from lincs/).
RAW_FILES: dict[str, str] = {
    "inst_info": "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
    "gene_info": "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
    "pert_info": "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
    "cell_info": "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
    # the decompressed Level 3 matrix; read with raw h5py
    "gctx": "GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx",
}


@dataclass
class Paths:
    """Filesystem layout. Built artifacts are per population:
    data/<population>/... and runs/<population>/..., so a second population
    (step A's core5_24h) never touches v1's files. Blank fields are filled from
    `population` / `data_dir` in __post_init__, once: to move a build, construct
    a new Paths(population=..., data_dir=...) (not dataclasses.replace), so the
    derived paths follow. Overriding a derived dir (e.g. nuisance_dir) is fine.
    """
    population: str = DEFAULT_POPULATION
    # The download: four TSVs + the decompressed GCTX.
    raw_dir: str = str(PROJECT_ROOT / "lincs" / "GSE70138")
    # Everything build_dataset writes: table, expr.npy, gene order, QC, encoders.
    data_dir: str = ""
    # The tabular HF dataset build_dataset writes (one row per retained well).
    tabular_dataset_dir: str = ""
    # Nuisance details: compound vocab, covariate encoder, splits, alpha nets.
    nuisance_dir: str = ""
    # Diffusion training output: one subdir per arm, each with its own arch.json.
    train_output_dir: str = ""

    def __post_init__(self):
        self.data_dir = self.data_dir or str(PROJECT_ROOT / "data" / self.population)
        self.tabular_dataset_dir = self.tabular_dataset_dir or os.path.join(self.data_dir, "lincs_tabular")
        self.nuisance_dir = self.nuisance_dir or os.path.join(self.data_dir, "nuisances")
        self.train_output_dir = self.train_output_dir or str(PROJECT_ROOT / "runs" / self.population)
        object.__setattr__(self, "_derived", True)

    def __setattr__(self, name, value):
        if name in ("population", "data_dir") and getattr(self, "_derived", False):
            raise AttributeError(
                f"Paths.{name} is fixed after construction (the derived paths would not "
                f"follow); build a new Paths(population=..., data_dir=...) instead.")
        object.__setattr__(self, name, value)

    def raw(self, key: str) -> str:
        return os.path.join(self.raw_dir, RAW_FILES[key])

    # Y out of band: (N, n_genes) float32 raw Level 3, aligned to table row order.
    @property
    def expr_npy(self) -> str:
        return os.path.join(self.data_dir, "expr.npy")

    @property
    def gene_order_json(self) -> str:
        return os.path.join(self.data_dir, "gene_order.json")

    @property
    def population_qc_json(self) -> str:
        return os.path.join(self.data_dir, "population_qc.json")

    @property
    def context_encoder_json(self) -> str:
        return os.path.join(self.data_dir, "context_encoder.json")


@dataclass
class CaseConfig:
    name: str = "lincs_l1000_compound_dose"
    action: ActionSpec = field(default_factory=ActionSpec)
    covariates: CovariateSpec = field(default_factory=CovariateSpec)
    outcome: OutcomeSpec = field(default_factory=OutcomeSpec)
    population: PopulationSpec = field(default_factory=PopulationSpec)
    # None -> Paths(population=population.name)
    paths: Paths | None = None
    seed: int = 42

    # THE ADJUSTMENT SET -- the C fields vs E. v1: () (IMPLEMENT.md §3.3).
    adjustment_set: Sequence[str] = field(default_factory=tuple)

    # THE ENVIRONMENT SET -- the E fields, GIVEN, not inferred.
    environment_set: Sequence[str] | None = None

    # Which resolved step-C injection `syn_effect` switches on (§3.8.2, §3.8.4):
    # a file name under the split dir, falling back to the base build's.
    # "syn_meta.json" is step C; "syn_meta_compound_r1.json" is step C2. It lives
    # here, not on OutcomeSpec, because OutcomeSpec is embedded verbatim in
    # decisions_record, and a new field there would invalidate every build.
    syn_meta_name: str = "syn_meta.json"

    def __post_init__(self):
        if self.paths is None:
            self.paths = Paths(population=self.population.name)
        if self.paths.population != self.population.name:
            raise ValueError(
                f"paths are for population {self.paths.population!r} but the config's "
                f"population is {self.population.name!r}; artifacts would mix populations.")
        if not set(self.action.control_values) <= set(self.population.pert_types):
            raise ValueError("ActionSpec.control_values must be retained by PopulationSpec.pert_types")
        unknown = [c for c in self.adjustment_set if c not in self.covariates.categorical_cols]
        if unknown:
            raise ValueError(
                f"adjustment_set names {unknown}, which have no cov_vec column "
                f"(adjustable fields: {list(self.covariates.categorical_cols)}).")
        not_carried = [c for c in self.adjustment_set if c not in CONTEXT_FIELDS]
        if not_carried:
            raise ValueError(
                f"adjustment_set names {not_carried}, which the context tensor does not "
                f"carry, so the generator cannot condition on them.")
        # E can go into
        #   alpha            -- needs a cov_vec column (adjustable=True)
        #   V-REx penalty    -- needs only a context column
        _eset = self.environment_set or ()
        e_unknown = [c for c in _eset if c not in CONTEXT_FIELDS]
        if e_unknown:
            raise ValueError(f"environment_set names unknown context fields {e_unknown}")
        both = sorted(set(self.adjustment_set) & set(_eset))
        if both:
            raise ValueError(
                f"{both} declared as both C and E; the roles are distinct in the "
                f"causal structure.")


def default_config() -> CaseConfig:
    return CaseConfig()


def decisions_record(cfg: CaseConfig) -> dict:
    """The frozen Phase 0 decisions a build was made under (IMPLEMENT.md §3.12)."""
    a = cfg.action
    return {
        "population": asdict(cfg.population),
        "population_key": cfg.population.key(),
        "outcome": asdict(cfg.outcome),
        "dose": {
            "encoding": "continuous log10_conc (uM); dose_level is eval/strata only",
            "dose_unit": a.dose_unit,
            "dose_levels_um": list(a.dose_levels_um),
            "dose_level_tol_log10": a.dose_level_tol_log10,
            "dose_level_offgrid_decimals": a.dose_level_offgrid_decimals,
            "control_log10_sentinel": a.control_log10_sentinel,
        },
        "fields": [asdict(f) for f in FIELDS],
        "adjustable": list(ADJUSTABLE),
    }


def add_adjustment_set_cli(parser) -> None:
    """Register `--adjustment_set` (and the population selectors)"""
    parser.add_argument("--population", default=None, choices=sorted(POPULATIONS),
                        help="Which declared population (spec.POPULATIONS). Default: the one "
                             "the --data_dir build was made for, else mcf7_24h. A build and "
                             "a population that disagree are refused.")
    parser.add_argument("--population_compounds", default=None, help="Comma-separated pert_ids or a file path; restricts the action space (controls always kept). Redefines the estimand.")
    parser.add_argument("--environment_set", default=None,  help="Comma-separated E fields for THIS arm; '' = explicitly empty, omit = role=E defaults. alpha sees C UNION E; V-REx runs over E.")
    parser.add_argument("--adjustment_set", default=None, help="Comma-separated C fields for THIS arm; '' = explicitly empty, omit = declared default. Generator and BOTH DR legs must match.")


def add_syn_cli(parser) -> None:
    """Register the step-C knobs (IMPLEMENT.md §3.8.2).

    `--syn_effect` is the switch: 0 (the default, and v1) leaves `syn_c` inert.
    Nonzero needs a resolved `syn_meta.json` (`python -m src.data.synthetic`),
    which carries beta and the direction v so the trainer and the oracle cannot
    drift onto different ground truths.
    """
    parser.add_argument("--syn_effect", type=float, default=None,
                        help="Step-C injected effect size; beta = this x the median "
                             "responder ||tau_hat||. Default: OutcomeSpec (0 = inert).")
    parser.add_argument("--syn_seed", type=int, default=None,
                        help="The syn_c ASSIGNMENT seed. It must equal the one the table "
                             "was built with (population_qc.json), so this exists to make "
                             "a mismatch explicit, not to re-draw syn_c.")
    parser.add_argument("--syn_meta", default=None,
                        help="Which resolved injection --syn_effect switches on: "
                             "syn_meta.json (step C, the default) or e.g. "
                             "syn_meta_compound_r1.json (step C2, §3.8.4).")


def check_syn_args(cfg: CaseConfig, args) -> None:
    """Refuse `--syn_meta` when `--syn_effect` is 0 (the default).

    `syn_effect` alone switches the injection on, so a `--syn_meta` without it
    would be dropped silently: the run would train or score on uninjected data
    while its command line says step C2. Called by the trainer and by evaluate;
    the test scripts build uninjected configs on purpose and do not call it.
    """
    if getattr(args, "syn_meta", None) and float(cfg.outcome.syn_effect) == 0:
        raise SystemExit(f"--syn_meta {args.syn_meta} selects an injection, but --syn_effect "
                         f"is 0, which switches it off. Pass --syn_effect (steps C and C2 "
                         f"use 1.0).")


def config_from_args(args) -> CaseConfig:
    """`default_config()` with an adjustment_set (and `--population`, if given). """
    pop_name = getattr(args, "population", None)
    cfg = default_config() if pop_name is None else CaseConfig(population=POPULATIONS[pop_name])
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
        cfg.population = replace(cfg.population, compounds=tuple(names) or None)
    raw_e = getattr(args, "environment_set", None)
    if raw_e is not None:
        cfg.environment_set = tuple(c.strip() for c in raw_e.split(",") if c.strip())
    # Step C (§3.8.2). OutcomeSpec is frozen, so this is a replace(); its
    # __post_init__ re-validates. `syn_c` itself is already in the table -- only
    # the injection is switched on here.
    syn = {k: getattr(args, k) for k in ("syn_effect", "syn_seed")
           if getattr(args, k, None) is not None}
    if syn:
        cfg.outcome = replace(cfg.outcome, **syn)
    if getattr(args, "syn_meta", None):
        cfg.syn_meta_name = str(args.syn_meta)
    if raw is not None or raw_e is not None:
        cfg.__post_init__()
    return cfg


def add_paths_cli(parser, *, nuisance_dir: bool = True) -> None:
    """Register `--data_dir` (a build other than data/<population>, e.g. a --limit
    smoke build) and `--nuisance_dir` (a split dir, e.g. from build_tiered_split)."""
    parser.add_argument("--data_dir", default=None, help="Build dir (default data/<population>/); e.g. data/mcf7_24h_limit1500 for the smoke build.")
    if nuisance_dir:
        parser.add_argument("--nuisance_dir", default=None, help="Split dir (default <data_dir>/nuisances/): splits.json, vocab, encoders, nuisances.")


def apply_paths_args(cfg: CaseConfig, args, *, require_splits: bool = True) -> CaseConfig:
    """Point cfg.paths at `--data_dir` / `--nuisance_dir` when given (see add_paths_cli)."""
    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        # runs follow the build: data/mcf7_24h_limit1500 -> runs/mcf7_24h_limit1500
        data_dir = os.path.abspath(data_dir)
        # The build names its own population; `--population` may confirm it but
        # never contradict it, so one flag cannot point a command at the wrong
        # table. Without `--population` the build's population is adopted.
        built = population_of_build(data_dir)
        if built is not None and built != cfg.population.name:
            if getattr(args, "population", None) is not None:
                raise SystemExit(f"--population {cfg.population.name} but {data_dir} is a build of "
                                 f"{built!r} (its population_qc.json); they must agree")
            if built not in POPULATIONS:
                raise SystemExit(f"{data_dir} is a build of {built!r}, which spec.POPULATIONS "
                                 f"does not declare")
            cfg.population = replace(POPULATIONS[built], compounds=cfg.population.compounds)
        cfg.paths = Paths(population=cfg.population.name, raw_dir=cfg.paths.raw_dir, data_dir=data_dir,
                          train_output_dir=str(PROJECT_ROOT / "runs" / os.path.basename(data_dir)))
    nz = getattr(args, "nuisance_dir", None)
    if nz:
        nz = os.path.abspath(nz)
        if require_splits and not os.path.isfile(os.path.join(nz, "splits.json")):
            raise FileNotFoundError(f"{nz} has no splits.json; build it with `python -m src.data.build_tiered_split`")
        cfg.paths.nuisance_dir = nz
    return cfg
