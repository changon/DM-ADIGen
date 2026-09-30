"""We adopt the approach from DiTs with adaLN conditioning, to more `statistical' settings.

The main non-triviality for adapting DiTs is deciding how conditioning is inputted into the network. 
In essence, we will embed each covariate based on role (action, confounder, environment) and type (continuous or categorical), and add it up embeddings for the adaLN application.

    A  action        The intervention whose effect is the estimand. Under CFG, it will look something like v = v_null + w * (v_action - v_null)
    C  confounder    The adjustment set. C is adjusted for across the URR and outcome model.
    E  environment   Environment info over which we learn causal invariants (plate, well position, imaging site). E is inputted into the URR, but not the outcome model, so is separate.

Based on kind/type this, we add the following layers to process it:
    kind="cat" (categorical)   -> nn.Embedding(cardinality)   + one extra "null" row if role A (relevant if CFG is on)
    kind="cont" (continuous) -> MLP over (v - loc) / scale  + a learned null VECTOR if role A or if the field is `nullable` (relevant if CFG is on)

Note, interpretation is a bit odd. in usual CFG, no class label is given so it's a marginal/posterior weighted avg, or a class averaged score. 
    Here, class is a treatment specification so no token provided is a marginal over tx, though C is \textit{not} dropped. Hence, this is an output conditional on the adjustment set.

Additional Remarks:
--- Interestingly NA can be expected at times. For example, a well may get something known as DMSO which would mean there is no compound nor concentration to record.
Here, NaN is given a special embedding that is learned, and separate from the other tx info.

--- Categorical fields carry their own `levels` and this is detailed in `arch.json`.
A checkpoint has its own index->meaning map and a re-ordered encoder will need extra work. 
`Field.encode` gives error on unseen levels.

--- In RxRx19a, one could technically make a joint table over (compound x dose x disease x cell_type),
 but this would call for ~1500*14*4*2 rows that are 
(a) largely empty and (b) fail to directly leverage shared info between common compounds

"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as _dc_field, replace
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

__all__ = [
    "ROLES",
    "ROLE_ACTION",
    "ROLE_CONFOUNDER",
    "ROLE_ENV",
    "Field",
    "CondSpec",
    "CondEmbedder",
]

# variable role, or purpose as described above
ROLE_ACTION = "A"
ROLE_CONFOUNDER = "C"
ROLE_ENV = "E"
ROLES = (ROLE_ACTION, ROLE_CONFOUNDER, ROLE_ENV)

# same as variable type
KIND_CAT = "cat"
KIND_CONT = "cont"
KINDS = (KIND_CAT, KIND_CONT)

@dataclass(frozen=True)
class Field:
    """One conditioning variable.

    name        key in the `cond` dict handed to the model.
    role        "A" | "C" | "E"  (see module docstring). Only A is droppable if doing CFG
    kind        "cat" | "cont". Variable type.
    cardinality categorical only: number of real levels (excludes the null row).
    levels      categorical only, optional: the level values, in index order. This is related in arch.json to be a convention to follow.
    dim         continuous only: number of components (1 for a scalar).
    loc, scale  continuous only: standardisation, v -> (v - loc) / scale. Recorded so standardization is kept consistent.
    nullable    continuous only: the field may be "not applicable" on some rows, as described in remarks above, signaled using the NaN. Gets a learned null vector even when role != A.
    n_freqs     continuous only: 0 (default) feeds the standardised value straight to the MLP. 
                                 >0 prepends `n_freqs` Fourier features per component. Helpful if target has a reasonable structure to model as such.
    """

    name: str
    role: str
    kind: str
    cardinality: int | None = None
    levels: tuple[str, ...] | None = None
    dim: int = 1
    loc: float = 0.0
    scale: float = 1.0
    nullable: bool = False
    n_freqs: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Field.name must be non-empty")
        if self.role not in ROLES:
            raise ValueError(f"{self.name}: role must be one of {ROLES}, got {self.role!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == KIND_CAT:
            if not self.cardinality or self.cardinality < 1:
                raise ValueError(f"{self.name}: categorical fields need cardinality >= 1")
            if self.levels is not None and len(self.levels) != self.cardinality:
                raise ValueError(
                    f"{self.name}: {len(self.levels)} levels declared but "
                    f"cardinality={self.cardinality}")
            if self.nullable:
                raise ValueError(
                    f"{self.name}: `nullable` is a continuous-field concept. A "
                    f"categorical field that can be absent should declare an "
                    f"explicit level for it.")
            if self.n_freqs:
                raise ValueError(f"{self.name}: n_freqs applies to continuous fields only")
        else:
            if self.dim < 1:
                raise ValueError(f"{self.name}: continuous fields need dim >= 1")
            if self.scale == 0:
                raise ValueError(f"{self.name}: scale must be non-zero")
            if self.cardinality is not None or self.levels is not None:
                raise ValueError(f"{self.name}: cardinality/levels apply to categorical fields only")
            if self.n_freqs < 0:
                raise ValueError(f"{self.name}: n_freqs must be >= 0")

    # -- derived ------------------------------------------------------------
    @property
    def droppable(self) -> bool:
        """True if classifier-free guidance replaces this field with its null. Field helps ascertain the conditioning effect, from the intervention A, during CFG computation. 
            I.e. , only drop A here. O.w., we do not have a causal effect.
        """
        return self.role == ROLE_ACTION

    @property
    def needs_null(self) -> bool:
        """Continuous fields need a learned null vector if they can be dropped during CFG or can be not-applicable (NaN). 
        Categorical fields express the null as an extra table row instead."""
        return self.kind == KIND_CONT and (self.droppable or self.nullable)

    @property
    def n_rows(self) -> int:
        """Rows in the embedding table, including the CFG null row if any."""
        if self.kind != KIND_CAT:
            raise AttributeError(f"{self.name} is not categorical")
        return int(self.cardinality) + (1 if self.droppable else 0)

    @property
    def null_index(self) -> int:
        """Index of the CFG null row (the row past the last real level). This is og DiT's LabelEmbedder trick, generalised."""
        if self.kind != KIND_CAT or not self.droppable:
            raise AttributeError(f"{self.name} has no categorical null row")
        return int(self.cardinality)

    @property
    def in_features(self) -> int:
        """Width of the continuous field's MLP input. primarily bookkeeping maintained in arch.json"""
        if self.kind != KIND_CONT:
            raise AttributeError(f"{self.name} is not continuous")
        return self.dim * (1 + 2 * self.n_freqs)

    # -- data helpers -------------------------------------------------------
    def encode(self, values: Sequence[Any]) -> list[int]:
        """Map raw level values such as HRCE, Active SARS Cov 2... to indices that consumable by nnn.Embedding.
            Errors out if level value is not known.
        """
        if self.kind != KIND_CAT:
            raise AttributeError(f"{self.name} is not categorical")
        if self.levels is None:
            raise ValueError(
                f"{self.name}: encode() needs pinned `levels`; the spec declares "
                f"only cardinality={self.cardinality}")

        lookup = {lv: i for i, lv in enumerate(self.levels)}
        out: list[int] = []
        unseen: set[str] = set()
        for v in values:
            key = str(v)
            if key in lookup:
                out.append(lookup[key])
            else:
                unseen.add(key)

        if unseen:
            raise KeyError(
                f"{self.name}: level(s) {sorted(unseen)!r} are not in the pinned "
                f"vocabulary {list(self.levels)!r}. The checkpoint was trained "
                f"against that vocabulary; re-encoding against a different one "
                f"would silently relabel the covariate.")
        return out

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "role": self.role, "kind": self.kind}
        if self.kind == KIND_CAT:
            d["cardinality"] = int(self.cardinality)
            if self.levels is not None:
                d["levels"] = list(self.levels)
        else:
            d.update(dim=int(self.dim), loc=float(self.loc), scale=float(self.scale))
            if self.nullable:
                d["nullable"] = True
            if self.n_freqs:
                d["n_freqs"] = int(self.n_freqs)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Field":
        lv = d.get("levels")
        return cls(
            name=str(d["name"]),
            role=str(d["role"]),
            kind=str(d["kind"]),
            cardinality=(int(d["cardinality"]) if d.get("cardinality") is not None else None),
            levels=(tuple(str(x) for x in lv) if lv is not None else None),
            dim=int(d.get("dim", 1)),
            loc=float(d.get("loc", 0.0)),
            scale=float(d.get("scale", 1.0)),
            nullable=bool(d.get("nullable", False)),
            n_freqs=int(d.get("n_freqs", 0)),
        )


@dataclass(frozen=True)
class CondSpec:
    """The conditioning specification for the generator.

    Stored in `arch.json`, so a checkpoint carries its own description of
    - what it conditions on
    - in what order
    - with what vocabularies 
    
    Rebuilding a model from a checkpoint must follow this contract.
    """

    fields: tuple[Field, ...] = _dc_field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate field names in CondSpec: {sorted(dupes)}")

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{name!r} not in CondSpec (have {self.names})")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def by_role(self, role: str) -> tuple[Field, ...]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        return tuple(f for f in self.fields if f.role == role)

    def names_by_role(self, role: str) -> tuple[str, ...]:
        return tuple(f.name for f in self.by_role(role))

    @property
    def has_null(self) -> bool:
        """True if any field is droppable, i.e. the model has an unconditional branch and classifier-free guidance is available at sampling time."""
        return any(f.droppable for f in self.fields)

    def with_fields(self, *extra: Field) -> "CondSpec":
        """A copy with more fields appended (adding `experiment`, an E block...)."""
        return CondSpec(self.fields + tuple(extra))

    def validate_against(self, other_c_fields: Iterable[str]) -> None:
        """Assert this spec's ADJUSTMENT SET matches another leg's.

        If the generator and URR adjust for different confounder sets, we have a problem since they should adjust for the same thing.
        E fields are deliberately NOT compared based on initial doc string.
        """
        mine = set(self.names_by_role(ROLE_CONFOUNDER))
        theirs = set(other_c_fields)
        if mine != theirs:
            raise ValueError(
                f"adjustment-set mismatch between the two DR legs.\n"
                f"  generator C fields: {sorted(mine)}\n"
                f"  alpha      C fields: {sorted(theirs)}\n"
                f"  only in generator:  {sorted(mine - theirs)}\n"
                f"  only in alpha:      {sorted(theirs - mine)}\n"
                f"Both legs must adjust for the same confounders. (Environment "
                f"fields are exempt and are not compared here.)")

    # -- serialisation ------------------------------------------------------
    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.fields]

    @classmethod
    def from_list(cls, items: Iterable[Mapping[str, Any]]) -> "CondSpec":
        return cls(tuple(Field.from_dict(d) for d in items))

    def to_json(self) -> str:
        return json.dumps(self.to_list(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CondSpec":
        return cls.from_list(json.loads(s))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
def fourier_features(v: torch.Tensor, n_freqs: int, max_freq: float = 5.0
                     ) -> torch.Tensor:
    """(N, D) -> (N, D * 2 * n_freqs) sin/cos features.    """
    if n_freqs < 1:
        raise ValueError("n_freqs must be >= 1 to build Fourier features")
    freqs = torch.exp(
        math.log(max_freq) * torch.arange(n_freqs, dtype=torch.float32, device=v.device) / max(n_freqs - 1, 1)
    )                                              # (F,) 1 .. max_freq
    args = v.unsqueeze(-1) * freqs                 # (N, D, F)
    out = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (N, D, 2F)
    return out.flatten(1)


class _ContEmbedder(nn.Module):
    """MLP over a standardised continuous field. """

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.mlp(v.to(self.mlp[0].weight.dtype))


class CondEmbedder(nn.Module):
    """Turns a `{name: tensor}` conditioning dict into the adaLN vector `c`.

    forward(cond, drop=None, terms_out=None) -> (N, hidden_size)

      cond      per-field tensors.
                    Categorical: (N,) integer indices.
                    Continuous: (N,) or (N, dim) floats, with NaN marking "not applicable" on a `nullable` field.
      drop      optional (N,) bool. Where True, every ACTION field is replaced by its learned null
                    Note, here, C and E should be unaffected by the masking done.
      terms_out optional dict; if given, filled with per-field ||term|| means so we can see how the breakdown of the context embeddings look (prior to adding it up)
    """

    def __init__(self, spec: CondSpec, hidden_size: int, *, cfg_null: bool = True):
        super().__init__()
        self.spec = spec
        self.hidden_size = int(hidden_size)
        self.cfg_null = bool(cfg_null)

        cat, cont, nulls = {}, {}, {}
        for f in spec:
            if f.kind == KIND_CAT:
                cat[f.name] = nn.Embedding(
                    f.n_rows if self.cfg_null else int(f.cardinality), hidden_size)
            else:
                cont[f.name] = _ContEmbedder(f.in_features, hidden_size)
                if f.needs_null:
                    nulls[f.name] = nn.Parameter(torch.zeros(hidden_size))
        self.cat = nn.ModuleDict(cat)
        self.cont = nn.ModuleDict(cont)
        self.nulls = nn.ParameterDict(nulls)

    # -- init ---------------------------------------------------------------
    def init_weights(self) -> None:
        """std=0.02 on every weight, as og DiT does to its label and timestep embedders.

        A continuous term depends on `dim`, `n_freqs`, and how well `loc`/`scale` actually standardise the field.
        """
        for emb in self.cat.values():
            nn.init.normal_(emb.weight, std=0.02)
        for mod in self.cont.values():
            nn.init.normal_(mod.mlp[0].weight, std=0.02)
            nn.init.normal_(mod.mlp[2].weight, std=0.02)
            nn.init.constant_(mod.mlp[0].bias, 0)
            nn.init.constant_(mod.mlp[2].bias, 0)
        for p in self.nulls.values():
            nn.init.normal_(p, std=0.02)

    def calibrate(self, probes: Mapping[str, torch.Tensor] | None = None, target: float | None = None, n_probe: int = 4096) -> dict[str, float]:
        """Scale each continuous field's output layer so terms used in conditioning are comparable.

            With probes, we get values to know the loc/scale of training, enabling us to standardize meaningfully across all contexts.
            Without it, randn is used as a stand in for normalization. this is subject to conditioning imbalance.

        Returns the applied factors, keyed by field, for logging.
        """
        if target is None:
            target = 0.02 * self.hidden_size ** 0.5
        probes = probes or {}
        factors: dict[str, float] = {}

        with torch.no_grad():
            for name, mod in self.cont.items():
                f = self.spec[name]
                dev = mod.mlp[0].weight.device
                if name in probes:
                    x = probes[name].to(dev).float()
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                    if x.shape[-1] != f.dim:
                        raise ValueError(
                            f"{name}: probe has {x.shape[-1]} components, "
                            f"spec says dim={f.dim}")
                    x = x[~torch.isnan(x).any(dim=-1)]
                    if x.numel() == 0:
                        raise ValueError(
                            f"{name}: probe is empty after dropping NaN rows; "
                            f"calibration needs real values, not only absent ones.")
                    x = (x - f.loc) / f.scale
                else:
                    x = torch.randn(n_probe, f.dim, device=dev)   # already standardised
                if f.n_freqs:
                    x = torch.cat([x, fourier_features(x, f.n_freqs)], dim=-1)
                measured = mod(x).norm(dim=-1).mean()
                k = float(target / measured.clamp_min(1e-12))
                mod.mlp[2].weight.mul_(k)
                factors[name] = k
        return factors

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        cond: Mapping[str, torch.Tensor],
        drop: torch.Tensor | None = None,
        terms_out: dict[str, float] | None = None,
    ) -> torch.Tensor:
        missing = [f.name for f in self.spec if f.name not in cond]
        if missing:
            raise KeyError(
                f"conditioning dict is missing {missing}; the spec declares "
                f"{list(self.spec.names)}")
        if drop is not None and not self.cfg_null:
            raise ValueError("drop was given but this embedder was built without CFG null rows (cfg_null=False), so dropped samples have no index to route to.")

        ref = cond[self.spec.names[0]]
        n = ref.shape[0]
        device = ref.device
        c = torch.zeros(n, self.hidden_size, device=device, dtype=torch.get_default_dtype())

        for f in self.spec:
            term = self._term(f, cond, drop, n)
            c = c + term.to(c.dtype)
            if terms_out is not None:
                terms_out[f.name] = float(term.detach().float().norm(dim=-1).mean())

        if terms_out is not None:
            terms_out["__total__"] = float(c.detach().float().norm(dim=-1).mean())
        return c

    def _term(self, f: Field, cond: Mapping[str, torch.Tensor],
              drop: torch.Tensor | None, n: int) -> torch.Tensor:
        """One field's contribution. The adaLN sum and the token path share this."""
        v = cond[f.name]
        if v.shape[0] != n:
            raise ValueError(
                f"{f.name}: batch dim {v.shape[0]} != {n} for the other fields")
        return (self._cat_term(f, v, drop) if f.kind == KIND_CAT
                else self._cont_term(f, v, drop))

    def field_tokens(self, cond: Mapping[str, torch.Tensor],
                     drop: torch.Tensor | None = None,
                     roles: tuple[str, ...] = ("A",)) -> torch.Tensor:
        """Per-field terms kept SEPARATE, as (N, k, hidden) tokens. The adaLN path sums these into one vector, which forces one modulation for each patch, xattn maintains each token separately for the xattn
        """
        names = [f.name for f in self.spec if f.role in roles]
        if not names:
            raise ValueError(f"no field in the spec has role in {roles}")
        missing = [nm for nm in names if nm not in cond]
        if missing:
            raise KeyError(f"conditioning dict is missing {missing}")
        if drop is not None and not self.cfg_null:
            raise ValueError(
                "drop was given but this embedder was built without CFG null rows")
        n = int(cond[names[0]].shape[0])
        return torch.stack(
            [self._term(self.spec[nm], cond, drop, n) for nm in names], dim=1)

    def _cat_term(self, f: Field, v: torch.Tensor, drop: torch.Tensor | None) -> torch.Tensor:
        idx = v.long()
        if idx.dim() != 1:
            idx = idx.reshape(idx.shape[0])
        # Range check every step.
        if idx.numel():
            hi = int(f.cardinality)
            bad = (idx < 0) | (idx >= hi)
            if bool(bad.any()):
                lo_v, hi_v = int(idx.min()), int(idx.max())
                raise ValueError(
                    f"{f.name}: indices in [{lo_v}, {hi_v}] but the field declares "
                    f"cardinality={hi}"
                    + (f" (levels={list(f.levels)})" if f.levels else ""))
        if drop is not None and f.droppable:
            idx = torch.where(drop, torch.full_like(idx, f.null_index), idx)
        return self.cat[f.name](idx)

    def _cont_term(self, f: Field, v: torch.Tensor,
                   drop: torch.Tensor | None) -> torch.Tensor:
        x = v.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.shape[-1] != f.dim:
            raise ValueError(f"{f.name}: expected dim {f.dim}, got {x.shape[-1]}")

        # NaN == "not applicable". Record where, then fill in with a proper nan indicator for processing.
        na = torch.isnan(x).any(dim=-1)
        if bool(na.any()) and not f.needs_null:
            raise ValueError(
                f"{f.name}: NaN present but the field is neither droppable nor "
                f"declared nullable=True. A continuous field that can be "
                f"not-applicable must declare it, so it gets a learned null "
                f"instead of a sentinel value.")
        x = torch.nan_to_num(x, nan=0.0)

        x = (x - f.loc) / f.scale
        if f.n_freqs:
            x = torch.cat([x, fourier_features(x, f.n_freqs)], dim=-1)
        term = self.cont[f.name](x)

        if f.needs_null:
            null = self.nulls[f.name].to(term.dtype).expand_as(term)
            use_null = na
            if drop is not None and f.droppable:
                use_null = use_null | drop
            term = torch.where(use_null.unsqueeze(-1), null, term)
        return term
