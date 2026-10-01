"""Generation primitives for the LINCS evaluation suite.

    load_arm                  rebuild a trained arm from its arch.json + weights
    check_arm_against_data    refuse a checkpoint that was trained on other data
    SampleTarget / targets_from_rows   actions at real decision contexts
    generate_batch            sample (B, n_genes) vectors, optionally CFG-guided
    generate_for_rows         streaming driver: per-row means + a sample reservoir

Copied from `RxRx19a/src/eval/generation.py` (IMPLEMENT.md §3.1.1: always copy,
never import across trees), then adapted for a 978-d vector outcome:

  - **No image code.** `x.clamp(-1, 1)` is gone (z-space has no bound, and the
    clamp would truncate ~32% of z-scored gene values -- IMPLEMENT.md §2.2,
    §3.3), and so is `x.to(memory_format=torch.channels_last)` (it raises on a
    rank-2 tensor). `LatentCtx` / the image VAE / `_load_real_images` are not
    ported at all.
  - `build_generator_from_ckpt(ckpt_dir)` takes one argument here, not RxRx's
    `(cfg, ckpt_dir)`.
  - `SampleTarget` loses RxRx's `infected` bit (meaningless on LINCS, §2.2).
  - **`check_arm_against_data` is new and required.** `build_generator_from_ckpt`
    deliberately leaves the gene-order check to its caller, and `load_expr_meta`
    only guards split-vs-expr_meta, never checkpoint-vs-data.
  - **`generate_for_rows` is new.** At E3's 16 samples per real row, `--pool all`
    is 597k samples x 978 floats = 2.3 GB per arm, so nothing of that size is
    held: per-row sums are accumulated and a pre-drawn subset of raw samples is
    kept for the distribution metrics.

The architecture and the conditioning contract both come from the checkpoint's
`arch.json`; the schedule comes from it too, via `make_eval_scheduler_for_ckpt`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import CONTEXT_FIELDS, context_for_rows  # noqa: E402
from src.data.dataset import cond_from_arrays  # noqa: E402
from src.models import build_generator_from_ckpt, read_arch_spec  # noqa: E402
from src.processes import make_eval_scheduler_for_ckpt  # noqa: E402
from src.spec import CaseConfig  # noqa: E402

# arch.json fields that must agree between a checkpoint and the data it is
# scored on. A mismatch means the arm learned a different outcome definition,
# a different population, or a different split, so tau is not comparable.
IDENTITY_FIELDS = ("population", "table_fingerprint", "split_fingerprint",
                   "plate_center", "normalize_mean", "normalize_std",
                   "syn_effect", "syn_seed", "syn_beta", "syn_v_sha1")

# Substream id for the reservoir draw (see `_rng` in evaluate.py for the
# same idea: every draw site names itself, so draws do not depend on order).
_STREAM_RESERVOIR = 11


def row_seed(seed: int, row: int, rep: int = 0) -> int:
    """Noise seed for one generated sample.

    Depends on (seed, row, rep) alone, so a sample is identical however the rows
    were batched -- copied from RxRx's `_row_seed`, with the replicate index
    added so `--n_per_row > 1` does not repeat one row's noise.
    """
    h = hashlib.blake2b(f"{seed}|gen|{int(row)}|{int(rep)}".encode(), digest_size=4)
    return int.from_bytes(h.digest(), "little")


# ---------------------------------------------------------------------------
# Model rebuild
# ---------------------------------------------------------------------------

def _cond_spec(model: torch.nn.Module):
    """The conditioning contract of a (possibly wrapped) model."""
    return getattr(model, "module", model).cond_spec


def supports_cfg_null(model: torch.nn.Module) -> bool:
    """True if the arm was trained with CFG dropout, i.e. it has a real
    action-marginal branch to guide away from."""
    m = getattr(model, "module", model)
    return float(getattr(m, "class_dropout_prob", 0.0)) > 0.0


def checkpoint_dir(run_dir: str, epoch: int) -> str:
    d = os.path.join(run_dir, f"checkpoint-{int(epoch):04d}")
    if not os.path.isdir(d):
        have = sorted(x for x in os.listdir(run_dir) if x.startswith("checkpoint-")) \
            if os.path.isdir(run_dir) else []
        raise FileNotFoundError(
            f"{d} does not exist. {run_dir} holds {have or 'no checkpoints'}.")
    return d


def load_weights(model: torch.nn.Module, ckpt_dir: str, which: str) -> str:
    """Load `which` weights into `model`. 'ema' is `model_1.safetensors`: the EMA
    copy accelerate prepared second, which is what Phase 4 scores (E1)."""
    if which not in ("ema", "train"):
        raise ValueError(f"which must be 'ema' or 'train', got {which!r}")
    fname = "model_1.safetensors" if which == "ema" else "model.safetensors"
    path = os.path.join(ckpt_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # strict: a silently partial load would score a half-initialised net.
    model.load_state_dict(safetensors_load_file(path), strict=True)
    return path


def load_arm(run_dir: str, *, epoch: int = 499, which: str = "ema",
             device: torch.device | str = "cuda", sampler: str = "ddim",
             n_inference_steps: int | None = None):
    """Rebuild one trained arm: (model, scheduler, arch, ckpt_dir).

    The model is returned in `.eval()` mode, which is load-bearing: with
    `model.training` true and `class_dropout_prob > 0`,
    `layers.resolve_timestep_and_drop` draws its OWN CFG mask, so ~10% of eval
    rows would silently go unconditional. Every v1 arm has
    `class_dropout_prob = 0.1`.

    `calibrate_conditioning` is deliberately NOT called: it rescales the
    continuous-field weights in place and the trainer already did it before the
    EMA copy, so its factors are inside both checkpointed state dicts.
    """
    ckpt_dir = checkpoint_dir(run_dir, epoch)
    arch = read_arch_spec(ckpt_dir)
    if arch is None:
        raise FileNotFoundError(f"no arch.json for {ckpt_dir} or its parent")
    model = build_generator_from_ckpt(ckpt_dir)
    load_weights(model, ckpt_dir, which)
    model = model.to(device).eval()
    scheduler = make_eval_scheduler_for_ckpt(ckpt_dir, sampler)
    if n_inference_steps is not None:
        scheduler.set_timesteps(int(n_inference_steps), device=device)
    return model, scheduler, arch, ckpt_dir


def check_arm_against_data(arch: dict, cfg: CaseConfig, splits: dict,
                           expr_meta: dict, *, n_compounds: int | None = None,
                           syn: dict | None = None) -> dict:
    """Refuse to score a checkpoint against data it was not trained on.

    `build_generator_from_ckpt`'s docstring assigns the gene-order check to the
    caller (eval), and nothing else compares a checkpoint's provenance with the
    split and outcome definition in front of it. Returns the checked values for
    the result JSON.
    """
    bad: list[str] = []

    gene_ids = [int(g) for g in expr_meta["pr_gene_id"]]
    if "gene_pr_ids" in arch and [int(g) for g in arch["gene_pr_ids"]] != gene_ids:
        bad.append(f"gene order differs from the data's {len(gene_ids)} genes")
    if "gene_order_sha1" in arch:
        # Same recipe train_diffusion used to write the field.
        want = hashlib.sha1(json.dumps(gene_ids).encode()).hexdigest()
        if str(arch["gene_order_sha1"]) != want:
            bad.append(f"gene_order_sha1 {arch['gene_order_sha1'][:12]} != {want[:12]}")
    if int(arch.get("n_genes", -1)) != int(cfg.outcome.n_genes):
        bad.append(f"n_genes {arch.get('n_genes')} != cfg {cfg.outcome.n_genes}")

    # A tiered arm (steps C / A) trains on a THINNED split but is scored on the
    # unablated pool (§3.8.1), so its own split_fingerprint cannot match the
    # eval's. The check that still holds -- and is stronger than skipping -- is
    # that the arm's UNTHINNED pool (`nu_rows.npy`) plus its holdout reproduce
    # exactly the split being scored.
    tier = arch.get("tier") or {}
    if tier.get("active"):
        from src.data.splits import SPLITS_FILENAME, split_fingerprint
        nz = arch.get("nuisance_dir") or ""
        try:
            with open(os.path.join(nz, SPLITS_FILENAME)) as fh:
                ts = json.load(fh)
            nu = np.load(os.path.join(nz, "nu_rows.npy"))
            base = split_fingerprint(nu, np.asarray(ts["holdout_idx"], dtype=np.int64),
                                     np.asarray(ts["reserve_idx"], dtype=np.int64))
        except (OSError, KeyError, ValueError) as e:
            bad.append(f"tiered arm: cannot recover its unthinned split from {nz!r} ({e})")
            base = None
        if base is not None and base != splits.get("split_fingerprint"):
            bad.append(f"tiered arm: its unthinned pool fingerprints {base} but the "
                       f"eval split is {splits.get('split_fingerprint')}")
        arch = dict(arch)
        arch.pop("split_fingerprint", None)   # checked above, in the only way it can be

    have = {"population": cfg.population.name,
            "table_fingerprint": expr_meta.get("table_fingerprint"),
            "split_fingerprint": splits.get("split_fingerprint"),
            "plate_center": expr_meta.get("plate_center"),
            "normalize_mean": expr_meta.get("normalize_mean"),
            "normalize_std": expr_meta.get("normalize_std"),
            "syn_effect": float(cfg.outcome.syn_effect),
            "syn_seed": int(cfg.outcome.syn_seed),
            # The injection the eval itself applied; an arm trained on a different
            # beta or direction is not comparable with this oracle (§3.8.2).
            "syn_beta": (float(syn["beta"]) if syn else 0.0),
            "syn_v_sha1": (str(syn["v_sha1"]) if syn else None)}
    for k in IDENTITY_FIELDS:
        if k not in arch:
            continue
        a, b = arch[k], have[k]
        same = (abs(float(a) - float(b)) < 1e-12 if isinstance(b, float)
                else int(a) == int(b) if isinstance(b, int) and not isinstance(b, bool)
                else str(a) == str(b))
        if not same:
            bad.append(f"{k}: checkpoint {a!r} != data {b!r}")

    # Roles: the generator's C must equal the one eval conditions on, or the
    # sampled distribution is not the one the estimand is defined over.
    arch_c = tuple(str(c) for c in (arch.get("adjustment_set") or ()))
    if arch_c != tuple(cfg.adjustment_set):
        bad.append(f"adjustment_set: checkpoint {list(arch_c)} != cfg {list(cfg.adjustment_set)}")
    if bool(arch.get("include_env", False)) and not bool(arch.get("environment_set")):
        bad.append("include_env is set but environment_set is empty")
    if n_compounds is not None:
        for f in arch.get("cond_spec", []):
            if f.get("name") == "compound" and int(f.get("cardinality", -1)) != int(n_compounds):
                bad.append(f"compound cardinality {f.get('cardinality')} != "
                           f"nuisance_meta n_compounds {n_compounds}")

    if bad:
        raise RuntimeError(
            "checkpoint / data mismatch; tau would not be comparable:\n  - "
            + "\n  - ".join(bad))
    return dict(have, gene_order_sha1=arch.get("gene_order_sha1"),
                adjustment_set=list(arch_c))


# ---------------------------------------------------------------------------
# Sampling targets
# ---------------------------------------------------------------------------

@dataclass
class SampleTarget:
    """One action (A) at one decision context (C, E).

    `context` is a full row of `ContextEncoder` codes in `CONTEXT_FIELDS` order.
    v1's cond_spec holds only role-A fields, so no context column is read -- but
    `cond_from_arrays` indexes this tensor for any C/E field, so it must still be
    the right width for steps C and A.
    """

    compound_idx: int
    log10_conc: float
    is_control: int              # 0/1
    context: tuple[int, ...]     # (F,) ContextEncoder codes, CONTEXT_FIELDS order

    def __post_init__(self):
        if len(self.context) != len(CONTEXT_FIELDS):
            raise ValueError(
                f"SampleTarget.context has {len(self.context)} entries, expected "
                f"{len(CONTEXT_FIELDS)} (one per CONTEXT_FIELDS). Pass a row of "
                f"context_for_rows(cfg, meta, pick), not individual field indices.")


def targets_from_rows(cfg: CaseConfig, meta, pick: np.ndarray) -> list[SampleTarget]:
    """SampleTargets for `pick` rows of the tabular dataset, in order.

    `meta` is the HF dataset or a pandas frame carrying `compound_idx`,
    `log10_conc`, `is_control` and the `CONTEXT_SOURCE_COLUMNS`.
    """
    pick = np.asarray(pick, dtype=np.int64)
    compound_idx = np.asarray(meta["compound_idx"], dtype=np.int64)[pick]
    log10_conc = np.asarray(meta["log10_conc"], dtype=np.float32)[pick]
    is_control = np.asarray(meta["is_control"], dtype=np.int64)[pick]
    ctx = context_for_rows(cfg, meta, pick)
    return [SampleTarget(int(c), float(l), int(ic), tuple(int(v) for v in row))
            for c, l, ic, row in zip(compound_idx, log10_conc, is_control, ctx)]


# ---------------------------------------------------------------------------
# Sampling driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    scheduler,
    targets: list[SampleTarget],
    n_inference_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float = 1.0,
    row_seeds: "list[int] | None" = None,
) -> torch.Tensor:
    """Generate one batch of gene vectors for `targets`. Returns (B, n_genes).

    Never clamped: samples live in z-space (IMPLEMENT.md §3.3).

    `row_seeds` seeds each row's init noise separately. Without it the whole
    batch shares one generator, so a row's sample depends on its position in the
    batch and changes whenever batching does.
    """
    m = getattr(model, "module", model)
    n_genes = int(m.n_genes)
    B = len(targets)
    if row_seeds is None:
        g = torch.Generator(device=device).manual_seed(int(seed))
        x = torch.randn(B, n_genes, generator=g, device=device)
    else:
        if len(row_seeds) != B:
            raise ValueError(f"row_seeds has {len(row_seeds)} entries for {B} targets")
        x = torch.stack([
            torch.randn(n_genes, device=device,
                        generator=torch.Generator(device=device).manual_seed(int(s)))
            for s in row_seeds])

    spec = _cond_spec(model)
    compound = torch.tensor([t.compound_idx for t in targets], dtype=torch.long, device=device)
    lx = torch.tensor([t.log10_conc for t in targets], dtype=torch.float32, device=device)
    ic = torch.tensor([t.is_control for t in targets], dtype=torch.long, device=device)
    ctx = torch.tensor(np.array([t.context for t in targets]), dtype=torch.long, device=device)

    use_guidance = abs(guidance_scale - 1.0) > 1e-6
    null_cfg = use_guidance and supports_cfg_null(model)
    scheduler.set_timesteps(int(n_inference_steps), device=device)
    # One conditioning dict for the whole trajectory.
    cond = cond_from_arrays(spec, compound=compound, log10_conc=lx,
                           is_control=ic, context=ctx, device=device)
    if use_guidance and not null_cfg:
        ref_cond = cond_from_arrays(spec, compound=torch.zeros_like(compound), log10_conc=lx,
                                   is_control=torch.ones_like(ic), context=ctx, device=device)
    drop_all = torch.ones(B, dtype=torch.bool, device=device) if null_cfg else None

    # Iterate `timesteps` AS GIVEN: diffusers descends T->0, FlowMatching
    # ascends noise->data. Both start from pure noise. `step` gets no
    # `generator=` kwarg -- FlowMatching.step does not accept one.
    for t in scheduler.timesteps:
        pred = model(x, t, cond, return_dict=False)[0]
        if use_guidance:
            if null_cfg:
                # Same cond dict with every role-A field dropped: the reference
                # is the action marginal, not an untreated well.
                pred_ref = model(x, t, cond, drop=drop_all, return_dict=False)[0]
            else:
                pred_ref = model(x, t, ref_cond, return_dict=False)[0]
            pred = pred_ref + guidance_scale * (pred - pred_ref)
        x = scheduler.step(pred, t, x).prev_sample
    return x


def _reservoir_mask(group_of_row: np.ndarray, n_per_row: int, cap: int,
                    seed: int) -> np.ndarray:
    """Which (rep, row) items to retain for the distribution metrics.

    The subset is drawn up front from the known item count, so which samples are
    kept depends only on `seed` -- not on `--gen_batch_size` and not on arrival
    order (an online reservoir would depend on both).
    """
    n_rows = int(group_of_row.size)
    item_group = np.tile(np.asarray(group_of_row, dtype=np.int64), int(n_per_row))
    keep = np.zeros(item_group.size, dtype=bool)
    if cap <= 0:
        return keep
    # A named stream, so adding a draw site elsewhere cannot shift this subset.
    rng = np.random.default_rng([int(seed), _STREAM_RESERVOIR])
    for g in np.unique(item_group):
        if g < 0:
            continue
        idx = np.flatnonzero(item_group == g)
        take = min(int(cap), idx.size)
        keep[rng.choice(idx, take, replace=False)] = True
    return keep


@torch.no_grad()
def generate_for_rows(
    model: torch.nn.Module,
    scheduler,
    cfg: CaseConfig,
    meta,
    rows: np.ndarray,
    *,
    n_per_row: int,
    n_inference_steps: int,
    device: torch.device,
    seed: int = 0,
    guidance_scale: float = 1.0,
    batch_size: int = 1024,
    group_of_row: np.ndarray | None = None,
    reservoir_cap: int = 4096,
    log_every: int = 20,
) -> dict:
    """Sample `n_per_row` wells at each of `rows`, streaming.

    Returns
        row_mean (n_rows, G) float32   mean sample per row -- what tau uses
        row_var  (n_rows, G) float32   within-row variance across replicates
        reservoir dict{group -> (m, G) float32}   raw samples for MMD / Frechet
        n_samples, n_per_row, forward_calls

    Nothing of size n_rows * n_per_row * G is ever materialised.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n_rows = int(rows.size)
    if n_rows == 0:
        raise ValueError("no rows to generate at")
    if n_per_row < 1:
        raise ValueError(f"n_per_row must be >= 1, got {n_per_row}")
    m = getattr(model, "module", model)
    n_genes = int(m.n_genes)

    targets = targets_from_rows(cfg, meta, rows)
    if group_of_row is None:
        group_of_row = np.zeros(n_rows, dtype=np.int64)
    group_of_row = np.asarray(group_of_row, dtype=np.int64)
    if group_of_row.size != n_rows:
        raise ValueError(f"group_of_row has {group_of_row.size} entries for {n_rows} rows")
    keep = _reservoir_mask(group_of_row, n_per_row, reservoir_cap, seed)

    # float64 accumulators: 597k samples per arm, so a float32 sum would lose
    # low-order bits on the long tail.
    tot = np.zeros((n_rows, n_genes), dtype=np.float64)
    tot_sq = np.zeros((n_rows, n_genes), dtype=np.float64)
    res: dict[int, list[np.ndarray]] = {}
    n_chunks = (n_rows + batch_size - 1) // batch_size
    done = 0
    for rep in range(int(n_per_row)):
        for ci in range(n_chunks):
            lo, hi = ci * batch_size, min((ci + 1) * batch_size, n_rows)
            chunk = list(range(lo, hi))
            batch = [targets[i] for i in chunk]
            seeds = [row_seed(seed, int(rows[i]), rep) for i in chunk]
            x = generate_batch(model, scheduler, batch, n_inference_steps, device,
                               seed=seed, guidance_scale=guidance_scale,
                               row_seeds=seeds)
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    f"non-finite samples at rep {rep} chunk {ci}; the sampler "
                    f"diverged (check --num_inference_steps and the schedule)")
            xn = x.detach().to(torch.float32).cpu().numpy()
            tot[lo:hi] += xn
            tot_sq[lo:hi] += xn.astype(np.float64) ** 2
            item0 = rep * n_rows
            sel = keep[item0 + lo: item0 + hi]
            if sel.any():
                for g in np.unique(group_of_row[lo:hi][sel]):
                    take = sel & (group_of_row[lo:hi] == g)
                    res.setdefault(int(g), []).append(xn[take].copy())
            done += hi - lo
            if log_every and (ci % log_every == 0 or hi == n_rows):
                print(f"[gen] rep {rep + 1}/{n_per_row} chunk {ci + 1}/{n_chunks} "
                      f"({done}/{n_rows * n_per_row} samples)", flush=True)

    n = float(n_per_row)
    row_mean = (tot / n).astype(np.float32)
    # Population variance across replicates; 0 when n_per_row == 1.
    row_var = np.maximum(tot_sq / n - (tot / n) ** 2, 0.0).astype(np.float32)
    return {"row_mean": row_mean, "row_var": row_var,
            "reservoir": {g: np.concatenate(v, axis=0) for g, v in res.items()},
            "n_samples": int(n_rows * n_per_row), "n_per_row": int(n_per_row),
            "n_rows": n_rows}
