"""ADIGen diffusion training for LINCS L1000 (IMPLEMENT.md §3.6).

Copied from RxRx19a/src/train/train_diffusion.py (always copy, §3.1.1), then
adapted. Same loop: Accelerate, EMA, AdamW (fused), warm-up + exponential LR
decay, arch.json, DR weights aligned to train_idx and normalised to mean 1,
optional V-REx over plate, optional FM-only cmean loss, resume from
checkpoint-NNNN. LINCS edits:

  * Y is the (B, 978) plate-centred z-scored gene vector (batch key `y`); the
    per-sample loss is rank-agnostic (`.flatten(1).mean(1)`) in the training
    loop and in `_validation_loss`, through one `_per_sample_loss`.
  * --arch {mlp, dit1d} (required), --mlp_size / --dit_size, --patch_size.
    No --latent / VAE / n_channels / resolution; no xattn (--cond_mode).
  * --dr_mode {conditional, weighted} (P7); --dr_weights_file picks the
    source (dr_weights_urr.npz from `net`, the default; _counts; _design).
  * Paths through spec.add_paths_cli (--data_dir, --nuisance_dir).
  * --plate_center (decision 6 ablation) reaches both datasets and arch.json.
  * Rows live on the device (`_RowBatcher`); no DataLoader.
  * V-REx env = spec.invariance_env_fields (plate), keyed on the context codes.
  * Fixes against the RxRx copy (IMPLEMENT.md §5; RxRx is not patched):
      (a) the V-REx batch sampler gets set_epoch every epoch (RxRx never
          called it: every epoch replayed the same batches);
      (b) a checkpoint is always saved at the last epoch;
      (c) --cmean_lambda > 0 under DDPM is refused (RxRx skipped it silently);
      (d) the cmean aux forward passes an explicit no-drop mask (RxRx's
          train-mode forward drew its own CFG mask, so ~10% of aux rows
          pulled the null branch toward mu_hat(a));
      (e) cmean with a non-empty C or --include_env is refused (mu_hat is
          keyed on the action only).
  * Run-dir safety: a fresh start into a dir holding checkpoints is refused;
    a resume must match every identity field of arch.json.
  * --mixed_precision defaults to "no" (TF32 on); fp16 is opt-in.
  * Tracking: wandb through accelerate (RxRx: tensorboard, absent from this
    env), --wandb_mode {online, offline, disabled}. The run id is kept in the
    run dir, so a resume continues the same wandb run. loss_history.jsonl
    (per epoch) is written either way.

Run (from lincs/):
  python -m src.train.train_diffusion --arch mlp --num_epochs 500
  python -m src.train.train_diffusion --arch mlp --num_epochs 500 --dr_mode weighted   # dr_weights_urr.npz
  python -m src.train.train_diffusion --arch dit1d --num_epochs 500 [--dit_size S --patch_size 10]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import socket
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import (  # noqa: E402
    CONTEXT_COL, PLATE_CENTER_MODES, CaseConfig, add_adjustment_set_cli, add_paths_cli, add_syn_cli, check_syn_args,
    apply_paths_args, config_from_args, format_role_summary, invariance_env_fields, role_tag)
from src.models import (  # noqa: E402
    ARCHS, DIT_SIZES, MLP_SIZES, arch_spec, build_generator, read_arch_spec,
    resolve_arch_kwargs, write_arch_spec)
from src.processes import make_train_flow_matching, make_train_scheduler, training_target  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.nuisances.weight_norm import NORM_MODES, group_normalize, train_groups  # noqa: E402
from src.data.dataset import (  # noqa: E402
    LincsDataset, build_cond_spec, cond_from_batch, dose_probe)

HISTORY_FILENAME = "loss_history.jsonl"
WANDB_ID_FILENAME = "wandb_run_id.txt"
RUN_LOCK_FILENAME = ".train.lock"
# arch.json keys too bulky for the wandb config (they stay in arch.json).
_WANDB_SKIP_KEYS = ("gene_pr_ids", "cond_spec", "train_args")
VAL_SEED = 1234          # validation noise: the same draws every epoch, so val_loss is comparable
VAL_ROWS_SEED = 12345    # the val_cap subsample of the holdout

# arch.json keys not compared on resume: provenance of the latest launch.
# `dr_weight_stats` are floating-point summaries (ESS, max) whose last bits can
# differ between CPU types. The settings that produce them ARE compared, and so
# are the exact integer counts (`dr_weight_groups`), which would move if the
# grouping or the cap routine changed between a run and its resume.
_NON_IDENTITY_KEYS = ("train_args", "dr_weight_stats")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class _EnvStratifiedBatchSampler(torch.utils.data.Sampler):
    """Yield batches drawn from a few environments for invariance penalty

    Shuffling spreads a 256-row batch over ~100 plates (~2-3 rows each).
    Instead, we draw `envs_per_batch` environments per batch so we don't model over sampling noise (256/8 = 32 here).

    Sampling is with replacement within an env when the env is short, so small plates still participate instead of being dropped.
    """

    def __init__(self, env_ids: np.ndarray, batch_size: int, envs_per_batch: int,
                 seed: int = 0):
        self.batch_size = int(batch_size)
        self.envs_per_batch = max(2, int(envs_per_batch))
        self.per_env = max(1, self.batch_size // self.envs_per_batch)
        self.rows_by_env = {}
        for e in np.unique(env_ids):
            self.rows_by_env[int(e)] = np.where(env_ids == e)[0]
        self.envs = np.array(sorted(self.rows_by_env))
        self.envs_per_batch = min(self.envs_per_batch, len(self.envs))
        self.n_batches = max(1, len(env_ids) // self.batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        for _ in range(self.n_batches):
            chosen = rng.choice(self.envs, self.envs_per_batch, replace=False)
            batch = []
            for e in chosen:
                rows = self.rows_by_env[int(e)]
                take = rng.choice(rows, self.per_env,
                                  replace=len(rows) < self.per_env)
                batch.extend(int(i) for i in take)
            yield batch


class _RowBatcher:
    """Fixed rows held on the device; `epoch(e)` yields {key: tensor[idx]} batches.

    Replaces RxRx's DataLoader over a per-row Dataset: at batch 256 the per-row
    dict + default collate costs more than an MLP step. The shuffle is seeded by
    (seed, epoch), so a resumed run replays the batches of an uninterrupted one.
    With `batch_sampler` (V-REx), batches come from it instead.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], batch_size: int, device, *,
                 shuffle: bool, seed: int = 0, batch_sampler: _EnvStratifiedBatchSampler | None = None):
        self.t = {k: v.to(device) for k, v in tensors.items()}
        sizes = {k: v.shape[0] for k, v in self.t.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"row tensors must share their first dim, got {sizes}")
        self.n = next(iter(sizes.values()))
        self.batch_size = int(batch_size)
        self.device = device
        self.shuffle = shuffle
        self.seed = int(seed)
        self.batch_sampler = batch_sampler

    def __len__(self) -> int:
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        return math.ceil(self.n / self.batch_size)

    def _take(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.t.items()}

    def epoch(self, epoch: int):
        if self.batch_sampler is not None:
            self.batch_sampler.set_epoch(epoch)       # fix (a)
            for rows in self.batch_sampler:
                yield self._take(torch.as_tensor(rows, dtype=torch.long, device=self.device))
        elif self.shuffle:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(epoch))
            perm = torch.randperm(self.n, generator=g).to(self.device)
            for idx in perm.split(self.batch_size):
                yield self._take(idx)
        else:
            for s in range(0, self.n, self.batch_size):
                yield {k: v[s:s + self.batch_size] for k, v in self.t.items()}


def _row_tensors(ds: LincsDataset) -> dict[str, torch.Tensor]:
    """The dataset columns the loop reads (cond_from_batch + y), whole-split."""
    if ds.y is None:
        raise ValueError("the training dataset was built with load_y=False")
    return {"y": ds.y, "compound_idx": ds.compound_idx, "log10_conc": ds.log10_conc,
            "is_control": ds.is_control, "context": ds.context}


def select_val_rows(holdout_idx: np.ndarray, val_cap: int) -> np.ndarray:
    """The holdout rows validation uses: all, or a fixed `val_cap` subsample."""
    if val_cap > 0 and holdout_idx.shape[0] > val_cap:   # smaller val set
        return np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(holdout_idx, size=val_cap, replace=False))
    return holdout_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The AdamW step config. Used to build the optimizer and re-pinned after a resume. Note, fused runs the whole step as a single CUDA kernel instead of thousands of per-tensor launches; the state layout is unchanged, so checkpoints stay interchangeable with the non-fused path.
_ADAMW_STEP = {"fused": True, "foreach": False, "capturable": False}

def _pin_fused_adamw(optimizer) -> None:
    """Re-apply `_ADAMW_STEP` to every param group after `load_state`."""
    inner = getattr(optimizer, "optimizer", optimizer)   # unwrap AcceleratedOptimizer
    for group in inner.param_groups:
        group.update(_ADAMW_STEP)

def _spec_diff(was, now) -> str:
    """Field-level diff of two `CondSpec.to_list()` payloads, for error messages.
    """
    if not isinstance(was, list) or not isinstance(now, list):
        return f"  was: {was!r}\n  now: {now!r}"
    a = {f.get("name"): f for f in was}
    b = {f.get("name"): f for f in now}
    lines = []
    for name in sorted(set(a) | set(b)):
        if name not in a:
            lines.append(f"  + {name}: added ({b[name].get('role')})")
        elif name not in b:
            lines.append(f"  - {name}: removed (was {a[name].get('role')})")
        elif a[name] != b[name]:
            for k in sorted(set(a[name]) | set(b[name])):
                if a[name].get(k) != b[name].get(k):
                    lines.append(f"  ~ {name}.{k}: {a[name].get(k)!r} -> {b[name].get(k)!r}")
    return "\n".join(lines) or "  (specs differ but no field-level difference found)"


def _json_eq(a, b) -> bool:
    """Compare through JSON, so tuple-vs-list and int-vs-float from the file compare equal to the freshly built values."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _per_sample_loss(model, y: torch.Tensor, cond, noise_scheduler, fm, generator=None) -> torch.Tensor:
    """(B,) factual denoising MSE, rank-agnostic. DDPM: v-prediction on a random
    integer t; FM: velocity x0 - noise at t = tau * N. `generator` makes the
    noise reproducible (validation); None uses the global RNG (training)."""
    B = y.shape[0]
    noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
    if fm is not None:
        tau = fm.sample_tau(B, y.device, generator=generator)
        noisy = fm.add_noise(y, noise, tau)
        t = fm.model_timesteps(tau)
        target = fm.velocity_target(y, noise)
    else:
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,),
                          device=y.device, generator=generator).long()
        noisy = noise_scheduler.add_noise(y, noise, t)
        target = training_target(noise_scheduler, y, noise, t)
    pred = model(noisy, t, cond).sample
    # A (B, G) prediction that broadcasts against the target would only warn in mse_loss.
    assert pred.shape == target.shape, f"prediction {tuple(pred.shape)} vs target {tuple(target.shape)}"
    return F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1).mean(1)


@torch.no_grad()
def _validation_loss(model, noise_scheduler, val_batcher: _RowBatcher, device: torch.device,
                     cond_spec, seed: int = VAL_SEED, fm=None) -> float:
    """Held-out denoising MSE at *factual* conditioning, unweighted.

    Independent noise pairing, so val_loss is one metric comparable across arms.
    """
    was_training = model.training
    model.eval()
    assert not model.training, "validation must run under eval()"
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    gen = torch.Generator(device=device).manual_seed(seed)
    for batch in val_batcher.epoch(0):
        cond = cond_from_batch(batch, cond_spec, device)
        per = _per_sample_loss(model, batch["y"], cond, noise_scheduler, fm, generator=gen)
        total += per.double().sum()
        count += per.shape[0]
    if was_training:
        model.train()
    return float(total.item() / max(count, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True, choices=ARCHS, help="Denoiser backbone (§3.4): mlp (headline) or dit1d (ablation).")
    p.add_argument("--mlp_size", default="B", choices=tuple(MLP_SIZES), help="MLP hidden/depth (S 512/4, B 1024/6, L 2048/8).")
    p.add_argument("--dit_size", default="S", choices=tuple(DIT_SIZES), help="1D-DiT config from the paper (S ~33M default, B ~130M ablation).")
    p.add_argument("--patch_size", type=int, default=10, help="dit1d only: genes per token. 10 -> 98 tokens (978 padded to 980); 6 -> 163 tokens, no pad.")
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_epochs", type=int, default=2)
    p.add_argument("--lr_decay_every", type=int, default=200, help="ExponentialLR half-life, in epochs (~104 steps each at batch 256 on mcf7_24h).")
    p.add_argument("--reset_lr", type=float, default=0.0,
                   help="On resume only: restart the LR schedule at this value. load_state restores the saved scheduler, so --learning_rate is otherwise ignored when resuming. 0 = off (keep the checkpoint's decayed LR).")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA of the weights (eval samples from it: model_1.safetensors).")
    p.add_argument("--mixed_precision", default="no", choices=("no", "fp16", "bf16"), help="'no' = fp32 with TF32 matmuls (default). bf16 for large dit1d runs on Ampere+.")
    p.add_argument("--checkpoint_every", type=int, default=100, help="Save a checkpoint every N epochs; the last epoch is always saved.")
    p.add_argument("--resume_epoch", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--output_subdir", type=str, default=None, help="Run dir under runs/<build>/. Default: derived from arch, size, dr_mode, weights, method, extras, roles, split dir and seed.")
    p.add_argument("--log_every", type=int, default=10, help="Log step metrics (loss, grad norm, lr) to wandb every N steps.")
    p.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"), help="wandb tracking; credentials from ~/.netrc. 'disabled' for smoke runs.")
    p.add_argument("--wandb_project", default="lincs-adigen")
    p.add_argument("--wandb_entity", default="493302570", help="wandb entity (user or team) the runs log to.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_cap", type=int, default=2000, help="Max holdout rows used to estimate the validation. 0 = use the full holdout.")
    p.add_argument("--val_every", type=int, default=5,  help="Compute + log validation loss every N epochs (and at the last).")
    p.add_argument("--max_steps", type=int, default=0, help="Stop each epoch after N optimizer steps. 0 = full epoch.")
    p.add_argument("--class_dropout_prob", type=float, default=0.1, help="CFG dropout on the whole treatment (every role-A field); >0 gives guidance a true unconditional branch.")
    p.add_argument("--dr_mode", default="conditional", choices=("conditional", "weighted"), help="Training risk. 'weighted' weights the factual loss by per-row weights from --dr_weights_file; 'conditional' is unweighted. Weights are normalized to mean 1.")
    p.add_argument("--dr_weight_norm", default="global", choices=NORM_MODES, help="weighted: how the weights are normalised. 'global' (default, the v1 / step-C risk): one mean-1 normalisation over all train rows. 'group' (P2, STEP_A.md §4): within each target group (the positivity cell minus the confounder), so every group keeps its unweighted mass; needs a tiered split.")
    p.add_argument("--dr_weight_clip", type=float, default=None, help="weighted: cap each normalised weight at this value, the group (or global) total preserved. >= 1. Default: no cap.")
    p.add_argument("--dr_weights_file", default="dr_weights_urr.npz", help="weighted: {row_id, w} npz in the split dir: dr_weights_urr.npz (export --mode net, v1 primary), dr_weights_counts.npz, dr_weights_design.npz.")
    p.add_argument("--diffusion_method", default="ddpm", choices=("ddpm", "fm"), help="'ddpm' = VP cosine zero-SNR, v-prediction; 'fm' = flow matching (velocity). Not resume-compatible.")
    p.add_argument("--invariance_lambda", type=float, default=0.0, help="V-REx penalty weight: loss = mean_e[L_e] + lam*Var_e[L_e], e over spec.invariance_env_fields (plate; set with --environment_set).")
    p.add_argument("--envs_per_batch", type=int, default=8,  help="Envs drawn per batch, so each per-env mean uses batch_size/envs_per_batch rows. Ignored if lambda=0.")
    p.add_argument("--cmean_lambda", type=float, default=0.0, help="Weight on the tau=0 conditional-mean auxiliary loss ||v(eps,0,a)+eps - mu_hat(a)||^2 (P9). FM only. Costs one extra forward per step.")
    p.add_argument("--cmean_file", type=str, default="cmean.npz", help="Filename in the split dir from src.nuisances.precompute_cmean.")
    p.add_argument("--plate_center", default=None, choices=PLATE_CENTER_MODES, help="Default OutcomeSpec.plate_center (decision 6); 'none' is the ablation (needs expr_meta_none.json).")
    p.add_argument("--grad_checkpoint", type=int, default=0,  help="1 = gradient checkpointing.")
    p.add_argument("--include_env", type=int, default=0, help="0 (ADIGen) = generator ignores role-E; E enters --invariance_lambda. 1 = E in adaLN (ablation).")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    add_syn_cli(p)
    a = p.parse_args()
    a.grad_checkpoint = bool(a.grad_checkpoint)
    a.include_env = bool(a.include_env)
    if min(a.train_batch_size, a.num_epochs, a.checkpoint_every, a.val_every, a.log_every) < 1:
        p.error("--train_batch_size, --num_epochs, --checkpoint_every, --val_every and --log_every must be >= 1")
    if not 0.0 <= a.ema_decay < 1.0:
        p.error("--ema_decay must be in [0, 1)")
    if a.dr_mode != "weighted" and (a.dr_weight_norm != "global" or a.dr_weight_clip is not None):
        p.error("--dr_weight_norm / --dr_weight_clip apply to --dr_mode weighted only")
    if a.dr_weight_clip is not None and not a.dr_weight_clip >= 1.0:
        p.error("--dr_weight_clip must be >= 1 (the mean weight is 1)")
    return a


def _default_output_subdir(args, cfg: CaseConfig, plate_center: str) -> str:
    """Readable run name: every setting that changes the model or its data shows up."""
    size = args.mlp_size if args.arch == "mlp" else args.dit_size
    parts = [f"{args.arch}-{size}" + (f"-p{args.patch_size}" if args.arch == "dit1d" else "")]
    dr = args.dr_mode
    if args.dr_mode == "weighted":
        stem = os.path.splitext(os.path.basename(args.dr_weights_file))[0]
        dr += "-" + (stem[len("dr_weights_"):] if stem.startswith("dr_weights_") else stem)
        if args.dr_weight_norm != "global":
            dr += "-gn"
        if args.dr_weight_clip is not None:
            dr += f"-c{args.dr_weight_clip:g}"
    parts += [dr, args.diffusion_method]
    if args.cmean_lambda > 0:
        cstem = os.path.splitext(os.path.basename(args.cmean_file))[0]
        parts.append(f"cm{args.cmean_lambda:g}" + ("" if cstem == "cmean" else "-" + cstem.replace("cmean_", "")))
    if args.invariance_lambda > 0:
        parts.append(f"vrex{args.invariance_lambda:g}" + ("" if args.envs_per_batch == 8 else f"-e{args.envs_per_batch}"))
    if args.include_env:
        parts.append("env")
    if plate_center != cfg.outcome.plate_center:
        parts.append(f"pc-{plate_center}")
    if args.class_dropout_prob != 0.1:
        parts.append(f"cfg{args.class_dropout_prob:g}")
    for flag, default, tag in (("train_batch_size", 256, "bs"), ("learning_rate", 1e-4, "lr"),
                               ("ema_decay", 0.999, "ema"), ("mixed_precision", "no", "mp")):
        v = getattr(args, flag)
        if v != default:
            parts.append(f"{tag}{v:g}" if isinstance(v, float) else f"{tag}{v}")
    name = "_".join(parts) + role_tag(cfg)
    nz = os.path.abspath(cfg.paths.nuisance_dir)
    if nz != os.path.abspath(os.path.join(cfg.paths.data_dir, "nuisances")):
        base = os.path.basename(nz)
        name += "_" + (base[len("nuisances"):].lstrip("_") if base.startswith("nuisances") else base)
    return f"{name}_s{args.seed}"


def _resolve_resume(args, ckpt_root: str) -> tuple[str | None, int | None]:
    if args.resume_from_checkpoint and args.resume_epoch is not None:
        raise ValueError("Use only one of --resume_epoch / --resume_from_checkpoint")
    if args.resume_from_checkpoint:
        path = os.path.abspath(os.path.expanduser(args.resume_from_checkpoint))
        m = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
        if not m:
            raise ValueError(f"Cannot parse epoch from {path}")
        return path, int(m.group(1))
    if args.resume_epoch is not None:
        return os.path.join(ckpt_root, f"checkpoint-{args.resume_epoch:04d}"), args.resume_epoch
    return None, None


def _acquire_run_lock(ckpt_root: str):
    """Hold an exclusive lock on the run dir for the whole process, so two launches of
    one run (e.g. the same sbatch submitted twice) never write it together. A POSIX lock
    (fcntl.lockf, NFS-aware): released by the kernel / lock manager when the process dies.
    Returns the open file; keep a reference until exit."""
    import errno
    import fcntl
    os.makedirs(ckpt_root, exist_ok=True)
    fh = open(os.path.join(ckpt_root, RUN_LOCK_FILENAME), "a+")
    try:
        fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            fh.seek(0)
            holder = fh.read().strip() or "another process"
            fh.close()
            raise SystemExit(f"[init] {ckpt_root} is in use by {holder}; a second launch of the same run is refused")
        print(f"[init] WARNING could not lock {ckpt_root} ({e}); concurrent launches are not detected", flush=True)
        return fh
    fh.seek(0)
    fh.truncate()
    fh.write(f"job={os.environ.get('SLURM_JOB_ID', '-')} host={socket.gethostname()} pid={os.getpid()}\n")
    fh.flush()
    return fh


def _trim_history(path: str, keep_through_epoch: int | None) -> None:
    """Resume: drop history lines past the resumed checkpoint (a crashed run's
    tail). Fresh start: set aside a history left by a run that never checkpointed."""
    if not os.path.isfile(path):
        return
    if keep_through_epoch is None:
        os.replace(path, f"{path}.stale.{os.getpid()}")
        return
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and json.loads(ln)["epoch"] <= keep_through_epoch]
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    cfg: CaseConfig = apply_paths_args(config_from_args(args), args)
    check_syn_args(cfg, args)
    plate_center = args.plate_center or cfg.outcome.plate_center

    # Perf (math-preserving, resume-safe)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_subdir = args.output_subdir or _default_output_subdir(args, cfg, plate_center)
    ckpt_root = os.path.join(cfg.paths.train_output_dir, output_subdir)
    # A fresh start never overwrites a run that has checkpoints.
    resume_dir, completed_epoch = _resolve_resume(args, ckpt_root)
    # accelerate writes random_states_0.pkl last, so a torn save lacks it
    if resume_dir is not None and not all(os.path.isfile(os.path.join(resume_dir, f)) for f in
                                          ("model_1.safetensors", "scheduler.bin", "random_states_0.pkl")):
        raise SystemExit(f"[init] no complete checkpoint at {resume_dir}; nothing was changed")
    if resume_dir is not None and args.reset_lr > 0:
        _ss = torch.load(os.path.join(resume_dir, "scheduler.bin"), map_location="cpu", weights_only=False)
        if "_milestones" in _ss and _ss["last_epoch"] < _ss["_milestones"][0]:
            # LinearLR's chainable warm-up would scale reset_lr, and the milestone restarts the decay
            # from the checkpoint's base lr, so the reset would not hold.
            raise SystemExit(f"[init] --reset_lr inside the LR warm-up (step {_ss['last_epoch']} < "
                             f"{_ss['_milestones'][0]}) is not supported; resume from a checkpoint after it")
    _existing = sorted(glob.glob(os.path.join(ckpt_root, "checkpoint-*")))
    if resume_dir is None and _existing:
        raise SystemExit(
            f"[init] {ckpt_root} already holds {len(_existing)} checkpoint(s) (last {os.path.basename(_existing[-1])}). "
            f"Resume with --resume_epoch N, or train into a new --output_subdir.")
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=1,
                              log_with=None if args.wandb_mode == "disabled" else "wandb")
    if accelerator.num_processes > 1:
        raise SystemExit("[init] single-process only: the device-resident batcher does not shard rows")
    device = accelerator.device
    set_seed(args.seed)
    accelerator.print(f"[init] run dir {ckpt_root}")
    accelerator.print(f"[init] roles: {format_role_summary(cfg)}")

    # --- load nuisance metadata --------------------------------------------
    nz = cfg.paths.nuisance_dir
    with open(os.path.join(nz, "nuisance_meta.json")) as f:
        nmeta = json.load(f)
    n_compounds = int(nmeta["n_compounds"])

    # --- shared train/holdout split (held-out rows never enter training) ----
    splits = load_splits(cfg)
    train_idx = splits["train_idx"]
    tier = splits["tier"]
    accelerator.print(
        f"[init] split {splits['split_fingerprint']}: train={train_idx.shape[0]} "
        f"holdout={splits['holdout_idx'].shape[0]} (holdout_frac {splits['holdout_frac']}, seed={splits['seed']}); "
        f"tier {'ACTIVE ' + json.dumps({k: tier[k] for k in ('confounder', 'gamma', 'keep_frac') if k in tier}) if tier.get('active') else 'off'}")

    # --- rows in memory (z-space, plate-centred per --plate_center) ----------
    train_ds = LincsDataset(cfg, indices=train_idx, plate_center=plate_center)
    if not np.array_equal(train_ds.row_ids, train_idx):
        raise RuntimeError("LincsDataset rows are not train_idx")
    is_ctl_tr = train_ds.is_control.numpy() != 0
    lx_trt = train_ds.log10_conc.numpy()[~is_ctl_tr]           # treated train rows' raw dose
    gene_pr_ids = [int(g) for g in train_ds.expr_meta["pr_gene_id"]]
    with open(cfg.paths.gene_order_json) as f:
        if [int(g) for g in json.load(f)["pr_gene_id"]] != gene_pr_ids:
            raise RuntimeError("expr_meta.json and gene_order.json disagree on the gene order; re-run expr_stats")
    accelerator.print(f"[init] train rows {len(train_ds):,} (treated {int((~is_ctl_tr).sum()):,}, "
                      f"vehicle {int(is_ctl_tr.sum()):,}); y {tuple(train_ds.y.shape)} plate_center={plate_center}")

    # --- ADIGen DR weights (one per TRAIN row) ------------------------------
    dr_w = None
    dr_weights_mode = None
    dr_weights_sha1 = None
    dr_weight_stats = None
    if args.dr_mode == "weighted":
        wpath = os.path.join(nz, args.dr_weights_file)
        wz = np.load(wpath)
        if not np.array_equal(wz["row_id"], train_idx):
            raise RuntimeError(f"{args.dr_weights_file} was computed for a different train split; re-run its exporter")
        if "split_fingerprint" in wz.files and str(wz["split_fingerprint"]) != splits["split_fingerprint"]:
            raise RuntimeError(f"{args.dr_weights_file} is for split {wz['split_fingerprint']}, not {splits['split_fingerprint']}")
        if tier.get("active") and tuple(cfg.adjustment_set) != (tier["confounder"],):
            raise SystemExit(f"[init] this split thins on {tier['confounder']!r}: a weighted arm adjusts for it "
                             f"(--adjustment_set {tier['confounder']}), got C={list(cfg.adjustment_set)}")
        dr_weights_mode = str(wz["mode"]) if "mode" in wz.files else os.path.splitext(args.dr_weights_file)[0]
        dr_w = wz["w"].astype(np.float32) # dr weights
        dr_weights_sha1 = hashlib.sha1(np.ascontiguousarray(dr_w).tobytes()).hexdigest()
        if not np.all(np.isfinite(dr_w)) or (dr_w < 0).any():
            raise ValueError(f"{args.dr_weights_file}: weights must be finite and >= 0")

        # some weight checks: compute various statistics
        _raw_mean, _raw_max = float(dr_w.mean()), float(dr_w.max())
        _ess = lambda a: float((a.sum() ** 2) / max((a ** 2).sum(), 1e-12) / len(a))
        _ess_raw = _ess(dr_w.astype(np.float64))
        if args.dr_weight_norm == "global" and args.dr_weight_clip is None:
            # normalize to mean 1 so the gradient scale matches the conditional arm
            dr_w = (dr_w / max(float(dr_w.mean()), 1e-8)).astype(np.float32)
        else:
            # P2 (STEP_A.md §4). 'group': every target group keeps its unweighted
            # mass, so the weights only rebalance the confounder within a group.
            # 'global' with a cap is the same routine with one group.
            if args.dr_weight_norm == "group":
                if not tier.get("active"):
                    raise SystemExit("[init] --dr_weight_norm group needs a tiered split: the group is the "
                                     "positivity cell minus the confounder, and this split thins on nothing")
                _grp = train_groups(cfg, tier["confounder"], train_idx)
            else:
                _grp = np.zeros(len(dr_w), dtype=np.int8)
            _w64, dr_weight_stats = group_normalize(dr_w, _grp, cap=args.dr_weight_clip)
            if dr_weight_stats["max_group_sum_rel_err"] > 1e-9:
                raise RuntimeError(f"group normalisation left a group off its row count by "
                                   f"{dr_weight_stats['max_group_sum_rel_err']:.2e}")
            dr_w = _w64.astype(np.float32)
            # what the loss actually sees (float32), next to the float64 summary
            dr_weight_stats["n_exactly_one"] = int((dr_w == 1.0).sum())
            dr_weight_stats["max_f32"] = float(dr_w.max())
            accelerator.print(
                f"[init]   norm={args.dr_weight_norm} clip={args.dr_weight_clip}: "
                f"{dr_weight_stats['n_groups']:,} groups, max |group sum / n - 1| "
                f"{dr_weight_stats['max_group_sum_rel_err']:.1e}; capped {dr_weight_stats['n_capped']:,} rows "
                f"({100 * dr_weight_stats['capped_frac']:.2f}%) in {dr_weight_stats['n_groups_with_cap']:,} groups, "
                f"{dr_weight_stats['cap_passes']} pass(es)")
        accelerator.print(
            f"[init] dr_mode=weighted file={args.dr_weights_file} (mode {dr_weights_mode})\n"
            f"[init]   raw : mean={_raw_mean:.4f} max={_raw_max:.3f} ESS/n={_ess_raw:.3f}\n"
            f"[init]   used: mean={dr_w.mean():.4f} std={dr_w.std():.4f} "
            f"min={dr_w.min():.3f} max={dr_w.max():.3f} "
            f"ESS/n={_ess(dr_w.astype(np.float64)):.3f} "
            f"zeroed={100.0 * float((dr_w <= 0).mean()):.1f}%; "
            f"treated mean {dr_w[~is_ctl_tr].mean():.4f}, vehicle mean {dr_w[is_ctl_tr].mean():.4f}")
    else:
        accelerator.print(f"[init] dr_mode={args.dr_mode}")

    # --- conditional means for the auxiliary loss (precompute_cmean.py) -----
    cmean_mu_t = None
    cmean_id = None
    cmean_min_n = None
    cmean_sha1 = None
    if args.cmean_lambda > 0:
        if args.diffusion_method != "fm":        # fix (c)
            raise SystemExit("[init] --cmean_lambda > 0 is FM only (the tau=0 velocity identity); "
                             "use --diffusion_method fm or --cmean_lambda 0")
        if cfg.adjustment_set or args.include_env:   # fix (e)
            raise SystemExit("[init] mu_hat(a) is keyed on the action only, but this arm conditions on "
                             f"C={list(cfg.adjustment_set)} include_env={args.include_env}; cmean would pull "
                             "E[Y|a,c] toward E[Y|a]")
        _cz = np.load(os.path.join(nz, args.cmean_file))
        if not np.array_equal(np.asarray(_cz["train_idx"], dtype=np.int64), train_idx):
            raise RuntimeError(f"{args.cmean_file} was built for a different train split; re-run src.nuisances.precompute_cmean. Reusing it would leak a held-out arm's own mean into training.")
        if str(_cz["split_fingerprint"]) != splits["split_fingerprint"] or str(_cz["plate_center"]) != plate_center:
            raise RuntimeError(f"{args.cmean_file} is for split {_cz['split_fingerprint']} / plate_center "
                               f"{_cz['plate_center']}, not {splits['split_fingerprint']} / {plate_center}")
        cmean_id = np.asarray(_cz["row_gid"], dtype=np.int64)
        cmean_mu_t = torch.from_numpy(np.asarray(_cz["mu"], dtype=np.float32)).to(device)
        cmean_min_n = int(_cz["min_n"])
        cmean_sha1 = hashlib.sha1(np.ascontiguousarray(_cz["mu"]).tobytes()
                                  + np.ascontiguousarray(cmean_id).tobytes()).hexdigest()
        if tuple(cmean_mu_t.shape[1:]) != (cfg.outcome.n_genes,):
            raise ValueError(f"{args.cmean_file}: mu {tuple(cmean_mu_t.shape)} is not (K, {cfg.outcome.n_genes})")
        accelerator.print(
            f"[init] cmean_lambda={args.cmean_lambda}  arms={cmean_mu_t.shape[0]:,} (min_n {cmean_min_n})  "
            f"rows covered={100.0*float((cmean_id>=0).mean()):.1f}%  "
            f"(uncovered rows are masked out of the aux loss)")

    # --- environments for the V-REx invariance penalty ---------------------
    env_ids = None
    env_fields = tuple(invariance_env_fields(cfg))
    if args.invariance_lambda > 0:
        # Derived from the roles (--environment_set overrides): role-E MINUS anything promoted to C.
        if not env_fields:
            raise ValueError(
                "invariance_lambda > 0 but the environment set is EMPTY: every "
                "role-E field is promoted to C for this arm. There is nothing to "
                "be invariant across -- either lower the adjustment set or set "
                "--invariance_lambda 0.")
        cols = [CONTEXT_COL[f] for f in env_fields]
        _, env_ids = np.unique(train_ds.context[:, cols].numpy(), axis=0, return_inverse=True)
        env_ids = env_ids.reshape(-1).astype(np.int64)
        _per_env = args.train_batch_size // args.envs_per_batch
        _counts = np.bincount(env_ids)
        if _counts.size < args.envs_per_batch:
            raise ValueError(
                f"invariance env {list(env_fields)} yields only {_counts.size} "
                f"environments, fewer than --envs_per_batch={args.envs_per_batch}.")
        accelerator.print(
            f"[init] invariance: V-REx lambda={args.invariance_lambda} "
            f"env={list(env_fields)} n_envs={_counts.size} "
            f"min_rows/env={_counts.min()} (need >={_per_env}) "
            f"envs_per_batch={args.envs_per_batch}")

    # --- batches: every row tensor on the device ------------------------------
    rows = _row_tensors(train_ds)
    if dr_w is not None:
        rows["dr_w"] = torch.from_numpy(dr_w)
    if env_ids is not None:
        rows["env_id"] = torch.from_numpy(env_ids)
    if cmean_id is not None:
        # -1 = this arm has no precomputed mean (too few train rows); those rows are masked out of the auxiliary loss.
        rows["cmean_id"] = torch.from_numpy(cmean_id)
    env_sampler = (_EnvStratifiedBatchSampler(env_ids, args.train_batch_size, args.envs_per_batch, seed=args.seed)
                   if env_ids is not None else None)
    train_batcher = _RowBatcher(rows, args.train_batch_size, device, shuffle=True, seed=args.seed,
                                batch_sampler=env_sampler)

    # --- validation rows (held-out denoising loss) ---------------------------
    val_idx = select_val_rows(splits["holdout_idx"], args.val_cap)
    val_ds = LincsDataset(cfg, indices=val_idx, plate_center=plate_center)
    val_batcher = _RowBatcher(_row_tensors(val_ds), args.train_batch_size, device, shuffle=False)
    accelerator.print(f"[init] validation rows={val_idx.shape[0]} (holdout, val_cap={args.val_cap})")

    # --- denoiser ----------------------------------------------------------
    cond_spec = build_cond_spec(cfg, n_compounds, lx_trt, include_env=args.include_env, dose_encoding="scalar")
    accelerator.print(
        f"[init] cond_spec A={[f.name for f in cond_spec if f.role == 'A']} "
        f"C={[f.name for f in cond_spec if f.role == 'C']} "
        f"E={[f.name for f in cond_spec if f.role == 'E']}")
    size = args.mlp_size if args.arch == "mlp" else args.dit_size
    arch_kwargs = resolve_arch_kwargs(args.arch, size, patch_size=args.patch_size if args.arch == "dit1d" else None)
    G = cfg.outcome.n_genes
    model = build_generator(cond_spec, arch=args.arch, n_genes=G, arch_kwargs=arch_kwargs,
                            class_dropout_prob=args.class_dropout_prob)
    n_params = sum(p_.numel() for p_ in model.parameters())
    accelerator.print(
        f"[init] {args.arch}-{size} params={n_params/1e6:.1f}M {arch_kwargs} "
        f"cfg_dropout={args.class_dropout_prob}")

    # Nothing prunes checkpoints, so say up front what the cadence will cost.
    _n_ckpt = math.ceil(args.num_epochs / args.checkpoint_every)
    _gb = n_params * 16 / 1e9          # model + EMA + 2 AdamW moments, fp32
    accelerator.print(
        f"[init] checkpoints: every {args.checkpoint_every} epochs + the last -> "
        f"<= {_n_ckpt} x {_gb:.2f}GB = {_n_ckpt * _gb:.1f}GB")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # calibrate conditioning prior to ema cpy
    _factors = model.calibrate_conditioning(dose_probe(lx_trt))
    accelerator.print(f"[init] cond calibration {_factors}")

    model_ema = deepcopy(model)
    for p_ in model_ema.parameters():
        p_.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,  **_ADAMW_STEP)
    from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
    steps_per_epoch = len(train_batcher)
    _gamma = 0.5 ** (1.0 / (steps_per_epoch * args.lr_decay_every))
    if args.lr_warmup_epochs > 0:
        warmup = LinearLR( optimizer, start_factor=0.01, end_factor=1.0,  total_iters=int(args.lr_warmup_epochs * steps_per_epoch),)
        decay = ExponentialLR(optimizer, gamma=_gamma)
        lr_scheduler = SequentialLR(  optimizer, schedulers=[warmup, decay],  milestones=[int(args.lr_warmup_epochs * steps_per_epoch)], )
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=_gamma)

    # ddpm: diffusers VP scheduler (zero-SNR, v-pred).
    # fm: straight-line flow matching (src/processes/flow_matching.py)
    fm = None
    if args.diffusion_method == "fm":
        noise_scheduler = None
        fm = make_train_flow_matching()
        accelerator.print(
            f"[init] flow matching: N={fm.num_train_timesteps} "
            f"target=velocity(x0-noise) tau~uniform coupling=independent")
    else:
        noise_scheduler = make_train_scheduler(True)
        accelerator.print(
            f"[init] schedule: zero_snr=1 "
            f"prediction_type={noise_scheduler.config.prediction_type}")

    model, model_ema, optimizer, lr_scheduler = accelerator.prepare(model, model_ema, optimizer, lr_scheduler)

    # --- arch.json: identity of the run ----------------------------------------
    # file names as paths under the split dir, so ./x.npz, x.npz and an absolute path compare equal
    _rel_to_nz = lambda f: os.path.relpath(os.path.realpath(os.path.join(nz, f)), os.path.realpath(nz))  # noqa: E731
    spec = arch_spec(
        cond_spec,
        arch=args.arch,
        size=size,
        n_genes=G,
        arch_kwargs=arch_kwargs,
        class_dropout_prob=args.class_dropout_prob,
        # --- outcome / data --------------------------------------------------
        population=cfg.population.name,
        gene_pr_ids=gene_pr_ids,
        gene_order_sha1=hashlib.sha1(json.dumps(gene_pr_ids).encode()).hexdigest(),
        plate_center=plate_center,
        normalize_mean=cfg.outcome.normalize_mean,
        normalize_std=cfg.outcome.normalize_std,
        syn_effect=float(cfg.outcome.syn_effect),
        syn_seed=int(cfg.outcome.syn_seed),
        # The resolved step-C injection (§3.8.2). Recorded so eval can refuse an
        # arm whose ground truth differs from the oracle's: syn_effect alone does
        # not pin beta, which depends on a measured scale.
        syn_beta=(float(train_ds.syn_meta["beta"]) if train_ds.syn_meta else 0.0),
        syn_vec_seed=(int(train_ds.syn_meta["vec_seed"]) if train_ds.syn_meta else None),
        syn_v_sha1=(str(train_ds.syn_meta["v_sha1"]) if train_ds.syn_meta else None),
        # Step C2 (§3.8.4): which injection. v_sha1 already pins it (in compound
        # mode it hashes the whole direction matrix); these make it readable.
        syn_meta_name=(str(train_ds.syn_meta["name"]) if train_ds.syn_meta else None),
        syn_mode=(str(train_ds.syn_meta["mode"]) if train_ds.syn_meta else None),
        syn_rho=(float(train_ds.syn_meta.get("rho", 0.0)) if train_ds.syn_meta else None),
        split_fingerprint=splits["split_fingerprint"],
        table_fingerprint=train_ds.expr_meta["table_fingerprint"],
        nuisance_dir=os.path.realpath(nz),
        tier=tier,
        # --- roles / risk ------------------------------------------------------
        adjustment_set=list(cfg.adjustment_set),
        environment_set=list(env_fields),
        include_env=args.include_env,
        dr_mode=args.dr_mode,
        dr_weights_file=_rel_to_nz(args.dr_weights_file) if args.dr_mode == "weighted" else None,
        dr_weights_mode=dr_weights_mode,
        dr_weights_sha1=dr_weights_sha1,
        # P2 (STEP_A.md §4). None = the global mean-1 normalisation with no cap,
        # i.e. every run trained before these flags existed, so those runs still
        # resume and still read as the original `dr` arm.
        dr_weight_norm=(args.dr_weight_norm if args.dr_weight_norm != "global" else None),
        dr_weight_clip=(float(args.dr_weight_clip) if args.dr_weight_clip is not None else None),
        dr_weight_groups=(None if dr_weight_stats is None else
                          {k: int(dr_weight_stats[k]) for k in
                           ("n_groups", "n_capped", "n_groups_with_cap", "n_exactly_one")}),
        dr_weight_stats=dr_weight_stats,
        cmean_lambda=float(args.cmean_lambda),
        cmean_file=_rel_to_nz(args.cmean_file) if args.cmean_lambda > 0 else None,
        cmean_min_n=cmean_min_n,
        cmean_sha1=cmean_sha1,
        invariance_lambda=float(args.invariance_lambda),
        envs_per_batch=int(args.envs_per_batch) if args.invariance_lambda > 0 else None,
        # --- schedules -----------------------------------------------------------
        zero_snr=True,
        diffusion_method=args.diffusion_method,
        # Training-time tau distribution. Sampling never reads it
        tau_dist=(fm.tau_dist if fm is not None else None),
        tau_ln_m=(fm.tau_ln_m if fm is not None else None),
        tau_ln_s=(fm.tau_ln_s if fm is not None else None),
        # Training-time noise/data pairing. Sampling never reads it
        coupling=("independent" if fm is not None else None),
        # Index scale the denoiser was trained against: N for DDPM, the FM embedder scale (t = tau * N).
        num_train_timesteps=int(
            fm.num_train_timesteps if fm is not None
            else noise_scheduler.config.num_train_timesteps),
        # --- optimisation that fixes the batches / weights --------------------
        seed=int(args.seed),
        train_batch_size=int(args.train_batch_size),
        mixed_precision=args.mixed_precision,
        ema_decay=float(args.ema_decay),
        val_cap=int(args.val_cap),
        # provenance of the latest launch (not compared on resume)
        train_args={k: v for k, v in sorted(vars(args).items())},
    )

    # --- identity: every arch.json field except _NON_IDENTITY_KEYS must match ---
    def _identity_diff(prior: dict) -> list[str]:
        bad = []
        for _key in sorted((set(prior) | set(spec)) - set(_NON_IDENTITY_KEYS)):
            _was, _now = prior.get(_key), spec.get(_key)
            if _json_eq(_was, _now):
                continue
            bad.append(f"{_key}:\n" + (_spec_diff(_was, _now) if _key == "cond_spec"
                                         else f"  was: {_was!r}\n  now: {_now!r}"))
        return bad

    _run_lock = _acquire_run_lock(ckpt_root)   # held until the process exits
    accelerator.print(f"[init] holding {_run_lock.name}")
    # re-read under the lock: a previous holder may have checkpointed while this launch initialised
    _existing = sorted(glob.glob(os.path.join(ckpt_root, "checkpoint-*")))
    if resume_dir is None and _existing:
        raise SystemExit(
            f"[init] {ckpt_root} already holds {len(_existing)} checkpoint(s) (last {os.path.basename(_existing[-1])}). "
            f"Resume with --resume_epoch N, or train into a new --output_subdir.")
    _prior = read_arch_spec(ckpt_root) if os.path.isfile(os.path.join(ckpt_root, "arch.json")) else None
    _fork = False
    if resume_dir is not None:
        # the arch.json of the run the checkpoint came from, and the run dir's own if it has one
        _src_root = os.path.dirname(os.path.normpath(resume_dir))
        _fork = os.path.realpath(_src_root) != os.path.realpath(ckpt_root)
        if _fork and _existing:
            # a fork (--resume_from_checkpoint of another run) only goes into a run dir without checkpoints
            raise SystemExit(f"[init] forking {resume_dir} into {ckpt_root}, which already holds checkpoints; "
                             f"use a new --output_subdir")
        _pairs = [(_src_root, read_arch_spec(resume_dir))]
        if _prior is not None:
            _pairs.append((ckpt_root, _prior))
        elif _existing:
            raise SystemExit(f"[init] {ckpt_root} holds checkpoints but no arch.json")
        for _where, _p in _pairs:
            if _p is None:
                raise SystemExit(f"[init] resuming {resume_dir} but {_where} has no arch.json")
            bad = _identity_diff(_p)
            if bad:
                raise SystemExit(
                    f"[init] {_where} is not resume-compatible; these arch.json fields differ:\n"
                    + "\n".join(bad) + "\nTrain into a NEW --output_subdir, or match the checkpoint.")
    elif _prior is not None and _identity_diff(_prior):
        # no checkpoint yet, but another run's arch.json: a concurrent launch with the same name, or a crashed one
        raise SystemExit(
            f"[init] {ckpt_root} holds the arch.json of a different run (no checkpoint yet; another launch may be "
            f"writing it). Train into a new --output_subdir, or remove that dir if the other run is dead:\n"
            + "\n".join(_identity_diff(_prior)))

    # Record the architecture at checkpoints so the eval scripts rebuild the right model
    history_path = os.path.join(ckpt_root, HISTORY_FILENAME)
    if accelerator.is_main_process:
        write_arch_spec(ckpt_root, spec)
        # a fork starts a fresh history (its epochs <= N live in the source run)
        _trim_history(history_path, None if _fork else completed_epoch)

    # --- wandb (through accelerate); a resume continues the same run ----------
    if args.wandb_mode != "disabled":
        id_path = os.path.join(ckpt_root, WANDB_ID_FILENAME)
        run_id = None
        if resume_dir is not None and os.path.isfile(id_path):
            with open(id_path) as f:
                run_id = f.read().strip() or None
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]
            if accelerator.is_main_process:
                with open(id_path, "w") as f:
                    f.write(run_id + "\n")
        accelerator.init_trackers(
            args.wandb_project,
            config={**{k: v for k, v in spec.items() if k not in _WANDB_SKIP_KEYS},
                    "args": spec["train_args"], "n_params": n_params, "run_dir": ckpt_root},
            init_kwargs={"wandb": {
                "id": run_id, "resume": "allow", "name": output_subdir,
                "group": os.path.basename(cfg.paths.train_output_dir),
                "tags": [args.arch, args.dr_mode, args.diffusion_method],
                "dir": ckpt_root, "mode": args.wandb_mode, "entity": args.wandb_entity}},
        )
        accelerator.print(f"[init] wandb {args.wandb_mode}: project={args.wandb_project} run={output_subdir} id={run_id}")

    start_epoch = 0
    if resume_dir is not None:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(resume_dir)
        accelerator.load_state(resume_dir)
        # Re-pin.
        _pin_fused_adamw(optimizer)
        start_epoch = completed_epoch + 1
        accelerator.print(f"[resume] from {resume_dir} -> starting epoch {start_epoch}")

        # load_state restores the scheduler, so a resumed run keeps the decayed LR and ignores --learning_rate. --reset_lr restarts the decay from a chosen value.
        _sch = getattr(lr_scheduler, "scheduler", lr_scheduler)
        _decay = _sch._schedulers[-1] if isinstance(_sch, SequentialLR) else _sch
        _half_life = math.log(0.5) / math.log(_decay.gamma) / steps_per_epoch
        if abs(_decay.gamma - _gamma) > 1e-15:
            accelerator.print(f"[resume] WARNING the checkpoint's LR half-life ({_half_life:.1f} epochs) is kept; "
                              f"--lr_decay_every {args.lr_decay_every} is ignored on resume")
        if args.reset_lr > 0:
            # The decay is chainable (lr <- lr * gamma per step), so setting the group lr restarts it from
            # reset_lr. The scheduler object keeps its class (RxRx swapped in a fresh ExponentialLR, whose
            # saved state a later resume could not load into the SequentialLR it builds).
            _inner_opt = getattr(optimizer, "optimizer", optimizer)
            for _g in _inner_opt.param_groups:
                _g["lr"] = _g["initial_lr"] = args.reset_lr
            _sch._last_lr = [args.reset_lr for _ in _inner_opt.param_groups]
            accelerator.print(
                f"[resume] LR schedule reset: lr={args.reset_lr:g} "
                f"half-life={_half_life:.1f} epochs"
            )

    if start_epoch >= args.num_epochs:
        accelerator.print(f"[done] start_epoch {start_epoch} >= num_epochs {args.num_epochs}")
        accelerator.end_training()
        return

    _ema_params = list(model_ema.parameters())
    _model_params = list(model.parameters())

    def update_ema(decay: float):
        with torch.no_grad():
            torch._foreach_mul_(_ema_params, decay)
            torch._foreach_add_(_ema_params, _model_params, alpha=1.0 - decay)

    _unwrapped = accelerator.unwrap_model(model)

    # --- training loop -----------------------------------------------------
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        _t0 = perf_counter()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_loss_count = 0
        grad_norms = []
        for batch in train_batcher.epoch(epoch):
            y = batch["y"]                                   # (B, G)

            # =========== ADIGen arms: FACTUAL-pair loss, per-sample weight ==========
            # Conditional and weighted differ in the weight choice
            #
            #   conditional : w_i = 1                  -> the plain conditional model
            #   weighted    : w_i = DR weight (URR alpha / counts / design)

            # Every field the spec declares, by name -- A (compound/dose/is_control), C (the adjustment set) and E. Controls carry dose=NaN, not 0.0.
            cond = cond_from_batch(batch, cond_spec, device)
            per = _per_sample_loss(model, y, cond, noise_scheduler, fm)   # (B,)

            if args.dr_mode == "weighted":
                w_i = batch["dr_w"].float()      # (B,)
                loss = (w_i * per).mean()
            else:
                w_i = torch.ones_like(per)
                loss = per.mean()

            # --- Invariance penalty (V-REx)
            if args.invariance_lambda > 0 and "env_id" in batch:
                e_ids = batch["env_id"]
                wper = w_i * per
                uniq = torch.unique(e_ids)
                if uniq.numel() >= 2:
                    e_means = torch.stack([wper[e_ids == e].mean() for e in uniq])
                    _vrex = e_means.var()
                    loss = e_means.mean() + args.invariance_lambda * _vrex

                    # report
                    _inv_pen = float(args.invariance_lambda * _vrex.detach())
                    _inv_base = float(e_means.mean().detach())
                    if global_step % 100 == 0:
                        accelerator.print(
                            f"[vrex] step={global_step} base={_inv_base:.5f} "
                            f"pen={_inv_pen:.5f} ratio={_inv_pen/max(_inv_base,1e-9):.3f} "
                            f"n_envs={int(uniq.numel())}")

            # --- conditional-mean auxiliary loss (P9; FM only, checked at init)
            if args.cmean_lambda > 0:
                gid = batch["cmean_id"]
                keep = gid >= 0 # arms with enough train rows for mu_hat
                if bool(keep.any()):
                    eps0 = torch.randn_like(y[keep])
                    n0 = int(keep.sum())
                    t0 = fm.model_timesteps(torch.zeros(n0, device=device))
                    cond0 = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in cond.items()}
                    # fix (d): no CFG drop in the aux pass; the null branch is not supervised toward mu_hat(a)
                    drop0 = (torch.zeros(n0, dtype=torch.bool, device=device)
                             if _unwrapped.class_dropout_prob > 0 else None)
                    mu_pred = model(eps0, t0, cond0, drop=drop0).sample + eps0
                    mu_tgt = cmean_mu_t[gid[keep]]
                    assert mu_pred.shape == mu_tgt.shape, f"{tuple(mu_pred.shape)} vs {tuple(mu_tgt.shape)}"
                    _aux = F.mse_loss(mu_pred.float(), mu_tgt)
                    loss = loss + args.cmean_lambda * _aux
                    if global_step % 100 == 0:
                        accelerator.print(
                            f"[cmean] step={global_step} aux={float(_aux.detach()):.5f} "
                            f"lam*aux={float(args.cmean_lambda*_aux.detach()):.5f} "
                            f"covered={100.0*float(keep.float().mean()):.0f}%")

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 4.0) # guard, used in DoubleGen.
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            update_ema(args.ema_decay)

            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1
            grad_norms.append(grad_norm.detach())
            if accelerator.trackers and global_step % args.log_every == 0:
                accelerator.log({"train/loss": loss.detach().item(),
                                 "train/grad_norm": float(grad_norm),
                                 "train/lr": lr_scheduler.get_last_lr()[0],
                                 "train/w_mean_abs": float(w_i.abs().mean())},
                                step=global_step)
            global_step += 1

            # conditional block for quick testing
            if args.max_steps and epoch_loss_count >= args.max_steps:
                accelerator.print(
                    f"[epoch {epoch}] --max_steps={args.max_steps} reached, "
                    f"ending epoch early")
                break

        # --- epoch train + validation loss -------------------------------
        train_loss = float((epoch_loss_sum / max(epoch_loss_count, 1)).item())
        gn = torch.stack(grad_norms).float() if grad_norms else torch.zeros(1)
        _sec = perf_counter() - _t0
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch + 1 == args.num_epochs)
        val_loss = val_loss_ema = None
        if do_val:
            val_loss = _validation_loss(model, noise_scheduler, val_batcher, device, cond_spec, fm=fm)
            val_loss_ema = _validation_loss(model_ema, noise_scheduler, val_batcher, device, cond_spec, fm=fm)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"[epoch {epoch}] train_loss={train_loss}")
        accelerator.log({"epoch": epoch, "train/loss_epoch": train_loss,
                         "train/grad_norm_max": float(gn.max()), "train/sec_per_epoch": _sec,
                         **({"val/loss": val_loss, "val/loss_ema": val_loss_ema} if do_val else {})},
                        step=global_step)
        if accelerator.is_main_process:
            with open(history_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_loss_ema": val_loss_ema,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm_mean": float(gn.mean()),
                    "grad_norm_max": float(gn.max()),
                    "steps": epoch_loss_count,
                    "sec": round(_sec, 2),
                }) + "\n")
            _val_str = (f" val_loss={val_loss:.5f} val_loss_ema={val_loss_ema:.5f}" if do_val else "")
            accelerator.print(
                f"[epoch {epoch}] train_loss={train_loss:.5f}{_val_str} "
                f"lr={lr_scheduler.get_last_lr()[0]:.3g} |g| {float(gn.mean()):.3f} "
                f"{epoch_loss_count / max(_sec, 1e-9):.1f} it/s"
            )

        # --- checkpoint --------------------------------------------------
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.num_epochs:   # fix (b)
            with torch.no_grad():
                accelerator.wait_for_everyone()
                state_dir = os.path.join(ckpt_root, f"checkpoint-{epoch:04d}")
                # random_states_0.pkl is written last and marks a complete save (resume checks it);
                # drop a stale one first when a resumed run re-saves an existing dir
                _rs = os.path.join(state_dir, "random_states_0.pkl")
                if os.path.isfile(_rs):
                    os.remove(_rs)
                accelerator.save_state(state_dir)
                accelerator.print(f"[epoch {epoch}] saved accelerate state -> {state_dir}")
                accelerator.wait_for_everyone()

    accelerator.end_training()
    accelerator.print(f"[done] {ckpt_root}")


if __name__ == "__main__":
    main()
