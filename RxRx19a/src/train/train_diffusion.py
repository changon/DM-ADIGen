"""ADIGen diffusion training for RxRx19a.

Run:
  accelerate launch -m src.train.train_diffusion \
      --num_epochs 200 --checkpoint_every 20 --dit_size B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import (  # noqa: E402
    CaseConfig, add_adjustment_set_cli, config_from_args,
    invariance_env_fields)
from src.models import (  # noqa: E402
    COND_MODES, DIT_SIZES, arch_spec, build_generator, read_arch_spec,
    write_arch_spec)
from src.processes import make_train_scheduler, training_target  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.data.dataset import (  # noqa: E402
    build_cond_spec, cond_from_batch, dose_probe)
from datasets import load_from_disk as _lfd_cm  # noqa: E402

# Dataset extension.
class _TrainDataset(torch.utils.data.Dataset):
    """Wraps RxRx19aDataset to attach the per-row DR weight and env id.

    `indices` (optional) restricts the wrapped dataset to a subset of the set;
    `dr_w` and `env_id` must be aligned with this.
    """

    def __init__(self, cfg: CaseConfig, indices: np.ndarray | None = None,
                 latents=None, dr_w: np.ndarray | None = None,
                 env_id: np.ndarray | None = None,
                 cmean_id: np.ndarray | None = None):
        from src.data.dataset import RxRx19aDataset
        self.inner = RxRx19aDataset(cfg, indices=indices, load_images=True, latents=latents)
        self.dr_w = dr_w
        self.env_id = env_id
        self.cmean_id = cmean_id
        for _name, _a in (("dr_w", dr_w), ("env_id", env_id), ("cmean_id", cmean_id)):
            if _a is not None and len(_a) != len(self.inner):
                raise ValueError(
                    f"{_name} ({len(_a)}) must align with the restricted "
                    f"dataset ({len(self.inner)})")

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        item = self.inner[idx]
        if self.dr_w is not None:
            item["dr_w"] = torch.tensor(float(self.dr_w[idx]), dtype=torch.float32)
        if self.env_id is not None:
            item["env_id"] = torch.tensor(int(self.env_id[idx]), dtype=torch.long)
        if self.cmean_id is not None:
            # -1 = this action has no precomputed mean (too few train rows, or held out); those rows are masked out of the auxiliary loss.
            item["cmean_id"] = torch.tensor(int(self.cmean_id[idx]), dtype=torch.long)
        return item


class _EnvStratifiedBatchSampler(torch.utils.data.Sampler):
    """Yield batches drawn from a few environments for invariance penalty

    Shuffling spreads a 128-row batch over ~57 plates (~2 rows each),
    Instead, we draw `envs_per_batch` environments per batch so we don't model over smapling noise (128/8 = 16 here).

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


@torch.no_grad()
def _validation_loss(
    model,
    noise_scheduler,
    val_loader,
    accelerator: Accelerator,
    device: torch.device,
    cond_spec,
    seed: int = 1234,
    fm=None,
) -> float:
    """Held-out denoising MSE at *factual* conditioning, unweighted.

    Independent noise pairing, so val_loss is one metric comparable across arms.
    """
    was_training = model.training
    model.eval()
    assert not model.training, "validation must run under eval()"
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    for batch in val_loader:
        image = batch["image"]
        B = image.shape[0]
        cond = cond_from_batch(batch, cond_spec, device)

        noise = torch.randn(image.shape, generator=gen, device=device, dtype=image.dtype)
        if fm is not None:
            tau = fm.sample_tau(B, device, generator=gen)
            noisy = fm.add_noise(image, noise, tau)
            t = fm.model_timesteps(tau)
            target = fm.velocity_target(image, noise)
        else:
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device, generator=gen).long()
            noisy = noise_scheduler.add_noise(image, noise, t)
            target = training_target(noise_scheduler, image, noise, t)

        pred = model(noisy, t, cond).sample
        per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        total += per.sum()
        count += B

    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    if was_training:
        model.train()
    return float((total / count.clamp(min=1)).item())


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

@dataclass
class TrainArgs:
    num_epochs: int
    lr_decay_every: int
    train_batch_size: int
    learning_rate: float
    lr_warmup_epochs: int
    reset_lr: float
    checkpoint_every_epochs: int
    resume_epoch: int | None
    resume_from_checkpoint: str | None
    output_subdir: str
    num_workers: int
    log_every_n_steps: int
    seed: int
    val_cap: int
    val_every: int
    max_steps: int
    dit_size: str
    patch_size: int
    cond_mode: str
    class_dropout_prob: float
    grad_checkpoint: bool
    latent: bool
    dr_mode: str
    dr_weights_file: str
    nuisance_dir: str
    diffusion_method: str
    invariance_lambda: float
    cmean_lambda: float
    cmean_file: str
    invariance_env: str
    envs_per_batch: int
    adjustment_set: str | None
    environment_set: str | None
    population_compounds: str | None
    include_env: bool


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--lr_decay_every", type=int, default=50)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_epochs", type=int, default=2)
    p.add_argument("--reset_lr", type=float, default=0.0,
                   help="On resume only: restart the LR schedule at this value. load_state restores the saved scheduler, so --learning_rate is otherwise ignored when resuming. 0 = off (keep the checkpoint's decayed LR).")
    p.add_argument("--checkpoint_every", type=int, default=20, help="Save a checkpoint every N epochs.")
    p.add_argument("--resume_epoch", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--output_subdir", type=str, default="compound_conc")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_cap", type=int, default=2000, help="Max holdout rows used to estimate the validation. 0 = use the full holdout.")
    p.add_argument("--max_steps", type=int, default=0, help="Stop each epoch after N optimizer steps. 0 = full epoch.")
    p.add_argument("--val_every", type=int, default=1,  help="Compute + log validation loss every N epochs.")
    p.add_argument("--dit_size", type=str, default="B", choices=tuple(DIT_SIZES), help="DiT config from the paper (S/B/L/XL).")
    p.add_argument("--patch_size", type=int, default=8,  help="DiT patch size. 8 -> 256 tokens at 128px; 4 -> 1024 tokens (sharper, ~4x the attention cost).")
    p.add_argument("--cond_mode", type=str, default="adaln", choices=COND_MODES, help="Conditioning transport. 'adaln' sums every field into one vector, so one modulation applies to all patches. 'xattn' keeps that path and ADDS role-A fields as cross-attention tokens (zero-initialised), letting a patch weight a field by its own content.")
    p.add_argument("--class_dropout_prob", type=float, default=None, help="CFG dropout on the whole treatment; >0 gives guidance a true unconditional branch. Default 0.1.")
    p.add_argument("--dr_mode", type=str, default="conditional", choices=("conditional", "knn_dr"),  help="Training risk. 'knn_dr' weights the factual loss by per-row weights from --dr_weights_file (design, kNN-DR, or URR); 'conditional' is unweighted. Weights are normalized to mean 1.")
    p.add_argument("--diffusion_method", type=str, default="ddpm", choices=("ddpm", "fm"), help="'ddpm' = VP cosine + DDIM/DDPM; 'fm' = flow matching via Euler ODE. Not resume-compatible.")
    p.add_argument("--invariance_lambda", type=float, default=0.0, help="V-REx penalty weight: loss = mean_e[L_e] + lam*Var_e[L_e]")
    p.add_argument("--cmean_lambda", type=float, default=0.0, help="Weight on the tau=0 conditional-mean auxiliary loss ||v(eps,0,a)+eps - mu_hat(a)||^2. The FM target is a SINGLE sample, so the treatment is a small share of its variance and the loss-optimal velocity shrinks it; mu_hat cuts that noise by n_eff. FM only. Costs one extra forward per step.")
    p.add_argument("--cmean_file", type=str, default="cmean.npz", help="Filename in the nuisance dir from src/nuisances/precompute_cmean.py.")
    p.add_argument("--invariance_env", type=str, default=None, choices=["plate", "experiment", "cell_type"])
    p.add_argument("--envs_per_batch", type=int, default=8,  help="Envs drawn per batch, so each per-env mean uses batch_size/envs_per_batch rows. Ignored if lambda=0.")
    p.add_argument("--dr_weights_file", type=str, default="dr_weights_knn.npz",  help="Filename in nuisance dir of the dr knn")
    p.add_argument("--nuisance_dir", type=str, default="", help="Override cfg.paths.nuisance_dir with a prebuilt split dir (e.g. from src/data/build_tiered_split.py). Supplies splits.json, nuisance_meta, vocab, dr weights and cmean.")
    p.add_argument("--latent", type=int, default=0,  help="1 = denoise precomputed VAE latents (80,16,16) instead of pixels (5,128,128). .")
    p.add_argument("--grad_checkpoint", type=int, default=0,  help="1 = gradient checkpointing.")
    p.add_argument("--include_env", type=int, default=0, help="0 (ADIGen) = generator ignores role-E; E enters the URR and --invariance_lambda. 1 = E in adaLN (ablation).")
    add_adjustment_set_cli(p)
    a = p.parse_args()
    
    if a.class_dropout_prob is None: # CFG dropout defaults to the DiT paper's 0.1.
        a.class_dropout_prob = 0.1
    return TrainArgs(
        num_epochs=a.num_epochs,
        lr_decay_every=a.lr_decay_every,
        train_batch_size=a.train_batch_size,
        learning_rate=a.learning_rate,
        lr_warmup_epochs=a.lr_warmup_epochs,
        reset_lr=float(a.reset_lr),
        checkpoint_every_epochs=a.checkpoint_every,
        resume_epoch=a.resume_epoch,
        resume_from_checkpoint=a.resume_from_checkpoint,
        output_subdir=a.output_subdir,
        num_workers=a.num_workers,
        log_every_n_steps=a.log_every,
        seed=a.seed,
        val_cap=int(a.val_cap),
        val_every=int(a.val_every),
        max_steps=int(a.max_steps),
        dit_size=a.dit_size,
        patch_size=int(a.patch_size),
        cond_mode=a.cond_mode,
        class_dropout_prob=float(a.class_dropout_prob),
        grad_checkpoint=bool(a.grad_checkpoint),
        latent=bool(a.latent),
        dr_mode=a.dr_mode,
        dr_weights_file=a.dr_weights_file,
        nuisance_dir=a.nuisance_dir,
        cmean_lambda=a.cmean_lambda,
        cmean_file=a.cmean_file,
        diffusion_method=a.diffusion_method,
        invariance_lambda=float(a.invariance_lambda),
        invariance_env=a.invariance_env,
        envs_per_batch=int(a.envs_per_batch),
        adjustment_set=a.adjustment_set,
        environment_set=a.environment_set,
        population_compounds=a.population_compounds,
        include_env=bool(a.include_env),
    )


def _resolve_resume(args: TrainArgs, ckpt_root: str) -> tuple[str | None, int | None]:
    import re
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


def main():
    args = _parse_args()
    cfg: CaseConfig = config_from_args(args)

    # Perf (math-preserving, resume-safe)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Training manipulations (thinning, confounded retention, ...) are baked into a prebuilt split dir
    if args.nuisance_dir:
        if not os.path.isfile(os.path.join(args.nuisance_dir, "splits.json")):
            raise FileNotFoundError(f"{args.nuisance_dir} has no splits.json")
        cfg.paths.nuisance_dir = args.nuisance_dir
        print(f"[train] nuisance_dir override: {cfg.paths.nuisance_dir}")

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1, log_with="tensorboard", project_dir=os.path.join(cfg.paths.train_output_dir, args.output_subdir, "logs"),)
    device = accelerator.device

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(cfg.paths.train_output_dir, args.output_subdir), exist_ok=True)
        accelerator.init_trackers("rxrx19a_doublegen")

    # --- load nuisance metadata --------------------------------------------
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as f:
        nmeta = json.load(f)

    n_compounds = int(nmeta["n_compounds"])

    # --- shared train/holdout split (held-out rows never enter training) ----
    splits = load_splits(cfg)
    train_idx = splits["train_idx"]
    accelerator.print(
        f"[init] train={train_idx.shape[0]} holdout={splits['holdout_idx'].shape[0]} "
        f"({splits['holdout_frac']:.0%}, seed={splits['seed']})"
    )

    # --- latent mode -------------------------------------------------------
    # The DiT denoises precomputed VAE latents (80,16,16) instead of pixels (5,128,128): 
    latent_spec = None
    if args.latent:
        from src.data.dataset import LatentSpec, default_latent_path
        latent_spec = LatentSpec.load(default_latent_path(cfg))
        accelerator.print(
            f"[init] LATENT mode: vae={latent_spec.vae} "
            f"denoising ({latent_spec.n_channels}, {latent_spec.resolution}, "
            f"{latent_spec.resolution})")

    # --- ADIGen kNN-DR weights (one per TRAIN row) -------------------------
    # w_i = alpha*_i + (1/k)#{j: i in NN(donor X', A_j)} - (1/k)sum alpha*_j
    dr_w = None
    if args.dr_mode == "knn_dr":
        # load weights
        wz = np.load(os.path.join(cfg.paths.nuisance_dir, args.dr_weights_file))
        if not np.array_equal(wz["row_id"], train_idx):
            raise RuntimeError(
                f"{args.dr_weights_file} was computed for a different train split; "
                "re-run src.nuisances.fit_knn_dr"
            )
        dr_w = wz["w"].astype(np.float32) # dr weights

        # some weight checks: compute various statistics
        _raw_mean, _raw_max = float(dr_w.mean()), float(dr_w.max())
        _ess = lambda a: float((a.sum() ** 2) / max((a ** 2).sum(), 1e-12) / len(a))
        _ess_raw = _ess(dr_w.astype(np.float64))
        # normalize to mean 1 so the gradient scale matches the conditional arm
        dr_w = (dr_w / max(float(dr_w.mean()), 1e-8)).astype(np.float32)
        accelerator.print(
            f"[init] dr_mode=knn_dr\n"
            f"[init]   raw : mean={_raw_mean:.4f} max={_raw_max:.3f} ESS/n={_ess_raw:.3f}\n"
            f"[init]   used: mean={dr_w.mean():.4f} std={dr_w.std():.4f} "
            f"min={dr_w.min():.3f} max={dr_w.max():.3f} "
            f"ESS/n={_ess(dr_w.astype(np.float64)):.3f} "
            f"zeroed={100.0 * float((dr_w <= 0).mean()):.1f}%")
    else:
        accelerator.print(f"[init] dr_mode={args.dr_mode}")

    # --- load conditional means for the auxiliary loss. computed in precompute_cmean.py ---------------------
    cmean_mu = None
    cmean_id = None
    if args.cmean_lambda > 0:
        _cz = np.load(os.path.join(cfg.paths.nuisance_dir, args.cmean_file),  allow_pickle=True)
        if not np.array_equal(np.asarray(_cz["train_idx"], dtype=np.int64), train_idx):
            raise RuntimeError(f"{args.cmean_file} was built for a different train split; re-run src/nuisances/precompute_cmean.py. Reusing it would leak a held-out arm's own mean into training.")
        _meta_c = _lfd_cm(cfg.paths.tabular_dataset_dir)
        _cp = np.asarray(_meta_c["compound_idx"], dtype=np.int64)
        _lxc = np.asarray(_meta_c["log10_conc"], dtype=np.float64)
        _icc = np.asarray(_meta_c["is_control"], dtype=np.int64)
        _d = np.where(_icc == 1, np.nan, np.round(_lxc, 3))
        _keys = np.array([f"{int(c)}|{'ctl' if int(k)==1 else f'{v:+.3f}'}" for c, v, k in zip(_cp, _d, _icc)])[train_idx]
        _lut = {k: i for i, k in enumerate(_cz["keys"].tolist())}
        cmean_id = np.array([_lut.get(k, -1) for k in _keys], dtype=np.int64)
        cmean_mu = torch.from_numpy(np.asarray(_cz["mu"], dtype=np.float32))
        accelerator.print(
            f"[init] cmean_lambda={args.cmean_lambda}  actions={len(_lut):,}  "
            f"rows covered={100.0*float((cmean_id>=0).mean()):.1f}%  "
            f"(uncovered rows are masked out of the aux loss)")

    # --- environments for the V-REx invariance penalty ---------------------
    env_ids = None
    if args.invariance_lambda > 0:
        from datasets import load_from_disk as _lfd
        _meta = _lfd(cfg.paths.tabular_dataset_dir)
        # Derived from the roles unless explicitly overridden. The environment
        # index is the role-E block MINUS anything promoted to C for this arm.
        if args.invariance_env:
            _env_fields = tuple(
                f.strip() for f in args.invariance_env.split(",") if f.strip())
            _src = "override"
        else:
            _env_fields = invariance_env_fields(cfg)
            _src = "spec.FIELDS role=E"
        if not _env_fields:
            raise ValueError(
                "invariance_lambda > 0 but the environment set is EMPTY: every "
                "role-E field is promoted to C for this arm. There is nothing to "
                "be invariant across -- either lower the adjustment set or set "
                "--invariance_lambda 0.")
        # context codes are recomputed at load, so build the key properly from generator
        from src.data.build_dataset import ContextEncoder
        from src.spec import CONTEXT_COL
        _enc = ContextEncoder.load_or_build(cfg)
        _missing = [f for f in _env_fields if f not in CONTEXT_COL]
        if _missing:
            raise ValueError(
                f"invariance env names {_missing}, which the context tensor does "
                f"not carry. Add them to spec.FIELDS and re-run build_dataset.")
        import pandas as _pd
        _cols = {c: _meta[c] for c in ("cell_type", "experiment", "plate",
                                       "well", "site", "disease_condition")}
        _codes = _enc.encode(_pd.DataFrame(_cols))
        _key = ["|".join(str(_codes[i, CONTEXT_COL[f]]) for f in _env_fields)
                for i in range(len(_codes))]
        _key = np.asarray(_key)[train_idx]
        _levels = {v: i for i, v in enumerate(sorted(set(_key.tolist())))}
        env_ids = np.array([_levels[v] for v in _key], dtype=np.int64)
        _per_env = args.train_batch_size // args.envs_per_batch
        _counts = np.bincount(env_ids)
        if len(_levels) < args.envs_per_batch:
            raise ValueError(
                f"invariance env {list(_env_fields)} yields only {len(_levels)} "
                f"environments, fewer than --envs_per_batch={args.envs_per_batch}.")
        accelerator.print(
            f"[init] invariance: V-REx lambda={args.invariance_lambda} "
            f"env={list(_env_fields)} ({_src}) n_envs={len(_levels)} "
            f"min_rows/env={_counts.min()} (need >={_per_env}) "
            f"envs_per_batch={args.envs_per_batch}")

    # wrap in a DataLoader to handle batching and shuffling, and other regularization choices.
    train_ds = _TrainDataset(cfg, indices=train_idx, latents=latent_spec, dr_w=dr_w, env_id=env_ids, cmean_id=cmean_id)
    cmean_mu_t = None if cmean_mu is None else cmean_mu.to(device)
    if env_ids is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=_EnvStratifiedBatchSampler(env_ids, args.train_batch_size, args.envs_per_batch, seed=args.seed),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )

    # --- validation loader (held-out denoising loss) -----------------------
    from src.data.dataset import RxRx19aDataset
    holdout_idx = splits["holdout_idx"]
    if args.val_cap > 0 and holdout_idx.shape[0] > args.val_cap: # smaller val test
        val_idx = np.sort(np.random.default_rng(12345).choice(holdout_idx, size=args.val_cap, replace=False))
    else:
        val_idx = holdout_idx

    val_ds = RxRx19aDataset(cfg, indices=val_idx, load_images=True, latents=latent_spec)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    accelerator.print(f"[init] validation rows={val_idx.shape[0]} "
                      f"(holdout, val_cap={args.val_cap})")

    # --- denoiser ----------------------------------------------------------

    # n channels, and res
    gen_channels = latent_spec.n_channels if latent_spec else cfg.image.n_channels 
    gen_res = latent_spec.resolution if latent_spec else cfg.image.resolution

    _lx_train = np.asarray(train_ds.inner.ds["log10_conc"], dtype=np.float64) # dose
    _treated = ~np.asarray(train_ds.inner.ds["is_control"], dtype=bool) # treated

    # build spec
    cond_spec = build_cond_spec(cfg, n_compounds, _lx_train[_treated],  include_env=args.include_env, dose_encoding="scalar")
    accelerator.print(
        f"[init] cond_spec A={[f.name for f in cond_spec if f.role == 'A']} "
        f"C={[f.name for f in cond_spec if f.role == 'C']} "
        f"E={[f.name for f in cond_spec if f.role == 'E']}")
    model = build_generator(
        cfg, cond_spec,
        n_channels=gen_channels,
        resolution=gen_res,
        dit_size=args.dit_size,
        patch_size=args.patch_size,
        class_dropout_prob=args.class_dropout_prob,
        cond_mode=args.cond_mode,
    )
    n_params = sum(p_.numel() for p_ in model.parameters())
    accelerator.print(
        f"[init] dit params={n_params/1e6:.1f}M size={args.dit_size} "
        f"patch={args.patch_size} cfg_dropout={args.class_dropout_prob}")

    # Nothing prunes checkpoints, so say up front what the cadence will cost.
    _n_ckpt = args.num_epochs // max(args.checkpoint_every_epochs, 1)
    _gb = n_params * 16 / 1e9          # model + EMA + 2 AdamW moments, fp32
    accelerator.print(
        f"[init] checkpoints: every {args.checkpoint_every_epochs} epochs -> "
        f"{_n_ckpt} x {_gb:.2f}GB = {_n_ckpt * _gb:.1f}GB")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # calibrate conditioning prior to ema cpy
    _factors = model.calibrate_conditioning(dose_probe(_lx_train[_treated]))
    accelerator.print(f"[init] cond calibration {_factors}")

    from copy import deepcopy
    model_ema = deepcopy(model)
    for p_ in model_ema.parameters():
        p_.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,  **_ADAMW_STEP)
    from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
    steps_per_epoch = len(train_loader)
    if args.lr_warmup_epochs > 0:
        warmup = LinearLR( optimizer, start_factor=0.01, end_factor=1.0,  total_iters=int(args.lr_warmup_epochs * steps_per_epoch),)
        decay = ExponentialLR(optimizer, gamma=0.5 ** (1.0 / (steps_per_epoch * args.lr_decay_every)))
        lr_scheduler = SequentialLR(  optimizer, schedulers=[warmup, decay],  milestones=[int(args.lr_warmup_epochs * steps_per_epoch)], )
    else:
        lr_scheduler = ExponentialLR(  optimizer, gamma=0.5 ** (1.0 / (steps_per_epoch * args.lr_decay_every))  )

    # ddpm: diffusers VP scheduler (+ optional v-pred). 
    # fm: straight-line flow matching (src/processes/flow_matching.py)
    fm = None
    if args.diffusion_method == "fm":
        from src.processes import make_train_flow_matching
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

    model, model_ema, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare( model, model_ema, optimizer, train_loader, val_loader, lr_scheduler )

    # --- resume ------------------------------------------------------------
    ckpt_root = os.path.join(cfg.paths.train_output_dir, args.output_subdir)

    # checks for resuming. look at specs and other arch details
    _prior = read_arch_spec(ckpt_root)
    if _prior is not None:
        _checks = [
            ("diffusion_method", _prior.get("diffusion_method", "ddpm"),
             args.diffusion_method, "changes the regression target"),
            # cond_spec identifies every field's role, cardinality, levels and (for dose) loc/scale. 
            ("cond_spec", _prior.get("cond_spec"), cond_spec.to_list(),
             "changes what the model conditions on"),
        ]
        # zero_snr only reaches make_train_scheduler on the ddpm path
        if args.diffusion_method != "fm":
            _checks.insert(0, ("zero_snr", bool(_prior.get("zero_snr", False)),  True,  "changes prediction_type, i.e. what the net outputs"))
        for _key, _was, _now, _why in _checks:
            # look through JSON so tuple-vs-list and int-vs-float from the file compare equal to the freshly built values.
            if json.dumps(_was, sort_keys=True) == json.dumps(_now, sort_keys=True):
                continue
            _detail = (_spec_diff(_was, _now) if _key == "cond_spec"
                       else f"  was: {_was!r}\n  now: {_now!r}")
            raise SystemExit(
                f"[init] {_key} mismatch: {ckpt_root} is not resume-compatible "
                f"({_why}).\n{_detail}\n"
                f"Train into a NEW --output_subdir, or match the checkpoint.")

    # Record the architecture at checkpoints so the eval scripts rebuild the right model
    if accelerator.is_main_process:
        write_arch_spec(ckpt_root, arch_spec(
            cond_spec,
            dit_size=args.dit_size,
            patch_size=args.patch_size,
            class_dropout_prob=args.class_dropout_prob,
            cond_mode=args.cond_mode,
            n_channels=int(gen_channels),
            resolution=int(gen_res),
            # --- run-specific keys -----------
            latent=bool(args.latent),                                  # evaluate
            vae=latent_spec.vae if latent_spec else None,              # evaluate
            # denoises per-channel standardized latents? evaluate undoes it prior to decoding
            latent_norm=bool(latent_spec.normalized) if latent_spec else False,
            zero_snr=True,                                             # schedules
            diffusion_method=args.diffusion_method,                    # schedules
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
        ))

    resume_dir, completed_epoch = _resolve_resume(args, ckpt_root)
    start_epoch = 0
    if resume_dir is not None:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(resume_dir)
        accelerator.load_state(resume_dir)
        # Re-pin.
        _pin_fused_adamw(optimizer)
        start_epoch = completed_epoch + 1
        accelerator.print(f"[resume] from {resume_dir} -> starting epoch {start_epoch}")

        # load_state restores scheduler.bin, so a resumed run keeps the decayed LR and ignores --learning_rate. --reset_lr restarts the decay from a chosen value.

        if args.reset_lr > 0:
            _inner_opt = getattr(optimizer, "optimizer", optimizer)
            for _g in _inner_opt.param_groups:
                _g["lr"] = _g["initial_lr"] = args.reset_lr
            _fresh = ExponentialLR(
                _inner_opt, gamma=0.5 ** (1.0 / (steps_per_epoch * args.lr_decay_every))
            )
            if hasattr(lr_scheduler, "scheduler"):
                lr_scheduler.scheduler = _fresh
            else:
                lr_scheduler = _fresh
            accelerator.print(
                f"[resume] LR schedule reset: lr={args.reset_lr:g} "
                f"half-life={args.lr_decay_every} epochs"
            )

    if start_epoch >= args.num_epochs:
        accelerator.print(f"[done] start_epoch {start_epoch} >= num_epochs {args.num_epochs}")
        return

    _ema_params = list(model_ema.parameters())
    _model_params = list(model.parameters())

    def update_ema(decay: float = 0.999):
        with torch.no_grad():
            torch._foreach_mul_(_ema_params, decay)
            torch._foreach_add_(_ema_params, _model_params, alpha=1.0 - decay)

    # --- training loop -----------------------------------------------------
    history_path = os.path.join(ckpt_root, "loss_history.jsonl")

    global_step = start_epoch * steps_per_epoch
    # Windowed throughput meter (main process only).
    _perf_t = perf_counter()
    _perf_step0 = global_step
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_loss_count = 0
        for batch in train_loader:
            image = batch["image"]                         # (B, C, H, W)
            B = image.shape[0]

            # =========== ADIGen arms: FACTUAL-pair loss, per-sample weight ==========
            # Conditional and knn_dr differ in the weight choice
            #
            #   cond.   : w_i = 1                  -> the plain conditional model
            #   knn_dr  : w_i = DR weight

            # Every field the spec declares, by name -- A (compound/dose/is_control), C (the adjustment set) and E. Controls carry dose=NaN, not 0.0.
            cond = cond_from_batch(batch, cond_spec, device)

            noise = torch.randn_like(image)
            if fm is not None:
                tau = fm.sample_tau(B, device)
                noisy = fm.add_noise(image, noise, tau)
                t = fm.model_timesteps(tau)
                target = fm.velocity_target(image, noise)
            else:
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (B,), device=device).long()
                noisy = noise_scheduler.add_noise(image, noise, t)
                target = training_target(noise_scheduler, image, noise, t)

            pred = model(noisy, t, cond).sample
            per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))  # (B,)

            if args.dr_mode == "knn_dr":
                w_i = batch["dr_w"].to(device).float()      # (B,)
                loss = (w_i * per).mean()
            else:
                w_i = torch.ones_like(per)
                loss = per.mean()

            # --- Invariance penalty (V-REx)
            if args.invariance_lambda > 0 and "env_id" in batch:
                e_ids = batch["env_id"].to(device)
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

            # --- conditional-mean auxiliary loss. Add auxiliary loss to help the model learn conditional means properly
            if args.cmean_lambda > 0 and fm is not None and "cmean_id" in batch:
                gid = batch["cmean_id"].to(device)
                keep = gid >= 0 # use treatment combos that have enough data for enforcing this constraint
                if bool(keep.any()):
                    eps0 = torch.randn_like(image[keep])
                    n0 = int(keep.sum())
                    t0 = fm.model_timesteps(torch.zeros(n0, device=device))
                    cond0 = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in cond.items()}
                    mu_pred = model(eps0, t0, cond0).sample + eps0
                    _aux = F.mse_loss(mu_pred, cmean_mu_t[gid[keep]])
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
            update_ema(0.999)

            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1
            if global_step % args.log_every_n_steps == 0:
                accelerator.log({"loss": loss.detach().item(),
                                 "grad_norm": float(grad_norm),
                                 "lr": lr_scheduler.get_last_lr()[0],
                                 "w_mean_abs": float(w_i.abs().mean())},
                                step=global_step)

            global_step += 1
            if accelerator.is_main_process and global_step % 50 == 0:
                _now = perf_counter()
                _ips = (global_step - _perf_step0) / max(_now - _perf_t, 1e-6)
                accelerator.print(
                    f"[perf] epoch={epoch} global_step={global_step} "
                    f"{_ips:.3f} it/s"
                )
                _perf_t = _now
                _perf_step0 = global_step

            # conditional block for quick testing 
            if args.max_steps and epoch_loss_count >= args.max_steps:
                accelerator.print(
                    f"[epoch {epoch}] --max_steps={args.max_steps} reached, "
                    f"ending epoch early")
                break

        # --- epoch train + validation loss -------------------------------
        train_loss_local = epoch_loss_sum / max(epoch_loss_count, 1)
        train_loss = float(accelerator.reduce(train_loss_local.clone(), reduction="mean").item())
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch + 1 == args.num_epochs)
        
        val_loss = (
            _validation_loss(model, noise_scheduler, val_loader, accelerator, device,
                             cond_spec, fm=fm)
            if do_val else None
        )
        accelerator.log(
            {"train_loss_epoch": train_loss,
             **({"val_loss": val_loss} if do_val else {})},
            step=global_step,
        )
        if accelerator.is_main_process:
            with open(history_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }) + "\n")
            _val_str = f" val_loss={val_loss:.5f}" if val_loss is not None else ""
            accelerator.print(
                f"[epoch {epoch}] train_loss={train_loss:.5f}{_val_str}"
            )

        # --- checkpoint --------------------------------------------------
        if (epoch + 1) % args.checkpoint_every_epochs == 0:
            with torch.no_grad():
                accelerator.wait_for_everyone()
                state_dir = os.path.join(ckpt_root, f"checkpoint-{epoch:04d}")
                accelerator.save_state(state_dir)
                accelerator.print(f"[epoch {epoch}] saved accelerate state -> {state_dir}")

                # Prune to the last KEEP_LAST checkpoint-NNNN state dirs so a frequent cadence does not fill disk.
                accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
