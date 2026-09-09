"""Domain feature extractor for RxRx19a evaluation.

A ResNet18 with a 5-channel conv1, fine-tuned on real RxRx19a images to classify `disease_condition`. 
The penultimate-layer features (512-d) are used as the embedding space for FID / MMD in `dist_metrics.py`.

Trained once and cached.

Run:
  python -m src.eval.feature_extractor --epochs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import RxRx19aDataset  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.spec import (  # noqa: E402
    CaseConfig, add_adjustment_set_cli, config_from_args, default_config)


# ---------------------------------------------------------------------------
# Disease-condition label encoder
# ---------------------------------------------------------------------------

def build_disease_label_encoder(cfg: CaseConfig) -> dict[str, int]:
    """Returns {disease_condition: int_id}. Sorted alphabetically for stability."""
    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    labels = sorted({str(v) for v in ds["disease_condition"]})
    return {name: i for i, name in enumerate(labels)}


def build_celltype_label_encoder(cfg: CaseConfig) -> dict[str, int]:
    """Returns {cell_type: int_id}. Sorted for stability (e.g. HRCE/VERO)."""
    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    labels = sorted({str(v) for v in ds["cell_type"]})
    return {name: i for i, name in enumerate(labels)}


def apply_label_encoder(values: Iterable[str], enc: dict[str, int]) -> np.ndarray:
    return np.array([enc[str(v)] for v in values], dtype=np.int64)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DomainResNet18(nn.Module):
    """ResNet18 with first conv replaced to accept `n_channels` (5 for RxRx).
    `forward` returns class logits. `embed(x)` returns the 512-d penultimate features used for FID/MMD.
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.embedding_dim = net.fc.in_features
        net.fc = nn.Linear(self.embedding_dim, n_classes)
        self.net = net

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        n = self.net
        z = n.conv1(x)
        z = n.bn1(z)
        z = n.relu(z)
        z = n.maxpool(z)
        z = n.layer1(z); z = n.layer2(z); z = n.layer3(z); z = n.layer4(z)
        z = n.avgpool(z)
        return torch.flatten(z, 1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(self.embed(x))


# ---------------------------------------------------------------------------
# Training-data wrapper that also yields disease label
# ---------------------------------------------------------------------------

class _LabeledRxRx(Dataset):
    def __init__(self, cfg: CaseConfig, indices: np.ndarray, label_enc: dict[str, int]):
        self.inner = RxRx19aDataset(cfg, indices=indices, load_images=True)
        ds = load_from_disk(cfg.paths.tabular_dataset_dir).select(indices.tolist())
        self.labels = apply_label_encoder(ds["disease_condition"], label_enc)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        item = self.inner[i]
        return item["image"], int(self.labels[i])


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def feature_extractor_paths(cfg: CaseConfig) -> tuple[str, str]:
    """The canonical (full-data) extractor checkpoint path used by eval."""
    out_dir = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    return (
        os.path.join(out_dir, "feature_extractor.pt"),
        os.path.join(out_dir, "feature_extractor_meta.json"),
    )


def load_feature_extractor(
    cfg: CaseConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[DomainResNet18, dict[str, int]]:
    """Load a previously trained extractor + the disease label encoder."""
    cfg = cfg or default_config()
    ckpt_path, meta_path = feature_extractor_paths(cfg)
    if not os.path.isfile(ckpt_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Feature extractor not found. Train it first with "
            f"`python -m src.eval.feature_extractor`. Looked at: {ckpt_path}"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    model = DomainResNet18(
        n_channels=int(meta["n_channels"]),
        n_classes=int(meta["n_classes"]),
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, meta["label_encoder"]


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def _split_within(pool: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Internal 90/10 used for early stopping. `pool` is the canonical
    train_idx, so neither side leaks into the holdout used by evaluate.py."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool.shape[0])
    n_val = int(round(pool.shape[0] * val_frac))
    return pool[perm[n_val:]], pool[perm[:n_val]]


def train(
    cfg: CaseConfig,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    num_workers: int = 4,
    seed: int = 0,
    force_retrain: bool = False,
) -> None:
    ckpt_path, meta_path = feature_extractor_paths(cfg)
    if os.path.isfile(ckpt_path) and not force_retrain:
        print(f"[feature_extractor] checkpoint already exists at {ckpt_path}; skipping.")
        return
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    label_enc = build_disease_label_encoder(cfg)
    n_classes = len(label_enc)
    print(f"[feature_extractor] disease classes ({n_classes}): {list(label_enc)}")

    # Pull the canonical train index — the same one nuisance + diffusion saw, to ensure proper train/test splits
    splits = load_splits(cfg)
    pool = splits["train_idx"]
    train_idx, val_idx = _split_within(pool, val_frac, seed)
    train_ds = _LabeledRxRx(cfg, train_idx, label_enc)
    val_ds = _LabeledRxRx(cfg, val_idx, label_enc)
    print(f"[feature_extractor] pool n={pool.shape[0]} "
          f"(holdout={splits['holdout_idx'].shape[0]} excluded), "
          f"train n={len(train_ds)}, val n={len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainResNet18(n_channels=cfg.image.n_channels, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_val = -1.0
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        print(f"[feature_extractor] epoch {ep+1}/{epochs} "
              f"train_loss={tr_loss/max(n_seen,1):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)
            with open(meta_path, "w") as f:
                json.dump({
                    "n_channels": cfg.image.n_channels,
                    "n_classes": n_classes,
                    "label_encoder": label_enc,
                    "val_acc": val_acc,
                    "epochs_run": ep + 1,
                    "pool_size": int(pool.shape[0]),
                    "holdout_size_excluded": int(splits["holdout_idx"].shape[0]),
                }, f, indent=2)

    print(f"[feature_extractor] best val_acc={best_val:.4f}; saved -> {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force_retrain", action="store_true")
    add_adjustment_set_cli(p)
    a = p.parse_args()
    train(
        config_from_args(a),
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        val_frac=a.val_frac, num_workers=a.num_workers, seed=a.seed,
        force_retrain=a.force_retrain,
    )


if __name__ == "__main__":
    main()
