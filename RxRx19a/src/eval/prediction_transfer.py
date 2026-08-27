"""Real vs. Real+Generated prediction-transfer test.

We train a ResNet18 (fresh-init, same architecture as the domain feature extractor) on three sources of training data and evaluate all three on the same held-out real test set:

  - real-only            : the augmentation-free baseline
  - generated-only (TSTR): a fidelity check; only useful as context
  - real + generated     : the augmentation condition

See how each compare, ideally all in alignment, perhaps with gains in the final augmentaiton case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from src.eval.feature_extractor import DomainResNet18


# ---------------------------------------------------------------------------
# Tensor-backed dataset (images + labels live in CPU memory)
# ---------------------------------------------------------------------------

class _TensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        assert images.shape[0] == labels.shape[0]
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, i: int):
        return self.images[i], int(self.labels[i].item())


# ---------------------------------------------------------------------------
# Train/eval one classifier
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    label_name: str
    source: Literal["real", "generated", "real_plus_generated"]
    n_classes: int
    n_train: int
    test_acc: float
    test_macro_f1: float


def _train_one(
    train_ds: Dataset,
    test_ds: Dataset,
    n_channels: int,
    n_classes: int,
    *,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    device = torch.device(device if isinstance(device, str) else device)

    model = DomainResNet18(n_channels=n_channels, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    for _ep in range(epochs):
        model.train()
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
        sched.step()

    # Test eval ---------------------------------------------------------------
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    acc = float((preds == targets).mean())
    f1 = float(_macro_f1(preds, targets, n_classes))
    return acc, f1


def _macro_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_transfer(
    real_train_images: torch.Tensor,
    real_train_labels: torch.Tensor,
    real_test_images: torch.Tensor,
    real_test_labels: torch.Tensor,
    gen_images: torch.Tensor,
    gen_labels: torch.Tensor,
    n_channels: int,
    n_classes: int,
    label_name: str,
    *,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> list[TransferResult]:
    real_train = _TensorDataset(real_train_images, real_train_labels)
    real_test = _TensorDataset(real_test_images, real_test_labels)
    gen = _TensorDataset(gen_images, gen_labels)
    mixed = ConcatDataset([real_train, gen])

    results: list[TransferResult] = []
    for source, ds in [
        ("real", real_train),
        ("generated", gen),
        ("real_plus_generated", mixed),
    ]:
        acc, f1 = _train_one(
            ds, real_test, n_channels=n_channels, n_classes=n_classes,
            epochs=epochs, batch_size=batch_size, lr=lr,
            num_workers=num_workers, device=device, seed=seed,
        )
        results.append(TransferResult(
            label_name=label_name, source=source, n_classes=n_classes,
            n_train=len(ds), test_acc=acc, test_macro_f1=f1,
        ))
    return results


def run_fillin_fidelity(
    real_train_images: torch.Tensor,
    real_train_labels: torch.Tensor,
    gen_images: torch.Tensor,
    gen_labels: torch.Tensor,
    n_channels: int,
    n_classes: int,
    label_name: str,
    *,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> TransferResult:
    """E2 -- fill-in fidelity (TRTS: Train on Real, Test on Synthetic).

    Train a classifier on REAL images only, then ask it to predict the label the
    generator was *conditioned* to produce, scored on the GENERATED images.

    """
    real_train = _TensorDataset(real_train_images, real_train_labels)
    gen_test = _TensorDataset(gen_images, gen_labels)
    acc, f1 = _train_one(
        real_train, gen_test, n_channels=n_channels, n_classes=n_classes,
        epochs=epochs, batch_size=batch_size, lr=lr,
        num_workers=num_workers, device=device, seed=seed,
    )
    return TransferResult(
        label_name=label_name, source="real_train__gen_test", n_classes=n_classes,
        n_train=len(real_train), test_acc=acc, test_macro_f1=f1,
    )
