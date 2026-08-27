"""Recursion OpenPhenom encoder for RxRx19a real & generated images.

OpenPhenom-S/16 (`recursionpharma/OpenPhenom`) is a channel-agnostic MAE, so the 5-channel RxRx19a images go in as-is and come out as a 384-d embedding per img
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

_MAES_DIR = Path(__file__).resolve().parents[2] / "third_party" / "maes_microscopy"

def load_openphenom(repo: str = "recursionpharma/OpenPhenom", device="cuda"):
    """Load OpenPhenom-S/16 (cached weights) onto `device` in eval mode."""
    if str(_MAES_DIR) not in sys.path:
        sys.path.insert(0, str(_MAES_DIR))
    from huggingface_hub import snapshot_download
    from huggingface_mae import MAEModel  # noqa: E402
    local = snapshot_download(repo)
    return MAEModel.from_pretrained(local).to(device).eval()


class TVN:
    """Typical Variation Normalization (PCA CenterScale) fit on negative controls.

    Variation among identically treated wells is nuisance, adjust on the covariance 
    
    `groups` centers each batch on its own control mean first; unseen groups fall back to the global mean, which is the only option for generated images.
    """

    def __init__(self, reg: float = 1e-3):
        self.reg = reg          # ridge on the eigenvalues, as a fraction of the mean
        self.mean_ = None       # global control mean
        self.group_means_ = {}  # per-batch control means
        self.W_ = None          # (d, d) whitening map

    def fit(self, X: np.ndarray, groups=None) -> "TVN":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        if groups is not None:
            groups = np.asarray(groups)
            for g in np.unique(groups):
                m = groups == g
                if int(m.sum()) >= 2:                 # a 1-well "batch" mean is noise
                    self.group_means_[g] = X[m].mean(0)
            Xc = np.stack([x - self.group_means_.get(g, self.mean_)
                           for x, g in zip(X, groups)])
        cov = np.cov(Xc, rowvar=False)
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None) + self.reg * float(np.mean(np.clip(w, 0.0, None)))
        self.W_ = V / np.sqrt(w)                      # (d, d): project then scale
        return self

    def transform(self, X: np.ndarray, groups=None) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("TVN.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        if groups is None:
            Xc = X - self.mean_
        else:
            groups = np.asarray(groups)
            Xc = np.stack([x - self.group_means_.get(g, self.mean_)
                           for x, g in zip(X, groups)])
        return (Xc @ self.W_).astype(np.float32)

    def fit_transform(self, X, groups=None) -> np.ndarray:
        return self.fit(X, groups).transform(X, groups)


def _to_uint8_256(x: torch.Tensor, size: int) -> torch.Tensor:
    """(C,H,W) or (N,C,H,W) float -> uint8 in [0,255], resized to `size`.

    Accepts raw [0,255], [0,1], or [-1,1]; rescales to [0,255] by range heuristic.
    """
    import torchvision.transforms.functional as TF
    x = x.float()
    mx = float(x.max())
    if mx <= 1.5:                                  # normalized input
        x = (x + 1.0) * 127.5 if float(x.min()) < -0.01 else x * 255.0
    x = x.clamp(0, 255)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = TF.resize(x, [size, size], antialias=True)
    return x.to(torch.uint8)


def openphenom_embed(cfg, indices, model, size: int = 256, batch: int = 32,
                     device="cuda", num_workers: int = 8) -> np.ndarray:
    """Embed REAL RxRx19a rows (by tabular index) -> (n, 384)."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    idx = np.asarray(indices)

    class _UintDS(Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            row = ds[int(idx[i])]
            chans = []
            for p in row["channel_paths"]:
                img = Image.open(p)
                if img.mode != "L":
                    img = img.convert("L")
                chans.append(TF.pil_to_tensor(img))       # (1, H, W) uint8
            x = torch.cat(chans, 0).float()               # (C, H, W)
            return _to_uint8_256(x, size)

    loader = DataLoader(_UintDS(), batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    out = []
    with torch.no_grad():
        for xb in loader:
            out.append(model.predict(xb.to(device)).float().cpu().numpy())
    return np.concatenate(out, 0)


def openphenom_embed_tensor(images: torch.Tensor, model, size: int = 256,
                            batch: int = 32, device="cuda") -> np.ndarray:
    """Embed an in-memory image tensor (n, C, H, W) -> (n, 384).

    For GENERATED images (the `ate_gen` plug-in). Handles [0,255]/[0,1]/[-1,1].
    """
    out = []
    with torch.no_grad():
        for s in range(0, images.shape[0], batch):
            xb = _to_uint8_256(images[s:s + batch], size).to(device)
            out.append(model.predict(xb).float().cpu().numpy())
    return np.concatenate(out, 0)
