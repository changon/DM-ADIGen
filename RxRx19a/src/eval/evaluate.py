"""The RxRx19a evaluation suite

    python -m src.eval.evaluate --source real
    python -m src.eval.evaluate --source generated --dit_subdir <arm> --truth runs/eval_artifacts/rescue_panel.json

  EFFECTS   per-compound treatment effect on the rescue axis (below), hits-vs-negatives separation and AUROC over the curated panel.
  ACCURACY  (--truth) MSE / bias / Spearman of those effects against the real-data oracle. (do we get causal items back?)
  QUALITY   FID + KID (Inception) and FID + MMD (domain ResNet18), marginal and per dose-bin,

The estimand
------------

    mu_mock  = mean phi over real Mock wells               (healthy reference)
    mu_uinf  = mean phi over real untreated-infected wells (the vehicle baseline)
    u_hat    = (mu_uinf - mu_mock) / ||mu_uinf - mu_mock|| ;  G = ||mu_uinf - mu_mock||
    Y(x)     = < phi(x) - mu_mock , u_hat >     # 0 at Mock, G at untreated-infected
    rescue(c,dose) = (G - Y(c,dose)) / G        # 1 = full rescue to Mock,
                                                # 0 = no better than vehicle,
                                                # <0 = worse than vehicle

"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import load_splits  # noqa: E402
from src.eval.dist_metrics import (  # noqa: E402
    DOSE_BIN_NAMES, _extract, compute_all, compute_per_slice, dose_bin)
from src.eval.feature_extractor import load_feature_extractor  # noqa: E402
from src.eval.openphenom_encoder import (  # noqa: E402
    TVN, load_openphenom, openphenom_embed)
from src.eval.prediction_transfer import run_fillin_fidelity  # noqa: E402
from src.eval.generation import _load_real_images  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, config_from_args)

# --- curated panel -----------------------------------------------------------
PANEL_HITS = [
    "GS-441524",                 # remdesivir active metabolite (canonical RxRx19a hit)
    "Remdesivir (GS-5734)",
    "Aloxistatin",               # E-64d, cathepsin/cysteine-protease inhibitor
    "Chloroquine",
    "Hydroxychloroquine Sulfate",
    "Amodiaquine",
    "Mefloquine",
    "Bafilomycin A1",
]
PANEL_INACTIVES = [
    "Haloperidol",               # antipsychotic
    "Migalastat",                # Fabry-disease chaperone
    "Ribavirin",                 # antiviral but morphologically inactive here
    "Tenofovir Disoproxil Fumarate",
    "Oseltamivir carboxylate",
    "Indomethacin",              # NSAID
    "methylprednisolone-sodium-succinate",
    "Camostat",                  # TMPRSS2 inhibitor; weak morphological signal
]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda.")
    add_adjustment_set_cli(p)
    p.add_argument("--pool", choices=("all", "train", "holdout", "reserve"), default="all",  help="Real rows to measure on: all / the generator's own train split / eval holdout / the tiered split's reserve wells (needs --nuisance_dir; k never-trained wells per scored arm = the OOS truth). NOTE `all` is ~80%% train rows plus the reserve, so it is an IN-SAMPLE number; `train` vs `holdout` separates recall from generalisation but holds out no ACTION.")
    p.add_argument("--no_population_filter", action="store_true", help="Score on every cell_type, not just cfg.population.cell_type. Reproduces the pre-2026-09-02 behaviour, in which ~59%% of a panel compound's rows were VERO -- a population the generator never trained on and has no field to condition on.")
    p.add_argument("--min_dose_n", type=int, default=20,  help="Minimum real rows for a (compound,dose) group to count.")
    p.add_argument("--cap_per_group", type=int, default=400,   help="Cap images per (compound,dose) group (speed). 0 = no cap.")
    p.add_argument("--cap_ref", type=int, default=2000,   help="Cap images per reference centroid (Mock / untreated-inf). 0 = no cap, which removes the anchor-subsampling drift entirely (G moves ~0.9%% between draws at 2000) and makes the ruler exactly reproducible. Free on the cached domain/inception encoders; on OpenPhenom it embeds 22,376 anchor wells instead of 4,000.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,  help="Output JSON path (default: runs/eval_artifacts/rescue_panel.json).")
    p.add_argument("--save_centroids", action="store_true", help="Write every per-(compound,dose) centroid plus the anchors to <out>_centroids.npz (whitened encoder space). Two such files (real + generated) give the full displacement vectors the scalar Y projects away; compare with scripts/vector_ate.py.")
    p.add_argument("--smoke", action="store_true",  help="Tiny run: 2 hits + 2 negs, small caps.")
    p.add_argument("--encoder", choices=("domain", "openphenom", "inception"), default="domain",  help="Feature space: domain ResNet18, Recursion openphenom, or inception.")
    p.add_argument("--source", choices=("real", "generated", "augmented", "roundtrip"), default="real",  help="Dataset the estimand uses: real (oracle), generated (every row replaced), or augmented (survivors + fill-in).")
    p.add_argument("--dit_subdir", default="dit_naive_fm",   help="Generator arm to sample from (--source generated|augmented).")
    p.add_argument("--nuisance_dir", default="", help="Prebuilt split dir (src/data/build_tiered_split.py). Required by --source augmented: its thinned train defines which rows stay real vs get generated fill-in.")
    p.add_argument("--gen_epoch", type=int, default=99)
    p.add_argument("--which_wgt", default="ema", choices=("train", "ema"))
    # Decoder-only override. finetune_decoder freezes the encoder, so latents on
    # disk and every checkpoint stay valid and only the decode path changes.
    p.add_argument("--vae_path", default="", help="Local fine-tuned VAE dir used to decode (--source generated|augmented|roundtrip). Default: the latent spec's VAE.")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--num_inference_steps", type=int, default=250)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--gen_per_row", type=int, default=1, help="Independent samples per requested row; embeddings are averaged per row before the estimand.")
    p.add_argument("--vehicle_anchor", choices=("real", "own"), default="real", help="Baseline for ate_*: 'real' = the standardized frame (anchors, axis and vehicle all fixed from real data; only the image supply varies across runs); 'own' = each source's own vehicle (cancels any additive domain offset; the pre-2026-09-08 convention). The generated vehicle is measured and vehicle_offset_G recorded either way; the two conventions differ by exactly that offset.")
    p.add_argument("--anchor_pool", choices=("holdout", "population"), default="holdout", help="Wells the frame (anchors, axis, G, TVN fit) may be built from. 'holdout' = the dedicated frame set: vehicle wells from the never-trained holdout only (Mock never splits), so train wells touch only the model, holdout only the ruler, reserve only the truth. 'population' = the pre-2026-09-08 behaviour. One frame for every pool, so panels across pools are directly comparable.")
    p.add_argument("--inception_channels", type=int, nargs=3, default=(1, 2, 3),  help="The 3 channels feeding Inception. Must match the real oracle's, or the two land in different spaces.")
    p.add_argument("--openphenom_repo", default="recursionpharma/OpenPhenom",  help="HuggingFace repo id for OpenPhenom weights.")
    p.add_argument("--op_img_size", type=int, default=256,  help="Resize edge for OpenPhenom (patch16 crop256).")
    p.add_argument("--tvn", action="store_true", help="Typical Variation Normalization, fit on control wells. Raw OpenPhenom space is plate/batch dominated.")
    p.add_argument("--g_max", type=float, default=50.0, help="Sanity ceiling on the infect_gap normalizer G; 0 disables. Healthy G is ~3.5-4 (openphenom) / ~10-12 (domain); the node-dependent blowup produced ~380-396.")
    p.add_argument("--tvn_fit", choices=("vehicle", "controls"), default="vehicle", help="Rows defining 'typical variation'. vehicle keeps the mock-vs-infected axis out of the whitened variance.")
    p.add_argument("--tvn_center", choices=("global", "experiment", "plate"),  default="experiment",  help="Batch key centered on its own control mean before whitening.")
    p.add_argument("--tvn_reg", type=float, default=1e-3,  help="Eigenvalue ridge, as a fraction of the mean eigenvalue.")
    p.add_argument("--op_batch", type=int, default=32,  help="OpenPhenom forward batch size.")
    # -- accuracy-vs-truth block -------------------------------------------
    p.add_argument("--truth", default=None,  help="Prior --source real JSON. Enables the ACCURACY block. Ignored when --source real.")
    # -- image-quality + conditioning-fidelity blocks -----------------------
    p.add_argument("--quality_n", type=int, default=2048,  help="Generated images kept for the QUALITY and FIDELITY blocks, paired with real at the same conditioning. 0 disables both.")
    p.add_argument("--no_fidelity", action="store_true", help="Skip the TRTS fidelity classifier -- the only check that a generator responds to its conditioning.")
    p.add_argument("--fidelity_n_real", type=int, default=4000, help="Real images used to train the TRTS classifier.")
    p.add_argument("--fidelity_epochs", type=int, default=8)
    return p.parse_args()


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC of `scores` separating labels (1 = positive). Higher score should mean more positive. NaN if a class is empty."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank = (start + csum + 1) / 2.0
    ranks = avg_rank[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the Spearman convention)."""
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(1, len(v) + 1)
    # Average within tied groups so a flat estimator does not get a spurious ordering from argsort's stability.
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq)); np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _accuracy(est: np.ndarray, truth: np.ndarray) -> dict:
    """How close, and how correctly ordered, is `est` relative to `truth`?"""
    d = est - truth
    return {
        "n": int(len(est)),
        "mse": float(np.mean(d ** 2)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "mae": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),                       # signed: est - truth
        "spearman_rho": _corr(_rankdata(est), _rankdata(truth)),
        "pearson_r": _corr(est, truth),
        "truth_sd": float(np.std(truth)),                # scale MSE against this
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

# note during sampling from the generator, it takes a given well-site (dataset row), then for each row, samples images for the row condition (treatment combo). standardizes this whole process.
def _rng(seed: int, *key) -> np.random.Generator: # well subsampling consistency, same pick every run
    """An independent stream per (seed, purpose). Every draw site names itself, so a draw no longer depends on how many draws ran before it """
    h = hashlib.blake2b("|".join(str(k) for k in key).encode(), digest_size=8)
    return np.random.default_rng([seed, int.from_bytes(h.digest(), "little")])


def _row_seed(seed: int, row: int) -> int: # which noise a generated row starts from, same sample 
    """Noise seed for one generated row. Depends on (seed, row) alone, so a row's sample is identical however the rows were batched."""
    h = hashlib.blake2b(f"{seed}|gen|{int(row)}".encode(), digest_size=4)
    return int.from_bytes(h.digest(), "little")


def main():
    #### Load initial args ####
    args = _parse_args()
    if args.smoke:
        global PANEL_HITS, PANEL_INACTIVES
        PANEL_HITS = PANEL_HITS[:2]
        PANEL_INACTIVES = PANEL_INACTIVES[:2]
        args.cap_per_group = 60
        args.cap_ref = 120
        args.min_dose_n = 5

    if args.truth and args.source != "real" and not os.path.isfile(args.truth):
        raise FileNotFoundError(
            f"--truth {args.truth!r} does not exist. It must be a --source real "
            f"results JSON readable FROM THIS NODE.")

    torch.set_num_threads(max(1, args.num_workers))
    device = torch.device(args.device)
    cfg = config_from_args(args)
    # The split dir is the experimental contract, holding split dependent info such as pool split, reserve well, dataset mask (for what to fill in). derived from the same prebuilt dir the generator was trained under, 
    if args.nuisance_dir:
        if not os.path.isfile(os.path.join(args.nuisance_dir, "splits.json")):
            raise FileNotFoundError(f"{args.nuisance_dir} has no splits.json")
        cfg.paths.nuisance_dir = args.nuisance_dir
        print(f"[eval] nuisance_dir override: {cfg.paths.nuisance_dir}", flush=True)

    # -- vocab / panel resolution -------------------------------------------
    with open(os.path.join(cfg.paths.nuisance_dir, "compound_vocab.json")) as f:
        vocab = json.load(f)
    name_to_idx = {n: i for n, i in vocab.items()}

    def resolve(names):
        out = []
        for nm in names:
            if nm in name_to_idx:
                out.append((nm, name_to_idx[nm]))
            else:
                print(f"[eval] WARNING: '{nm}' not in vocab; skipping")
        return out

    hits = resolve(PANEL_HITS)
    negs = resolve(PANEL_INACTIVES)
    panel = [(nm, i, 1) for nm, i in hits] + [(nm, i, 0) for nm, i in negs]
    print(f"[eval] panel: {len(hits)} hits + {len(negs)} negatives")

    #### Load proper datasets ####
    # -- metadata + pool mask -----------------------------------------------
    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    comp = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic = np.asarray(meta["is_control"], dtype=np.int64)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    infected = (dis == cfg.population.disease_condition).astype(np.int64)

    # `load_splits` restricts training to cfg.population (disease_condition AND cell_type) --- on HRCE and the relevant infected so we can do causal inference. (take out VERO, bc VERO panels were more limited)
    cell = np.array([str(x) for x in meta["cell_type"]])
    in_pop = np.ones(len(comp), dtype=bool)
    if cfg.population.cell_type is not None and not args.no_population_filter:
        in_pop &= cell == cfg.population.cell_type

    pool_mask = in_pop.copy()
    # pool_mask are rows that the estimand scores, or which wells' embeddings become treatment effects. this can be all, trian, holdout, or reserve
    # anchor_ok are the rows that the ruler can be built from, that is, the mock and vehicle centroid.
    anchor_ok = in_pop
    if args.pool in ("train", "holdout"): # Both come from load_splits, which already applied cfg.population. for training and validation.
        sp = load_splits(cfg)
        sel = sp["train_idx" if args.pool == "train" else "holdout_idx"]
        pool_mask = np.zeros(len(comp), dtype=bool)
        pool_mask[np.asarray(sel, dtype=np.int64)] = True
        in_split = np.zeros(len(comp), dtype=bool)
        for _k in ("train_idx", "holdout_idx"):
            in_split[np.asarray(sp[_k], dtype=np.int64)] = True
        anchor_ok = in_pop & (pool_mask | ~in_split)
        print(f"[eval] pool={args.pool}: {int(pool_mask.sum()):,} of "
              f"{int(in_pop.sum()):,} population rows; anchors restricted to "
              f"{int(anchor_ok.sum()):,} rows", flush=True)
    elif args.pool == "reserve": # test set. the k held out wells uniformly.
        if not args.nuisance_dir:
            raise RuntimeError("--pool reserve needs --nuisance_dir: reserve.json lives there")
        _res = json.load(open(os.path.join(cfg.paths.nuisance_dir, "reserve.json")))
        _rwells = {w for a in _res["arms"] for w in a["reserve_wells"]}
        _wid = np.array([str(x) for x in meta["well_id"]])
        pool_mask = in_pop & np.isin(_wid, sorted(_rwells))
        if args.min_dose_n == 20:
            args.min_dose_n = 4
            print("[eval] pool=reserve: min_dose_n relaxed 20 -> 4 rows (k reserve wells/arm; pass --min_dose_n to override)")
        print(f"[eval] pool=reserve: {len(_rwells)} wells -> "
              f"{int(pool_mask.sum()):,} rows across "
              f"{len(_res['arms'])} scored arms", flush=True)

    # three roles of splits: train wells -> model, holdout -> projection embeddings, reserve -> truth.
    if args.anchor_pool == "holdout":
        sp = load_splits(cfg)
        in_hold = np.zeros(len(comp), dtype=bool)
        in_hold[np.asarray(sp["holdout_idx"], dtype=np.int64)] = True
        in_split = in_hold.copy()
        in_split[np.asarray(sp["train_idx"], dtype=np.int64)] = True
        anchor_ok = in_pop & (in_hold | ~in_split)
        print(f"[eval] anchor_pool=holdout: frame restricted to "
              f"{int(anchor_ok.sum()):,} rows", flush=True)

    #### load a chosen embedding model: can be openphenom, inception , or domain encoder (trained on its own based on data.)
    if args.encoder == "openphenom":
        op_model = load_openphenom(args.openphenom_repo, device)
        print(f"[eval] encoder=OpenPhenom ({args.openphenom_repo}); channel-agnostic, native 5-ch, 384-d embeddings", flush=True)

        def _embed_tensor(imgs):
            from src.eval.openphenom_encoder import openphenom_embed_tensor
            f = openphenom_embed_tensor(imgs, op_model, size=args.op_img_size, batch=args.op_batch, device=device)
            return f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

        def embed_raw(idx):
            return openphenom_embed(cfg, idx, op_model, size=args.op_img_size, batch=args.op_batch, device=device, num_workers=args.num_workers)
    elif args.encoder == "inception":
        from src.eval.dist_metrics import (
            _InceptionFeatures, _to_inception_input)
        incept = _InceptionFeatures().to(device).eval()
        ch = tuple(args.inception_channels)
        print(f"[eval] encoder=Inception (fixed ImageNet, channels={ch})", flush=True)

        def _embed_tensor(imgs):
            f = _extract(incept, imgs, pre=lambda x: _to_inception_input(x, ch), batch_size=args.batch_size, device=device)
            return f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

        def embed_raw(idx):
            return _embed_tensor(_load_real_images(cfg, idx, num_workers=args.num_workers, batch_size=args.batch_size))
    else: # we train our own.
        model, _ = load_feature_extractor(cfg, device=device)

        class _Embed(torch.nn.Module):
            def __init__(self, base):
                super().__init__(); self.base = base
            def forward(self, x):
                return self.base.embed(x)

        feat_model = _Embed(model).to(device).eval()

        def _embed_tensor(imgs):
            f = _extract(feat_model, imgs, batch_size=args.batch_size, device=device)
            return f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

        def embed_raw(idx):
            return _embed_tensor(_load_real_images(
                cfg, idx, num_workers=args.num_workers, batch_size=args.batch_size))

    # ---- if already computed the embeddings based on the above models, simply load it from the cache ----
    _cdir = os.path.join(cfg.paths.train_output_dir, "eval_artifacts", "feat_cache")
    if (args.encoder in ("domain", "inception") and os.path.isfile(os.path.join(_cdir, "meta.json"))):
        _cm = json.load(open(os.path.join(_cdir, "meta.json")))
        if int(_cm.get("n_rows", -1)) == len(comp):
            _arr = np.load(os.path.join(_cdir, f"{args.encoder}.npy"), mmap_mode="r")

            def embed_raw(idx):  # noqa: F811
                return np.asarray(_arr[np.asarray(idx)], dtype=np.float32)
            _real_embed_raw_cached = True
            print(f"[eval] FEATURE CACHE hit ({args.encoder}, "
                  f"{_cm['n_rows']:,} rows) -- no live embedding of real images",
                  flush=True)
        else:
            print(f"[eval] feature cache not usable (rows {_cm.get('n_rows')} vs "
                  f"{len(comp)}) -- embedding live", flush=True)

    print(f"[eval] encoder={args.encoder}", flush=True)

    # -- setup the image store for the QUALITY block ------------------------------
    # The estimand keeps only embeddings for saving space, but QUALITY metrics such as FID needs full images. _stash keeps a uniform quality_n sample as they pass.
    _real_embed_raw = embed_raw     # the real embedder, before embed_raw is rebound
    _qual: list[list] = []          # [row_id, image] pairs
    _qual_seen = 0
    _stash_rng = _rng(args.seed, "stash")   # own stream: draws happen mid-generation

    def _stash(rows, imgs):
        nonlocal _qual_seen
        for j, i in enumerate(rows):
            _qual_seen += 1
            if len(_qual) < args.quality_n:
                _qual.append([int(i), imgs[j].detach().cpu()])
            else:
                k = int(_stash_rng.integers(0, _qual_seen))
                if k < args.quality_n:
                    _qual[k] = [int(i), imgs[j].detach().cpu()]

    # --- CORE Section: Get the sampled images. We will utilize 3 differnet options: ---
    # -- roundtrip: take a round trip to test the lossiness and see what would happen in ideal scneario with a perfect generator (load image, encode, decode, and embed)
    # -- generated: load dit and sample, decode, embed
    # -- augmented: real embeddigns for rows the data split kept, and fill-in the removed ones based on mask.
    if args.source == "roundtrip":
        from src.data.dataset import (
            LatentSpec, decode_latents, default_latent_path, encode_images, load_vae)
        _lspec = LatentSpec.load(default_latent_path(cfg))
        # Encode always uses the spec VAE; only the decoder may be overridden.
        _vae = load_vae(_lspec.vae, device)
        _dec_vae = load_vae(args.vae_path, device) if args.vae_path else _vae
        print(f"[eval] source=roundtrip  encode={_lspec.vae}  decode={args.vae_path or _lspec.vae}  (no diffusion)", flush=True)

        def embed_roundtrip(idx):
            idx = np.asarray(idx)
            imgs = _load_real_images(cfg, idx, num_workers=args.num_workers, batch_size=args.batch_size)
            outs = []
            for s0 in range(0, len(imgs), args.batch_size):
                chunk = imgs[s0:s0 + args.batch_size].to(device)
                z = encode_images(_vae, chunk, device)
                outs.append(decode_latents(_dec_vae, z, cfg.image.n_channels).cpu())
            rt = torch.cat(outs, dim=0)
            if args.quality_n:
                _stash(idx, rt)
            return _embed_tensor(rt)

        embed_raw = embed_roundtrip

    if args.source in ("generated", "augmented"):
        from src.data.build_dataset import context_for_rows
        from src.eval.generation import (LatentCtx, SampleTarget, _build_model, _generate_batch, _load_weights)
        from src.processes import make_eval_scheduler_for_ckpt

        gen_subdir = args.dit_subdir
        ckpt_dir = os.path.join(cfg.paths.train_output_dir, gen_subdir, f"checkpoint-{args.gen_epoch:04d}")
        print(f"[eval] source={args.source}  generator={ckpt_dir}", flush=True)

        # load model
        model = _build_model(cfg, ckpt_dir)
        _load_weights(model, ckpt_dir, which=args.which_wgt)
        latent_ctx = LatentCtx.from_ckpt(cfg, ckpt_dir, device, vae_path=args.vae_path)
        if args.vae_path:
            print(f"[eval] decoding through {args.vae_path}", flush=True)
        model = model.to(device, memory_format=torch.channels_last).eval()
        scheduler = make_eval_scheduler_for_ckpt(ckpt_dir, "ddim")

        # Full context row per dataset row, per checkpoint's CondSpec
        ctx_all = context_for_rows(cfg, meta, np.arange(len(comp)))
        _cache: dict[int, np.ndarray] = {}

        # fx to embed a generated sample
        def embed_generated(idx):
            idx = np.asarray(idx)
            todo = [int(i) for i in idx if int(i) not in _cache]
            for s in range(0, len(todo), args.gen_batch_size):
                chunk = todo[s:s + args.gen_batch_size]
                targets = [SampleTarget(int(comp[i]), float(lx[i]), int(ic[i]), tuple(int(v) for v in ctx_all[i]), int(infected[i])) for i in chunk]
                reps = []
                for r in range(args.gen_per_row):
                    imgs = _generate_batch(
                        model=model, scheduler=scheduler, cfg=cfg, targets=targets,
                        n_inference_steps=args.num_inference_steps, device=device,
                        seed=args.seed,
                        row_seeds=[_row_seed(args.seed + 7919 * r, i) for i in chunk],
                        guidance_scale=args.guidance_scale, latent_ctx=latent_ctx)
                    if args.quality_n and r == 0:
                        _stash(chunk, imgs)
                    reps.append(_embed_tensor(imgs))
                f = np.mean(reps, axis=0)
                for j, i in enumerate(chunk):
                    _cache[i] = f[j]
            return np.stack([_cache[int(i)] for i in idx])

        # if generated, 
        if args.source == "generated":
            embed_raw = embed_generated
        else:
            # augmented: real where the split kept the row, generated fill-in where splits removed it. Hence, Survivors = thinned train + holdout (never thinned), read from the prebuilt --nuisance_dir.
            if not args.nuisance_dir:
                raise RuntimeError( "--source augmented needs --nuisance_dir: the thinned split defines which rows stay real vs get generated fill-in.")
            print(f"[eval] augmented: splits from {cfg.paths.nuisance_dir}", flush=True)
            sp_rare = load_splits(cfg)
            kept = np.zeros(len(comp), dtype=bool)
            kept[np.asarray(sp_rare["train_idx"], dtype=np.int64)] = True
            kept[np.asarray(sp_rare["holdout_idx"], dtype=np.int64)] = True
            real_embed = embed_raw # embed according to each case.
            _emb_dim = int(real_embed(np.where(pool_mask)[0][:1]).shape[1])

            def embed_raw(idx):  # noqa: F811
                idx = np.asarray(idx)
                out = np.empty((len(idx), _emb_dim), dtype=np.float32)
                m = kept[idx]
                if m.any():
                    out[m] = real_embed(idx[m])
                if (~m).any():
                    out[~m] = embed_generated(idx[~m])
                return out
            print(f"[eval] augmented: {int(kept[np.where(pool_mask)[0]].mean()*100)}% of "
                  f"pooled rows are REAL, remainder generated fill-in", flush=True)

    # Now we can move to measurement. 

    # -- optional TVN batch correction. Fit on control wells (raw embeddings), then every later embed call is whitened (rescale embeddings by variances to have comparable embeddings). 
    # note, these are the wells used bc the vehicle centroid is computed on this, so we need to standardize this chioce for equal meaasuring across experiments and embeddings
    embed_indices = embed_raw
    tvn_meta = None
    if args.tvn:
        # get the rows, fit_idx, that define whose embeddings are typical variation. it is drawn from infected and untreated wells (vehicle). restrict to some subset for computational reasons. 
        batch_key = (None if args.tvn_center == "global" else np.array([str(x) for x in meta[args.tvn_center]]))
        fit_mask = anchor_ok & (ic == 1)  # fit on the anchors' rows
        if args.tvn_fit == "vehicle":
            fit_mask &= infected == 1
        fit_idx = np.where(fit_mask)[0]
        if args.cap_ref and len(fit_idx) > args.cap_ref:
            fit_idx = fit_idx[_rng(args.seed, "tvn_fit").choice(len(fit_idx), args.cap_ref, replace=False)]

        print(f"[eval] TVN: fitting on {len(fit_idx)} {args.tvn_fit} rows (center={args.tvn_center}, reg={args.tvn_reg})", flush=True)
        
        # embed these guys that we wish to whiten on.
        _fit_raw = _real_embed_raw(fit_idx)  # ALWAYS real/training wells
        import hashlib as _hl  # TVNDIAG stage hashes: first differing hash vs a healthy log locates a fork
        print(f"[eval] TVNDIAG fit_idx sha={_hl.sha256(np.ascontiguousarray(fit_idx).tobytes()).hexdigest()[:16]} n={len(fit_idx)}", flush=True)
        print(f"[eval] TVNDIAG raw_emb sha={_hl.sha256(np.round(_fit_raw, 3).tobytes()).hexdigest()[:16]} mean={_fit_raw.mean():.6f} std={_fit_raw.std():.6f} rowstd={_fit_raw.std(0).mean():.6f}", flush=True)

        # run tvn
        tvn = TVN(reg=args.tvn_reg).fit(_fit_raw, None if batch_key is None else batch_key[fit_idx])
        _w = tvn.eigs_raw_
        print(f"[eval] TVNDIAG eigs min={_w.min():.3e} med={np.median(_w):.3e} max={_w.max():.3e} sum={_w.sum():.3e}", flush=True)
        print(f"[eval] TVN: {len(tvn.group_means_)} batch means", flush=True)

        def embed_indices(idx):  # noqa: F811
            return tvn.transform(embed_raw(idx), None if batch_key is None else batch_key[np.asarray(idx)])

        tvn_meta = {"fit": args.tvn_fit, "center": args.tvn_center,
                    "fit_source": "real",
                    "reg": args.tvn_reg, "n_fit_rows": int(len(fit_idx)),
                    "n_batches": len(tvn.group_means_)}

    # TVN embed the rest of the data, if active arg
    if args.tvn:
        def _anchor_embed_indices(idx):
            return tvn.transform( _real_embed_raw(idx), None if batch_key is None else batch_key[np.asarray(idx)])
    else:
        _anchor_embed_indices = _real_embed_raw

    def centroid(mask: np.ndarray, cap: int, embed=None, key: str = ""):
        """`key` names the group so its subsample is drawn from its own stream."""
        idx = np.where(mask)[0]
        n = len(idx)
        if n == 0:
            return None, 0
        if cap and n > cap:
            idx = idx[_rng(args.seed, "centroid", key, cap).choice( n, cap, replace=False)]
        feats = (embed or embed_indices)(idx)
        return feats.mean(0), n  # report TRUE n (pre-cap) for context

    # --- NOW we really begin the next CORE Component: Evaluation in full ---
    # recall:
    # - μ_mock — centroid of real Mock wells (healthy, never infected): the "fully rescued" end.
    # - μ_uinf — centroid of real untreated-infected (vehicle) wells: the "sick, no drug" end.
    # build below.

    anchor_embed = _anchor_embed_indices   # real data

    # -- references: Mock and untreated-infected vehicle --------------------
    # get masks
    mock_mask = anchor_ok & (ic == 1) & (dis == "Mock")
    uinf_mask = anchor_ok & (ic == 1) & (infected == 1)
    print(f"[eval] mock n={int(mock_mask.sum())}  untreated-infected n={int(uinf_mask.sum())}")

    # compute centroids
    mu_mock, _ = centroid(mock_mask, args.cap_ref, embed=anchor_embed, key="mock")
    mu_uinf, _ = centroid(uinf_mask, args.cap_ref, embed=anchor_embed, key="uinf")
    if mu_mock is None or mu_uinf is None:
        raise RuntimeError("missing Mock or untreated-infected reference rows")

    # get the direction for projecting
    axis = mu_uinf - mu_mock
    G = float(np.linalg.norm(axis))          # infect_gap (signed scale)
    print(f"[eval] TVNDIAG anchors |mu_mock|={np.linalg.norm(mu_mock):.4f} |mu_uinf|={np.linalg.norm(mu_uinf):.4f} G={G:.4f}", flush=True)
    if args.g_max and not (0.1 < G < args.g_max): # g max check, by inverse covariance problems
        raise RuntimeError(
            f"infect_gap G={G:.3f} outside sanity band (0.1, {args.g_max}) -- known node-dependent embedding/TVN blowup; do not trust this run. "
            f"Rerun on a different node or lower --g_max only if the encoder legitimately changed scale.")
    u_hat = axis / (G + 1e-12) 

    def Y(mu_vec):
        return float((mu_vec - mu_mock) @ u_hat)

    # get outcomes via projection
    Y_uinf = Y(mu_uinf)                       # == G by construction

    # now compute ATE contrasts.
    mu_uinf_src = (mu_uinf if args.source == "real" else centroid(uinf_mask, args.cap_ref, key="uinf")[0])
    Y_uinf_src = Y(mu_uinf_src)
    Y_base = Y_uinf_src if args.vehicle_anchor == "own" else Y_uinf
    print(f"[eval] infect_gap G={G:.3f}  Y_uinf={Y_uinf:.3f}  Y_mock=0")
    print(f"[eval] vehicle_anchor={args.vehicle_anchor}; within-source vehicle Y={Y_uinf_src:.3f}; offset vs real vehicle {(Y_uinf_src - Y_uinf) / G:+.3f} G (should be ~0)", flush=True)

    def _ate(y):
        """0 = no better than vehicle, 1 = full rescue to mock; baseline per --vehicle_anchor."""
        return (Y_base - y) / G

    # Compute compound ATE's and metrics. Do this by dose and averaged over dose: first is for intervention level estimand checks, and the second is for pooled computations
    # -- compound by dose is in dose_rows
    # -- compound with averaged dose is mu pool, y pool, and rescue pool
    # note, for each compound, get infected wells, and for each dose, average dose well embeddings to get a centroid. Then, project average onto sick healthy axis and compute the ATE.
    results = []
    _cent: dict[str, np.ndarray] = {}
    if args.save_centroids:
        _cent["__mu_mock__"] = mu_mock
        _cent["__mu_uinf__"] = mu_uinf
        _cent["__mu_uinf_src__"] = mu_uinf_src
    for nm, cid, label in panel: # per panel compound
        c_mask = pool_mask & (comp == cid) & (infected == 1)
        n_c = int(c_mask.sum())
        if n_c < args.min_dose_n:
            print(f"[eval] {nm[:28]:28} idx={cid}: only {n_c} rows -> skip")
            results.append({"name": nm, "idx": cid, "label": label, "skipped": True, "n": n_c})
            continue

        doses = np.unique(np.round(lx[c_mask], 3))
        dose_rows = []
        for d in doses:
            dmask = c_mask & (np.round(lx, 3) == d)
            nd = int(dmask.sum())
            if nd < args.min_dose_n:
                continue
            mu_d, _ = centroid(dmask, args.cap_per_group, key=f"{cid}@{d:+.3f}")
            if args.save_centroids:
                _cent[f"{cid}@{d:+.3f}"] = mu_d
            y_d = Y(mu_d)
            dose_rows.append({"log10_conc": float(d), "n": nd,
                              "Y": y_d, "rescue": (G - y_d) / G,
                              "ate": _ate(y_d),
                              "cos_mock": _cos(mu_d, mu_mock),
                              "cos_uinf": _cos(mu_d, mu_uinf)
                            })

        mu_pool, _ = centroid(c_mask, args.cap_per_group, key=f"{cid}@pooled")
        if args.save_centroids:
            _cent[f"{cid}@pooled"] = mu_pool
        y_pool = Y(mu_pool)
        rescue_pool = (G - y_pool) / G

        # Dose slope, or ATE gained per 10x concentration. to test if we learn dose effects reasonbly well. 
        ate_slope = None
        if len(dose_rows) >= 3:
            xs = np.array([r["log10_conc"] for r in dose_rows], dtype=np.float64)
            ys = np.array([r["ate"] for r in dose_rows], dtype=np.float64)
            ate_slope = float(np.polyfit(xs, ys, 1)[0])

        # get best dose
        best = max(dose_rows, key=lambda r: r["rescue"]) if dose_rows else None

        # get all the rest into one dict
        rec = {
            "name": nm, "idx": cid, "label": label, "n": n_c,
            "rescue_pooled": rescue_pool, "Y_pooled": y_pool,
            "rescue_best": (best["rescue"] if best else rescue_pool),
            "best_dose_log10_conc": (best["log10_conc"] if best else None),
            "ate_pooled": _ate(y_pool),
            "ate_slope": ate_slope,
            "ate_max": (max(r["ate"] for r in dose_rows) if dose_rows else _ate(y_pool)),
            "cos_mock_pooled": _cos(mu_pool, mu_mock),
            "cos_uinf_pooled": _cos(mu_pool, mu_uinf),
            "dose_curve": dose_rows,
        }
        results.append(rec)
        tag = "HIT" if label == 1 else ("neg" if label == 0 else "---")
        print(f"[eval] {nm[:28]:28} [{tag}] n={n_c:5d}  "
              f"rescue_best={rec['rescue_best']:+.3f}  "
              f"rescue_pool={rescue_pool:+.3f}")

    # -- Preliminary check on ATEs, check if compounds work (label 1) vs. doing nothing (label 0)
    ok = [r for r in results if not r.get("skipped")]
    labels = np.array([r["label"] for r in ok])
    rescue_best = np.array([r["rescue_best"] for r in ok])
    rescue_pool = np.array([r["rescue_pooled"] for r in ok])

    def split_mean(v):
        return (float(np.mean(v[labels == 1])) if (labels == 1).any() else float("nan"),
                float(np.mean(v[labels == 0])) if (labels == 0).any() else float("nan"))

    def _lab_block(v):
        h, n = split_mean(v)
        return {"hits_mean": h, "neg_mean": n, "separation": h - n, "auroc": _auroc(v, labels)}

    h_best, n_best = split_mean(rescue_best)
    h_pool, n_pool = split_mean(rescue_pool)

    # Now, EFFECTS: the estimates themselves, plus the projections.
    effects = {
        "infect_gap_G": G,
        "vehicle_anchor": args.vehicle_anchor,
        "anchor_pool": args.anchor_pool,
        "Y_vehicle_within_source": Y_uinf_src,
        "vehicle_offset_G": (Y_uinf_src - Y_uinf) / G,
        "n_scored": int(len(ok)),
        "n_skipped": int(len(results) - len(ok)),
        "n_hits": int((labels == 1).sum()),
        "n_negatives": int((labels == 0).sum()),
    }

    # ACCURACY, half one: scored against the curated HIT/NEGATIVE labels derived from literature.
    vs_literature = {
        "rescue_best": {"hits_mean": h_best, "neg_mean": n_best,
                        "separation": h_best - n_best,
                        "auroc": _auroc(rescue_best, labels)},
        "rescue_pooled": {"hits_mean": h_pool, "neg_mean": n_pool,
                          "separation": h_pool - n_pool,
                          "auroc": _auroc(rescue_pool, labels)},
        "ate_pooled": _lab_block(np.array([r.get("ate_pooled", 0.0) for r in ok])),
        "ate_slope": _lab_block(np.array([(r.get("ate_slope") or 0.0) for r in ok])),
    }

    # -- ACCURACY: this run's effects vs the real-data oracle (Test set) ---------------
    vs_oracle = None
    if args.truth and args.source == "real":
        print("[eval] --truth ignored on --source real (that is the oracle itself)")
    elif args.truth:
        with open(args.truth) as f:
            truth_doc = json.load(f)
        if truth_doc.get("source") != "real":
            raise RuntimeError(f"--truth {args.truth} has source={truth_doc.get('source')!r}; the  accuracy block is only meaningful against a --source real oracle.")

        # CHECK: Truth and this run must be computed from same stuff, or vs_oracle compares projections onto two different axes. 
        _t_eff = truth_doc.get("effects", {})
        for _k, _mine in (("anchor_pool", args.anchor_pool), ("vehicle_anchor", args.vehicle_anchor)):
            _theirs = _t_eff.get(_k)
            if _theirs is None:
                print(f"[eval] WARNING: --truth predates {_k}; frames may differ")
            elif _theirs != _mine:
                raise RuntimeError(
                    f"--truth was built with {_k}={_theirs!r} but this run uses {_mine!r}; rebuild the oracle under the same frame.")
        
        # now derive the pairs to compare
        t_by_idx = {int(r["idx"]): r for r in truth_doc["per_compound"]  if not r.get("skipped")}
        pairs = [(r, t_by_idx[int(r["idx"])]) for r in ok if int(r["idx"]) in t_by_idx]
        if not pairs:
            raise RuntimeError("--truth shares no scored compound with this run; check that both used the same --encoder and --pool.")
        
        # get oracle
        vs_oracle = {"truth_path": os.path.abspath(args.truth),
                     "truth_encoder": truth_doc.get("encoder"),
                     "truth_pool": truth_doc.get("pool"),
                     # If these disagree the two runs scored different compound sets, and `vs_literature.auroc` is not directly comparable to the oracle's.
                     "n_matched": int(len(pairs)),
                     "n_unmatched": int(len(ok) - len(pairs))}

        # Compound level check: pair up the values against the truth's by compound, and compute accuracies,...
        for key in ("rescue_best", "rescue_pooled", "ate_pooled", "ate_slope", "ate_max"):
            kp = [(e, t) for e, t in pairs if e.get(key) is not None and t.get(key) is not None]
            if not kp:
                continue
            est = np.array([e[key] for e, _ in kp], dtype=np.float64)
            tru = np.array([t[key] for _, t in kp], dtype=np.float64)
            vs_oracle[key] = _accuracy(est, tru)
            vs_oracle[key]["n"] = int(len(kp))

        # (compound, dose) level check, matched one to one against the balanced-design truth. 
        est_d, tru_d = [], []
        for e, t in pairs:
            tc = {round(r["log10_conc"], 3): r for r in t.get("dose_curve", [])}
            for r in e.get("dose_curve", []):
                m = tc.get(round(r["log10_conc"], 3))
                if m is not None and "ate" in r and "ate" in m:
                    est_d.append(r["ate"]); tru_d.append(m["ate"])
        
        # don't emit correlation if < 2 points
        if len(est_d) >= 3:
            vs_oracle["ate_per_arm"] = _accuracy(np.array(est_d, dtype=np.float64),  np.array(tru_d, dtype=np.float64))
            vs_oracle["ate_per_arm"]["n"] = int(len(est_d))

        if truth_doc.get("encoder") != args.encoder:
            print(f"[eval] WARNING: truth encoder={truth_doc.get('encoder')} but this run used {args.encoder}; the two are not on a common scale.")

    # -- QUALITY: distributional metrics on the row-matched image pairs -----
    quality = None
    if _qual:
        # derive and load data
        gen_t = torch.stack([im for _, im in _qual], dim=0)
        q_rows = np.asarray([i for i, _ in _qual], dtype=np.int64)
        n_bins = len(np.unique(dose_bin(lx[q_rows], ic[q_rows])))
        print(f"[eval] quality: {len(q_rows)} row-matched (real, generated) pairs sampled from {_qual_seen} generated, spanning {n_bins}/4 dose bins",  flush=True)
        real_t = _load_real_images(cfg, q_rows, num_workers=args.num_workers, batch_size=args.batch_size)

        # The domain extractor is loaded independently of --encoder: the domain FID/MMD are the comparable numbers across arms
        try:
            qual_model, _ = load_feature_extractor(cfg, device=device)
        except FileNotFoundError as e:
            print(f"[eval] domain feature extractor not found ({e}); Inception-only quality metrics.")
            qual_model = None

        # compute all metrics now
        metrics = compute_all(real_images=real_t, gen_images=gen_t, domain_model=qual_model, device=device, batch_size=args.batch_size)
        quality = {"n_pairs": int(len(q_rows)), "n_generated": int(_qual_seen), "marginal": [m.to_dict() for m in metrics]}
        q_bins = dose_bin(lx[q_rows], ic[q_rows])
        if qual_model is not None:
            quality["per_dose_bin_domain"] = compute_per_slice(
                real_images=real_t, gen_images=gen_t,
                real_slice_id=q_bins, gen_slice_id=q_bins,
                slice_names=DOSE_BIN_NAMES, domain_model=qual_model,
                device=device, batch_size=args.batch_size)

        # -- FIDELITY: did the generator render the dose it was CONDITIONED on? train a classifier via run_fillin_fidelity based on dose, andd predict on the generated to see if it aligns with the dose conditioned on.
        if not args.no_fidelity and n_bins < 2:
            print(f"[eval] fidelity: SKIPPED -- the retained rows span only {n_bins} dose bin, so TRTS has nothing to discriminate. Raise --quality_n.")
        elif not args.no_fidelity:
            f_pool = np.where(pool_mask)[0]
            f_idx = f_pool[_rng(args.seed, "fidelity").choice( len(f_pool), min(args.fidelity_n_real, len(f_pool)), replace=False)]
            print(f"[eval] fidelity: TRTS on dose_bin, {len(f_idx)} real train rows", flush=True)
            f_real = _load_real_images(cfg, f_idx, num_workers=args.num_workers, batch_size=args.batch_size)
            r = run_fillin_fidelity(
                f_real, torch.tensor(dose_bin(lx[f_idx], ic[f_idx]), dtype=torch.long),
                gen_t, torch.tensor(q_bins, dtype=torch.long),
                n_channels=cfg.image.n_channels, n_classes=4, label_name="dose_bin",
                epochs=args.fidelity_epochs, batch_size=128,
                num_workers=args.num_workers, device=device, seed=args.seed)
            quality["fidelity_dose_bin"] = {
                "test_acc": r.test_acc, "test_macro_f1": r.test_macro_f1,
                "chance": float(np.bincount(q_bins, minlength=4).max() / len(q_bins)),
                "n_classes_present": int(n_bins),
            }

    # construct output object
    out = {
        "estimand": "rescue(c,dose) = (G - Y(c,dose))/G, Y = proj on (mu_uinf - mu_mock); vehicle-anchored, dose-resolved, REAL data oracle.",
        "pool": args.pool, "device": args.device,
        "nuisance_dir": args.nuisance_dir or None,
        "encoder": args.encoder, "tvn": tvn_meta,
        # Provenance. Without this a generated/augmented panel is indistinguishable from the real oracle except by filename
        "source": args.source,
        "anchors": "real",
        "generator": (None if args.source in ("real", "roundtrip") else {
            "dit_subdir": args.dit_subdir, "gen_epoch": args.gen_epoch,
            "which_wgt": args.which_wgt, "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "gen_per_row": args.gen_per_row,
        }),
        "seed": args.seed,
        "decoder_vae": args.vae_path or None,
        "population_cell_type": (None if args.no_population_filter else cfg.population.cell_type),
        "n_rows_in_population": int(in_pop.sum()),
        # per_compound stays top level: ten downstream scorers index it.
        "per_compound": results,
        "effects": effects,
        "accuracy": {"vs_literature": vs_literature, "vs_oracle": vs_oracle},
        "quality": quality,
    }

    ### NOW Final write - out. All components are written here. ###
    # Tag non-default encoders into the filename: `rescue_panel.json` 
    # TVN belongs in the tag: it changes the feature space, so a raw and a whitened run are different results, not reruns of the same one.
    etag = "" if args.encoder == "domain" else f"_{args.encoder}"
    if args.tvn:
        etag += f"_tvn-{args.tvn_fit}-{args.tvn_center}"
    
    # Same for the dataset axis: a generated run depends on which arm, epoch and seed produced it. Untagged they would all overwrite rescue_panel.json -- the real-data oracle downstream scorers read as ground truth.
    if args.source == "real":
        stag = ""
    elif args.source == "roundtrip":
        stag = "_roundtrip"
    else:
        stag = f"_{args.source}_{gen_subdir}_ep{args.gen_epoch:04d}"
        if abs(args.guidance_scale - 1.0) > 1e-6:
            stag += f"_g{args.guidance_scale:g}"
        if args.gen_per_row != 1:
            stag += f"_m{args.gen_per_row}"
        if args.seed != 0:
            stag += f"_s{args.seed}"

    cttag = ("" if (args.no_population_filter or cfg.population.cell_type is None) else f"_{cfg.population.cell_type}")
    ptag = cttag if args.pool == "all" else f"{cttag}_pool{args.pool}"
    out_path = args.out or os.path.join(cfg.paths.train_output_dir, "eval_artifacts", f"rescue_panel{etag}{ptag}{stag}.json")
    oracle_path = os.path.join(cfg.paths.train_output_dir, "eval_artifacts", f"rescue_panel{etag}{cttag}.json")
    if ((args.source != "real" or args.pool != "all") and os.path.abspath(out_path) == os.path.abspath(oracle_path)):
        raise RuntimeError(f"refusing to write a --source {args.source} --pool {args.pool} panel onto the real-data oracle path {oracle_path}; pass an explicit --out")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if args.save_centroids:
        cent_path = out_path[:-len(".json")] + "_centroids.npz"
        np.savez_compressed(cent_path, **_cent)
        print(f"[eval] wrote {cent_path} ({len(_cent)} centroids)")

    print(f"\n=== EFFECTS (n={effects['n_scored']} scored, {effects['n_hits']} hits / {effects['n_negatives']} negatives) ===")
    for nm, v in (("ATE vs vehicle (pooled)", "ate_pooled"),
                  ("ATE dose slope", "ate_slope"),
                  ("rescue (best dose)", "rescue_best"),
                  ("rescue (dose-pooled)", "rescue_pooled")):
        d = vs_literature[v]
        print(f"{nm:>22}: hits {d['hits_mean']:+.3f} vs neg {d['neg_mean']:+.3f} sep={d['separation']:+.3f}  AUROC={d['auroc']:.3f}")

    if vs_oracle is not None:
        a = vs_oracle["rescue_best"]
        print("\n=== ACCURACY vs oracle (rescue_best, n=%d compounds) ===" % a["n"])
        print(f"MSE={a['mse']:.4f}  RMSE={a['rmse']:.4f}  bias={a['bias']:+.4f}  (oracle sd={a['truth_sd']:.4f})")
        print(f"Spearman rho={a['spearman_rho']:+.3f}  Pearson r={a['pearson_r']:+.3f}")
        if a["rmse"] > a["truth_sd"]:
            print("  NOTE: RMSE exceeds the oracle's own spread -- this estimator  carries no usable per-compound signal.")

    if quality is not None:
        print(f"\n=== IMAGE QUALITY ({quality['n_pairs']} row-matched pairs) ===")
        for m in quality["marginal"]:
            print(f"  {m['name']:>16}: {m['value']:.4f}")
        fid = quality.get("fidelity_dose_bin")
        if fid:
            print(f"  {'dose_bin TRTS':>16}: acc={fid['test_acc']:.4f} (chance {fid['chance']:.3f})  macroF1={fid['test_macro_f1']:.4f}")
            if fid["test_acc"] <= fid["chance"] + 0.02:
                print("  NOTE: TRTS at chance -- the generator is not rendering the dose it was conditioned on, so any effect estimate is noise.")

    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
