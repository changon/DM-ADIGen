"""GPU smoke test of the Phase 2/3 generator path (IMPLEMENT.md §3.11). Imports
PyTorch: run it on a GPU node, never on the dev box.

  1. synthetic (no data): a toy CondSpec and (B, 978) inputs, for mlp and
     dit1d (patch 10 and 6):
       - output (B, G), exactly 0 at init (adaLN-Zero);
       - timestep as int / float / 0-dim / (B,) gives one output;
       - drop raises at class_dropout_prob = 0; at p > 0 train mode draws its
         own mask, and drop=all nulls every role-A field;
       - return_dict=False; arch.json round trip (build_generator_from_ckpt)
         reproduces the outputs; gradient checkpointing gives the same outputs
         and gradients; dit1d patchify / unpatchify round trip (pad sliced off);
       - a fixed batch is overfit under DDPM and FM (rank-agnostic loss);
       - parameter counts at the v1 sizes; one AlphaNet URR step.
  2. real (--data_dir build): LincsDataset batch -> cond -> both backbones,
     forward + backward; the trainer's _RowBatcher equals stacked
     LincsDataset rows and is seeded per epoch; every cmean*.npz in the split
     dir equals the group means of LincsDataset.y (an independent check of
     precompute_cmean's numpy path).
  3. --ckpt_dir RUN (repeatable): rebuild from arch.json alone, strict-load the
     EMA weights (model_1.safetensors) of its last checkpoint, and reproduce
     that epoch's val_loss_ema from loss_history.jsonl.
Exits nonzero on any failure.

    python -m src.tests.test_lincs_shapes --device cuda [--data_dir ...] [--synthetic 0] [--real 0] [--ckpt_dir runs/...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import LincsDataset, build_cond_spec, cond_from_batch, dose_probe  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.models import (  # noqa: E402
    arch_spec, build_generator, build_generator_from_ckpt, read_arch_spec, resolve_arch_kwargs,
    write_arch_spec)
from src.models.conditioning import CondSpec, Field  # noqa: E402
from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.nuisances.fit_urr import urr_loss  # noqa: E402
from src.processes import make_train_flow_matching, make_train_scheduler  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)
from src.train.train_diffusion import (  # noqa: E402
    HISTORY_FILENAME, _per_sample_loss, _row_tensors, _RowBatcher, _validation_loss, select_val_rows)

FAILS: list[str] = []
G = 978


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, KeyError):
        return True
    return False


# === 1. synthetic ================================================================

def toy_spec(n_comp: int = 6, with_c: bool = True) -> CondSpec:
    """The v1 action fields, plus (with_c) a role-C field, which CFG must never drop."""
    fields = [
        Field(name="compound", role="A", kind="cat", cardinality=n_comp),
        Field(name="is_control", role="A", kind="cat", cardinality=2),
        Field(name="dose", role="A", kind="cont", loc=0.0, scale=1.0, nullable=True),
    ]
    if with_c:
        fields.append(Field(name="syn_c", role="C", kind="cat", cardinality=2, levels=("0", "1")))
    return CondSpec(tuple(fields))


def toy_cond(B: int, dev, n_comp: int = 6, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(1, n_comp, (B,), generator=g)
    ctl = torch.rand(B, generator=g) < 0.2
    comp[ctl] = 0
    dose = torch.randn(B, generator=g)
    dose[ctl] = float("nan")
    syn_c = torch.randint(0, 2, (B,), generator=g)
    return {"compound": comp.to(dev), "is_control": ctl.long().to(dev), "dose": dose.to(dev),
            "syn_c": syn_c.to(dev)}


def _jitter(model: torch.nn.Module, scale: float = 0.02, seed: int = 0) -> None:
    """Move every weight off its init so zero-initialised paths carry signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p.device) * scale)


CASES = (("mlp", "S", None), ("dit1d", "S", 10), ("dit1d", "S", 6))


def synthetic_section(dev) -> None:
    # full fp32 here: TF32 kernels differ with the batch size (~1e-4), which the per-row checks would see
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        _synthetic_section(dev)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def _synthetic_section(dev) -> None:
    B = 32
    spec = toy_spec()
    cond = toy_cond(B, dev)
    x = torch.randn(B, G, device=dev)
    for arch, size, patch in CASES:
        tag = f"{arch}-{size}" + (f"/p{patch}" if patch else "")
        kw = resolve_arch_kwargs(arch, size, patch_size=patch)
        torch.manual_seed(0)
        m = build_generator(spec, arch=arch, n_genes=G, arch_kwargs=kw, class_dropout_prob=0.1).to(dev).eval()
        out0 = m(x, 500, cond).sample
        check(tuple(out0.shape) == (B, G) and bool((out0 == 0).all()),
              f"{tag}: output {tuple(out0.shape)}, exactly 0 at init (adaLN-Zero)")
        _jitter(m)
        with torch.no_grad():
            outs = [m(x, t, cond).sample for t in
                    (500, 500.0, torch.tensor(500, device=dev), torch.full((B,), 500, device=dev),
                     torch.full((B,), 500.0, device=dev))]
            o400 = m(x, 400, cond).sample
            tv = torch.linspace(0, 999, B, device=dev)
            ov = m(x, tv, cond).sample
            rows = [0, B // 2, B - 1]
            orow = torch.cat([m(x[i:i + 1], tv[i], {k: v[i:i + 1] for k, v in cond.items()}).sample for i in rows])
        d_forms = max(float((o - outs[0]).abs().max()) for o in outs)
        d_400 = float((o400 - outs[0]).abs().max())
        d_row = float((orow - ov[rows]).abs().max())
        check(d_forms < 1e-5 and float(outs[0].abs().mean()) > 0,
              f"{tag}: timestep int / float / 0-dim / (B,) long / (B,) float give one output "
              f"(max |diff| {d_forms:.1e}; |out| {float(outs[0].abs().mean()):.3g})")
        check(d_400 > 1e-3 and d_row < 1e-5,
              f"{tag}: t=400 vs 500 differ (max |diff| {d_400:.2e}); a (B,) t of distinct values acts per row "
              f"(batch vs one-row calls, max |diff| {d_row:.1e})")
        with torch.no_grad():
            tup = m(x, 500, cond, return_dict=False)
        check(isinstance(tup, tuple) and len(tup) == 1 and torch.equal(tup[0], outs[0]), f"{tag}: return_dict=False -> (out,)")

        # CFG drop
        other = {**toy_cond(B, dev, seed=1), "syn_c": cond["syn_c"]}         # other A, same C
        flip_c = {**cond, "syn_c": 1 - cond["syn_c"]}                        # same A, other C
        ones = torch.ones(B, dtype=torch.bool, device=dev)
        mixed = torch.arange(B, device=dev) % 2 == 0
        with torch.no_grad():
            d1 = m(x, 500, cond, drop=ones).sample
            d2 = m(x, 500, other, drop=ones).sample
            dc = m(x, 500, flip_c, drop=ones).sample
            d0 = m(x, 500, cond, drop=torch.zeros(B, dtype=torch.bool, device=dev)).sample
            dm = m(x, 500, cond, drop=mixed).sample
            m.train()
            r1, r2 = m(x, 500, cond).sample, m(x, 500, cond).sample
            m.eval()
        check(torch.allclose(d1, d2, atol=1e-5) and not torch.allclose(d1, outs[0], atol=1e-5)
              and not torch.allclose(dc, d1, atol=1e-5)
              and torch.allclose(d0, outs[0], atol=1e-5) and not torch.equal(r1, r2)
              and torch.allclose(dm[mixed], d1[mixed], atol=1e-5) and torch.allclose(dm[~mixed], outs[0][~mixed], atol=1e-5),
              f"{tag}: drop=all nulls every role-A field and never the role-C one; a mixed mask acts per row; "
              f"drop=none = eval output; train mode draws its own mask")
        m0 = build_generator(spec, arch=arch, n_genes=G, arch_kwargs=kw, class_dropout_prob=0.0).to(dev)
        check(_raises(lambda: m0(x, 500, cond, drop=torch.ones(B, dtype=torch.bool, device=dev)))
              and _raises(lambda: m(x[:, :G - 1], 500, cond)),
              f"{tag}: drop refused at class_dropout_prob=0; a (B, G-1) sample refused")

        # arch.json round trip
        with tempfile.TemporaryDirectory() as td:
            write_arch_spec(td, arch_spec(spec, arch=arch, size=size, n_genes=G, arch_kwargs=kw, class_dropout_prob=0.1))
            m2 = build_generator_from_ckpt(os.path.join(td, "checkpoint-0000")).to(dev).eval()
            m2.load_state_dict(m.state_dict(), strict=True)
            with torch.no_grad():
                o2 = m2(x, 500, cond).sample
            check(torch.allclose(o2, outs[0], atol=1e-6) and type(m2) is type(m),
                  f"{tag}: arch.json -> build_generator_from_ckpt (checkpoint-NNNN reads its parent) reproduces the output")

        # gradient checkpointing: same outputs and grads (same CFG mask via the seed)
        grads = []
        for ck in (False, True):
            m.zero_grad(set_to_none=True)
            m.train()
            (m.enable_gradient_checkpointing if ck else m.disable_gradient_checkpointing)()
            torch.manual_seed(7)
            o = m(x, 500, cond).sample
            o.square().mean().backward()
            grads.append((o.detach(), [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]))
        m.disable_gradient_checkpointing()
        (oa, ga), (ob, gb) = grads
        check(torch.allclose(oa, ob, atol=1e-5) and len(ga) == len(gb)
              and all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(ga, gb)),
              f"{tag}: gradient checkpointing gives the same outputs and gradients ({len(ga)} tensors)")
        if arch == "dit1d":
            xp = m.patchify(x)
            check(tuple(xp.shape) == (B, m.n_tokens, patch) and torch.equal(m.unpatchify(xp), x)
                  and m.pad == (-G) % patch and bool((xp.reshape(B, -1)[:, G:] == 0).all()),
                  f"{tag}: {m.n_tokens} tokens, pad {m.pad}; patchify/unpatchify round trip slices the pad off")

    # overfit a fixed batch: the rank-agnostic loss trains both backbones under both processes
    B = 64
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(6, G, generator=g)
    cond = toy_cond(B, dev, seed=3)
    y = (mu[cond["compound"].cpu()] + 0.1 * torch.randn(B, G, generator=g)).to(dev)
    for arch, size, patch in CASES[:2]:
        for method in ("ddpm", "fm"):
            torch.manual_seed(0)
            m = build_generator(toy_spec(), arch=arch, n_genes=G,
                                arch_kwargs=resolve_arch_kwargs(arch, size, patch_size=patch),
                                class_dropout_prob=0.0).to(dev).train()
            sched = make_train_scheduler(True) if method == "ddpm" else None
            fm = make_train_flow_matching() if method == "fm" else None
            opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
            losses = []
            for _ in range(300):
                per = _per_sample_loss(m, y, cond, sched, fm)
                loss = per.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            a, b = np.mean(losses[:20]), np.mean(losses[-20:])
            # the fit uses the conditioning: the same noise draws score worse with shuffled compounds
            m.eval()
            perm = torch.randperm(B, generator=torch.Generator().manual_seed(9)).to(dev)
            shuf = {**cond, "compound": cond["compound"][perm], "is_control": cond["is_control"][perm],
                    "dose": cond["dose"][perm]}
            with torch.no_grad():
                l_true = float(np.mean([float(_per_sample_loss(m, y, cond, sched, fm, generator=torch.Generator(device=dev).manual_seed(s_)).mean()) for s_ in range(8)]))
                l_shuf = float(np.mean([float(_per_sample_loss(m, y, shuf, sched, fm, generator=torch.Generator(device=dev).manual_seed(s_)).mean()) for s_ in range(8)]))
            check(per.shape == (B,) and np.isfinite(losses).all() and b < 0.6 * a and l_true < l_shuf,
                  f"{arch}-{size} {method}: fixed batch overfit, loss {a:.4f} -> {b:.4f} (per-sample {tuple(per.shape)}); "
                  f"eval loss with its own conditioning {l_true:.4f} < shuffled compounds {l_shuf:.4f}")

    # parameter counts at the v1 sizes (compound vocab of mcf7_24h)
    spec_v1 = toy_spec(1751, with_c=False)
    for arch, size, patch, band in (("mlp", "B", None, (30e6, 45e6)), ("dit1d", "S", 10, (28e6, 38e6))):
        m = build_generator(spec_v1, arch=arch, n_genes=G, arch_kwargs=resolve_arch_kwargs(arch, size, patch_size=patch),
                            class_dropout_prob=0.1)
        n = sum(p.numel() for p in m.parameters())
        check(band[0] <= n <= band[1], f"{arch}-{size}{'/' + str(patch) if patch else ''}: {n/1e6:.1f}M parameters "
              f"(IMPLEMENT.md §3.4: MLP-B ~35M, DiT-S/10 ~33M)")

    # one AlphaNet URR step (§3.11)
    B = 128
    cond = toy_cond(B, dev, seed=5)
    cov = torch.zeros(B, 3, device=dev)
    net = AlphaNet(n_compounds=6, cov_dim=3, cov_idx=[], positive=True).to(dev)
    args_net = (cov, cond["compound"], torch.nan_to_num(cond["dose"], nan=-10.0), cond["is_control"].float())
    trt = torch.nonzero(args_net[3] < 0.5).squeeze(-1)
    jt = trt[torch.randint(0, trt.numel(), (B, 8), device=dev)]
    before = [p.detach().clone() for p in net.parameters()]
    loss, _ = urr_loss(net, *args_net, args_net[1][jt], args_net[2][jt], args_net[3][jt])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    check(bool(torch.isfinite(loss)) and any(not torch.equal(a, p) for a, p in zip(before, net.parameters())),
          f"one AlphaNet URR step: loss {float(loss):+.4f}, weights updated")


# === 2. real data ===================================================================

def real_section(cfg, dev, batch: int) -> None:
    nz = cfg.paths.nuisance_dir
    s = load_splits(cfg)
    tr = s["train_idx"]
    n_compounds = json.load(open(os.path.join(nz, "nuisance_meta.json")))["n_compounds"]
    ds = LincsDataset(cfg, indices=tr)
    trt = ds.is_control.numpy() == 0
    spec = build_cond_spec(cfg, n_compounds, ds.log10_conc.numpy()[trt])

    batcher = _RowBatcher(_row_tensors(ds), batch, dev, shuffle=True, seed=0)
    idx = [0, len(ds) - 1, 5, 17, len(ds) // 2]
    got = batcher._take(torch.tensor(idx, device=dev))
    stacked = {k: torch.stack([ds[i][k] for i in idx]).to(dev) for k in got}
    e3a, e3b, e4 = next(batcher.epoch(3)), next(batcher.epoch(3)), next(batcher.epoch(4))
    n_seen = sum(b["y"].shape[0] for b in batcher.epoch(0))
    check(all(torch.equal(stacked[k], got[k]) for k in got) and torch.equal(e3a["y"], e3b["y"])
          and not torch.equal(e3a["y"], e4["y"]) and n_seen == len(ds) and len(batcher) == -(-len(ds) // batch),
          f"_RowBatcher: batch rows = stacked LincsDataset rows (keys {sorted(got)}); seeded per epoch; "
          f"one epoch covers {n_seen:,} rows in {len(batcher)} batches")

    for arch, size, patch in (("mlp", "B", None), ("dit1d", "S", 10)):
        torch.manual_seed(0)
        m = build_generator(spec, arch=arch, n_genes=G, arch_kwargs=resolve_arch_kwargs(arch, size, patch_size=patch),
                            class_dropout_prob=0.1).to(dev).train()
        fac = m.calibrate_conditioning(dose_probe(ds.log10_conc.numpy()[trt]))
        cond = cond_from_batch(e3a, spec, dev)
        ok = True
        for method in ("ddpm", "fm"):
            per = _per_sample_loss(m, e3a["y"], cond, make_train_scheduler(True) if method == "ddpm" else None,
                                   make_train_flow_matching() if method == "fm" else None)
            per.mean().backward()
            gn = [p.grad for p in m.parameters() if p.grad is not None]
            ok &= per.shape == (e3a["y"].shape[0],) and bool(torch.isfinite(per).all()) \
                and all(bool(torch.isfinite(g_).all()) for g_ in gn)
            m.zero_grad(set_to_none=True)
        check(ok, f"real {arch}-{size}: batch {tuple(e3a['y'].shape)} -> cond {list(spec.names)} -> DDPM and FM "
                  f"per-sample loss + backward finite; calibrate {fac}")

    files = sorted(glob.glob(os.path.join(nz, "cmean*.npz")))
    by_pc = {cfg.outcome.plate_center: ds}
    for f in files:
        z = np.load(f)
        if not np.array_equal(z["train_idx"], tr):
            check(False, f"{os.path.basename(f)}: train_idx is not this split's")
            continue
        pc = str(z["plate_center"])
        if pc not in by_pc:
            by_pc[pc] = LincsDataset(cfg, indices=tr, plate_center=pc)
        dsp = by_pc[pc]
        gid_np = z["row_gid"]
        gid = torch.from_numpy(gid_np).to(dev)
        keep = gid >= 0
        K = z["mu"].shape[0]
        y = dsp.y.to(dev).double()
        sums = torch.zeros(K, G, dtype=torch.float64, device=dev).index_add_(0, gid[keep], y[keep])
        cnt = torch.bincount(gid[keep], minlength=K)
        mu = (sums / cnt[:, None]).float().cpu().numpy()
        err = float(np.abs(mu - z["mu"]).max()) if K else 0.0
        # one arm per group: compound and vehicle flag constant, doses within one dose_level (+-0.05 log10)
        kk = gid_np >= 0
        df = {"g": gid_np[kk], "c": dsp.compound_idx.numpy()[kk], "v": dsp.is_control.numpy()[kk],
              "d": dsp.log10_conc.numpy()[kk]}
        same = True
        for arr in ("c", "v"):
            lo = np.full(K, np.iinfo(np.int64).max)
            hi = np.full(K, np.iinfo(np.int64).min)
            np.minimum.at(lo, df["g"], df[arr].astype(np.int64))
            np.maximum.at(hi, df["g"], df[arr].astype(np.int64))
            same &= bool((lo == hi).all())
        dlo, dhi = np.full(K, np.inf), np.full(K, -np.inf)
        np.minimum.at(dlo, df["g"], df["d"])
        np.maximum.at(dhi, df["g"], df["d"])
        spread = float((dhi - dlo).max()) if K else 0.0
        check(np.array_equal(cnt.cpu().numpy(), z["counts"]) and err < 1e-4 and same and spread <= 0.1 + 1e-6,
              f"{os.path.basename(f)} (plate_center {pc}): mu = group means of LincsDataset.y over "
              f"{int(keep.sum()):,} covered train rows, {K} arms (max |diff| {err:.1e}); each group one compound / "
              f"vehicle flag, log10 dose spread <= {spread:.3f}; min_n {int(z['min_n'])}")
    if not files:
        print(f"skip  no cmean*.npz in {nz}")


# === 3. rebuild a trained run from arch.json ===========================================

def ckpt_section(run_dir: str, dev) -> None:
    """Rebuild a trained run from arch.json alone; its EMA (model_1) and training (model)
    weights must reproduce the logged val_loss_ema / val_loss of that checkpoint's epoch."""
    from safetensors.torch import load_file

    run_dir = os.path.abspath(run_dir)
    tag = os.path.basename(run_dir)
    arch = read_arch_spec(run_dir)
    hpath = os.path.join(run_dir, HISTORY_FILENAME)
    hist = [json.loads(ln) for ln in open(hpath) if ln.strip()] if os.path.isfile(hpath) else []
    logged = {h["epoch"]: h for h in hist if h.get("val_loss_ema") is not None}
    # the newest complete checkpoint whose epoch has a logged validation
    ckpts = []
    for c in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        mm = re.search(r"checkpoint-(\d+)$", c)
        if (mm and int(mm.group(1)) in logged
                and all(os.path.isfile(os.path.join(c, f)) for f in ("model.safetensors", "model_1.safetensors"))):
            ckpts.append((int(mm.group(1)), c))
    if arch is None or not ckpts:
        check(False, f"{tag}: no arch.json, or no complete checkpoint at an epoch with a logged validation")
        return
    epoch, last = max(ckpts)

    ta = arch["train_args"]
    cfg = apply_paths_args(config_from_args(argparse.Namespace(**ta)), argparse.Namespace(**ta))
    s = load_splits(cfg)
    val_idx = select_val_rows(s["holdout_idx"], int(arch.get("val_cap", ta["val_cap"])))
    val_ds = LincsDataset(cfg, indices=val_idx, plate_center=arch["plate_center"])
    check([int(g) for g in val_ds.expr_meta["pr_gene_id"]] == arch["gene_pr_ids"],
          f"{tag}: arch.json gene order = the data's (sha1 {arch['gene_order_sha1'][:12]})")
    batcher = _RowBatcher(_row_tensors(val_ds), int(ta["train_batch_size"]), dev, shuffle=False)
    if arch["diffusion_method"] == "fm":
        sched, fm = None, make_train_flow_matching(int(arch["num_train_timesteps"]), tau_dist=arch["tau_dist"],
                                                   tau_ln_m=arch["tau_ln_m"], tau_ln_s=arch["tau_ln_s"])
    else:
        sched, fm = make_train_scheduler(bool(arch["zero_snr"])), None
    mp = arch.get("mixed_precision", "no")

    for fname, key, what in (("model_1.safetensors", "val_loss_ema", "EMA"), ("model.safetensors", "val_loss", "training")):
        m = build_generator_from_ckpt(last).to(dev).eval()
        try:
            m.load_state_dict(load_file(os.path.join(last, fname), device=str(dev)), strict=True)
        except RuntimeError as e:
            check(False, f"{tag}: strict load of {os.path.basename(last)}/{fname} failed: {e}")
            continue
        # the trainer's prepared model runs its forward under autocast when mixed precision is on
        with torch.autocast(dev.type, dtype=torch.bfloat16 if mp == "bf16" else torch.float16, enabled=mp != "no"):
            got = _validation_loss(m, sched, batcher, dev, m.cond_spec, fm=fm)
        want = logged[epoch][key]
        rel = abs(got - want) / max(abs(want), 1e-12)
        n = sum(p.numel() for p in m.parameters())
        check(rel < 1e-3, f"{tag}: {type(m).__name__} ({n/1e6:.1f}M) rebuilt from arch.json alone + strict-loaded "
                          f"{os.path.basename(last)}/{fname} ({what}): {key} {got:.6f} vs logged {want:.6f} "
                          f"at epoch {epoch} (rel {rel:.1e}; mixed_precision {mp})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--synthetic", type=int, default=1)
    p.add_argument("--real", type=int, default=1)
    p.add_argument("--ckpt_dir", action="append", default=[], help="A trained run dir (repeatable).")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[shapes] --device cuda but CUDA is unavailable")
    dev = torch.device(args.device)
    # the trainer's math settings, so validation losses reproduce
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[shapes] torch {torch.__version__} device {dev}"
          + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""), flush=True)

    if args.synthetic:
        synthetic_section(dev)
    if args.real:
        real_section(apply_paths_args(config_from_args(args), args), dev, args.batch)
    for d in args.ckpt_dir:
        ckpt_section(d, dev)

    if FAILS:
        print(f"[shapes] {len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("[shapes] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
