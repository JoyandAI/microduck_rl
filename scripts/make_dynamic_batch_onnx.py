#!/usr/bin/env python3
"""Rewrite an exported policy ONNX so its batch axis is symbolic (dynamic).

WHY THIS EXISTS
`scripts/export.py` bakes obs-normalisation into the graph (mandatory) but leaves
the batch axis annotated as the literal `1`. `onnxruntime` then REFUSES any other
batch size, so every evaluator that wants N independent DR draws has to fall back
to one `sess.run` per environment per control step. At 50 Hz that is
`N * 1000` ORT calls for a 20 s / 64 env measurement — the dominant cost of a
straightness matrix by ~50x.

The graph itself is already batch-agnostic (an MLP over the last axis plus a
normaliser); only the *declared* shape is narrow. This script rewrites that
declaration and then PROVES the rewrite is safe:

  1. batch-1 output must be bit-identical to the original graph;
  2. batch-N output must equal N stacked batch-1 runs.

Only step 3 (`--write`) touches disk, and only at the path you give — never the
delivered artefact.

Usage:
    # verify only (default) — nothing is written
    uv run python scripts/make_dynamic_batch_onnx.py --onnx walk_hd1910_12000.onnx

    # write the dynamic copy, then verify it
    uv run python scripts/make_dynamic_batch_onnx.py --onnx walk_hd1910_12000.onnx \
        --write /tmp/walk_dynbatch.onnx --check-batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _dynamic_value_info(value_info, dim_name: str):
    """Rebuild a value_info with axis 0 replaced by the symbolic dim `dim_name`."""
    t = value_info.type.tensor_type
    dims: list = []
    for k, d in enumerate(t.shape.dim):
        if k == 0:
            dims.append(dim_name)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        elif d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(dim_name)
    return helper.make_tensor_value_info(value_info.name, t.elem_type, dims)


def symbolise(model: onnx.ModelProto, dim_name: str = "batch") -> int:
    """Make the batch axis symbolic on every graph input/output. Returns #changed."""
    changed = 0
    init_names = {t.name for t in model.graph.initializer}
    for slot in ("input", "output"):
        seq = getattr(model.graph, slot)
        for k, vi in enumerate(seq):
            if vi.name in init_names or len(vi.type.tensor_type.shape.dim) < 2:
                continue
            d0 = vi.type.tensor_type.shape.dim[0]
            if d0.HasField("dim_param") and d0.dim_param == dim_name:
                continue
            seq[k].CopyFrom(_dynamic_value_info(vi, dim_name))
            changed += 1
    return changed


def _run(sess: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    i, o = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    return sess.run([o], {i: obs.astype(np.float32)})[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--write", default=None,
                    help="output path for the dynamic-batch copy (omit = verify only)")
    ap.add_argument("--check-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    orig_path = args.onnx
    ref = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    in_shape = ref.get_inputs()[0].shape
    obs_dim = int(in_shape[1])
    fixed = isinstance(in_shape[0], int)
    print(f"[i] {orig_path}: input {in_shape} -> "
          f"{'batch FIXED at 1 (per-env inference required)' if fixed else 'already dynamic'}")

    model = onnx.load(orig_path)
    before = onnx.load(orig_path)
    n_changed = symbolise(model)
    if not fixed and n_changed == 0:
        print("[i] nothing to do — already dynamic; verifying it anyway")

    tmp = Path("/tmp") / f"_dynbatch_{Path(orig_path).stem}.onnx"
    onnx.save(model, tmp)
    dyn = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    print(f"[i] rewritten: {[(i.name, i.shape) for i in dyn.get_inputs()]} "
          f"({n_changed} value_info dims symbolised)")

    rng = np.random.default_rng(args.seed)

    # ---- proof 1: batch-1 must be bit-identical to the original ----------
    x1 = rng.standard_normal((1, obs_dim), dtype=np.float32)
    a, b = _run(ref, x1), _run(dyn, x1)
    identical = bool(np.array_equal(a, b))
    print(f"[1] batch-1 identical to original : {identical} "
          f"(max|diff| = {float(np.max(np.abs(a - b))):.3e})")
    if not identical:
        print("[FAIL] the rewrite changed the graph's numerics; refusing to continue")
        return 1

    # ---- proof 2: batch-N must equal N stacked batch-1 runs --------------
    n = max(2, args.check_batch)
    xn = rng.standard_normal((n, obs_dim), dtype=np.float32)
    batched = _run(dyn, xn)
    stacked = np.concatenate([_run(ref, xn[k:k + 1]) for k in range(n)], axis=0)
    maxdiff = float(np.max(np.abs(batched - stacked)))
    ok = maxdiff < 1e-5 and batched.shape == stacked.shape
    print(f"[2] batch-{n} == {n} single runs  : {ok} "
          f"(shape {batched.shape} vs {stacked.shape}, max|diff| = {maxdiff:.3e})")
    if not ok:
        print("[FAIL] batching is not equivalent; do NOT use the dynamic copy")
        return 1

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.CopyFrom(before)
        symbolise(model)
        onnx.save(model, out)
        chk = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        print(f"[i] wrote {out} -> input {chk.get_inputs()[0].shape}")
        print("[OK] dynamic-batch copy verified (this is an EVAL copy; the "
              "delivered artefact is untouched)")
    else:
        print("[OK] rewrite verified; pass --write <path> to keep a dynamic copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
