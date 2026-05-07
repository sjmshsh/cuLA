#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
bench_kda_fused_fwd.py — Benchmark: cuLA fully-fused KDA forward vs FLA Triton baseline

Automatically selects the cuLA fully-fused implementation based on the current
GPU architecture:
  - sm100 (Blackwell) → cula.kda.blackwell_fused_fwd.flash_kda_prefill
  - sm90  (Hopper)    → cula.kda.hopper_fused_fwd.cula_kda_prefill

Compares:
  - Accuracy: RMSE, relative max diff between cuLA fully-fused and FLA Triton
  - Performance: kernel execution time (ms) with CUDA events

Modes (--mode, default: all):
  - fixed:    Fixed-length sequences, various (B, T, H, HV) configs.
              GVA rows (HV > H) are mixed in alongside non-GVA rows.
  - varlen:   Variable-length sequences with 2-3x length variation.
              Non-GVA base configs plus a GVA subset (H=16, HV=64).
  - gva:      Dedicated GVA benchmark (fixed + varlen) using prepare_gva_inputs.
              Covers multiple GVA ratios (2x / 4x / 8x); compares cuLA vs FLA.
  - overhead: GVA overhead benchmark — cuLA GVA vs cuLA non-GVA at the same
              total head count (HV).  Both paths present identical tensor shapes
              to the kernel, so a near-zero overhead% proves that GVA adds no
              measurable kernel latency regression.
  - both:     Fixed-length + varlen only (legacy alias, no gva/overhead).
  - all:      Run all of the above.

Under GVA (HV > H), q/k are expanded from H to HV heads via
`repeat_interleave(..., dim=2)`, equivalent to the einops pattern
`repeat(x, "... h d -> ... (h g) d")`.  This keeps FLA's `chunk_kda`
(which does not natively support GVA) and cuLA's SM100 fully-fused forward
(which requires q/k/v to share the head dim) on the same input layout.

Usage:
  python bench_kda_fused_fwd.py [--mode fixed|varlen|gva|overhead|both|all] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_fused_fwd.py --mode overhead --ncu
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    build_gva_fixed_configs,
    build_gva_varlen_configs,
    build_varlen_configs,
    exclusive_cumsum,
    prepare_gva_inputs,
    prepare_safe_gate_inputs,
    set_seed,
)
from cula.utils import get_device_sm_version, get_kda_fused_fwd

# ============================================================
# Resolve cuLA fully-fused implementation at import time
# ============================================================
_device = torch.device("cuda")
_major, _minor = get_device_sm_version(_device)
_SM_TAG = f"sm{_major}{_minor}"
cula_kda_fused_fwd = get_kda_fused_fwd(_device)

# ============================================================
# Constants
# ============================================================
# Default number of Q/K heads. Each benchmark config may additionally specify
# HV (number of V heads) to enable GVA (HV > H must be a positive multiple of H).
H, D = 64, 128
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False
HAS_INIT_STATE = False


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    if n_iters is None:
        n_iters = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / n_iters


def accuracy_stats(ref, out):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return rmse, rel_max, mean_diff


def run_fla(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
    return fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )


def run_cula(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
    return cula_kda_fused_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )


# ============================================================
# Config normalization helpers
# ============================================================
def _normalize_fixed_config(cfg):
    """Accept either (B, T) or (B, T, H_qk, HV) and return the 4-tuple form.

    For the 2-tuple legacy form, defaults to H_qk=HV=H (no GVA).
    """
    if len(cfg) == 2:
        B, T = cfg
        return B, T, H, H
    if len(cfg) == 4:
        return cfg
    raise ValueError(f"Fixed config must be (B, T) or (B, T, H, HV), got {cfg!r}")


def _normalize_varlen_config(cfg):
    """Accept (seq_lens, total_len, dist) or (seq_lens, total_len, dist, H_qk, HV).

    For the 3-tuple legacy form, defaults to H_qk=HV=H (no GVA).
    """
    if len(cfg) == 3:
        seq_lens, total_len, dist = cfg
        return seq_lens, total_len, dist, H, H
    if len(cfg) == 5:
        return cfg
    raise ValueError(
        f"Varlen config must be (seq_lens, total_len, dist) or (seq_lens, total_len, dist, H, HV), got {cfg!r}"
    )


# ============================================================
# Fixed-length benchmark (GVA-aware via per-config HV)
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(f" Fixed-Length Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton  (GVA when HV > H)")
    print("=" * 100)
    results = []

    for cfg in configs:
        B, T, H_qk, HV = _normalize_fixed_config(cfg)
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(
            B, T, H_qk, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
            num_v_heads=HV,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "H": H_qk,
                "HV": HV,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark (GVA-aware via per-config HV)
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 100)
    print(f" Varlen Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton  (GVA when HV > H)")
    print("=" * 100)
    results = []

    for cfg in configs:
        seq_lens, total_len, dist, H_qk, HV = _normalize_varlen_config(cfg)
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(
            1, T, H_qk, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
            num_v_heads=HV,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        results.append(
            {
                "tag": tag,
                "dist": dist,
                "T_total": T,
                "n_seqs": n_seqs,
                "H": H_qk,
                "HV": HV,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# GVA-dedicated benchmarks (uses prepare_gva_inputs)
# ============================================================
def bench_gva_fixed(configs):
    """Fixed-length GVA benchmark using :func:`prepare_gva_inputs`.

    All configs must have HV > H.  Data is prepared the same way as in the
    KimiDeltaAttention layer: q/k/g/beta are first generated with H heads and
    then expanded to HV heads via einops repeat before being fed to both cuLA
    and FLA.
    """
    print("\n" + "=" * 100)
    print(f" GVA Fixed-Length Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []

    for B, T, H_qk, HV in configs:
        assert HV > H_qk, f"GVA requires HV > H, got H={H_qk} HV={HV}"
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_gva_inputs(
            B, T, H_qk, HV, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]
        gva_ratio = inputs["gva_ratio"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "H": H_qk,
                "HV": HV,
                "gva_ratio": gva_ratio,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


def bench_gva_varlen(configs):
    """Varlen GVA benchmark using :func:`prepare_gva_inputs`.

    Configs are 5-tuples (seq_lens, total_len, dist, H, HV) as produced by
    :func:`~benchmarks.utils.build_gva_varlen_configs`.
    All configs must have HV > H.
    """
    print("\n" + "=" * 100)
    print(f" GVA Varlen Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []

    for seq_lens, total_len, dist, H_qk, HV in configs:
        assert HV > H_qk, f"GVA requires HV > H, got H={H_qk} HV={HV}"
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_gva_inputs(
            1, T, H_qk, HV, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]
        gva_ratio = inputs["gva_ratio"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        results.append(
            {
                "tag": tag,
                "dist": dist,
                "T_total": T,
                "n_seqs": n_seqs,
                "H": H_qk,
                "HV": HV,
                "gva_ratio": gva_ratio,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# GVA overhead benchmark
# (proves GVA adds no kernel cost vs a plain non-GVA run)
# ============================================================
def bench_gva_overhead(configs):
    """Quantify the kernel overhead introduced by GVA vs a plain non-GVA run.

    For every ``(B, T, H_qk, HV)`` config this function runs **cuLA only**
    with two different input preparations — both produce tensors of identical
    shape ``(1, B*T, HV, D)`` entering the kernel:

    * **baseline** – standard non-GVA: H = HV unique q/k heads, prepared via
      :func:`prepare_safe_gate_inputs` with ``num_v_heads=HV``.
    * **GVA**      – grouped q/k: H < HV heads expanded to HV via
      ``repeat_interleave``, prepared via :func:`prepare_gva_inputs`.

    Because the kernel receives identically-shaped tensors in both cases, the
    extra work that GVA adds is *only* the ``repeat_interleave`` call done in
    Python before the kernel is launched.  A near-zero ``overhead%`` column in
    the report confirms that the GVA feature introduces no measurable kernel
    latency regression.

    Note: FLA is intentionally excluded; the comparison is purely cuLA vs cuLA.
    """
    print("\n" + "=" * 100)
    print(f" GVA Overhead Benchmark: cuLA GVA vs cuLA non-GVA  (same kernel shape, {_SM_TAG})")
    print("=" * 100)
    results = []

    for cfg in configs:
        B, T, H_qk, HV = _normalize_fixed_config(cfg)
        assert HV > H_qk, f"GVA overhead bench requires HV > H, got H={H_qk} HV={HV}"
        gva_ratio = HV // H_qk
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        # ── baseline: non-GVA with HV heads (H == HV) ────────────────────────
        inp_base = prepare_safe_gate_inputs(
            B, T, HV, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
            num_v_heads=HV,
        )
        common_base = dict(
            q=inp_base["q"], k=inp_base["k"], v=inp_base["v"],
            g=inp_base["g"], beta=inp_base["beta"],
            scale=inp_base["scale"], A_log=inp_base["A_log"],
            dt_bias=inp_base["dt_bias"], init_state=inp_base["init_state"],
            cu_seqlens=cu_seqlens, lower_bound=inp_base["lower_bound"],
        )

        # ── GVA: H_qk heads expanded to HV ───────────────────────────────────
        inp_gva = prepare_gva_inputs(
            B, T, H_qk, HV, D, device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
        )
        common_gva = dict(
            q=inp_gva["q"], k=inp_gva["k"], v=inp_gva["v"],
            g=inp_gva["g"], beta=inp_gva["beta"],
            scale=inp_gva["scale"], A_log=inp_gva["A_log"],
            dt_bias=inp_gva["dt_bias"], init_state=inp_gva["init_state"],
            cu_seqlens=cu_seqlens, lower_bound=inp_gva["lower_bound"],
        )

        # ── performance ───────────────────────────────────────────────────────
        ms_base = time_kernel(lambda: run_cula(**common_base))
        ms_gva  = time_kernel(lambda: run_cula(**common_gva))
        overhead_pct = (ms_gva - ms_base) / ms_base * 100.0 if ms_base > 0 else 0.0

        results.append(
            {
                "B": B,
                "T": T,
                "H": H_qk,
                "HV": HV,
                "gva_ratio": gva_ratio,
                "ms_base": ms_base,
                "ms_gva": ms_gva,
                "overhead_pct": overhead_pct,
            }
        )

        del inp_base, inp_gva
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results, gva_fixed_results=None, gva_varlen_results=None, overhead_results=None):
    sep = "=" * 120
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: cula_kda_fused_fwd (fully-fused)")
    print(f"                  cuLA {_SM_TAG} fully-fused vs FLA Triton")
    print(f"                  D={D}  dtype=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    print(f"                  GVA rows are those with HV > H (H, HV shown per row).")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                  Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 110}")
        print(
            f"  {'B':>3s}  {'T':>6s}  {'H':>3s}  {'HV':>3s}  {'GVA':>4s}  │  "
            f"{'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 110}")
        for r in fixed_results:
            gva_tag = "yes" if r["HV"] > r["H"] else "no"
            print(
                f"  {r['B']:3d}  {r['T']:6d}  {r['H']:3d}  {r['HV']:3d}  {gva_tag:>4s}  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 110}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 120}")
        print(
            f"  {'Config':>45s}  {'H':>3s}  {'HV':>3s}  {'GVA':>4s}  │  "
            f"{'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 120}")
        for r in varlen_results:
            gva_tag = "yes" if r["HV"] > r["H"] else "no"
            print(
                f"  {r['tag']:>45s}  {r['H']:3d}  {r['HV']:3d}  {gva_tag:>4s}  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 120}")

    if gva_fixed_results:
        print("\n  [GVA Fixed-Length]  (data prepared via prepare_gva_inputs)")
        print(f"  {'─' * 116}")
        print(
            f"  {'B':>3s}  {'T':>6s}  {'H':>3s}  {'HV':>3s}  {'ratio':>5s}  │  "
            f"{'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 116}")
        for r in gva_fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  {r['H']:3d}  {r['HV']:3d}  {r['gva_ratio']:4d}x  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 116}")

    if gva_varlen_results:
        print("\n  [GVA Varlen]  (data prepared via prepare_gva_inputs)")
        print(f"  {'─' * 126}")
        print(
            f"  {'Config':>45s}  {'H':>3s}  {'HV':>3s}  {'ratio':>5s}  │  "
            f"{'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 126}")
        for r in gva_varlen_results:
            print(
                f"  {r['tag']:>45s}  {r['H']:3d}  {r['HV']:3d}  {r['gva_ratio']:4d}x  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 126}")

    if overhead_results:
        print("\n  [GVA Overhead]  cuLA GVA vs cuLA non-GVA — same kernel shape, same HV heads")
        print("  (near-zero overhead% proves GVA adds no kernel latency)")
        print(f"  {'─' * 96}")
        print(
            f"  {'B':>3s}  {'T':>6s}  {'H':>3s}  {'HV':>3s}  {'ratio':>5s}  │  "
            f"{'base(ms)':>10s}  {'gva(ms)':>10s}  {'overhead%':>10s}"
        )
        print(f"  {'─' * 96}")
        for r in overhead_results:
            flag = "  ✓" if abs(r["overhead_pct"]) < 3.0 else "  !"
            print(
                f"  {r['B']:3d}  {r['T']:6d}  {r['H']:3d}  {r['HV']:3d}  {r['gva_ratio']:4d}x  │  "
                f"{r['ms_base']:10.4f}  {r['ms_gva']:10.4f}  {r['overhead_pct']:+9.2f}%{flag}"
            )
        print(f"  {'─' * 96}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_kda_fused_fwd: cuLA fully-fused KDA forward vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["fixed", "varlen", "gva", "overhead", "both", "all"],
        help=(
            "Which benchmark mode to run (default: all). "
            "'all' = fixed + varlen + gva + overhead.  "
            "'both' = fixed + varlen only (legacy alias).  "
            "'gva' runs the dedicated GVA benchmark (fixed + varlen, "
            "data prepared via prepare_gva_inputs with multiple GVA ratios).  "
            "'overhead' compares cuLA-GVA vs cuLA-non-GVA at the same total "
            "head count to prove GVA adds no kernel latency regression."
        ),
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="Sanitizer mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--init_state",
        action="store_true",
        help="Use non-zero initial state (default: False)",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE, HAS_INIT_STATE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")
    if args.init_state:
        HAS_INIT_STATE = True
        print("[init_state] using non-zero initial state")

    print(
        f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}  →  using {cula_kda_fused_fwd.__module__}.{cula_kda_fused_fwd.__name__}"
    )

    # ------------------------------------------------------------------
    # Fixed-length configs: (B, T) → H_qk=HV=H (no GVA); (B, T, H_qk, HV)
    # activates GVA when HV > H. The two forms can be freely mixed.
    # ------------------------------------------------------------------
    fixed_configs = [
        # Non-GVA (H_qk == HV == H):
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (2, 512),
        (2, 1024),
        (2, 4096),
        (2, 8192),
        (2, 16384),
        # GVA (HV > H, same D=128):
        (1, 1024, 16, 64),
        (1, 4096, 16, 64),
        (1, 8192, 16, 64),
        (1, 4096, 32, 64),
        (1, 8192, 32, 64),
        (2, 4096, 16, 64),
        (2, 8192, 16, 64),
    ]

    # Varlen configs: 3-tuples (seq_lens, total_len, dist) default to no GVA;
    # extend to 5-tuples (..., H_qk, HV) to activate GVA on varlen workloads.
    varlen_configs_base = build_varlen_configs(
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )
    # A small GVA subset reuses the non-GVA varlen shapes with (H_qk=16, HV=64).
    gva_varlen_mixed = [(seq_lens, T, dist, 16, 64) for (seq_lens, T, dist) in varlen_configs_base if T <= 8192]
    varlen_configs = list(varlen_configs_base) + gva_varlen_mixed

    # ------------------------------------------------------------------
    # Dedicated GVA configs (multiple GVA ratios, uses prepare_gva_inputs)
    # ------------------------------------------------------------------
    gva_fixed_configs = build_gva_fixed_configs(
        batch_sizes=(1, 2),
        seq_lens=(1024, 4096, 8192),
        h_hv_pairs=((8, 32), (16, 64), (32, 64), (16, 128)),
    )
    gva_varlen_configs = build_gva_varlen_configs(
        h_hv_pairs=((16, 64), (32, 64)),
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192),
        dists=("uniform", "random", "skewed"),
    )

    # Overhead configs: (B, T, H_qk, HV) — HV > H required.
    # For each row the benchmark runs cuLA twice:
    #   baseline → non-GVA with HV heads (H == HV)
    #   gva      → GVA with H_qk heads expanded to HV
    # Both kernel inputs have shape (1, B*T, HV, D), so overhead% ≈ 0 proves
    # that GVA adds no kernel latency regression.
    overhead_configs = [
        (1, 1024,  16,  64),   # 4x GVA ratio
        (1, 4096,  16,  64),
        (1, 8192,  16,  64),
        (1, 16384, 16,  64),
        (1, 4096,  32,  64),   # 2x GVA ratio
        (1, 8192,  32,  64),
        (1, 4096,   8,  64),   # 8x GVA ratio
        (1, 8192,   8,  64),
        (2, 4096,  16,  64),
        (2, 8192,  16,  64),
    ]

    fixed_res, varlen_res, gva_fixed_res, gva_varlen_res, overhead_res = [], [], [], [], []

    if args.mode in ("fixed", "both", "all"):
        fixed_res = bench_fixed(fixed_configs)

    if args.mode in ("varlen", "both", "all"):
        varlen_res = bench_varlen(varlen_configs)

    if args.mode in ("gva", "all"):
        gva_fixed_res = bench_gva_fixed(gva_fixed_configs)
        gva_varlen_res = bench_gva_varlen(gva_varlen_configs)

    if args.mode in ("overhead", "all"):
        overhead_res = bench_gva_overhead(overhead_configs)

    print_report(fixed_res, varlen_res, gva_fixed_res, gva_varlen_res, overhead_res)

    return fixed_res, varlen_res, gva_fixed_res, gva_varlen_res, overhead_res


if __name__ == "__main__":
    main()
