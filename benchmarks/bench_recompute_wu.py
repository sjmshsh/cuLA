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

import argparse
import os
import pathlib
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra
from fla.ops.kda.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd

import cula.cudac as cula_cuda
from benchmarks.utils import SEED, exclusive_cumsum, generate_random_seq_lens, prepare_intra_inputs
from cula.kda.chunk_intra import chunk_kda_fwd_intra as cula_chunk_kda_fwd_intra

# Constant params
B, H, D = 2, 64, 128
BT = 64  # chunk size

# Varlen benchmark params
NUM_SEQS = 8
TOTAL_LEN = 8192
MIN_SEQ_LEN = 63
VARIANCE = 1.0

DISABLE_RECOMPUTE = False  # Whether to disable recompute (compute QG in forward)
GROUP_SIZE = 1  # GVA group size: HV = GROUP_SIZE * H. 1 means no GVA.


def accuracy_stats(a, b):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    a, b = a.float(), b.float()
    diff = a - b
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.abs().max().item()
    denom = b.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.abs().mean().item()
    return rmse, rel_max, mean_diff


def prepare_recompute_wu_inputs(B, T, H, D, device, cu_seqlens=None, chunk_size=BT):
    """Prepare inputs for recompute_w_u benchmarking.

    Runs chunk_kda_fwd_intra (FLA) to produce Akk, then returns
    all tensors needed for recompute_w_u_fwd / recompute_w_u_cuda.
    """
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
        B, T, H, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
    )

    # Run FLA chunk_kda_fwd_intra to get Akk (shared input for both impls)
    _, _, _, _, Aqk, Akk = fla_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=False,
    )

    return q, k, v, g, beta, Akk, cu_seqlens, chunk_indices


def run_fla_recompute_wu(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, disable_recompute):
    """Run FLA recompute_w_u_fwd."""
    return fla_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def run_cula_recompute_wu(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, chunk_size, disable_recompute):
    """Run cuLA recompute_w_u_cuda (MHA: all tensors share the same head dim)."""
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if disable_recompute else None
    kg = torch.empty_like(k) if gk is not None else None

    cula_cuda.recompute_w_u_cuda(
        k, v, beta, Akk, gk, cu_seqlens, chunk_indices, w, u, kg, chunk_size, q if disable_recompute else None, qg
    )
    return w, u, qg, kg


# ==============================================================================
# GVA helpers
# ==============================================================================

def prepare_recompute_wu_inputs_gva(B, T, HQK, HV, D, device, cu_seqlens=None, chunk_size=BT):
    """Prepare GVA inputs for recompute_w_u benchmarking.

    Produces Akk via cuLA's GVA-aware chunk_kda_fwd_intra so the tensor lives in
    HV-head space (shape [B, T, HV, BT]).  Both FLA (k replicated to HV) and cuLA
    (k compact in HQK) receive the same Akk.
    """
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
        B, T, HQK, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size, num_v_heads=HV
    )

    # Use cuLA GVA intra to produce Akk in HV space.
    _, _, _, _, _, Akk = cula_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=False,
    )

    return q, k, v, g, beta, Akk, cu_seqlens, chunk_indices


def run_fla_recompute_wu_gva(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, disable_recompute, group_size):
    """FLA reference for GVA recompute_w_u.

    FLA does not natively support GVA, so k and q are replicated to HV heads via
    repeat_interleave before the call — mirroring the strategy in bench_kda_chunk_intra.py.
    """
    k_hv = k.repeat_interleave(group_size, dim=2).contiguous()
    q_hv = q.repeat_interleave(group_size, dim=2).contiguous()
    return fla_recompute_w_u_fwd(
        k=k_hv,
        v=v,
        beta=beta,
        A=Akk,
        q=q_hv if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def run_cula_recompute_wu_gva(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, chunk_size, disable_recompute):
    """Run cuLA recompute_w_u_cuda with GVA layout.

    k/q live in HQK head space; v/gk/beta/Akk/w/u/kg/qg all live in HV head space.
    """
    B_flat, T, HV, Dv = v.shape
    w = torch.empty(B_flat, T, HV, Dv, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = torch.empty(B_flat, T, HV, Dv, device=q.device, dtype=q.dtype) if disable_recompute else None
    kg = torch.empty(B_flat, T, HV, Dv, device=k.device, dtype=k.dtype) if gk is not None else None

    cula_cuda.recompute_w_u_cuda(
        k, v, beta, Akk, gk, cu_seqlens, chunk_indices, w, u, kg, chunk_size, q if disable_recompute else None, qg
    )
    return w, u, qg, kg


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
def benchmark_recompute_wu_uniform():
    device = torch.device("cuda")
    chunk_size = BT
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 90)
    print(
        f"  Uniform-Length RecomputeWU Benchmark: cuLA vs FLA Triton  B={B} H={H} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 90)
    print(
        f"{'B':>4} {'T':>7} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 90)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
            B, T, H, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy: run once and compare
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        # Compare w, u, qg, kg
        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = accuracy_stats(t_fla, t_cula)
        # Use max across all outputs for display
        rmse = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: run_fla_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE),
        )
        ms_cula = triton.testing.do_bench(
            lambda: run_cula_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 90)


# ==============================================================================
# Varlen benchmark
# ==============================================================================
def benchmark_recompute_wu_varlen():
    device = torch.device("cuda")
    chunk_size = BT
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 100)
    print(
        f"  Varlen RecomputeWU Benchmark: cuLA vs FLA Triton  NUM_SEQS={NUM_SEQS} H={H} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 100)
    print(
        f"{'total_len':>10} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 100)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
            1, T, H, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        # Compare w, u, qg, kg
        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = accuracy_stats(t_fla, t_cula)
        # Use max across all outputs for display
        rmse = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: run_fla_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE),
        )
        ms_cula = triton.testing.do_bench(
            lambda: run_cula_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{total_len:>10} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 100)


# ==============================================================================
# GVA uniform seqlen benchmark
# ==============================================================================
def benchmark_recompute_wu_gva_uniform(group_size: int):
    """Benchmark GVA (HV > HQK) recompute_w_u: cuLA vs FLA Triton (k replicated to HV).

    FLA does not natively support GVA, so the reference replicates k/q along the
    head axis to HV before calling recompute_w_u_fwd.
    """
    device = torch.device("cuda")
    chunk_size = BT
    HQK = H
    HV = HQK * group_size
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 100)
    print(
        f"  GVA Uniform RecomputeWU Benchmark: cuLA vs FLA Triton  "
        f"B={B} HQK={HQK} HV={HV} (group_size={group_size}) D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 100)
    print(
        f"{'B':>4} {'T':>7} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 100)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs_gva(
            B, T, HQK, HV, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy: run once and compare
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu_gva(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE, group_size
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu_gva(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = accuracy_stats(t_fla, t_cula)
        rmse = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: run_fla_recompute_wu_gva(
                k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE, group_size
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: run_cula_recompute_wu_gva(
                k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 100)


# ==============================================================================
# GVA varlen benchmark
# ==============================================================================
def benchmark_recompute_wu_gva_varlen(group_size: int):
    """Varlen GVA benchmark for recompute_w_u: cuLA vs FLA Triton (k replicated to HV)."""
    device = torch.device("cuda")
    chunk_size = BT
    HQK = H
    HV = HQK * group_size
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 110)
    print(
        f"  GVA Varlen RecomputeWU Benchmark: cuLA vs FLA Triton  "
        f"NUM_SEQS={NUM_SEQS} HQK={HQK} HV={HV} (group_size={group_size}) D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 110)
    print(
        f"{'total_len':>10} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 110)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs_gva(
            1, T, HQK, HV, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu_gva(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE, group_size
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu_gva(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = accuracy_stats(t_fla, t_cula)
        rmse = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: run_fla_recompute_wu_gva(
                k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE, group_size
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: run_cula_recompute_wu_gva(
                k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{total_len:>10} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bench_recompute_wu: cuLA vs FLA Triton for recompute_w_u")
    parser.add_argument(
        "--disable_recompute",
        action="store_true",
        help="Disable recompute in both FLA and cuLA (pre-compute QG)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=1,
        help="GVA group size: HV = group_size * H. 1 (default) runs the non-GVA benchmark. "
             "Values > 1 run GVA benchmarks comparing cuLA (k in HQK space) vs FLA (k replicated to HV).",
    )
    args = parser.parse_args()

    if args.disable_recompute:
        DISABLE_RECOMPUTE = True
        print("[Disable recompute] pre-compute QG in forward")

    GROUP_SIZE = args.group_size

    if GROUP_SIZE == 1:
        benchmark_recompute_wu_uniform()
        benchmark_recompute_wu_varlen()
    else:
        benchmark_recompute_wu_gva_uniform(GROUP_SIZE)
        benchmark_recompute_wu_gva_varlen(GROUP_SIZE)
