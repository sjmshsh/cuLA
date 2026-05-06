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

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Adapted from flash-linear-attention: https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_kda.py


import pytest
import torch
import torch.nn.functional as F
from fla.ops import chunk_kda as fla_chunk_kda
from fla.ops.kda.gate import naive_kda_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close, device

from cula.utils import get_kda_fused_fwd

pytestmark = pytest.mark.sm90_only

# A single, less-common seed for the whole module; data generation is then made deterministic
# but visibly different from the upstream KDA test suite (which uses 42 throughout).
_SEED = 0xC4FA  # 50,426


def _make_qk(B: int, T: int, H: int, D: int, dtype: torch.dtype, *, generator: torch.Generator) -> torch.Tensor:
    """Q/K live close to a unit sphere in real workloads (l2-norm is applied
    inside the model). We sample from a normal and rescale, which exercises a
    wider numerical range than the original `rand(0,1)` initialization."""
    x = torch.randn(B, T, H, D, dtype=torch.float32, generator=generator) * 0.5
    return x.to(dtype)


def _make_v(B: int, T: int, HV: int, D: int, dtype: torch.dtype, *, generator: torch.Generator) -> torch.Tensor:
    """V is an activation; sample N(0, 0.5^2) which is closer to typical post-norm hidden states
    than uniform [0, 1]."""
    return (torch.randn(B, T, HV, D, dtype=torch.float32, generator=generator) * 0.5).to(dtype)


def _make_g(
    B: int, T: int, HV: int, D: int, *, dtype: torch.dtype, mask_p: float, gate_logit_normalizer: float,
    use_gate_in_kernel: bool, generator: torch.Generator,
) -> torch.Tensor:
    """Gate (forget) tensor in log space."""
    g = torch.randn(B, T, HV, D, dtype=torch.float if not use_gate_in_kernel else dtype, generator=generator)
    if use_gate_in_kernel:
        return g
    g = F.logsigmoid(g) / gate_logit_normalizer
    drop_mask = torch.rand(g.shape, dtype=g.dtype, generator=generator) > mask_p
    return g * drop_mask


def _make_beta(B: int, T: int, HV: int, beta_dtype: torch.dtype, *, generator: torch.Generator) -> torch.Tensor:
    """β is gating scalar in [0, 1]; we sample sigmoid(N(0,1))."""
    return torch.randn(B, T, HV, dtype=torch.float32, generator=generator).sigmoid().to(beta_dtype)


def _make_h0(B: int, HV: int, D: int, *, generator: torch.Generator) -> torch.Tensor:
    """Initial recurrent state. We use a small magnitude so that the recurrence
    does not blow up across long sequences."""
    return torch.randn(B, HV, D, D, dtype=torch.float32, generator=generator) * 0.05


# =============================================================================
# Fixed-length test
# =============================================================================
#
# Cases are grouped by GVA "heads_per_group = HV / H":
#   - group=1  → degenerates to plain MHA (sanity baseline)
#   - group=2/4/8/16 → real GVA paths
# We deliberately mix small and large T (incl. non-multiple-of-chunk-size 63/65/1500),
# different B, and toggles for use_qk_l2norm_in_kernel / use_gate_in_kernel /
# use_initial_state / output_final_state to exercise as many code paths as
# possible without blowing up the matrix.
# =============================================================================
_FIXED_CASES = [
    # ---------------- group = 1 (MHA baseline) ----------------
    (1, 63,   1, 1, 128, 1,    0,   False, False, True, True,  True,  False, torch.bfloat16),
    (2, 500,  3, 3, 128, 1,    0,   False, False, True, True,  True,  False, torch.bfloat16),
    (2, 1000, 3, 3, 128, 1,    0.5, False, False, True, True,  True,  False, torch.bfloat16),
    (3, 1024, 4, 4, 128, 0.1,  0,   False, False, True, True,  True,  False, torch.bfloat16),
    (4, 1024, 4, 4, 128, 1,    0,   True,  False, True, True,  True,  False, torch.bfloat16),  # qk_l2norm=True
    (2, 1500, 4, 4, 128, 10,   0,   False, True,  True, True,  True,  False, torch.bfloat16),  # gate_in_kernel
    (4, 2048, 8, 8, 128, 1,    0,   False, True,  True, True,  True,  False, torch.bfloat16),

    # ---------------- group = 2 ----------------
    (1, 64,   1, 2, 128, 1,    0,   False, False, True, True,  True,  False, torch.bfloat16),
    (2, 512,  2, 4, 128, 1,    0,   False, False, True, True,  True,  False, torch.bfloat16),
    (3, 1024, 4, 8, 128, 0.1,  0,   False, False, True, True,  True,  False, torch.bfloat16),
    (1, 65,   2, 4, 128, 1,    0,   False, False, True, True,  True,  True,  torch.bfloat16),  # deterministic
    (2, 65,   2, 4, 128, 1,    0,   False, False, True, False, False, False, torch.bfloat16),  # init=F outstate=F
    (4, 768,  4, 8, 128, 1,    0.3, True,  False, True, True,  True,  False, torch.bfloat16),  # qk_l2norm + dropout

    # ---------------- group = 4 ----------------
    (1, 65,   1, 4, 128, 1,    0,   False, False, True, False, True,  False, torch.bfloat16),  # init=F
    (2, 1024, 2, 8, 128, 1,    0,   False, True,  True, True,  True,  False, torch.bfloat16),  # gate_in_kernel
    (1, 256,  2, 8, 128, 1,    0,   False, False, True, True,  False, False, torch.bfloat16),  # outstate=False (skip path)
    (2, 4096, 2, 8, 128, 1,    0,   False, True,  True, True,  True,  False, torch.bfloat16),  # long T

    # ---------------- group = 8 ----------------
    (1, 2048, 1, 8, 128, 1,    0,   False, True,  True, True,  True,  False, torch.bfloat16),
    (2, 1024, 2, 16, 128, 1,   0,   False, False, True, True,  True,  False, torch.bfloat16),
    (1, 512,  1, 8, 128, 1,    0,   False, False, True, True,  False, False, torch.bfloat16),  # outstate=False

    # ---------------- group = 16 ----------------
    (1, 512,  1, 16, 128, 1,   0,   False, False, True, True,  True,  False, torch.bfloat16),
    (1, 256,  2, 32, 128, 1,   0,   False, False, True, True,  True,  False, torch.bfloat16),
]


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=["beta_fp32", "beta_bf16"])
@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "HV",
        "D",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "safe_gate",
        "use_initial_state",
        "output_final_state",
        "deterministic",
        "dtype",
    ),
    [
        pytest.param(
            *case,
            id=("B{}-T{}-H{}-HV{}-D{}-gln{}-mask_p{}-l2norm{}-gate{}-safe_gate{}-init{}-outstate{}-deterministic{}-{}").format(
                *case
            ),
        )
        for case in _FIXED_CASES
    ],
)
def test_safe_gate_chunk(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    use_initial_state: bool,
    output_final_state: bool,
    deterministic: bool,
    dtype: torch.dtype,
    beta_dtype: torch.dtype,
):
    from fla.ops.kda.gate import naive_kda_lowerbound_gate

    cula_kda_fused_fwd = get_kda_fused_fwd(device)

    # Use a torch.Generator so each tensor draws from an independent stream while
    # the overall test is still deterministic for a fixed _SEED.
    gen = torch.Generator(device="cpu").manual_seed(_SEED)

    q = _make_qk(B, T, H, D, dtype, generator=gen)
    k = _make_qk(B, T, H, D, dtype, generator=gen)
    v = _make_v(B, T, HV, D, dtype, generator=gen)
    g = _make_g(
        B, T, HV, D,
        dtype=dtype,
        mask_p=mask_p,
        gate_logit_normalizer=gate_logit_normalizer,
        use_gate_in_kernel=use_gate_in_kernel,
        generator=gen,
    )
    beta = _make_beta(B, T, HV, beta_dtype, generator=gen)
    h0 = _make_h0(B, HV, D, generator=gen)

    if deterministic:
        # Hand-crafted inputs that produce closed-form outputs; useful as a stronger
        # correctness anchor than just RMSE-against-reference.
        assert H == 2 and HV == 4 and not use_gate_in_kernel
        q = torch.zeros(B, T, H, D, dtype=dtype)
        k = torch.zeros(B, T, H, D, dtype=dtype)
        v = torch.zeros(B, T, HV, D, dtype=dtype)
        g = torch.zeros(B, T, HV, D, dtype=torch.float)
        q[:, :, 0, 0] = 1
        q[:, :, 1, 1] = 1
        k[:, :, 0, 0] = 1
        k[:, :, 1, 0] = 1
        for i in range(HV):
            v[:, :, i] = i + 1
        beta = torch.ones(B, T, HV, dtype=beta_dtype)
        h0 = torch.zeros(B, HV, D, D, dtype=torch.float32)

    A_log = dt_bias = None
    if use_gate_in_kernel:
        A_log = torch.randn(HV, dtype=torch.float, generator=gen)
        dt_bias = torch.randn(HV * D, dtype=torch.float, generator=gen)

    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
        naive_kda_gate_fn = naive_kda_lowerbound_gate
    else:
        lower_bound = None
        naive_kda_gate_fn = naive_kda_gate

    # NOTE: for inference scenarios, we only use the transposed state layout for better
    # decoding performance.
    h0_vk = h0.transpose(-1, -2).contiguous()
    if use_gate_in_kernel:
        A_log, dt_bias = (x.to(device).requires_grad_(False) for x in (A_log, dt_bias))
    q, k, v, g, beta, h0, h0_vk = (
        x.to(device).requires_grad_(False) for x in (q, k, v, g, beta, h0, h0_vk)
    )
    initial_state = h0.clone() if use_initial_state else None
    initial_state_vk = h0_vk.clone() if use_initial_state else None

    # GVA reference: replicate Q/K across each group so the naive/MHA reference can
    # consume them as if HV-many heads were present.
    heads_per_group = HV // H
    q_ref = q.repeat_interleave(heads_per_group, dim=2)
    k_ref = k.repeat_interleave(heads_per_group, dim=2)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q_ref.clone(), p=2, dim=-1),
        k=F.normalize(k_ref.clone(), p=2, dim=-1),
        v=v.clone(),
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        initial_state=initial_state,
        output_final_state=output_final_state,
    )

    ref_fla, ref_ht_fla = fla_chunk_kda(
        q=F.normalize(q_ref.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q_ref.clone(),
        k=F.normalize(k_ref.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k_ref.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    ref_fla_trans, ref_ht_fla_trans = fla_chunk_kda(
        q=F.normalize(q_ref.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q_ref.clone(),
        k=F.normalize(k_ref.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k_ref.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=initial_state_vk,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )

    tri, tri_ht = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=initial_state_vk,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    assert_close("o", ref, tri, 0.005)
    assert_close("o", ref_fla, tri, 0.005)
    assert_close("o", ref_fla_trans, tri, 0.005)
    if output_final_state:
        assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
        assert_close("ht", ref_ht_fla, tri_ht.transpose(-1, -2), 0.005)
        assert_close("ht", ref_ht_fla_trans, tri_ht, 0.005)
    else:
        assert ref_ht is None
        assert ref_ht_fla is None
        assert ref_ht_fla_trans is None
        assert tri_ht is None, "wrapper must surface None when output_final_state=False"


# =============================================================================
# Variable-length (cu_seqlens) test
# =============================================================================
_VARLEN_CASES = [
    # ---------------- group = 1 (MHA baseline) ----------------
    (4, 4, 128, 0.1, [0, 15],                                              torch.bfloat16, True, True, True),
    (4, 4, 128, 0.9, [0, 256, 500, 1000],                                  torch.bfloat16, True, True, True),
    (4, 4, 128, 0.5, [0, 256, 500, 1000],                                  torch.bfloat16, True, True, True),
    (4, 4, 128, 0,   [0, 15, 100, 300, 1200, 2000],                        torch.bfloat16, True, True, True),
    (4, 4, 128, 0,   [0, 100, 300, 1200, 3000, 4096],                      torch.bfloat16, True, True, True),

    # ---------------- group = 2 ----------------
    (2, 4, 128, 0,   [0, 63, 130],                                         torch.bfloat16, True, True, True),
    (2, 4, 128, 0,   [0, 17, 64, 65, 130],                                 torch.bfloat16, True, False, True),  # init=False
    (3, 6, 128, 0.2, [0, 257, 800, 1500],                                  torch.bfloat16, True, True, True),

    # ---------------- group = 4 ----------------
    (1, 4, 128, 0,   [0, 1],                                               torch.bfloat16, True, True, True),
    (1, 4, 128, 0,   [0, 63, 64, 65],                                      torch.bfloat16, True, True, True),
    (4, 16, 128, 0.5, [0, 15, 100, 300],                                   torch.bfloat16, True, True, False),  # outstate=False (skip path)
    (2, 8, 128, 0,   [0, 256, 1024, 4096],                                 torch.bfloat16, True, True, True),

    # ---------------- group = 8 ----------------
    (1, 8, 128, 0,   [0, 65, 200, 1024],                                   torch.bfloat16, True, True, True),
    (1, 8, 128, 0,   [0, 1024, 2048],                                      torch.bfloat16, True, False, False),  # init=F outstate=F

    # ---------------- group = 16 ----------------
    (1, 16, 128, 0,  [0, 257, 1024],                                       torch.bfloat16, True, True, True),

    # ---------------- group = 1, varlen at scale (simulated traces) ----------------
    (
        32, 32, 128, 0,
        [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192],
        torch.bfloat16, True, True, True,
    ),
    (
        32, 32, 128, 0,
        [0, 652, 1255, 1600, 2083, 2345, 2756, 3172, 3767, 4096, 4891, 5236, 5543, 6255, 6480, 6947, 7616, 8192],
        torch.bfloat16, True, True, True,
    ),
    (
        32, 32, 128, 0,
        [0, 315, 973, 1283, 2162, 2459, 2678, 2998, 3781, 4096, 4503, 5459, 6318, 6669, 6979, 7583, 8192],
        torch.bfloat16, True, True, True,
    ),
    (
        32, 32, 128, 0,
        [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192],
        torch.bfloat16, True, True, True,
    ),

    # ---------------- group = 4, varlen at scale ----------------
    (
        8, 32, 128, 0,
        [0, 255, 1024, 2049, 3072, 4097, 5120, 6144, 7168, 8192],
        torch.bfloat16, True, True, True,
    ),
]


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=["beta_fp32", "beta_bf16"])
@pytest.mark.parametrize(
    ("H", "HV", "D", "mask_p", "cu_seqlens", "dtype", "safe_gate", "use_initial_state", "output_final_state"),
    [
        pytest.param(
            *case,
            id="H{}-HV{}-D{}-mask_p{}-cu_seqlens{}-{}-safe_gate{}-init{}-outstate{}".format(*case),
        )
        for case in _VARLEN_CASES
    ],
)
def test_safe_gate_chunk_varlen(
    H: int,
    HV: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    safe_gate: bool,
    use_initial_state: bool,
    output_final_state: bool,
    beta_dtype: torch.dtype,
):
    cula_kda_fused_fwd = get_kda_fused_fwd(device)

    gen = torch.Generator(device="cpu").manual_seed(_SEED)

    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = (torch.randn((1, T, H, D), dtype=torch.float32, generator=gen) * 0.5).to(dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, generator=gen), p=2, dim=-1).to(dtype)
    v = (torch.randn((1, T, HV, D), dtype=torch.float32, generator=gen) * 0.5).to(dtype)
    g = F.logsigmoid(torch.randn(1, T, HV, D, dtype=torch.float, generator=gen))
    drop_mask = torch.rand(g.shape, dtype=g.dtype, generator=gen) > mask_p
    g = g * drop_mask + (~drop_mask) * (-1000)
    if safe_gate:
        g = g.clamp(-5, 0)

    beta = _make_beta(1, T, HV, beta_dtype, generator=gen)
    h0 = _make_h0(N, HV, D, generator=gen)
    # NOTE: for inference scenarios, we only use the transposed state layout for better
    # decoding performance.
    h0_vk = h0.transpose(-1, -2).contiguous()

    q, k, v, g, beta, h0, h0_vk = (
        x.to(device).requires_grad_(False) for x in (q, k, v, g, beta, h0, h0_vk)
    )
    initial_state = h0.clone() if use_initial_state else None
    initial_state_vk = h0_vk.clone() if use_initial_state else None
    heads_per_group = HV // H
    q_ref = q.repeat_interleave(heads_per_group, dim=2)
    k_ref = k.repeat_interleave(heads_per_group, dim=2)

    tri, tri_ht = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state_vk,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
    )

    ref_fla, ref_ht_fla = fla_chunk_kda(
        q=F.normalize(q_ref.clone(), p=2, dim=-1),
        k=k_ref.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
    )

    ref_fla_trans, ref_ht_fla_trans = fla_chunk_kda(
        q=F.normalize(q_ref.clone(), p=2, dim=-1),
        k=k_ref.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state_vk,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        transpose_state_layout=True,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q_ref[:, cu_seqlens[i] : cu_seqlens[i + 1]], p=2, dim=-1),
            k=k_ref[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i] if use_initial_state else None,
            output_final_state=output_final_state,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0) if output_final_state else None

    assert_close("o", ref, tri, 0.005)
    assert_close("o", ref_fla, tri, 0.005)
    assert_close("o", ref_fla_trans, tri, 0.005)
    if output_final_state:
        assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
        assert_close("ht", ref_ht_fla, tri_ht.transpose(-1, -2), 0.005)
        assert_close("ht", ref_ht_fla_trans, tri_ht, 0.005)
    else:
        assert ref_ht is None
        assert ref_ht_fla is None
        assert ref_ht_fla_trans is None
        assert tri_ht is None, "wrapper must surface None when output_final_state=False"


# =============================================================================
# Regression: output_final_state=False must skip the state buffer and still
# produce correct outputs on a GVA configuration.
# =============================================================================
@pytest.mark.parametrize(
    ("B", "T", "H", "HV"),
    [
        (1, 256, 1, 4),
        (2, 1024, 2, 8),
        (1, 512, 1, 16),
    ],
    ids=["small-g4", "medium-g4", "wide-g16"],
)
def test_output_final_state_skip_under_gva(B: int, T: int, H: int, HV: int):
    """Sanity check for the C++ side optimization: when output_final_state=False
    we must (a) still get correct outputs and (b) get None back as the second
    return value (no leaked tensor)."""
    cula_kda_fused_fwd = get_kda_fused_fwd(device)
    gen = torch.Generator(device="cpu").manual_seed(_SEED)
    D = 128
    dtype = torch.bfloat16

    q = _make_qk(B, T, H, D, dtype, generator=gen).to(device)
    k = _make_qk(B, T, H, D, dtype, generator=gen).to(device)
    v = _make_v(B, T, HV, D, dtype, generator=gen).to(device)
    g = _make_g(
        B, T, HV, D,
        dtype=dtype, mask_p=0.0, gate_logit_normalizer=1.0, use_gate_in_kernel=False,
        generator=gen,
    ).clamp(-5, 0).to(device)
    beta = _make_beta(B, T, HV, torch.float32, generator=gen).to(device)

    # Run twice: once with output_final_state=True (reference path), once with =False.
    o_full, ht_full = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        output_final_state=True, safe_gate=True, lower_bound=-5.0,
    )
    o_skip, ht_skip = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        output_final_state=False, safe_gate=True, lower_bound=-5.0,
    )

    assert ht_full is not None, "with output_final_state=True we must get a state tensor"
    assert ht_skip is None, "with output_final_state=False the wrapper must surface None"
    assert_close("o", o_full, o_skip, 0.005)


# =============================================================================
# API contract: GVA shape validation.
# =============================================================================
@pytest.mark.parametrize(
    ("H", "HV", "expect_error"),
    [
        (2, 4, False),  # valid: HV / H = 2
        (2, 8, False),  # valid: HV / H = 4
        (4, 4, False),  # valid: HV == H (MHA)
        (3, 4, True),   # invalid: 4 % 3 != 0
        (3, 7, True),   # invalid: 7 % 3 != 0
    ],
)
def test_gva_shape_validation(H: int, HV: int, expect_error: bool):
    """The Python wrapper must reject HV that is not a positive multiple of H
    *before* anything reaches the kernel."""
    cula_kda_fused_fwd = get_kda_fused_fwd(device)
    B, T, D = 1, 64, 128
    dtype = torch.bfloat16
    gen = torch.Generator(device="cpu").manual_seed(_SEED)

    q = _make_qk(B, T, H, D, dtype, generator=gen).to(device)
    k = _make_qk(B, T, H, D, dtype, generator=gen).to(device)
    v = _make_v(B, T, HV, D, dtype, generator=gen).to(device)
    g = _make_g(
        B, T, HV, D,
        dtype=dtype, mask_p=0.0, gate_logit_normalizer=1.0, use_gate_in_kernel=False,
        generator=gen,
    ).clamp(-5, 0).to(device)
    beta = _make_beta(B, T, HV, torch.float32, generator=gen).to(device)

    if expect_error:
        with pytest.raises(AssertionError):
            cula_kda_fused_fwd(
                q=F.normalize(q, p=2, dim=-1),
                k=F.normalize(k, p=2, dim=-1),
                v=v, g=g, beta=beta,
                output_final_state=False, safe_gate=True, lower_bound=-5.0,
            )
    else:
        o, _ = cula_kda_fused_fwd(
            q=F.normalize(q, p=2, dim=-1),
            k=F.normalize(k, p=2, dim=-1),
            v=v, g=g, beta=beta,
            output_final_state=False, safe_gate=True, lower_bound=-5.0,
        )
        assert o.shape == (B, T, HV, D)
