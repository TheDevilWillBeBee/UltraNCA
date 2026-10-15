"""Tiled relaxed-EMD nearest pairs for the compiled texture loss (TF32 or 3xTF32).
Only row/column maxima and their indices leave each tile; no dense similarity matrix is stored.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# First maximum, including the first NaN, as in torch.max. Panels are also reduced in index order.
@triton.jit
def _max_index(a, ai, b, bi):
    take = (b > a) | ((b == a) & (bi < ai)) | ((b != b) & ((a == a) | (bi < ai)))
    return tl.where(take, b, a), tl.where(take, bi, ai)


@triton.jit
def _pairs(X, Y, NX, NY, PX, IX, PY, IY, N: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
           X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
           Y0: tl.constexpr, Y1: tl.constexpr, Y2: tl.constexpr,
           NX0: tl.constexpr, NX1: tl.constexpr, NY0: tl.constexpr, NY1: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
    im, jn, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ii, jj, kk = im * BM + tl.arange(0, BM), jn * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(tl.cdiv(C, BK)):
        cc = k * BK + kk
        x = tl.load(X + b * X0 + ii[:, None] * X1 + cc[None, :] * X2, (ii[:, None] < N) & (cc[None, :] < C), 0)
        y = tl.load(Y + b * Y0 + jj[None, :] * Y1 + cc[:, None] * Y2, (jj[None, :] < M) & (cc[:, None] < C), 0)
        if PREC == 'tf32':  # match cuBLAS rounding; Triton's default TF32 conversion truncates
            x = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)
            y = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[y], dtype=tl.float32, is_pure=True, pack=1)
        acc = tl.dot(x, y, acc, input_precision=PREC)
    nx = tl.load(NX + b * NX0 + ii * NX1, ii < N, 0)
    ny = tl.load(NY + b * NY0 + jj * NY1, jj < M, 0)
    sim = acc / (nx[:, None] * ny[None, :] + 1.e-10)
    sim = tl.where((ii[:, None] < N) & (jj[None, :] < M), sim, -float('inf'))
    vx, ix = tl.reduce((sim, tl.broadcast_to(jj[None, :], (BM, BN))), 1, _max_index)
    vy, iy = tl.reduce((sim, tl.broadcast_to(ii[:, None], (BM, BN))), 0, _max_index)
    ox = (b * tl.cdiv(M, BN) + jn) * N + ii
    oy = (b * tl.cdiv(N, BM) + im) * M + jj
    tl.store(PX + ox, vx, ii < N); tl.store(IX + ox, ix, ii < N)
    tl.store(PY + oy, vy, jj < M); tl.store(IY + oy, iy, jj < M)


@triton_op('ultranca::remd_nearest', mutates_args={})
def nearest(x: torch.Tensor, y: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, c = x.shape
    m = y.shape[1]
    bm, bn, bk = 64, 64, 32
    px = torch.empty((b, triton.cdiv(m, bn), n), dtype=x.dtype, device=x.device)
    py = torch.empty((b, triton.cdiv(n, bm), m), dtype=x.dtype, device=x.device)
    ix, iy = torch.empty_like(px, dtype=torch.int32), torch.empty_like(py, dtype=torch.int32)
    prec = 'tf32' if torch.backends.cuda.matmul.allow_tf32 else 'tf32x3'
    wrap_triton(_pairs)[(triton.cdiv(n, bm), triton.cdiv(m, bn), b)](
        x, y, nx, ny, px, ix, py, iy, n, m, c, *x.stride(), *y.stride(), *nx.stride(), *ny.stride(),
        bm, bn, bk, prec, num_warps=4, enable_fp_fusion=False)
    vx, sx = px.max(1); vy, sy = py.max(1)
    j = ix.gather(1, sx[:, None, :]).squeeze(1).long()
    i = iy.gather(1, sy[:, None, :]).squeeze(1).long()
    return vx, j, vy, i
