"""Triton kernels of the compiled texture loss.
- nearest: tiled relaxed-EMD nearest pairs (TF32 or 3xTF32; fp16 for fp16-valued features, exact). Only row/column
  maxima and their indices leave each tile; no dense similarity matrix is stored.
- remd_grad_selected: compute only the selected REMD direction per image; reverse matches accumulate in FP32.
- cov_l1, cov_l1_grad: the covariance term mean |cov(x) - cov(y)| of fp16-valued features and its gradient, on fp16
  tensor cores with fp32 accumulation (exact products): cov = (v^T v - n m^T m) / (n - 1), per tile of the upper
  triangle, only the per-tile sums of |cov(x) - cov(y)| and the int8 signs leave the kernel.
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
        if PREC == 'fp16':  # fp16-valued inputs: exact products
            acc = tl.dot(x.to(tl.float16), y.to(tl.float16), acc)
        else:
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
def nearest(x: torch.Tensor, y: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor, fp16: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp16: x and y hold fp16 values (any float dtype): fp16 products, exact."""
    b, n, c = x.shape
    m = y.shape[1]
    bm, bn, bk = (64, 128, 32) if fp16 else (64, 64, 32)
    px = torch.empty((b, triton.cdiv(m, bn), n), dtype=torch.float32, device=x.device)
    py = torch.empty((b, triton.cdiv(n, bm), m), dtype=torch.float32, device=x.device)
    ix, iy = torch.empty_like(px, dtype=torch.int32), torch.empty_like(py, dtype=torch.int32)
    prec = 'fp16' if fp16 else 'tf32' if torch.backends.cuda.matmul.allow_tf32 else 'tf32x3'
    wrap_triton(_pairs)[(triton.cdiv(n, bm), triton.cdiv(m, bn), b)](
        x, y, nx, ny, px, ix, py, iy, n, m, c, *x.stride(), *y.stride(), *nx.stride(), *ny.stride(),
        bm, bn, bk, prec, num_warps=4, enable_fp_fusion=False)
    vx, sx = px.max(1); vy, sy = py.max(1)
    j = ix.gather(1, sx[:, None, :]).squeeze(1).long()
    i = iy.gather(1, sy[:, None, :]).squeeze(1).long()
    return vx, j, vy, i


@triton.jit
def _remd_bwd_selected(X, Y, NX, NY, I, J, SIDE, G, OUT, N: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                       G0: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """One direction per image. The reverse direction accumulates repeated nearest-x indices, as scatter_add."""
    block, b = tl.program_id(0), tl.program_id(1)
    p, c = block * BR + tl.arange(0, BR), tl.arange(0, BC)
    x_side = tl.load(SIDE + b)
    if x_side:
        valid = p < N
        ix = p
        iy = tl.load(J + b * N + p, valid, 0).to(tl.int32)
        weight = -tl.load(G + b * G0) / N
    else:
        valid = p < M
        ix = tl.load(I + b * M + p, valid, 0).to(tl.int32)
        iy = p
        weight = -tl.load(G + b * G0) / M
    x = tl.load(X + (b * N + ix[:, None]) * C + c[None, :], valid[:, None] & (c[None, :] < C), 0).to(tl.float32)
    y = tl.load(Y + (b * M + iy[:, None]) * C + c[None, :], valid[:, None] & (c[None, :] < C), 0).to(tl.float32)
    nx = tl.load(NX + b * N + ix, valid, 0)
    ny = tl.load(NY + b * M + iy, valid, 0)
    d = nx * ny + 1.e-10
    dot = tl.sum(x * y, 1)
    factor = dot * ny / (nx * d * d)
    factor = tl.where(tl.abs(factor) < float('inf'), factor, 0.)  # nan_to_num(0, 0, 0), including zero vectors
    grad = weight * (y / d[:, None] - factor[:, None] * x)
    out = OUT + (b * N + ix[:, None]) * C + c[None, :]
    if x_side:
        tl.store(out, grad, valid[:, None] & (c[None, :] < C))
    else:
        tl.atomic_add(out, grad, valid[:, None] & (c[None, :] < C), sem='relaxed')


@triton_op('ultranca::remd_grad_selected', mutates_args={})
def remd_grad_selected(g: torch.Tensor, x: torch.Tensor, y: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor,
                       i: torch.Tensor, j: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:
    """REMD image gradient for contiguous feature sets. Evaluate only the selected direction, on the GPU;
    repeated reverse matches use FP32 atomics, with the same accumulation-order caveat as torch.scatter_add."""
    b, n, c = x.shape
    m = y.shape[1]
    assert all(v.is_contiguous() for v in (x, y, nx, ny, i, j, x_side))
    out = torch.zeros((b, n, c), device=x.device, dtype=torch.float32)
    wrap_triton(_remd_bwd_selected)[(triton.cdiv(max(n, m), 16), b)](
        x, y, nx, ny, i, j, x_side, g, out, n, m, c, g.stride(0), 16, triton.next_power_of_2(c),
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _cov_fwd(X, Y, MX, MY, TI, TJ, S, PART, n, m, T, C: tl.constexpr, X0, X1, Y0, Y1,
             BC: tl.constexpr, BK: tl.constexpr):
    """Tile (TI[t], TJ[t]) of the upper triangle of item b: d = cov(x) - cov(y) -> sign(d) (int8, both triangles),
    sum |d| (doubled off the diagonal)."""
    t, b = tl.program_id(0), tl.program_id(1)
    ti, tj = tl.load(TI + t), tl.load(TJ + t)
    ci, cj, kk = ti * BC + tl.arange(0, BC), tj * BC + tl.arange(0, BC), tl.arange(0, BK)
    ax = tl.zeros((BC, BC), tl.float32)
    for k in range(0, n, BK):
        r = (k + kk)[:, None]
        a = tl.load(X + b * X0 + r * X1 + ci[None, :], r < n, 0).to(tl.float16)
        v = tl.load(X + b * X0 + r * X1 + cj[None, :], r < n, 0).to(tl.float16)
        ax = tl.dot(tl.trans(a), v, ax)
    ay = tl.zeros((BC, BC), tl.float32)
    for k in range(0, m, BK):
        r = (k + kk)[:, None]
        a = tl.load(Y + b * Y0 + r * Y1 + ci[None, :], r < m, 0).to(tl.float16)
        v = tl.load(Y + b * Y0 + r * Y1 + cj[None, :], r < m, 0).to(tl.float16)
        ay = tl.dot(tl.trans(a), v, ay)
    mxi, mxj = tl.load(MX + b * C + ci), tl.load(MX + b * C + cj)
    myi, myj = tl.load(MY + b * C + ci), tl.load(MY + b * C + cj)
    d = (ax - n * mxi[:, None] * mxj[None, :]) / (n - 1) - (ay - m * myi[:, None] * myj[None, :]) / (m - 1)
    s = tl.where(d > 0, 1, tl.where(d < 0, -1, 0)).to(tl.int8)
    tl.store(S + b * C * C + ci[:, None] * C + cj[None, :], s)
    tl.store(S + b * C * C + cj[None, :] * C + ci[:, None], s)
    tl.store(PART + b * T + t, tl.sum(tl.abs(d)) * tl.where(ti == tj, 1.0, 2.0))


@triton_op('ultranca::cov_l1', mutates_args={})
def cov_l1(x: torch.Tensor, y: torch.Tensor, mx: torch.Tensor, my: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """mean |cov(x) - cov(y)| per item of fp16-valued x (b, n, c), y (b, m, c) (any float dtype, unit channel stride)
    with means mx, my (b, c) fp32 -> (b,), and sign(cov(x) - cov(y)) (b, c, c) int8."""
    b, n, c = x.shape
    m = y.shape[1]
    assert c % 64 == 0 and n > 1 and m > 1 and x.stride(2) == 1 and y.stride(2) == 1
    bc = 128 if c % 128 == 0 else 64
    tiles = c // bc
    ti, tj = torch.triu_indices(tiles, tiles, device=x.device).int()
    part = torch.empty(b, len(ti), device=x.device)
    sign = torch.empty(b, c, c, device=x.device, dtype=torch.int8)
    wrap_triton(_cov_fwd)[(len(ti), b)](x, y, mx, my, ti, tj, sign, part, n, m, len(ti), c, x.stride(0), x.stride(1),
                                         y.stride(0), y.stride(1), bc, 32, num_warps=bc // 16, num_stages=3)
    return part.sum(1) / (c * c), sign


@triton.jit
def _cov_bwd(X, MX, S, COEF, VEC, R, OUT, n, C: tl.constexpr, X0, X1, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
    """OUT = R + COEF[b] (x - mx) S + VEC[b] (rows broadcast), for an (BM, BN) tile of item b; OUT, R contiguous."""
    im, jn, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows, cols, kk = im * BM + tl.arange(0, BM), jn * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, C, BK):
        a = tl.load(X + b * X0 + rows[:, None] * X1 + (k + kk)[None, :], rows[:, None] < n, 0).to(tl.float32)
        a = (a - tl.load(MX + b * C + k + kk)[None, :]).to(tl.float16)
        s = tl.load(S + b * C * C + (k + kk)[:, None] * C + cols[None, :]).to(tl.float16)
        acc = tl.dot(a, s, acc)
    o = b * n * C + rows[:, None] * C + cols[None, :]
    r = tl.load(R + o, rows[:, None] < n, 0)
    out = r + tl.load(COEF + b) * acc + tl.load(VEC + b * C + cols)[None, :]
    tl.store(OUT + o, out, rows[:, None] < n)


@triton_op('ultranca::cov_l1_grad', mutates_args={})
def cov_l1_grad(x: torch.Tensor, mx: torch.Tensor, sign: torch.Tensor, coef: torch.Tensor, vec: torch.Tensor,
                r: torch.Tensor) -> torch.Tensor:
    """r + coef[b] (x - mx) sign + vec[b] per item: the covariance-term gradient (coef = g 2 / (c^2 (n - 1))) added to
    the other terms' gradients r (b, n, c) fp32 contiguous and vec (b, c)."""
    b, n, c = x.shape
    assert c % 64 == 0 and x.stride(2) == 1 and r.is_contiguous()
    bm, bn, bk = 64, 64, 32
    out = torch.empty_like(r)
    wrap_triton(_cov_bwd)[(triton.cdiv(n, bm), c // bn, b)](x, mx, sign, coef, vec, r, out, n, c, x.stride(0), x.stride(1),
                                                            bm, bn, bk, num_warps=4)
    return out
