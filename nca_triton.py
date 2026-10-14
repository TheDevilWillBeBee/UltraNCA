"""Fused NCA update step in Triton, a drop-in for train.step (select with `impl: triton` in the config).

Forward: one kernel per step computes the 3x3 perception (circular), w1 / b1 / relu / w2 and the masked residual update
for a tile of pixels without writing the 48 perception or 96 hidden channels to memory. Autograd saves only the
12-channel state; the backward kernel recomputes perception and hidden activations, and produces the perception
gradient and per-program partial weight gradients. A small stencil kernel maps the perception gradient back to the
state (the adjoint of the circular 3x3 filters). Matmuls run on tensor cores in TF32 (or exact fp32 with PREC='ieee').
Channels are padded 12 -> 16, perception 48 -> 64, hidden 96 -> 128 (tl.dot needs powers of two).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CP, HP = 16, 128                  # padded channels, padded hidden size
BP_FWD, WARPS_FWD = 128, 4        # forward: pixels per program, warps
BP_BWD, TPP, WARPS_BWD = 32, 16, 4  # backward: pixels per tile, tiles per program, warps
BP_ADJ = 128
PREC = 'tf32'                     # 'tf32' or 'ieee' (exact fp32)


@triton.jit
def _nbr(base, yy, xx, dy: tl.constexpr, dx: tl.constexpr, c, cm, H: tl.constexpr, W: tl.constexpr):
    idx = ((yy + dy + H) % H) * W + (xx + dx + W) % W
    return tl.load(base + c[None, :] * (H * W) + idx[:, None], mask=cm[None, :], other=0.0)


@triton.jit
def _perception(base, yy, xx, c, cm, H: tl.constexpr, W: tl.constexpr, BP: tl.constexpr, CP: tl.constexpr):
    """(BP, 4 * CP) tile, column 4c + f with f = identity, sobel_x, sobel_x^T, laplacian (as train.FILTERS)."""
    a = _nbr(base, yy, xx, -1, -1, c, cm, H, W)
    b = _nbr(base, yy, xx, -1, 0, c, cm, H, W)
    d = _nbr(base, yy, xx, -1, 1, c, cm, H, W)
    e = _nbr(base, yy, xx, 0, -1, c, cm, H, W)
    f = _nbr(base, yy, xx, 0, 0, c, cm, H, W)
    g = _nbr(base, yy, xx, 0, 1, c, cm, H, W)
    h = _nbr(base, yy, xx, 1, -1, c, cm, H, W)
    i = _nbr(base, yy, xx, 1, 0, c, cm, H, W)
    j = _nbr(base, yy, xx, 1, 1, c, cm, H, W)
    sx = (d - a) + 2.0 * (g - e) + (j - h)
    sy = (h - a) + 2.0 * (i - b) + (j - d)
    lap = (a + d + h + j) + 2.0 * (b + e + g + i) - 12.0 * f
    # join(join(f, sy), join(sx, lap)) -> [..., a, o] = filter 2o + a; reshape puts it at column 4c + 2a + o = 4c + filter
    return tl.reshape(tl.join(tl.join(f, sy), tl.join(sx, lap)), (BP, 4 * CP))


@triton.jit
def _fwd(x_ptr, m_ptr, w1_ptr, b1_ptr, w2_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
         CP: tl.constexpr, HP: tl.constexpr, BP: tl.constexpr, PREC: tl.constexpr):
    kb, t = tl.program_id(0), tl.program_id(1)
    k = kb // B
    q = t * BP + tl.arange(0, BP)
    yy, xx = q // W, q % W
    c = tl.arange(0, CP); cm = c < C
    r = tl.arange(0, 4 * CP); j = tl.arange(0, HP)
    base = x_ptr + kb * C * H * W
    p = _perception(base, yy, xx, c, cm, H, W, BP, CP)
    hid = tl.dot(p, tl.load(w1_ptr + k * 4 * CP * HP + r[:, None] * HP + j[None, :]), input_precision=PREC) \
        + tl.load(b1_ptr + k * HP + j)[None, :]
    u = tl.dot(tl.maximum(hid, 0.0), tl.load(w2_ptr + k * HP * CP + j[:, None] * CP + c[None, :]), input_precision=PREC)
    xc = tl.load(base + c[None, :] * (H * W) + q[:, None], mask=cm[None, :], other=0.0)
    m = tl.load(m_ptr + kb * H * W + q)
    tl.store(y_ptr + kb * C * H * W + c[None, :] * (H * W) + q[:, None], xc + u * m[:, None], mask=cm[None, :])


@triton.jit
def _bwd(x_ptr, m_ptr, g_ptr, w1_ptr, b1_ptr, w2_ptr, dp_ptr, pw1_ptr, pb1_ptr, pw2_ptr,
         B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, CP: tl.constexpr, HP: tl.constexpr,
         BP: tl.constexpr, TPP: tl.constexpr, PREC: tl.constexpr):
    kb, pid = tl.program_id(0), tl.program_id(1)
    k = kb // B
    c = tl.arange(0, CP); cm = c < C
    r = tl.arange(0, 4 * CP); j = tl.arange(0, HP)
    base = x_ptr + kb * C * H * W
    aw1 = tl.zeros((4 * CP, HP), tl.float32); aw2 = tl.zeros((HP, CP), tl.float32); ab1 = tl.zeros((HP,), tl.float32)
    for tt in range(TPP):
        q = (pid * TPP + tt) * BP + tl.arange(0, BP)
        yy, xx = q // W, q % W
        p = _perception(base, yy, xx, c, cm, H, W, BP, CP)
        w1 = tl.load(w1_ptr + k * 4 * CP * HP + r[:, None] * HP + j[None, :])
        hid = tl.dot(p, w1, input_precision=PREC) + tl.load(b1_ptr + k * HP + j)[None, :]
        gm = tl.load(g_ptr + kb * C * H * W + c[None, :] * (H * W) + q[:, None], mask=cm[None, :], other=0.0) \
            * tl.load(m_ptr + kb * H * W + q)[:, None]
        aw2 = tl.dot(tl.trans(tl.maximum(hid, 0.0)), gm, aw2, input_precision=PREC)
        w2t = tl.trans(tl.load(w2_ptr + k * HP * CP + j[:, None] * CP + c[None, :]))
        dh = tl.where(hid > 0, tl.dot(gm, w2t, input_precision=PREC), 0.0)
        ab1 += tl.sum(dh, 0)
        aw1 = tl.dot(tl.trans(p), dh, aw1, input_precision=PREC)
        dp = tl.dot(dh, tl.trans(w1), input_precision=PREC)                   # (BP, 4 * CP), column 4c + f
        tl.store(dp_ptr + (kb * C * 4 + r[None, :]) * (H * W) + q[:, None], dp, mask=r[None, :] < 4 * C)
    slot = kb * tl.num_programs(1) + pid
    tl.store(pw1_ptr + slot * 4 * CP * HP + r[:, None] * HP + j[None, :], aw1)
    tl.store(pw2_ptr + slot * HP * CP + j[:, None] * CP + c[None, :], aw2)
    tl.store(pb1_ptr + slot * HP + j, ab1)


@triton.jit
def _ldp(base, yy, xx, sy: tl.constexpr, sx: tl.constexpr, f: tl.constexpr, c, cm, H: tl.constexpr, W: tl.constexpr):
    idx = ((yy + sy + H) % H) * W + (xx + sx + W) % W
    return tl.load(base + (c[None, :] * 4 + f) * (H * W) + idx[:, None], mask=cm[None, :], other=0.0)


@triton.jit
def _adjoint(dp_ptr, g_ptr, dx_ptr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, CP: tl.constexpr, BP: tl.constexpr):
    """dx(y) = g(y) + sum_f sum_o K_f(o) dp_f(y - o): the transpose of the circular 3x3 perception filters."""
    n, t = tl.program_id(0), tl.program_id(1)
    q = t * BP + tl.arange(0, BP)
    yy, xx = q // W, q % W
    c = tl.arange(0, CP); cm = c < C
    base = dp_ptr + n * C * 4 * H * W
    off = n * C * H * W + c[None, :] * (H * W) + q[:, None]
    acc = tl.load(g_ptr + off, mask=cm[None, :], other=0.0)
    acc += _ldp(base, yy, xx, 0, 0, 0, c, cm, H, W) - 12.0 * _ldp(base, yy, xx, 0, 0, 3, c, cm, H, W)      # o = (0, 0)
    # o = (dy, dx) is read at y - o; f = 1 sobel_x, 2 sobel_x^T, 3 laplacian
    acc += -_ldp(base, yy, xx, 1, 1, 1, c, cm, H, W) - _ldp(base, yy, xx, 1, 1, 2, c, cm, H, W) + _ldp(base, yy, xx, 1, 1, 3, c, cm, H, W)
    acc += -2.0 * _ldp(base, yy, xx, 1, 0, 2, c, cm, H, W) + 2.0 * _ldp(base, yy, xx, 1, 0, 3, c, cm, H, W)
    acc += _ldp(base, yy, xx, 1, -1, 1, c, cm, H, W) - _ldp(base, yy, xx, 1, -1, 2, c, cm, H, W) + _ldp(base, yy, xx, 1, -1, 3, c, cm, H, W)
    acc += -2.0 * _ldp(base, yy, xx, 0, 1, 1, c, cm, H, W) + 2.0 * _ldp(base, yy, xx, 0, 1, 3, c, cm, H, W)
    acc += 2.0 * _ldp(base, yy, xx, 0, -1, 1, c, cm, H, W) + 2.0 * _ldp(base, yy, xx, 0, -1, 3, c, cm, H, W)
    acc += -_ldp(base, yy, xx, -1, 1, 1, c, cm, H, W) + _ldp(base, yy, xx, -1, 1, 2, c, cm, H, W) + _ldp(base, yy, xx, -1, 1, 3, c, cm, H, W)
    acc += 2.0 * _ldp(base, yy, xx, -1, 0, 2, c, cm, H, W) + 2.0 * _ldp(base, yy, xx, -1, 0, 3, c, cm, H, W)
    acc += _ldp(base, yy, xx, -1, -1, 1, c, cm, H, W) + _ldp(base, yy, xx, -1, -1, 2, c, cm, H, W) + _ldp(base, yy, xx, -1, -1, 3, c, cm, H, W)
    tl.store(dx_ptr + off, acc, mask=cm[None, :])


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1p, b1p, w2p, m):
        K, B, C, H, W = x.shape
        y = torch.empty_like(x)
        _fwd[(K * B, H * W // BP_FWD)](x, m, w1p, b1p, w2p, y, B, C, H, W, CP, HP, BP_FWD, PREC, num_warps=WARPS_FWD, num_stages=1)
        ctx.save_for_backward(x, m, w1p, b1p, w2p)
        return y

    @staticmethod
    def backward(ctx, g):
        x, m, w1p, b1p, w2p = ctx.saved_tensors
        K, B, C, H, W = x.shape
        g = g.contiguous()
        nprog = H * W // (BP_BWD * TPP)
        dp = torch.empty(K * B, C, 4, H, W, device=x.device)
        pw1 = torch.empty(K, B * nprog, 4 * CP, HP, device=x.device)
        pb1 = torch.empty(K, B * nprog, HP, device=x.device)
        pw2 = torch.empty(K, B * nprog, HP, CP, device=x.device)
        _bwd[(K * B, nprog)](x, m, g, w1p, b1p, w2p, dp, pw1, pb1, pw2, B, C, H, W, CP, HP, BP_BWD, TPP, PREC,
                             num_warps=WARPS_BWD, num_stages=1)
        dx = torch.empty_like(x)
        _adjoint[(K * B, H * W // BP_ADJ)](dp, g, dx, C, H, W, CP, BP_ADJ)
        return dx, pw1.sum(1), pb1.sum(1), pw2.sum(1), None


def step(x, w1, b1, w2, update_rate):
    """Same signature and math as train.step: x (K, B, C, H, W); w1 (K, hidden, 4C), b1 (K, hidden), w2 (K, C, hidden)."""
    K, B, C, H, W = x.shape
    Hd = w1.shape[1]
    w1p = F.pad(w1.transpose(1, 2), (0, HP - Hd, 0, 4 * (CP - C))).contiguous()   # (K, 4 * CP, HP), row 4c + f
    b1p = F.pad(b1, (0, HP - Hd)).contiguous()
    w2p = F.pad(w2.transpose(1, 2), (0, CP - C, 0, HP - Hd)).contiguous()        # (K, HP, CP)
    m = (torch.rand(K * B, H * W, device=x.device) + update_rate).floor()
    return _Step.apply(x.contiguous(), w1p, b1p, w2p, m)
