"""PyTorch wrapper of the fused CUDA NCA step (nca_cuda, built by build.sh). Same signature as train.step:
x (K, B, C, H, W), w1 (K, hidden, 4C), b1 (K, hidden), w2 (K, C, hidden); update_rate >= 1 updates every cell;
growing=True applies the alive mask of a growing NCA (alpha = channel 3). The update mask is generated in the kernel
from a per-call seed drawn from torch's CPU generator (reproducible under torch.manual_seed)."""
import torch

try:
    from . import nca_cuda
except ImportError:
    import nca_cuda


def _forward(x, w1, b1, w2, update_rate, seed, fp32, growing):  # -> y, alive mask (growing) for the backward
    K, B, C, H, W = x.shape
    y = torch.empty_like(x)
    n = (K * B, H, W) if growing else (0, 0, 0)
    alpha, alive = torch.empty(n, device=x.device), torch.empty(n, dtype=torch.uint8, device=x.device)
    nca_cuda.forward(x.view(K * B, C, H, W), w1, b1, w2, y.view(K * B, C, H, W), alpha, alive, B, seed, update_rate,
                     fp32, growing, torch.cuda.current_stream(x.device).cuda_stream)
    return y, alive


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, update_rate, seed, fp32, growing):
        y, alive = _forward(x, w1, b1, w2, update_rate, seed, fp32, growing)
        ctx.save_for_backward(x, w1, b1, w2, alive)
        ctx.update_rate, ctx.seed, ctx.fp32, ctx.growing = update_rate, seed, fp32, growing
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w1, b1, w2, alive = ctx.saved_tensors
        K, B, C, H, W = x.shape
        packed_bytes, partial_floats, chunk = nca_cuda.backward_workspace(C, w1.shape[1], ctx.fp32, K * B, H, W)
        packed = torch.empty(K * packed_bytes, dtype=torch.uint8, device=x.device)
        partial = torch.empty(K * B * partial_floats, device=x.device)
        chunk = CHUNK or chunk
        dp = torch.empty(K * B if chunk >= K * B else 2 * chunk, 4 * C, H, W, device=x.device)  # all, or 2 chunks
        gx, dw1, db1, dw2 = torch.empty_like(x), torch.empty_like(w1), torch.empty_like(b1), torch.empty_like(w2)
        nca_cuda.backward(x.view(K * B, C, H, W), gy.contiguous().view(K * B, C, H, W), alive, w1, b1, w2,
                          gx.view(K * B, C, H, W), dw1, db1, dw2, packed, dp, partial, B, ctx.seed, ctx.update_rate,
                          ctx.fp32, ctx.growing, torch.cuda.current_stream(x.device).cuda_stream)
        return gx, dw1, db1, dw2, None, None, None, None


CHUNK = None  # images per backward chunk (None: two chunks keep their dp within half of the L2 cache)


def new_seed():
    return int(torch.randint(0, 2 ** 62, (1,)).item())


def step(x, w1, b1, w2, update_rate, growing=False, fp32=False, seed=None):
    seed = new_seed() if seed is None else seed
    args = (x.contiguous(), w1.contiguous(), b1.contiguous(), w2.contiguous(), float(update_rate), seed, fp32, growing)
    if torch.is_grad_enabled() and (x.requires_grad or w1.requires_grad or b1.requires_grad or w2.requires_grad):
        return _Step.apply(*args)
    return _forward(*args)[0]  # inference: no autograd bookkeeping (~40 us less host time per step)


@torch.no_grad()
def rollout(x, w1, b1, w2, update_rate, steps, growing=False, fp32=False, seed=None):
    """`steps` steps of step() without gradients, in one call (no Python per step; step seeds derived from `seed`)."""
    seed = new_seed() if seed is None else seed
    x, w1, b1, w2 = x.contiguous(), w1.contiguous(), b1.contiguous(), w2.contiguous()
    K, B, C, H, W = x.shape
    y = torch.empty_like(x)
    tmp = torch.empty(K * B if steps > 1 else 0, C, H, W, device=x.device)  # ping-pong buffer
    n = (K * B, H, W) if growing else (0, 0, 0)
    alpha, alive = torch.empty(n, device=x.device), torch.empty(n, dtype=torch.uint8, device=x.device)
    nca_cuda.rollout(x.view(K * B, C, H, W), w1, b1, w2, y.view(K * B, C, H, W), tmp, alpha, alive, steps, B, seed,
                     float(update_rate), fp32, growing,
                     torch.cuda.current_stream(x.device).cuda_stream)
    return y


def update_mask(n, h, w, seed, update_rate, device='cuda'):
    """The mask the kernels use for `seed`: (n, h, w) of {0, 1}."""
    m = torch.empty(n, h, w, device=device)
    nca_cuda.update_mask(m, seed, float(update_rate))
    return m
