"""PyTorch wrapper of the fused CUDA NCA step (nca_cuda, built by build.sh). Same signature as batched_nca/train.step:
x (K, B, C, H, W), w1 (K, hidden, 4C), b1 (K, hidden), w2 (K, C, hidden). The update mask is generated in the kernel
from a per-call seed drawn from torch's CPU generator (reproducible under torch.manual_seed)."""
import torch

try:
    from . import nca_cuda
except ImportError:
    import nca_cuda


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, update_rate, seed, fp32):
        K, B, C, H, W = x.shape
        y = torch.empty_like(x)
        nca_cuda.forward(x.view(K * B, C, H, W), w1, b1, w2, y.view(K * B, C, H, W), B, seed, update_rate, fp32)
        ctx.save_for_backward(x, w1, b1, w2)
        ctx.update_rate, ctx.seed, ctx.fp32 = update_rate, seed, fp32
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w1, b1, w2 = ctx.saved_tensors
        K, B, C, H, W = x.shape
        packed_bytes, partial_floats = nca_cuda.backward_workspace(C, w1.shape[1], ctx.fp32, H, W)
        packed = torch.empty(K * packed_bytes, dtype=torch.uint8, device=x.device)
        partial = torch.empty(K * B * partial_floats, device=x.device)
        dp = torch.empty(K * B, 4 * C, H, W, device=x.device)
        gx, dw1, db1, dw2 = torch.empty_like(x), torch.empty_like(w1), torch.empty_like(b1), torch.empty_like(w2)
        nca_cuda.backward(x.view(K * B, C, H, W), gy.contiguous().view(K * B, C, H, W), w1, b1, w2,
                          gx.view(K * B, C, H, W), dw1, db1, dw2, packed, dp, partial, B, ctx.seed, ctx.update_rate, ctx.fp32)
        return gx, dw1, db1, dw2, None, None, None


def new_seed():
    return int(torch.randint(0, 2 ** 62, (1,)).item())


def step(x, w1, b1, w2, update_rate, fp32=False, seed=None):
    seed = new_seed() if seed is None else seed
    return _Step.apply(x.contiguous(), w1.contiguous(), b1.contiguous(), w2.contiguous(), float(update_rate), seed, fp32)


def update_mask(n, h, w, seed, update_rate, device='cuda'):
    """The mask the kernels use for `seed`: (n, h, w) of {0, 1}."""
    m = torch.empty(n, h, w, device=device)
    nca_cuda.update_mask(m, seed, float(update_rate))
    return m
