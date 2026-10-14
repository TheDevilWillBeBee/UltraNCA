"""Steady-state cost of one training iteration of train.py vs group size K: a 64-step rollout (the mean of 32-95)
forward + backward plus the OT loss, B = 2 samples per NCA, 128 x 128, TF32; warm-up (compilation) excluded.
python benchmark.py [--impl cuda triton ...] [--K 1 12 24 32] [--compile_loss 1]
"""
import argparse, time
import torch
import train as T
from loss import OTLoss

ap = argparse.ArgumentParser()
ap.add_argument('--impl', nargs='+', default=['eager', 'triton', 'cuda'])
ap.add_argument('--K', nargs='+', type=int, default=[1, 12, 24, 32])
ap.add_argument('--compile_loss', type=int, default=1)
a = ap.parse_args()
dev, B, S, STEPS = 'cuda', 2, 128, 64
torch.backends.cuda.matmul.allow_tf32 = True
ot = OTLoss().to(dev)
ot = torch.compile(ot) if a.compile_loss else ot


def step_fn(impl):
    if impl == 'eager':
        return T.step
    if impl == 'compile':
        return torch.compile(T.step)
    if impl == 'triton':
        import nca_triton
        return nca_triton.step
    from ncaops import nca_ops
    return nca_ops.step


for impl in a.impl:
    step = step_fn(impl)
    for K in a.K:
        g = torch.Generator(device=dev).manual_seed(0)
        w1 = (0.05 * torch.randn(K, 96, 48, device=dev, generator=g)).requires_grad_()
        b1 = torch.zeros(K, 96, device=dev).requires_grad_()
        w2 = (0.01 * torch.randn(K, 12, 96, device=dev, generator=g)).requires_grad_()
        with torch.no_grad():
            tf = ot.features(torch.rand(K, 3, S, S, device=dev))
        rows = torch.arange(K, device=dev).repeat_interleave(B)

        def iteration():
            x = 0.1 * torch.randn(K, B, 12, S, S, device=dev)
            for _ in range(STEPS):
                x = step(x, w1, b1, w2, 0.5)
            tex = ot(x[:, :, :3].reshape(K * B, 3, S, S) + 0.5, tf, rows)
            (tex.sum() + 10 * (x - x.clamp(-1, 1)).abs().sum()).backward()

        for _ in range(3):
            iteration()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        for _ in range(5):
            iteration()
        torch.cuda.synchronize()
        ms = 1000 * (time.time() - t) / 5
        print(f'{impl:8s} K={K:3d}: {ms:7.1f} ms / iteration, {768 * ms / K / 1000:5.1f} s per NCA for 768 iterations, '
              f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB', flush=True)
