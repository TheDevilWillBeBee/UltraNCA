"""Validate the CUDA NCA step against a PyTorch reference with the same update mask, and benchmark it.
python testing/test_nca.py [--bench]
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nca_ops

FILTERS = torch.stack([torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]]), torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                       torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]), torch.tensor([[1., 2, 1], [2, -12, 2], [1, 2, 1]])])[:, None]


def reference_step(x, w1, b1, w2, mask):
    """x (K, B, C, H, W), mask (K, B, 1, H, W)."""
    K, B, C, H, W = x.shape
    p = F.conv2d(F.pad(x.reshape(-1, 1, H, W), (1, 1, 1, 1), mode='circular'), FILTERS.to(x)).reshape(K, B, 4 * C, H * W)
    h = torch.relu(w1[:, None] @ p + b1[:, None, :, None])
    return x + (w2[:, None] @ h).reshape(K, B, C, H, W) * mask


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def make(K, B, C=12, HID=96, H=128, W=128, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    x = 0.3 * torch.randn(K, B, C, H, W, device='cuda', generator=g)
    w1 = 0.1 * torch.randn(K, HID, 4 * C, device='cuda', generator=g)
    b1 = 0.05 * torch.randn(K, HID, device='cuda', generator=g)
    w2 = 0.05 * torch.randn(K, C, HID, device='cuda', generator=g)
    return x, w1, b1, w2


def test_forward():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for K, B, H, W in ((3, 2, 128, 128), (2, 3, 37, 50), (1, 1, 8, 16)):  # also partial tiles
        x, w1, b1, w2 = make(K, B, H=H, W=W)
        seed = 1234
        mask = nca_ops.update_mask(K * B, H, W, seed, 0.5).view(K, B, 1, H, W)
        ref = reference_step(x, w1, b1, w2, mask)
        for fp32 in (True, False):
            y = nca_ops.step(x, w1, b1, w2, 0.5, fp32=fp32, seed=seed)
            print(f'forward K={K} B={B} {H}x{W} {"fp32 (3xTF32)" if fp32 else "tf32"}: relative error {rel(y, ref):.1e}')
    m = nca_ops.update_mask(4, 128, 128, 7, 0.5)
    print(f'mask: mean {m.mean().item():.4f} (expect 0.5), values {sorted(m.unique().tolist())}')


def test_backward():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for K, B, H, W in ((3, 2, 64, 64), (2, 3, 37, 50)):
        x, w1, b1, w2 = make(K, B, H=H, W=W)
        R = torch.randn_like(x)
        seed = 99
        mask = nca_ops.update_mask(K * B, H, W, seed, 0.5).view(K, B, 1, H, W)
        params = [t.clone().requires_grad_() for t in (x, w1, b1, w2)]
        ref = torch.autograd.grad((reference_step(*params, mask) * R).sum(), params)
        for fp32 in (True, False):
            params = [t.clone().requires_grad_() for t in (x, w1, b1, w2)]
            out = torch.autograd.grad((nca_ops.step(*params, 0.5, fp32=fp32, seed=seed) * R).sum(), params)
            errs = ' '.join(f'{n} {rel(a, b):.1e}' for n, a, b in zip(('dx', 'dw1', 'db1', 'dw2'), out, ref))
            print(f'backward K={K} B={B} {H}x{W} {"fp32 (3xTF32)" if fp32 else "tf32"}: relative error {errs}')
        out2 = torch.autograd.grad((nca_ops.step(*params, 0.5, seed=seed) * R).sum(), params)
        print('  deterministic (two tf32 runs bitwise equal):', all(torch.equal(a, b) for a, b in zip(out, out2)))


def bench(quick=False):
    """64-step rollout at K = 24, B = 2, 128 x 128 (the mean training rollout), forward only and forward + backward;
    then the backward's kernels individually (torch.profiler)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # batched_nca
    import train as T, nca_triton as NT
    torch.backends.cuda.matmul.allow_tf32 = True
    K, B, STEPS = 24, 2, 64
    _, w1, b1, w2 = make(K, B)
    w1, b1, w2 = (t.requires_grad_() for t in (w1, b1, w2))
    steps = {'eager': T.step, 'triton': NT.step, 'cuda tf32': nca_ops.step,
             'cuda fp32 (3xTF32)': lambda *a: nca_ops.step(*a, fp32=True)}
    if quick:
        steps = {'cuda tf32': nca_ops.step}

    def roll(st, grad):
        with torch.set_grad_enabled(grad):
            x = 0.1 * torch.randn(K, B, 12, 128, 128, device='cuda')
            for _ in range(STEPS):
                x = st(x, w1, b1, w2, 0.5)
            if grad:
                x[:, :, :3].square().mean().backward()

    for name, st in steps.items():
        for grad in (False, True):
            for _ in range(2):
                roll(st, grad)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); s.record()
            for _ in range(5):
                roll(st, grad)
            e.record(); torch.cuda.synchronize()
            print(f'{STEPS}-step rollout K={K} B={B} {"fwd+bwd" if grad else "fwd   "} {name:20s} {s.elapsed_time(e) / 5:7.1f} ms'
                  f'  peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.1f} GB', flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        roll(nca_ops.step, True); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=8, max_name_column_width=60))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--bench', action='store_true'); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    test_forward()
    test_backward()
    if a.bench or a.quick:
        bench(a.quick)
