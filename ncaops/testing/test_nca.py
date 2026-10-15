"""Validate the CUDA NCA step against a PyTorch reference with the same update mask, for every instantiated
(channels, hidden) size, with and without the stochastic update and the growing-NCA alive mask; optionally time it.
python ncaops/testing/test_nca.py [--bench]
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nca_ops

FILTERS = torch.stack([torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]]), torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                       torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]), torch.tensor([[1., 2, 1], [2, -12, 2], [1, 2, 1]])])[:, None]
# relative-error bounds: fp32 = 3xTF32; TF32 errors are those of TF32 itself (eager PyTorch TF32 vs FP32 is the same).
# A growing-NCA cell whose alive test (alpha > 0.1) flips under TF32 rounding differs by its whole state (and so do its
# 8 neighbours). Check the mask exactly against the unmasked CUDA step, then exclude confirmed flips from the
# forward error and count them (at most FLIPS across the shapes of each case).
BOUNDS = {True: (1e-5, 1e-5), False: (1e-2, 3e-2)}  # fp32: (forward, gradients)
FLIPS = {True: 0, False: 32}


def alive(x):  # GrowingNCA.get_living_mask: x (K, B, C, H, W) -> (K, B, 1, H, W)
    K, B, C, H, W = x.shape
    return (F.max_pool2d(x[:, :, 3].reshape(K * B, 1, H, W), 3, 1, 1) > 0.1).reshape(K, B, 1, H, W)


def reference_step(x, w1, b1, w2, mask, growing=False):
    """x (K, B, C, H, W), mask (K, B, 1, H, W)."""
    K, B, C, H, W = x.shape
    p = F.conv2d(F.pad(x.reshape(-1, 1, H, W), (1, 1, 1, 1), mode='circular'), FILTERS.to(x)).reshape(K, B, 4 * C, H * W)
    h = torch.relu(w1[:, None] @ p + b1[:, None, :, None])
    y = x + (w2[:, None] @ h).reshape(K, B, C, H, W) * mask
    return y * (alive(x) & alive(y)) if growing else y


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def make(K, B, C=12, HID=96, H=128, W=128, seed=0, growing=False):
    g = torch.Generator(device='cuda').manual_seed(seed)
    x = 0.3 * torch.randn(K, B, C, H, W, device='cuda', generator=g)
    if growing:  # a living disc in a dead background (small non-zero states, alpha <= 0.1)
        yy, xx = torch.meshgrid(torch.arange(H, device='cuda') - H / 2, torch.arange(W, device='cuda') - W / 2, indexing='ij')
        dead = yy ** 2 + xx ** 2 >= (min(H, W) / 3) ** 2
        x = torch.where(dead, 0.1 * x, x)
        x[:, :, 3] = torch.where(dead, x[:, :, 3].clamp(max=0.1), x[:, :, 3])
    w1 = 0.1 * torch.randn(K, HID, 4 * C, device='cuda', generator=g)
    b1 = 0.05 * torch.randn(K, HID, device='cuda', generator=g)
    w2 = 0.05 * torch.randn(K, C, HID, device='cuda', generator=g)
    return x, w1, b1, w2


def mask_for(K, B, H, W, seed, rate):
    return nca_ops.update_mask(K * B, H, W, seed, rate).view(K, B, 1, H, W)


def test():
    """Per case and precision: the largest relative error of the forward over several shapes (including partial
    tiles), of the gradients (dx, dw1, db1, dw2) over two shapes, and whether TF32 backward runs (also in several image chunks) are bitwise equal."""
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    ok = True
    for C, HID in ((12, 96), (16, 128)):
        for rate in (0.5, 1.0):
            for growing in (False, True):
                fwd, bwd, det = {True: 0., False: 0.}, {True: 0., False: 0.}, True
                flips = {True: 0, False: 0}
                # 129 rows also exercises a one-row final segment (including an empty half with NCA_FWD_TAIL).
                for K, B, H, W in ((3, 2, 128, 128), (2, 3, 37, 50), (1, 1, 8, 16), (3, 2, 80, 80), (17, 2, 129, 65)):
                    x, w1, b1, w2 = make(K, B, C, HID, H, W, growing=growing)
                    ref_raw = reference_step(x, w1, b1, w2, mask_for(K, B, H, W, 1234, rate))
                    ref_alive = alive(x) & alive(ref_raw) if growing else None
                    ref = ref_raw * ref_alive if growing else ref_raw
                    for fp32 in (True, False):
                        out = nca_ops.step(x, w1, b1, w2, rate, growing, fp32=fp32, seed=1234)
                        err = rel(out, ref)
                        if growing:
                            raw = nca_ops.step(x, w1, b1, w2, rate, False, fp32=fp32, seed=1234)
                            out_alive = alive(x) & alive(raw)
                            assert torch.equal(out, raw * out_alive), 'growing step must apply the exact alive mask'
                            flip = out_alive != ref_alive
                            flips[fp32] += flip.sum().item()
                            err = max(rel(raw, ref_raw), rel(out.masked_fill(flip, 0), ref.masked_fill(flip, 0)))
                        fwd[fp32] = max(fwd[fp32], err)
                for K, B, H, W in ((3, 2, 64, 64), (2, 3, 37, 50)):
                    x, w1, b1, w2 = make(K, B, C, HID, H, W, growing=growing)
                    R, mask = torch.randn_like(x), mask_for(K, B, H, W, 99, rate)
                    grads = lambda f: torch.autograd.grad((f(*params) * R).sum(), params)
                    params = [t.clone().requires_grad_() for t in (x, w1, b1, w2)]
                    ref = grads(lambda *p: reference_step(*p, mask, growing))
                    for fp32 in (True, False):
                        out = grads(lambda *p: nca_ops.step(*p, rate, growing, fp32=fp32, seed=99))
                        bwd[fp32] = max([bwd[fp32]] + [rel(a, b) for a, b in zip(out, ref)])
                    for chunk in (2, K * B, None):  # two streams, one launch, automatic choice for this GPU
                        nca_ops.CHUNK = chunk
                        det &= all(torch.equal(a, b) for a, b in zip(out, grads(lambda *p: nca_ops.step(*p, rate, growing, seed=99))))
                passed = det and all(fwd[p] < BOUNDS[p][0] and bwd[p] < BOUNDS[p][1] and flips[p] <= FLIPS[p] for p in (True, False))
                ok &= passed
                print(f'{C:2d}/{HID:3d} rate={rate} {"growing" if growing else "texture"}: fp32 fwd {fwd[True]:.1e} grad '
                      f'{bwd[True]:.1e} | tf32 fwd {fwd[False]:.1e} grad {bwd[False]:.1e} | deterministic {det} | '
                      + (f'alive flips {flips[True]} / {flips[False]} | ' if growing else '') + f'{"ok" if passed else "FAILED"}', flush=True)
    def step_seed(seed, i):  # splitmix64, as in nca_cuda.cu
        mask = (1 << 64) - 1
        z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    for C, HID in ((12, 96), (16, 128)):
        for growing in (False, True):
            x, w1, b1, w2 = make(2, 2, C, HID, 37, 50, growing=growing)
            original = x.clone()
            same = True
            for fp32 in (False, True):
                for rate in (0., 0.5, 1.):
                    ref = x
                    for steps in range(8):
                        if steps:
                            ref = nca_ops.step(ref, w1, b1, w2, rate, growing, fp32, seed=step_seed(99, steps - 1))
                        same &= torch.equal(nca_ops.rollout(x, w1, b1, w2, rate, steps, growing, fp32, seed=99), ref)
            same &= torch.equal(x, original)
            ok &= same
            print(f'rollout {C}/{HID} {"growing" if growing else "texture"}, both precisions, rates 0/0.5/1: {"ok" if same else "FAILED"}')

        # A dense living grid dies after four steps: both reused destinations must clear their stale states.
        x = torch.ones(1, 1, C, 8, 16, device='cuda'); x[:, :, 3] = 0.2
        w1 = torch.zeros(1, HID, 4 * C, device='cuda'); b1 = torch.ones(1, HID, device='cuda')
        w2 = torch.full((1, C, HID), 0.01, device='cuda'); w2[:, 3] = -0.025 / HID
        y = nca_ops.rollout(x, w1, b1, w2, 1., 7, True, seed=99)
        same = torch.equal(y, torch.zeros_like(y))
        ok &= same
        print(f'rollout {C}/{HID} dying grid: {"ok" if same else "FAILED"}')
    m = nca_ops.update_mask(4, 128, 128, 7, 0.5)
    print(f'mask: mean {m.mean().item():.4f} (expect 0.5), values {sorted(m.unique().tolist())}')
    print('all passed' if ok else 'SOME CASES FAILED')
    return ok


def bench():
    """Rollouts of the training workloads (texture: K = 32, B = 2, 128^2, 64 steps; growing: K = 32, B = 8, 72^2,
    80 steps, from noise in a disc), forward only and forward + backward; then the kernels of one fwd + bwd rollout."""
    torch.backends.cuda.matmul.allow_tf32 = True
    for C, HID, rate, growing, K, B, S, steps in ((12, 96, 0.5, False, 32, 2, 128, 64), (16, 128, 0.5, True, 32, 8, 72, 80)):
        x0, w1, b1, w2 = make(K, B, C, HID, S, S, growing=growing)
        w1, b1, w2 = (t.requires_grad_() for t in (w1, b1, w2))

        def roll(grad):
            with torch.set_grad_enabled(grad):
                x = x0
                for _ in range(steps):
                    x = nca_ops.step(x, w1, b1, w2, rate, growing)
                if grad:
                    x[:, :, :4].square().mean().backward()

        for grad in (False, True):
            for _ in range(2):
                roll(grad)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record()
            for _ in range(5):
                roll(grad)
            e.record(); torch.cuda.synchronize()
            print(f'{C}/{HID} {"growing" if growing else "texture"} K={K} B={B} {S}^2 {steps}-step rollout '
                  f'{"fwd+bwd" if grad else "fwd    "}: {s.elapsed_time(e) / 5:7.1f} ms', flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            roll(True); torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=8, max_name_column_width=50))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--bench', action='store_true')
    ok = test()
    if ap.parse_args().bench:
        bench()
    sys.exit(0 if ok else 1)
