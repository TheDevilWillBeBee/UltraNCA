"""Regression checks for the VGG OT optimizations: identical draws, repeated target IDs, and chunked gradients.
Run with python ncaops/testing/test_vgg_optimizations.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from loss_ms import MultiScaleOTLoss, _sample
from loss import _remd_grad
from loss_triton import remd_grad_selected


class ExpandedTargets(MultiScaleOTLoss):
    """The original target path: expand entire cached maps before selecting spatial positions."""

    def _patch_losses(self, gen, tgt, n_img, target_rows=None):
        if target_rows is not None:
            tgt = [[f[target_rows] for f in fs] for fs in tgt]
        return super()._patch_losses(gen, tgt, n_img)


def sampling():
    rows = torch.tensor([2, 0, 2, 1, 2], device='cuda')
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for M, n in ((16, 32), (32, 32), (128, 32), (4096, 32)):
            f = torch.randn(3, 64, M, device='cuda', dtype=dtype).transpose(1, 2)
            for compiled in (False, True):
                old = lambda f, rows: _sample(f[rows], n)
                new = lambda f, rows: _sample(f, n, rows)
                if compiled:
                    torch._dynamo.reset()
                    old, new = torch.compile(old), torch.compile(new)
                    old(f, rows); new(f, rows)
                torch.manual_seed(123)
                a = old(f, rows)
                torch.manual_seed(123)
                b = new(f, rows)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    print('direct target sampling: identical values and draws (eager/compiled, all/sampled/replacement)', flush=True)


def cached_gradients():
    rows = torch.tensor([2, 0, 2], device='cuda')
    target = torch.rand(3, 3, 128, 128, device='cuda')
    image = torch.rand_like(target)
    outer_grad = torch.tensor([0.3, -0.5, 2.0], device='cuda')
    for vgg in ('torch', 'cuda'):
        for scales in (1, 3):
            # Force a partial final chunk; every target ID is valid independently of chunk size.
            kw = dict(patch=32, num_scales=scales, n_samples=64, vgg=vgg, compile=False)
            m, ref = MultiScaleOTLoss(**kw).cuda(), ExpandedTargets(**kw).cuda()
            m._chunk_images = ref._chunk_images = lambda N, R: 2
            ref.vgg = m.vgg  # identical autotuned convolutions in both calls
            cache = m.targets(target)
            results = []
            for fn in (ref, m):
                x = image.clone().requires_grad_()
                torch.manual_seed(456)
                loss = fn(x, cache=cache, rows=rows)
                loss.backward(outer_grad)
                results.append((loss.detach(), x.grad))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            # The existing REMD scatter uses atomics; compare gradients relative to their scale, including
            # near-zero entries where a per-element relative tolerance is uninformative.
            a, b = results[0][1], results[1][1]
            rel = ((a - b).norm() / a.norm()).item()
            print(f'  {vgg}, scales={scales}: gradient relative L2 {rel:.2e}', flush=True)
            assert rel < 2e-5
            assert (a - b).abs().max() < 2e-4 * a.abs().max()
            torch.manual_seed(456)
            value = m(image, cache=cache, rows=rows)
            torch.testing.assert_close(value, results[1][0], rtol=0, atol=0)
    print('cached loss: matching values, weighted image gradients, and evaluation across scales/backends/chunks', flush=True)


def remd_backward():
    # Mixed directions, repeated matches (including all-to-one), zero vectors, unequal sets, and a scalar
    # expanded upstream gradient. Compare to the existing formula without changing nearest-pair selection.
    torch.manual_seed(789)
    for n, m, c in ((37, 53, 65), (1024, 1024, 64), (256, 256, 512)):
        x = torch.randn(3, n, c, device='cuda').relu().half().float()
        y = torch.randn(3, m, c, device='cuda').relu().half().float()
        x[:, 0] = 0
        y[:, 0] = 0
        nx, ny = x.norm(dim=2), y.norm(dim=2)
        i = torch.randint(n, (3, m), device='cuda')
        j = torch.randint(m, (3, n), device='cuda')
        i[1] = 1
        side = torch.tensor([True, False, False], device='cuda')
        for g in (torch.tensor([0.7, -0.3, 1.2], device='cuda'), torch.ones(1, device='cuda').expand(3)):
            expected = _remd_grad(g, x, y, nx, ny, i, j, side)
            for fn in (remd_grad_selected, torch.compile(remd_grad_selected, fullgraph=True)):
                actual = fn(g, x, y, nx, ny, i, j, side)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6)
                assert torch.isfinite(actual).all()
    print('selected REMD backward: both directions, collisions, zero vectors, and odd shapes match', flush=True)


def gradient_maxima():
    from ncaops.vgg_ops import _sampled_amax
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for mixed in (False, True):
            # Missing images, arbitrary ordering, a partial final block, and updates to existing maxima.
            rows = torch.tensor([0] * 19 + [250] * 17 + [490] * 11, device='cuda')
            if mixed:
                rows = rows[torch.randperm(len(rows), device='cuda')]
            g = torch.randn(len(rows), 65, device='cuda', dtype=dtype)
            for special in (False, True):
                if special:
                    g[0, 0], g[-1, -1] = float('nan'), float('inf')
                expected = torch.ones(5, device='cuda')
                actual = expected.clone()
                expected.scatter_reduce_(0, rows // 100, g.abs().float().amax(1), 'amax')
                _sampled_amax(actual, g, rows, 100)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    print('sampled gradient maxima: exact, including unordered rows, missing images, and NaN/Inf', flush=True)


def window_metadata():
    m = MultiScaleOTLoss(patch=32, num_scales=3, compile=False).cuda()
    # Compare to the former setup, including its random-number consumption.
    torch.manual_seed(12)
    size = torch.tensor([128 // 2 ** s for s in m.scale_of], device='cuda')
    hi = (128 - size + 1)[None].expand(3, -1)
    oy = (torch.rand(3, m.K, device='cuda') * hi).long().clamp_max(hi - 1)
    ox = (torch.rand(3, m.K, device='cuda') * hi).long().clamp_max(hi - 1)
    n = torch.arange(3, device='cuda')[:, None].expand(-1, m.K)
    f = (size // 32)[None].expand(3, -1)
    expected = [v.reshape(-1).int().contiguous() for v in (n, oy, ox, f)]
    for _ in range(2):  # cold and cached metadata
        torch.manual_seed(12)
        actual = m._windows(3, 128, 'cuda')
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = m._windows(3, 128, 'cuda')
    graph.replay()
    previous = captured[1].clone()
    graph.replay()
    assert not torch.equal(previous, captured[1]), 'graph replay must draw new crop positions'
    print('window metadata: identical draws, no capture-time host synchronization, fresh draws on replay', flush=True)


def single_scale():
    class GenericWindows(MultiScaleOTLoss):
        def _windows(self, N, R, device):
            # The former generic setup. Restore its unused crop draws so both implementations subsequently
            # sample the same feature positions; skipping these draws intentionally changes seeded sequences.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                size = torch.tensor([R // 2 ** s for s in self.scale_of], device=device)
                hi = (R - size + 1)[None]
                oy = (torch.rand(N, self.K, device=device) * hi).long().clamp_max(hi - 1)
                ox = (torch.rand(N, self.K, device=device) * hi).long().clamp_max(hi - 1)
                n = torch.arange(N, device=device)[:, None].expand(-1, self.K)
                f = (size // self.patch)[None].expand(N, -1)
                return [v.reshape(-1).int().contiguous() for v in (n, oy, ox, f)]

    torch.manual_seed(431)
    target = torch.rand(3, 3, 32, 32, device='cuda')
    image = torch.rand_like(target)
    rows = torch.tensor([2, 0, 2], device='cuda')
    outer_grad = torch.tensor([0.3, -0.5, 2.0], device='cuda')
    for vgg, acc16 in (('torch', False), ('cuda', False), ('cuda', True)):
        for mode, pairing in (('whole', 'pooled'), ('crops', 'pooled'), ('crops', 'paired')):
            kw = dict(patch=32, num_scales=1, n_samples=64, vgg=vgg, vgg_acc16=acc16,
                      target_mode=mode, pairing=pairing, compile=False)
            m, ref = MultiScaleOTLoss(**kw).cuda(), GenericWindows(**kw).cuda()
            ref.vgg = m.vgg
            m._chunk_images = ref._chunk_images = lambda N, R: 2
            before = torch.cuda.get_rng_state()
            cold, warm = m._windows(3, 32, 'cuda'), m._windows(3, 32, 'cuda')
            assert cold is warm
            assert torch.equal(before, torch.cuda.get_rng_state())
            for cached in ((False, True) if mode == 'whole' else (False,)):
                args = dict(cache=m.targets(target), rows=rows) if cached else dict(target=target, rows=rows)
                results = []
                for fn in (ref, m):
                    x = image.clone().requires_grad_()
                    torch.manual_seed(432)
                    value = fn(x, **args)
                    value.backward(outer_grad)
                    results.append((value.detach(), x.grad))
                torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                a, b = results[0][1], results[1][1]
                assert (a - b).norm() < 2e-5 * a.norm()
                torch.manual_seed(432)
                torch.testing.assert_close(m(image, **args), results[1][0], rtol=0, atol=0)
    print('single scale: identical values and matching weighted gradients for matched samples; all backends, target modes, '
          'partial chunks, and evaluation; no crop RNG draws', flush=True)


if __name__ == '__main__':
    sampling()
    cached_gradients()
    remd_backward()
    gradient_maxima()
    window_metadata()
    single_scale()
    print('all VGG optimization checks passed')
