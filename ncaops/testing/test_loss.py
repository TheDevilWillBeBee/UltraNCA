"""Check compiled VGG pooling and tiled REMD against PyTorch, including ties and non-finite inputs.
python ncaops/testing/test_loss.py
"""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from loss import _Pool2d, _REMD


def test():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        pool = torch.compile(_Pool2d.apply, fullgraph=True)
        for layout in (torch.contiguous_format, torch.channels_last):
            # ReLU activations include many ties; also check the first maximum and last NaN choices.
            x = torch.randn(2, 16, 12, 18, device='cuda', dtype=dtype).relu().contiguous(memory_format=layout)
            x[0, 0, :2, :2] = float('nan')
            x[0, 1, :2, :2] = -float('inf')
            x[0, 2, :2, :2] = float('inf')
            x[0, 3, :2, :2] = 1
            x.requires_grad_()
            ref = F.max_pool2d(x, 2)
            y = pool(x)
            torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)
            g = torch.randn_like(ref)
            dx, = torch.autograd.grad(y, x, g)
            dx_ref, = torch.autograd.grad(ref, x, g)
            torch.testing.assert_close(dx, dx_ref, rtol=0, atol=0)
            print(f'pool {dtype} {layout}: exact output and gradient')
    test_remd()
    print('all passed')


def test_remd():
    if not hasattr(torch.library, 'triton_op'):
        print('REMD tiling requires torch.library.triton_op; the loss uses the eager formulation on this version')
        return
    from loss_triton import nearest
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for n, m, c in ((1024, 1024, 64), (256, 256, 512), (37, 53, 65)):
            # Exactly representable operands isolate reduction/tie handling from TF32 quantization differences
            # between cuBLAS algorithms (odd shapes can select an IEEE implementation).
            x = torch.randint(-8, 9, (3, n, c), device='cuda').float() / 8
            y = torch.randint(-8, 9, (3, m, c), device='cuda').float() / 8
            y[:, 1] = y[:, 0]
            x[:, 1] = x[:, 0]
            nx, ny = x.norm(dim=2), y.norm(dim=2)
            sim = x @ y.transpose(1, 2) / (nx[:, :, None] * ny[:, None, :] + 1e-10)
            vy, j = sim.max(2); vx, i = sim.max(1)
            out = nearest(x, y, nx, ny)
            for got, ref in zip(out, (vy, j, vx, i)):
                torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)
            for a, b in zip(out, nearest(x, y, nx, ny)):
                assert torch.equal(a, b), 'nearest pairs must be deterministic'
            for a, b in zip(out, nearest(x[1:2], y[1:2], nx[1:2], ny[1:2])):
                assert torch.equal(a[1:2], b), 'images must be independent'

        # Compiled forward and gradients vs the eager REMD, for continuous VGG-like features and both layouts.
        compiled = torch.compile(_REMD.apply, fullgraph=True)
        for transposed in (False, True):
            x = torch.randn(3, 1024, 64, device='cuda').relu()
            y = torch.randn_like(x).relu()
            if transposed:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
                y = y.transpose(1, 2).contiguous().transpose(1, 2)
            x.requires_grad_()
            ref, out = _REMD.apply(x, y), compiled(x, y)
            torch.testing.assert_close(out, ref, atol=2e-6, rtol=2e-6)
            g = torch.tensor([0.7, -0.2, 1.3], device='cuda')
            dx, = torch.autograd.grad(out, x, g)
            dx_ref, = torch.autograd.grad(ref, x, g)
            torch.testing.assert_close(dx, dx_ref, atol=2e-6, rtol=2e-4)
        print(f'REMD tf32={tf32}: values, gradients, determinism and independent images ok')

    # All ties, and NaNs crossing tile boundaries: first maximum / first NaN must win.
    for val in (0., 1., float('nan'), float('inf')):
        x = torch.full((2, 73, 64), val, device='cuda')
        y = torch.ones(2, 139, 64, device='cuda')
        nx, ny = x.norm(dim=2), y.norm(dim=2)
        a, j, b, i = nearest(x, y, nx, ny)
        assert (i == 0).all() and (j == 0).all()
        if val == 0 or val == 1:
            torch.testing.assert_close(a, torch.full_like(a, val), rtol=0, atol=0)
            torch.testing.assert_close(b, torch.full_like(b, val), rtol=0, atol=0)
        else:
            assert a.isnan().all() and b.isnan().all()
    x = torch.ones(2, 73, 64, device='cuda'); x[:, 65] = float('nan')
    y = torch.ones(2, 139, 64, device='cuda'); y[:, 129] = float('nan')
    a, j, b, i = nearest(x, y, x.norm(dim=2), y.norm(dim=2))
    expected_j = torch.full_like(j, 129); expected_j[:, 65] = 0
    expected_i = torch.full_like(i, 65); expected_i[:, 129] = 0
    assert a.isnan().all() and b.isnan().all()
    assert torch.equal(j, expected_j) and torch.equal(i, expected_i)
    print('REMD ties and non-finite inputs: ok')


if __name__ == '__main__':
    test()
