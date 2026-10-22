"""Validate the multi-scale OT loss (loss_ms.py) against Cells2Pixels' AppearanceLoss (../Cells2Pixels, or $C2P):
(1) one scale with every feature position used (no sampling): values and image gradients equal to fp32 rounding;
(2) the patch builder (windows, 2 x 2 means, normalisation) and its backward against F.interpolate + autograd;
(3) the full loss (3 scales, 1024 samples, whole target) against AppearanceLoss over seeds (statistical);
(4) the crops modes run, their gradients are deterministic in the builder and finite;
(5) vgg='cuda' (ncaops/vgg_ops.py, both accumulations) against vgg='torch' in fp32: every position used: loss and
image gradient within the fp16 / bf16 error; with sampling, every mode: the mean over seeds within the noise.
python ncaops/testing/test_loss_ms.py [--bench]
"""
import os, sys, argparse, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
C2P = os.environ.get('C2P', os.path.abspath(os.path.join(HERE, '..', '..', '..', 'Cells2Pixels')))
sys.path.insert(1, C2P)
from loss_ms import MultiScaleOTLoss  # noqa: E402

dev = 'cuda'


def reference(num_scales, patch, R, path):
    from losses.appearance_loss import AppearanceLoss
    return AppearanceLoss(target_image_size=(R, R), patch_size=(patch, patch), num_scales=num_scales, relative_scale=1,
                          target_images_path={'rgb': path}, total_channels=3, output_channels={'rgb': [0, 1, 2]},
                          device=dev)


def exact(path):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    R, P = 256, 64
    ref = reference(1, P, R, path)
    ref.style_loss_fn.n_samples = 10 ** 6  # every position: no sampling
    target = ref.target_images  # (1, 3, R, R)
    ours = MultiScaleOTLoss(patch=P, num_scales=1, n_samples=10 ** 6).to(dev)
    torch.manual_seed(0)
    img = (target.roll(37, 2) * 0.8 + 0.1 * torch.rand_like(target)).repeat(2, 1, 1, 1)
    img[1] = torch.rand_like(img[1])
    a = img.clone().requires_grad_()
    la = torch.stack([ref({'rendered_images': a[i:i + 1]}, return_summary=False)[0] for i in range(2)])
    la.sum().backward()
    b = img.clone().requires_grad_()
    lb = ours(b, target.expand(2, -1, -1, -1))
    lb.sum().backward()
    rel = ((a.grad - b.grad).abs().max() / a.grad.abs().max()).item()
    print(f'(1) one scale, all positions: loss {la.tolist()} vs {lb.tolist()}, gradient rel. max diff {rel:.1e}')
    assert (la - lb).abs().max() < 1e-4 * la.abs().max() and rel < 1e-3


def builder():
    m = MultiScaleOTLoss(patch=32, num_scales=3).to(dev)
    N, R = 3, 128
    img = torch.rand(N, 3, R, R, device=dev, requires_grad=True)
    spec = m._windows(N, R, dev)
    x = m._build(img.detach(), spec, 32, torch.float32)
    refs = []
    for p in range(len(spec[0])):
        n, oy, ox, f = (int(v[p]) for v in spec)
        win = img[n:n + 1, :, oy:oy + 32 * f, ox:ox + 32 * f]
        refs.append(((F.interpolate(win, size=(32, 32), mode='bilinear', align_corners=False) - m.mean[:, None, None]) * m.istd[:, None, None]))
    ref = torch.cat(refs)
    g = torch.randn_like(ref)
    (ref * g).sum().backward()
    gb = m._build_backward(g.contiguous(memory_format=torch.channels_last), spec, N, R, m.K)
    e1, e2 = (x - ref).abs().max().item(), ((gb - img.grad).abs().max() / img.grad.abs().max()).item()
    print(f'(2) patch builder: forward max diff {e1:.1e}, backward rel. max diff {e2:.1e} (factors {sorted(set(spec[3].tolist()))})')
    assert e1 < 1e-5 and e2 < 1e-5


def statistical(path, seeds=8):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    R, P = 512, 128
    ref = reference(3, P, R, path)
    target = ref.target_images
    ours = MultiScaleOTLoss(patch=P, num_scales=3).to(dev)
    torch.manual_seed(0)
    img = (target.roll((91, 33), (2, 3)) * 0.9 + 0.05)
    va, vb = [], []
    for s in range(seeds):
        torch.manual_seed(s)
        with torch.no_grad():
            va.append(ref({'rendered_images': img}, return_summary=False)[0].item())
        vb.append(ours(img, target).item())
    va, vb = torch.tensor(va), torch.tensor(vb)
    print(f'(3) 3 scales, {R}^2 / {P}^2, whole target: Cells2Pixels {va.mean():.4f} +- {va.std():.4f}, ours {vb.mean():.4f} +- {vb.std():.4f}')
    assert (va.mean() - vb.mean()).abs() < 3 * (va.std() ** 2 / seeds + vb.std() ** 2 / seeds).sqrt() + 1e-3 * va.mean()


def modes(path):
    from PIL import Image
    import numpy as np
    R, P = 256, 64
    t = torch.from_numpy(np.array(Image.open(path).convert('RGB').resize((R, R), Image.BICUBIC))).permute(2, 0, 1).float().div(255)
    target = t[None].repeat(4, 1, 1, 1).to(dev)
    img = torch.rand(4, 3, R, R, device=dev)
    for mode, pairing in (('whole', 'pooled'), ('crops', 'pooled'), ('crops', 'paired')):
        m = MultiScaleOTLoss(patch=P, num_scales=3, target_mode=mode, pairing=pairing).to(dev)
        x = img.clone().requires_grad_()
        loss = m(x, target)
        loss.sum().backward()
        self_loss = m(target, target)
        print(f'(4) {mode:5s} {pairing:6s}: loss of noise {loss.mean():.3f}, of the target itself {self_loss.mean():.3f}, '
              f'gradient finite {bool(torch.isfinite(x.grad).all())}')
        assert torch.isfinite(x.grad).all() and (self_loss < loss).all()
    cache = MultiScaleOTLoss(patch=P, num_scales=3).to(dev)
    c = cache.targets(target)
    print(f'    whole-target cache: {sum(f.numel() * f.element_size() for fs in c for f in fs) / 4 / 2 ** 20:.1f} MB per target')
    cache(img, cache=c)


def cuda_vgg(path, seeds=8):
    from PIL import Image
    import numpy as np
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    R, P = 256, 64
    t = torch.from_numpy(np.array(Image.open(path).convert('RGB').resize((R, R), Image.BICUBIC))).permute(2, 0, 1).float().div(255)
    target = t[None].repeat(3, 1, 1, 1).to(dev)
    torch.manual_seed(0)
    img = (target.roll(37, 2) * 0.8 + 0.1 * torch.rand_like(target))
    img[1] = torch.rand_like(img[1])
    grads = {}
    for name, kw in (('torch', {}), ('cuda', dict(vgg='cuda')), ('cuda acc16', dict(vgg='cuda', vgg_acc16=True))):
        m = MultiScaleOTLoss(patch=P, num_scales=1, n_samples=10 ** 6, **kw).to(dev)  # every position: no sampling
        x = img.clone().requires_grad_()
        loss = m(x, target)
        loss.sum().backward()
        grads[name] = (loss.detach(), x.grad)
    lr, gr = grads['torch']
    for name in ('cuda', 'cuda acc16'):
        l, g = grads[name]
        el, eg = ((l - lr).abs() / lr).max().item(), ((g - gr).norm() / gr.norm()).item()
        print(f'(5) {name:10s}, one scale, all positions: loss rel. diff {el:.1e}, image gradient rel. L2 diff {eg:.1e}')
        assert el < 1e-2 and eg < 0.2
    torch.backends.cudnn.allow_tf32 = True
    for mode, pairing in (('whole', 'pooled'), ('crops', 'pooled'), ('crops', 'paired')):
        vals = {}
        for name, kw in (('torch', {}), ('cuda', dict(vgg='cuda'))):
            torch._dynamo.reset()
            m = MultiScaleOTLoss(patch=P, num_scales=3, target_mode=mode, pairing=pairing, **kw).to(dev)
            v = []
            for s in range(seeds):
                torch.manual_seed(s)
                x = img.clone().requires_grad_()
                loss = m(x, target)
                loss.sum().backward()
                assert torch.isfinite(x.grad).all()
                v.append(loss.detach())
            vals[name] = torch.stack(v)  # (seeds, images)
        a, b = vals['torch'], vals['cuda']
        tol = 3 * (a.var(0) / seeds + b.var(0) / seeds).sqrt() + 1e-2 * a.mean(0)
        print(f'(5) {mode:5s} {pairing:6s} 3 scales, sampled: torch {a.mean(0).tolist()} vs cuda {b.mean(0).tolist()} '
              f'(tolerance {tol.tolist()})')
        assert ((a.mean(0) - b.mean(0)).abs() < tol).all()


def bench():
    from PIL import Image  # noqa: F401
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for R, P in ((1024, 256), (512, 128), (256, 64)):
        N = 32
        img = torch.rand(N, 3, R, R, device=dev, requires_grad=True)
        target = torch.rand(N, 3, R, R, device=dev)
        for bf16 in (False, True):
            for mode, pairing in (('whole', 'pooled'), ('crops', 'pooled'), ('crops', 'paired')):
                torch._dynamo.reset()  # a compiled loss per configuration (else the recompile limit falls back to eager)
                m = MultiScaleOTLoss(patch=P, num_scales=3, target_mode=mode, pairing=pairing, vgg_bf16=bf16).to(dev)
                cache = m.targets(target[:8]) if mode == 'whole' and R <= 512 else None
                f = lambda: m(img, target).sum().backward()
                for _ in range(2):
                    f()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
                ms = 1000 * (time.time() - t0) / 3
                line = f'bench {R}^2 / {P}^2 N={N} {"bf16" if bf16 else "fp32"} {mode:5s} {pairing:6s}: {ms:7.1f} ms ({ms / N:5.2f} per image), peak {torch.cuda.max_memory_allocated() / 2 ** 30:4.1f} GB'
                if cache is not None:
                    g = lambda: m(img[:8], cache=cache).sum().backward()
                    g(); torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(3):
                        g()
                    torch.cuda.synchronize()
                    line += f'; cached whole targets {1000 * (time.time() - t0) / 3 / 8:5.2f} ms per image'
                print(line, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    path = os.path.join(C2P, 'data/textures_hr/bubbles.jpg')
    exact(path)
    builder()
    statistical(path)
    modes(path)
    cuda_vgg(path)
    print('all multi-scale loss checks passed')
    if a.bench:
        bench()
