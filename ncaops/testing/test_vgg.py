"""Validate the fused VGG16 features (ncaops/vgg_ops.py, both accumulate modes) against torchvision's VGG16 in fp32:
the relu1_1 ... relu5_1 maps and the image gradients for dense map gradients, sampled-position gradients and both, for
several image sizes. Errors are max |ours - fp32| / max |fp32|, printed next to torchvision's own error under bf16
autocast (in parentheses), the yardstick: the fused VGG must not be worse than 2x it.
Also checks the samples against the maps, flat positions, the normalized input, a backward that starts below relu5_1,
and determinism; --bench times the forward and forward + backward against torchvision (bf16 autocast, channels_last,
eager and torch.compile) at several batch sizes.
python ncaops/testing/test_vgg.py [--bench]
"""
import os, sys, argparse, time
import torch
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ncaops.vgg_ops import VGG16Features, LAYERS  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = 'cuda'
ref_net = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:max(LAYERS) + 1].to(dev).eval().requires_grad_(False)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]


def ref_features(img, bf16=False):
    x, out = (img - MEAN) / STD, []
    with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
        for i, layer in enumerate(ref_net):
            x = layer(x)
            if i in LAYERS:
                out.append(x.float())
    return out


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def check(vgg, N, H, W, seed=0):
    torch.manual_seed(seed)
    img = torch.rand(N, 3, H, W, device=dev)
    img = (img + torch.nn.functional.avg_pool2d(img, 9, 1, 4)) / 2  # some spatial structure
    gm = [torch.randn(N, c, H >> l, W >> l, device=dev) * 1e-3 for l, c in enumerate((64, 128, 256, 512, 512))]
    pos = [torch.rand(N, (H >> l) * (W >> l), device=dev).topk(min(64, (H >> l) * (W >> l)), 1).indices for l in range(5)]
    gs = [torch.randn(N, p.shape[1], c, device=dev) * 1e-2 for p, c in zip(pos, (64, 128, 256, 512, 512))]

    def ref_grad(bf16, use_maps, use_samples):
        x = img.clone().requires_grad_()
        loss = 0.0
        for f, g, p, s in zip(ref_features(x, bf16), gm, pos, gs):
            if use_maps:
                loss = loss + (f * g).sum()
            if use_samples:
                fs = f.flatten(2).transpose(1, 2).gather(1, p[..., None].expand(-1, -1, f.shape[1]))
                loss = loss + (fs * s).sum()
        loss.backward()
        return x.grad

    def our_grad(use_maps, use_samples, flat=False):
        x = img.clone().requires_grad_()
        if flat:  # the same positions as rows of the flattened maps, all images in one set
            rows = [p + torch.arange(N, device=dev)[:, None] * ((H >> l) * (W >> l)) for l, p in enumerate(pos)]
            maps, samples = vgg(x, positions=[r.reshape(1, -1) for r in rows], flat=True)
            samples = [s.reshape(N, -1, s.shape[-1]) for s in samples]
        else:
            maps, samples = vgg(x, positions=pos)
        loss = 0.0
        for f, g, s_out, s in zip(maps, gm, samples, gs):
            if use_maps:
                loss = loss + (f.float() * g).sum()
            if use_samples:
                loss = loss + (s_out * s).sum()
        loss.backward()
        return x.grad, maps, samples

    with torch.no_grad():
        ref, half = ref_features(img), ref_features(img, bf16=True)
    line = f'{"fp16" if vgg.acc16 else "fp32"} accumulation, N={N} {H}x{W}: features'
    gd, maps, samples = our_grad(True, False)
    for l in range(5):
        e, eh = rel(maps[l], ref[l]), rel(half[l], ref[l])
        line += f' {e:.1e} ({eh:.1e})'
        assert e < max(2 * eh, 1e-2), f'relu{l + 1}_1 features'
        f = maps[l].float().flatten(2).transpose(1, 2)
        assert torch.equal(samples[l], f.gather(1, pos[l][..., None].expand(-1, -1, f.shape[2]))), 'samples != maps at positions'
        assert maps[l].is_contiguous(memory_format=torch.channels_last) and maps[l].dtype == torch.float16
    line += '; image gradient'
    for name, um, us in (('maps', True, False), ('samples', False, True), ('both', True, True)):
        g = gd if name == 'maps' else our_grad(um, us)[0]
        e, eh = rel(g, ref_grad(False, um, us)), rel(ref_grad(True, um, us), ref_grad(False, um, us))
        line += f' {name} {e:.1e} ({eh:.1e})'
        assert e < max(2 * eh, 2e-2), f'image gradient ({name})'
    g2 = our_grad(True, True)[0]
    assert torch.equal(g2, our_grad(True, True)[0]), 'backward not deterministic'
    assert torch.equal(g2, our_grad(True, True, flat=True)[0]), 'flat positions != per-image positions'
    # normalized input: (N, H, W, 3) normalised, gradient in the same layout; against the reference, like the images
    xn = ((img - MEAN) / STD).permute(0, 2, 3, 1).contiguous().requires_grad_()
    mn = vgg(xn, normalized=True)
    assert all(rel(u, v) < max(2 * rel(h, v), 1e-2) for u, h, v in zip(mn, half, ref)), 'normalized input features'
    sum((f.float() * g).sum() for f, g in zip(mn, gm)).backward()
    e, eh = rel(xn.grad.permute(0, 3, 1, 2) / STD, ref_grad(False, True, False)), rel(ref_grad(True, True, False), ref_grad(False, True, False))
    line += f'; normalized input {e:.1e} ({eh:.1e})'
    assert e < max(2 * eh, 2e-2), 'normalized input gradient'
    # gradient from relu3_1 only: the layers above are skipped
    x = img.clone().requires_grad_()
    (vgg(x)[2].float() * gm[2]).sum().backward()
    xr, xh = img.clone().requires_grad_(), img.clone().requires_grad_()
    (ref_features(xr)[2] * gm[2]).sum().backward()
    (ref_features(xh, bf16=True)[2] * gm[2]).sum().backward()
    e, eh = rel(x.grad, xr.grad), rel(xh.grad, xr.grad)
    line += f'; relu3_1 only {e:.1e} ({eh:.1e})'
    assert e < max(2 * eh, 2e-2)
    print(line, flush=True)


def bench():
    """VGG alone at 256^2: forward (no grad), and forward + backward from gradients on all 5 maps, fused (both
    accumulations) against torchvision VGG16 (channels_last, bf16 autocast) eager and under torch.compile()."""
    torch.backends.cudnn.allow_tf32 = True
    net = ref_net.to(memory_format=torch.channels_last)
    vggs = [VGG16Features(acc).to(dev) for acc in ('fp32', 'fp16')]

    def tv(x):
        out = []
        with torch.autocast('cuda', torch.bfloat16):
            h = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
            for i, layer in enumerate(net):
                h = layer(h)
                if i in LAYERS:
                    out.append(h)
        return out
    tv_compiled = torch.compile(tv)
    for N in (16, 32, 64, 128):
        img = torch.rand(N, 3, 256, 256, device=dev)
        gm = [torch.randn(N, c, 256 >> l, 256 >> l, device=dev).contiguous(memory_format=torch.channels_last) * 1e-3
              for l, c in enumerate((64, 128, 256, 512, 512))]
        gh, gb = [g.half() for g in gm], [g.to(torch.bfloat16) for g in gm]

        def run(f, g, backward):
            if not backward:
                with torch.no_grad():
                    f(img)
                return
            torch.autograd.backward(f(img.clone().requires_grad_()), g)

        res = []
        for f, g in ((vggs[0], gh), (vggs[1], gh), (tv, gb), (tv_compiled, gb)):
            for backward in (False, True):
                for _ in range(2):
                    run(f, g, backward)
                torch.cuda.synchronize(); t = time.time()
                for _ in range(5):
                    run(f, g, backward)
                torch.cuda.synchronize(); res.append(1000 * (time.time() - t) / 5)
        print(f'bench N={N:3d} 256^2, forward / forward + backward (ms): fused fp32 acc. {res[0]:6.1f} / {res[1]:6.1f}, '
              f'fused fp16 acc. {res[2]:6.1f} / {res[3]:6.1f}, torchvision bf16 eager {res[4]:6.1f} / {res[5]:6.1f}, '
              f'compiled {res[6]:6.1f} / {res[7]:6.1f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    for acc in ('fp32', 'fp16'):
        vgg = VGG16Features(acc).to(dev)
        check(vgg, 2, 64, 64)
        check(vgg, 3, 128, 80, seed=1)
        check(vgg, 2, 256, 256, seed=2)
    print('all VGG checks passed')
    if a.bench:
        bench()
