"""Cost of the single-scale OT texture loss at 256^2 (VGG16 relu1_1 ... relu5_1 of the full render, relaxed EMD +
mean / covariance terms on 1024 random positions per layer; config_lppn.yml's default loss) with its backward to the
images, for N images, per implementation and VGG precision:
  eager    plain PyTorch: torchvision VGG16 (NCHW), the Cells2Pixels OT terms (dense cosine matrix, min over both axes),
           chunks of 32 images (gradients accumulated: 128 x 256^2 of fp32 activations does not fit in 24 GB)
  compile  the same per-chunk function under torch.compile() (default mode)
  current  loss_ms.MultiScaleOTLoss(patch=256, num_scales=1) with the torchvision VGG (compiled VGG + OT,
           channels_last, tiled REMD, image gradient computed in the forward, chunks bounded in memory)
  cuda     the same with vgg='cuda' (ncaops/vgg_ops.py: fused CUTLASS VGG, fp16 activations, fp32 accumulation;
           precision fp16 below), compiled OT terms
  cuda16   the same with fp16 accumulation in the forward convolutions (vgg_acc16)
--precision applies to eager / compile / current (fp32 = TF32 convolutions, or bf16 autocast).
Targets: the target features are computed at every call (targets that change every batch), T distinct targets
(default T = N: one per image), image n against target n % T; --cached: 8 fixed targets whose features are computed
once (the earlier setting: MultiScaleOTLoss.targets()), image n against target n % 8.
python benchmark_vgg.py [--impl eager compile current cuda cuda16] [--N 8 16 32 64 128] [--precision fp32 bf16]
                        [--targets T] [--cached] [--profile]
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from loss_ms import MultiScaleOTLoss, LAYERS

ap = argparse.ArgumentParser()
ap.add_argument('--impl', nargs='+', default=['eager', 'compile', 'current', 'cuda', 'cuda16'])
ap.add_argument('--N', nargs='+', type=int, default=[8, 16, 32, 64, 128])
ap.add_argument('--precision', nargs='+', default=['fp32', 'bf16'])
ap.add_argument('--profile', action='store_true', help='kernel-time breakdown (convolutions vs the rest) per run')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--targets', type=int, default=0, help='distinct targets (0: N, one per image)')
ap.add_argument('--cached', action='store_true', help='8 fixed targets, features computed once')
a = ap.parse_args()
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
R, NS = 256, 1024


class PlainOT(torch.nn.Module):
    """The straightforward implementation (Cells2Pixels losses/appearance_loss.OptimalTransportLoss, one scale)."""

    def __init__(self, bf16):
        super().__init__()
        self.vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:max(LAYERS) + 1].eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[:, None, None])
        self.bf16 = bf16

    def features(self, x):
        x, out = (x - self.mean) / self.std, []
        with torch.autocast('cuda', torch.bfloat16, enabled=self.bf16):
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in LAYERS:
                    out.append(x.float().flatten(2))  # (b, c, positions)
        return out

    @staticmethod
    def sample(f, n):  # (b, c, m) -> (b, n, c), without replacement
        b, c, m = f.shape
        if m <= n:
            return f.transpose(1, 2)
        idx = torch.rand(b, m, device=f.device).argsort(1)[:, None, :n]
        return f.gather(2, idx.expand(-1, c, -1)).transpose(1, 2)

    def forward(self, img, targets):  # img (b, 3, H, W); targets: per layer (b, c, m) -> per-image loss (b,)
        loss = 0.0
        for fx, fy in zip(self.features(img), targets):
            n = min(fx.shape[2], NS)
            x, y = self.sample(fx, n), self.sample(fy.float(), n)
            nx, ny = x.norm(dim=2, keepdim=True), y.norm(dim=2, keepdim=True)
            d = 1 - x @ y.transpose(1, 2) / (nx * ny.transpose(1, 2) + 1e-10)
            remd = torch.maximum(d.min(1)[0].mean(1), d.min(2)[0].mean(1))
            mx, my = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
            cx = (x - mx).transpose(1, 2) @ (x - mx) / (n - 1)
            cy = (y - my).transpose(1, 2) @ (y - my) / (n - 1)
            loss = loss + remd + (mx - my).abs().mean((1, 2)) + (cx - cy).abs().mean((1, 2))
        return loss


def make(impl, prec, N):
    bf16 = prec == 'bf16'
    T = 8 if a.cached else a.targets or N
    targets = torch.rand(T, 3, R, R, device=dev)
    if impl in ('current', 'cuda', 'cuda16'):
        kw = dict(vgg='cuda', vgg_acc16=impl == 'cuda16') if impl != 'current' else dict(vgg_bf16=bf16)
        m = MultiScaleOTLoss(patch=R, num_scales=1, target_mode='whole', **kw).to(dev)
        cache = m.targets(targets) if a.cached else None

        def run(img):
            rows = torch.arange(len(img), device=dev) % T
            if a.cached:
                m(img, cache=cache, rows=rows).sum().backward()
            else:
                m(img, targets, rows=rows).sum().backward()
        return run
    m = PlainOT(bf16).to(dev)
    if a.cached:
        with torch.no_grad():
            tf = [f.to(torch.bfloat16) for f in m.features(targets)]  # cached target features (b0, c, m), as `current`
    fn = torch.compile(m) if impl == 'compile' else m
    features = torch.compile(m.features) if impl == 'compile' else m.features  # targets: forward only

    def run(img):
        for c0 in range(0, len(img), 32):
            x = img[c0:c0 + 32]
            rows = torch.arange(c0, c0 + len(x), device=dev) % T
            if a.cached:
                tfr = [f[rows] for f in tf]
            else:
                with torch.no_grad():
                    tfr = [f.to(torch.bfloat16) for f in features(targets[rows])]
            fn(x, tfr).sum().backward()
    return run


def bench(impl, prec, N):
    torch._dynamo.reset()
    torch.manual_seed(0)
    run = make(impl, prec, N)
    img = torch.rand(N, 3, R, R, device=dev).requires_grad_()
    for _ in range(2):
        run(img)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(a.reps):
        img.grad = None
        t = time.time(); run(img); torch.cuda.synchronize(); ts.append(1000 * (time.time() - t))
    ms = float(np.median(ts))
    line = (f'{impl:8s} {prec:5s} N={N:4d}: {ms:8.1f} ms  ({ms / N:6.3f} ms per image), peak '
            f'{torch.cuda.max_memory_allocated() / 2 ** 30:5.1f} GB')
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run(img); torch.cuda.synchronize()
        conv = other = 0.0
        for e in prof.key_averages():  # VGG convolutions: cuDNN's kernels and the fused VGG's CUTLASS ones
            k, t = e.key.lower(), e.self_device_time_total / 1000
            if ('fprop' in k or 'dgrad' in k or 'xmma' in k or 'cutlass__5x_cudnn' in k or 'implicitgemmconvolution' in k) \
                    and 'triton' not in k:
                conv += t
            else:
                other += t
        line += f'; GPU {conv + other:7.1f} ms = convolutions {conv:7.1f} + other {other:7.1f}'
    print(line, flush=True)


for prec in a.precision:
    for impl in a.impl:
        p = prec
        if impl.startswith('cuda'):  # fp16 activations whatever --precision says: run once
            if prec != a.precision[0]:
                continue
            p = 'fp16'
        for N in a.N:
            try:
                bench(impl, p, N)
            except torch.OutOfMemoryError:
                print(f'{impl:8s} {p:5s} N={N:4d}: out of memory', flush=True)
                torch.cuda.empty_cache()
