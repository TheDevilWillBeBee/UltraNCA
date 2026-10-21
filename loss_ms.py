"""Multi-scale OT texture loss of Cells2Pixels (losses/appearance_loss.AppearanceLoss: RGB, no pseudo targets) for N
images with one target image each, batched over the images.

Per image and scale s = 0 .. num_scales - 1: the image (R x R) at scale 0, or int(2 s) random crops of R / 2^s at scale
s >= 1 (crops: other numbers of crops per scale), each downsampled to patch x patch (bilinear, align_corners=False, no antialiasing: a factor f >= 2 takes the
mean of 2 x 2 pixels), then VGG16 relu1_1 ... relu5_1. Per layer: relaxed EMD (cosine) + L1 of the means and of the
covariances between n = min(n_x, n_y, n_samples) random positions (without replacement) of the image's features and
of the target's. The image's loss is the sum over layers, the mean over a scale's terms, the mean over the scales.
Target side (target_mode):
- 'whole': the target resized to patch * 2^s at scale s (the original): one term per patch, each against the whole
  target. For fixed targets, targets() computes this once and forward() takes the cache; otherwise VGG runs at
  patch * 2^s on the target at every call.
- 'crops': the target cut and downsampled like the image (its own random windows; VGG at the patch resolution only,
  for targets that change at every call). pairing 'pooled': one term per scale between the features of all the
  image's crops of the scale and those of all the target's crops of the scale (both sides pooled: the crops of a
  scale must match the target as a set, not one by one); 'paired': one term per patch, patch k against target crop k.
  Scale 0 (one patch) is the same in every mode.
forward() returns the per-image loss (N,) and is differentiable w.r.t. the images: VGG runs forward and backward in
chunks of images (memory) during the forward pass, whose backward then scales the stored image gradient.
Sampling without replacement uses top-k of random keys; above 64 n positions (whole-target maps of the highest scale)
it samples with replacement (expected duplicates < 1 % of the samples), as Cells2Pixels' batched loss does.
"""
import logging
import torch
import torchvision
import triton
import triton.language as tl

from loss import _REMD, _moments

# compiling the REMD's Triton kernel logs one warning per inline-asm TF32 rounding (harmless; torch resets log levels)
logging.getLogger('torch._higher_order_ops.triton_kernel_wrap').addFilter(lambda r: 'TTIR mutation analysis' not in r.getMessage())

LAYERS = (1, 6, 11, 18, 25)  # relu1_1 ... relu5_1


@triton.jit
def _build_fwd(img, out, pn, poy, pox, pf, mean, istd, R, P, BLOCK: tl.constexpr):
    """Patch p: image pn[p], window offset (poy, pox), factor pf (1: copy; even: mean of 2 x 2 pixels at f i + f/2 - 1),
    normalised, written (patches, P, P, 3)."""
    p, blk = tl.program_id(0), tl.program_id(1)
    n, oy, ox, f = tl.load(pn + p), tl.load(poy + p), tl.load(pox + p), tl.load(pf + p)
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    m = idx < P * P
    y, x = idx // P, idx % P
    off = tl.maximum(f // 2 - 1, 0)
    d = (f > 1).to(tl.int32)  # second tap (f = 1: the same pixel four times)
    r0, c0 = oy + f * y + off, ox + f * x + off
    for ch in tl.static_range(3):
        base = img + (n * 3 + ch).to(tl.int64) * R * R
        v = (tl.load(base + r0 * R + c0, mask=m) + tl.load(base + r0 * R + c0 + d, mask=m)
             + tl.load(base + (r0 + d) * R + c0, mask=m) + tl.load(base + (r0 + d) * R + c0 + d, mask=m)) * 0.25
        v = (v - tl.load(mean + ch)) * tl.load(istd + ch)
        tl.store(out + (p.to(tl.int64) * P * P + idx) * 3 + ch, v.to(out.dtype.element_ty), mask=m)


@triton.jit
def _build_bwd(gout, gimg, poy, pox, pf, istd, K, R, P, BLOCK: tl.constexpr):
    """Image gradient: per pixel, the sum over the image's K patches (fixed order) of the patch gradients that read it."""
    n, blk = tl.program_id(0), tl.program_id(1)
    idx = blk * BLOCK + tl.arange(0, BLOCK)
    m = idx < R * R
    Y, X = idx // R, idx % R
    a0 = tl.zeros((BLOCK,), tl.float32)
    a1 = tl.zeros((BLOCK,), tl.float32)
    a2 = tl.zeros((BLOCK,), tl.float32)
    for k in range(K):
        p = n * K + k
        oy, ox, f = tl.load(poy + p), tl.load(pox + p), tl.load(pf + p)
        off = tl.maximum(f // 2 - 1, 0)
        u, v = Y - oy - off, X - ox - off
        ok = m & (u >= 0) & (v >= 0) & (u % f < 2) & (v % f < 2) & (u // f < P) & (v // f < P)
        w = tl.where(f > 1, 0.25, 1.0)
        src = gout + (p.to(tl.int64) * P * P + (u // f) * P + v // f) * 3
        a0 += w * tl.load(src, mask=ok, other=0.).to(tl.float32)
        a1 += w * tl.load(src + 1, mask=ok, other=0.).to(tl.float32)
        a2 += w * tl.load(src + 2, mask=ok, other=0.).to(tl.float32)
    base = gimg + n.to(tl.int64) * 3 * R * R + idx
    tl.store(base, a0 * tl.load(istd), mask=m)
    tl.store(base + R * R, a1 * tl.load(istd + 1), mask=m)
    tl.store(base + 2 * R * R, a2 * tl.load(istd + 2), mask=m)


def _sample(f, n):
    """n random positions per item of f (b, M, c) -> (b, n, c); all of them (any order) if M <= n."""
    b, M, c = f.shape
    if M <= n:
        return f
    if M > 64 * n:
        idx = torch.randint(0, M, (b, n), device=f.device)
    else:
        idx = torch.rand(b, M, device=f.device).topk(n, 1, sorted=False).indices
    return f.gather(1, idx[..., None].expand(-1, -1, c))


class MultiScaleOTLoss(torch.nn.Module):
    def __init__(self, patch=256, num_scales=3, n_samples=1024, target_mode='whole', pairing='pooled', vgg_bf16=False,
                 chunk_gb=6.0, compile=True, crops=None):
        super().__init__()
        assert target_mode in ('whole', 'crops') and pairing in ('pooled', 'paired')
        self.patch, self.num_scales, self.n, self.mode, self.pairing, self.bf16 = patch, num_scales, n_samples, target_mode, pairing, vgg_bf16
        self.chunk_gb = chunk_gb
        self.vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:max(LAYERS) + 1].eval().requires_grad_(False)
        self.vgg = self.vgg.to(memory_format=torch.channels_last)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('istd', 1 / torch.tensor([0.229, 0.224, 0.225]))
        # patches per scale: Cells2Pixels' 1, 2, 4, ... (int(2 s)) unless given
        self.crops = list(crops) if crops else [1 if s == 0 else max(1, int(2.0 * s)) for s in range(num_scales)]
        assert len(self.crops) == num_scales and self.crops[0] == 1, 'scale 0 is the whole image'
        self.K = sum(self.crops)
        self.scale_of = [s for s in range(num_scales) for _ in range(self.crops[s])]
        self.pooled = target_mode == 'crops' and pairing == 'pooled'
        # torch.compile fuses the OT terms' elementwise passes over (patches, n, c) and VGG's bias / relu epilogues
        self._loss_fn = torch.compile(self._loss, dynamic=False) if compile else self._loss
        self._features_fn = torch.compile(self.features, dynamic=False) if compile else self.features

    # ---------------------------------------------------------------- patches and features
    def _windows(self, N, R, device):
        """Patch specs of N images (image-major, K patches each): image, window offset, factor."""
        P = self.patch
        size = torch.tensor([R // 2 ** s for s in self.scale_of], device=device)
        assert all(R % 2 ** s == 0 and (R // 2 ** s) % P == 0 for s in self.scale_of), 'crops must be multiples of the patch'
        f = (size // P).int()
        assert all(v == 1 or v % 2 == 0 for v in f.tolist()), 'downsampling factors 1 or even'
        hi = (R - size + 1)[None].expand(N, -1)  # window offsets uniform in [0, R - size]
        oy = (torch.rand(N, self.K, device=device) * hi).long().clamp_max(hi - 1)
        ox = (torch.rand(N, self.K, device=device) * hi).long().clamp_max(hi - 1)
        n = torch.arange(N, device=device)[:, None].expand(-1, self.K)
        return [v.reshape(-1).int().contiguous() for v in (n, oy, ox, f[None].expand(N, -1))]

    def _build(self, img, spec, P, dtype):
        """img (N, 3, R, R) fp32 -> normalised patches (len(spec[0]), 3, P, P), channels_last."""
        R = img.shape[-1]
        out = torch.empty(len(spec[0]), P, P, 3, device=img.device, dtype=dtype)
        BLOCK = 256
        _build_fwd[(len(spec[0]), triton.cdiv(P * P, BLOCK))](img, out, *spec, self.mean, self.istd, R, P, BLOCK=BLOCK)
        return out.permute(0, 3, 1, 2)

    def _build_backward(self, g, spec, N, R, K):
        """Gradient of _build for the patch gradient g (patches, 3, P, P) -> (N, 3, R, R) (deterministic)."""
        P = g.shape[-1]
        g = g.permute(0, 2, 3, 1).contiguous()
        out = torch.empty(N, 3, R, R, device=g.device)
        BLOCK = 256
        _build_bwd[(N, triton.cdiv(R * R, BLOCK))](g, out, spec[1], spec[2], spec[3], self.istd, K, R, P, BLOCK=BLOCK)
        return out

    def features(self, x):
        """normalised images (b, 3, h, w) channels_last -> per layer (b, h' w', c) (views of channels_last maps)."""
        feats = []
        with torch.autocast('cuda', torch.bfloat16, enabled=self.bf16):
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in LAYERS:
                    b, c, h, w = x.shape
                    feats.append(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        return feats

    def _dtype(self):
        return torch.bfloat16 if self.bf16 else torch.float32

    @torch.no_grad()
    def targets(self, t):
        """Cache of the whole-target features of fixed targets t (N, 3, R, R) for forward(): per scale, per layer
        (N, positions, c) in bf16."""
        assert self.mode == 'whole'
        return [[f.to(torch.bfloat16) for f in fs] for fs in self._whole(t)]

    def _whole(self, t):  # per scale: features of the target resized to patch * 2^s
        N, _, R, _ = t.shape
        out = []
        for s in range(self.num_scales):
            P = self.patch * 2 ** s
            spec = [v.int().contiguous() for v in (torch.arange(N, device=t.device), torch.zeros(N, device=t.device),
                                                   torch.zeros(N, device=t.device), torch.full((N,), R // P, device=t.device))]
            assert R % P == 0 and (R // P == 1 or R // P % 2 == 0)
            out.append(self._features_fn(self._build(t, spec, P, self._dtype())))
        return out

    # ---------------------------------------------------------------- loss
    def _terms(self, x, y):  # OT terms between position sets x, y (b, n, c) -> (b,)
        mx, cx = _moments(x)
        my, cy = _moments(y)
        return _REMD.apply(x, y, True) + (mx - my).abs().mean((1, 2)) + (cx - cy).abs().mean((1, 2))

    def _weights(self, device):  # weight of each term of an image: per patch, or per scale (pooled)
        if self.pooled:
            return torch.full((self.num_scales,), 1 / self.num_scales, device=device)
        return torch.tensor([1 / (self.num_scales * self.crops[s]) for s in self.scale_of], device=device)

    def _patch_losses(self, gen, tgt, n_img):
        """gen: per layer (n_img K, M, c) of the image patches; tgt: per scale per layer (n_img, M_s, c) (whole) or per
        layer (n_img K, M, c) (crops). -> loss per term (n_img, K) or, pooled, (n_img, num_scales)"""
        K, loss = self.K, 0.0
        if self.pooled:  # one term per scale: all crops of the scale on both sides
            for x, y in zip(gen, tgt):
                x, y = x.reshape(n_img, K, *x.shape[1:]), y.reshape(n_img, K, *y.shape[1:])
                terms, k0 = [], 0
                for cnt in self.crops:
                    bx = x[:, k0:k0 + cnt].reshape(n_img, cnt * x.shape[2], -1)
                    by = y[:, k0:k0 + cnt].reshape(n_img, cnt * y.shape[2], -1)
                    n = min(bx.shape[1], by.shape[1], self.n)
                    terms.append(self._terms(_sample(bx, n).float(), _sample(by, n).float()))
                    k0 += cnt
                loss = loss + torch.stack(terms, 1)
            return loss
        for l, x in enumerate(gen):
            n = min(x.shape[1], self.n)
            xs = _sample(x, n).float()
            if self.mode == 'whole':  # per patch k: the target map of its scale
                ys = torch.stack([_sample(tgt[self.scale_of[k]][l], n) for k in range(K)], 1)
            else:  # paired: patch k against target crop k
                ys = _sample(tgt[l], n)
            loss = loss + self._terms(xs, ys.reshape(n_img * K, n, -1).float())
        return loss.reshape(n_img, K)

    def _loss(self, x, tgt, weights):
        """normalised patches x (n_img K, 3, P, P) -> per-image loss (n_img,)"""
        return (self._patch_losses(self.features(x), tgt, x.shape[0] // self.K) * weights).sum(1)

    def _chunk_images(self, N, R):
        per_patch = (self.patch / 256) ** 2 * (0.1 if self.bf16 else 0.2)  # GB: saved VGG activations and OT terms
        extra = 0.0
        if self.mode == 'whole':  # forward-only target maps up to R x R
            extra = (R / 256) ** 2 * (0.02 if self.bf16 else 0.04)
        return max(1, int(self.chunk_gb / (self.K * per_patch * (2 if self.mode == 'crops' else 1) + extra)))

    def forward(self, img, target=None, cache=None, rows=None):
        """img (N, 3, R, R) in [0, 1] (not clamped); target (T, 3, R, R) or, in whole mode, cache = targets(...) of T
        targets; rows (N,): the target of each image (default: image n -> target n) -> per-image loss (N,),
        differentiable w.r.t. img."""
        assert (target is None) != (cache is None) and (cache is None or self.mode == 'whole')
        rows = torch.arange(len(img), device=img.device) if rows is None else rows
        return _MSLoss.apply(img, self, target, cache, rows)


class _MSLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, m, target, cache, rows):
        N, _, R, _ = img.shape
        img = img.detach().float().contiguous()
        loss, grad = torch.empty(N, device=img.device), torch.empty_like(img)
        weights = m._weights(img.device)
        step = m._chunk_images(N, R)
        for a in range(0, N, step):
            b = min(N, a + step)
            n_img = b - a
            spec = m._windows(n_img, R, img.device)
            with torch.no_grad():
                r = rows[a:b]
                if m.mode == 'whole':
                    tgt = [[f[r] for f in fs] for fs in cache] if cache is not None else m._whole(target[r].float().contiguous())
                else:
                    tspec = m._windows(n_img, R, img.device)
                    tgt = m._features_fn(m._build(target[r].float().contiguous(), tspec, m.patch, m._dtype()))
            x = m._build(img[a:b], spec, m.patch, m._dtype())
            if not ctx.needs_input_grad[0]:  # evaluation: no VGG backward
                with torch.no_grad():
                    loss[a:b] = m._loss_fn(x, tgt, weights)
                continue
            with torch.enable_grad():
                x.requires_grad_()
                per_img = m._loss_fn(x, tgt, weights)
                gx, = torch.autograd.grad(per_img.sum(), x)
            loss[a:b] = per_img.detach()
            grad[a:b] = m._build_backward(gx, spec, n_img, R, m.K)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, g):
        grad, = ctx.saved_tensors
        return grad * g[:, None, None, None], None, None, None, None
