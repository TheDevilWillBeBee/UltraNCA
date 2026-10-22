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
VGG (vgg): 'torch' (torchvision, fp32 / TF32 or vgg_bf16 autocast, compiled with the OT terms) or 'cuda'
(ncaops/vgg_ops.py: fused CUTLASS kernels, fp16 activations with fp32 accumulation, or fp16 accumulation with
vgg_acc16; the image's positions are drawn before VGG runs, always without replacement, so that its backward
scatters only the sampled gradients; the OT terms, on fp16 tensor cores (exact for its fp16 features, _OT16), are
compiled on their own).
"""
import logging
import torch
import torchvision
import triton
import triton.language as tl

from loss import _REMD, _moments, _remd_grad
from loss_triton import nearest, cov_l1, cov_l1_grad

# compiling the REMD's Triton kernel logs one warning per inline-asm TF32 rounding (harmless; torch resets log levels,
# so a filter on the logger that emits them: torch/_higher_order_ops/triton_kernel_wrap.py logs as torch._dynamo)
logging.getLogger('torch._dynamo').addFilter(lambda r: 'TTIR mutation analysis' not in r.getMessage())

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
def _build_bwd(gout, gimg, poy, pox, pf, istd, K, R, P, CS, BLOCK: tl.constexpr):
    """Image gradient: per pixel, the sum over the image's K patches (fixed order) of the patch gradients (patches, P,
    P, 3) with pixel stride CS that read it."""
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
        src = gout + (p.to(tl.int64) * P * P + (u // f) * P + v // f) * CS
        a0 += w * tl.load(src, mask=ok, other=0.).to(tl.float32)
        a1 += w * tl.load(src + 1, mask=ok, other=0.).to(tl.float32)
        a2 += w * tl.load(src + 2, mask=ok, other=0.).to(tl.float32)
    base = gimg + n.to(tl.int64) * 3 * R * R + idx
    tl.store(base, a0 * tl.load(istd), mask=m)
    tl.store(base + R * R, a1 * tl.load(istd + 1), mask=m)
    tl.store(base + 2 * R * R, a2 * tl.load(istd + 2), mask=m)


class _OT16(torch.autograd.Function):
    """OT terms of fp16-valued x (b, n, c) (the image's, with gradient) and y (b, m, c) (the target's), any float
    dtypes, contiguous: relaxed EMD + L1 of the means + L1 of the covariances, per item (b,). Every product runs on fp16
    tensor cores with fp32 accumulation (exact for fp16 values); the covariances are never stored (loss_triton.cov_l1);
    the backward adds the three gradients in one pass (loss_triton.cov_l1_grad)."""

    @staticmethod
    def forward(ctx, x, y):
        mx, my = x.float().mean(1), y.float().mean(1)
        nx, ny = x.float().norm(dim=2), y.float().norm(dim=2)
        near_y, j, near_x, i = nearest(x, y, nx, ny, True)
        per_x, per_y = 1 - near_y.mean(1), 1 - near_x.mean(1)
        cov, sign = cov_l1(x, y, mx, my)
        ctx.save_for_backward(x, y, nx, ny, i, j, per_x > per_y, mx, my, sign)
        return torch.maximum(per_x, per_y) + (mx - my).abs().mean(1) + cov

    @staticmethod
    def backward(ctx, g):
        x, y, nx, ny, i, j, x_side, mx, my, sign = ctx.saved_tensors
        b, n, c = x.shape
        r = _remd_grad(g, x.float(), y.float(), nx, ny, i, j, x_side).contiguous()
        vec = g[:, None] * torch.sign(mx - my) / (c * n)  # the mean term
        return cov_l1_grad(x, mx, sign, g * 2 / (c * c * (n - 1)), vec, r), None


def _draw(sets, M, n, base):
    """n distinct random positions in [0, M) per set, offset by base (sets,) -> (sets, n) int64."""
    return torch.rand(sets, M, device=base.device).topk(n, 1, sorted=False).indices + base[:, None]


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
                 chunk_gb=6.0, compile=True, crops=None, vgg='torch', vgg_acc16=False):
        super().__init__()
        assert target_mode in ('whole', 'crops') and pairing in ('pooled', 'paired') and vgg in ('torch', 'cuda')
        self.patch, self.num_scales, self.n, self.mode, self.pairing, self.bf16 = patch, num_scales, n_samples, target_mode, pairing, vgg_bf16
        self.chunk_gb, self.fused = chunk_gb, vgg == 'cuda'
        if self.fused:
            from ncaops.vgg_ops import VGG16Features  # the vgg_cuda extension (ncaops/build.sh)
            assert patch % 16 == 0
            self.bf16 = False
            self.vgg = VGG16Features('fp16' if vgg_acc16 else 'fp32')
        else:
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
        # (fused VGG: the OT terms only)
        compiled = (lambda f: torch.compile(f, dynamic=False)) if compile else (lambda f: f)
        self._ot_fn = compiled(self._ot)
        if self.fused:
            self._loss_fn, self._features_fn = self._fused_loss, self.features
        else:
            self._loss_fn, self._features_fn = compiled(self._loss), compiled(self.features)

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
        g = g.permute(0, 2, 3, 1)  # (patches, P, P, 3); the fused VGG's: a view with pixel stride 8
        if g.stride(3) != 1 or g.stride(1) != P * g.stride(2) or g.stride(0) != P * P * g.stride(2):
            g = g.contiguous()
        out = torch.empty(N, 3, R, R, device=g.device)
        BLOCK = 256
        _build_bwd[(N, triton.cdiv(R * R, BLOCK))](g, out, spec[1], spec[2], spec[3], self.istd, K, R, P, g.stride(2), BLOCK=BLOCK)
        return out

    def features(self, x):
        """normalised images (b, 3, h, w) channels_last -> per layer (b, h' w', c) (views of channels_last maps)."""
        if self.fused:
            maps = self.vgg(x.permute(0, 2, 3, 1), normalized=True)
            return [f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]) for f in maps]
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
        (N, positions, c) in bf16 (fused VGG: fp16)."""
        assert self.mode == 'whole'
        return [[f.to(torch.float16 if self.fused else torch.bfloat16) for f in fs] for fs in self._whole(t)]

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
        if self.fused:  # fp16-valued features
            return _OT16.apply(x.contiguous(), y.contiguous())
        x, y = x.float(), y.float()
        mx, cx = _moments(x)
        my, cy = _moments(y)
        return _REMD.apply(x, y, True) + (mx - my).abs().mean((1, 2)) + (cx - cy).abs().mean((1, 2))

    def _weights(self, device):  # weight of each term of an image: per patch, or per scale (pooled)
        if self.pooled:
            return torch.full((self.num_scales,), 1 / self.num_scales, device=device)
        return torch.tensor([1 / (self.num_scales * self.crops[s]) for s in self.scale_of], device=device)

    def _patch_losses(self, gen, tgt, n_img):
        """gen: per layer (n_img K, M, c) of the image patches (or already sampled: (n_img K, n, c); pooled: per scale
        (n_img, n, c)); tgt: per scale per layer (n_img, M_s, c) (whole) or per layer (n_img K, M, c) (crops).
        -> loss per term (n_img, K) or, pooled, (n_img, num_scales)"""
        K, loss = self.K, 0.0
        if self.pooled:  # one term per scale: all crops of the scale on both sides
            for x, y in zip(gen, tgt):
                y = y.reshape(n_img, K, *y.shape[1:])
                if torch.is_tensor(x):
                    x = x.reshape(n_img, K, *x.shape[1:])
                terms, k0 = [], 0
                for s, cnt in enumerate(self.crops):
                    bx = x[s] if isinstance(x, list) else x[:, k0:k0 + cnt].reshape(n_img, cnt * x.shape[2], -1)
                    by = y[:, k0:k0 + cnt].reshape(n_img, cnt * y.shape[2], -1)
                    n = min(bx.shape[1], by.shape[1], self.n)
                    terms.append(self._terms(_sample(bx, n), _sample(by, n)))
                    k0 += cnt
                loss = loss + torch.stack(terms, 1)
            return loss
        for l, x in enumerate(gen):
            n = min(x.shape[1], self.n)
            xs = _sample(x, n)
            if self.mode == 'whole':  # per patch k: the target map of its scale
                ys = torch.stack([_sample(tgt[self.scale_of[k]][l], n) for k in range(K)], 1)
            else:  # paired: patch k against target crop k
                ys = _sample(tgt[l], n)
            loss = loss + self._terms(xs, ys.reshape(n_img * K, n, -1))
        return loss.reshape(n_img, K)

    def _loss(self, x, tgt, weights):
        """normalised patches x (n_img K, 3, P, P) -> per-image loss (n_img,)"""
        return (self._patch_losses(self.features(x), tgt, x.shape[0] // self.K) * weights).sum(1)

    def _ot(self, gen, tgt, weights, n_img):
        return (self._patch_losses(gen, tgt, n_img) * weights).sum(1)

    # ---------------------------------------------------------------- fused VGG: positions drawn before it runs
    def _rows(self, n_img, device):
        """Image positions per layer, as rows of the flattened (n_img K h w) positions of the layer's map: 1-D, the
        concatenation of position sets, and the sets' shapes (count, size). Sets: the patches, or (pooled) per scale,
        the crops of each image; n random positions, or all of them where a set has at most n."""
        K, rows, shapes = self.K, [], []
        for l in range(5):
            M = (self.patch >> l) ** 2
            groups = [(n_img * K, 1, 0)] if not self.pooled else \
                [(n_img, cnt, k0) for cnt, k0 in zip(self.crops, [sum(self.crops[:s]) for s in range(self.num_scales)])]
            r, sh = [], []
            for count, cnt, k0 in groups:  # count sets of cnt patches each, the first at patch k0 of each image
                base = (torch.arange(count, device=device) * (K if self.pooled else 1) + k0) * M
                if cnt * M > self.n:
                    r.append(_draw(count, cnt * M, self.n, base))
                else:
                    r.append(base[:, None] + torch.arange(cnt * M, device=device))
                sh.append(r[-1].shape)
            rows.append(torch.cat([v.reshape(-1) for v in r]))
            shapes.append(sh)
        return rows, shapes

    def _fused_loss(self, x, tgt, weights):
        """normalised patches x (n_img K, P, P, 3) -> per-image loss (n_img,). The image features are fp32 samples of
        the fused VGG, also where every position is used: their gradients stay fp32 until VGG scales them."""
        n_img = x.shape[0] // self.K
        rows, shapes = self._rows(n_img, x.device)
        _, samples = self.vgg(x, positions=rows, normalized=True, flat=True)
        gen = []
        for s, sh in zip(samples, shapes):
            if not self.pooled:  # one set shape (a split would copy the gradients back together)
                gen.append(s.reshape(*sh[0], -1))
                continue
            gen.append([p.reshape(a, b, -1) for p, (a, b) in zip(s.split([a * b for a, b in sh]), sh)])
        return self._ot_fn(gen, tgt, weights, n_img)

    def _chunk_images(self, N, R):
        # GB: saved VGG activations and OT terms per patch; forward-only target maps up to R x R
        per_patch = (self.patch / 256) ** 2 * (0.04 if self.fused else 0.1 if self.bf16 else 0.2)
        extra = 0.0
        if self.mode == 'whole':
            extra = (R / 256) ** 2 * (0.02 if self.bf16 or self.fused else 0.04)
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
            if m.fused:
                x = x.permute(0, 2, 3, 1)  # (patches, P, P, 3) contiguous
            if not ctx.needs_input_grad[0]:  # evaluation: no VGG backward
                with torch.no_grad():
                    loss[a:b] = m._loss_fn(x, tgt, weights)
                continue
            with torch.enable_grad():
                x.requires_grad_()
                per_img = m._loss_fn(x, tgt, weights)
                gx, = torch.autograd.grad(per_img.sum(), x)
            if m.fused:
                gx = gx.permute(0, 3, 1, 2)
            loss[a:b] = per_img.detach()
            grad[a:b] = m._build_backward(gx, spec, n_img, R, m.K)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, g):
        grad, = ctx.saved_tensors
        return grad * g[:, None, None, None], None, None, None, None
