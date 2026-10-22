"""VGG16 relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 (torchvision IMAGENET1K_V1 weights) on the fused CUDA kernels of
vgg_cuda.cu (built by build.sh): fp16 NHWC activations and gradients, CUTLASS convolutions whose epilogues apply bias +
relu in the forward and the relu mask in the backward, max pooling with a 1-byte argmax. Differentiable w.r.t. the
images.

    vgg = VGG16Features(accumulate='fp32').cuda()
    maps = vgg(images)                      # images (N, 3, H, W) in [0, 1], fp32, H and W multiples of 16
                                            # -> 5 relu maps (N, C, H / 2^l, W / 2^l), fp16, channels_last memory
    maps, samples = vgg(images, positions=[p1, None, p3, None, None])
                                            # also the features at positions p (N, S) int64 (flattened y * w + x index
                                            # per image, distinct per image) of the chosen layers: (N, S, C) fp32
    maps, samples = vgg(images, positions=[r1, ...], flat=True)
                                            # positions as rows of the layer's flattened (N h w) positions, any shape
                                            # (..., S), distinct overall: (..., S, C) fp32 (e.g. sets pooled over images)

Gradients may come from the maps (masked and added into the layer gradient) and from the samples (scattered at their
positions), in any combination; layers above the highest one with a gradient are skipped in the backward. Gradients
from the samples cost only their size: sample the positions here instead of gathering from the maps when the maps are
large and only samples are used (a gather from the maps gives them dense, mostly-zero gradients).
normalized=True takes the network input directly: (N, H, W, 3), ImageNet-normalised RGB, channels last (any float
dtype), and returns its gradient (fp32) in the same layout.

Precision: the convolutions take fp16 inputs and accumulate in fp32 (accumulate='fp32': features within ~1e-3 of fp32,
about twice TF32's error and a tenth of bf16 autocast's) or fp16 (accumulate='fp16': ~1.75x faster convolutions on
GeForce cards, whose tensor cores run fp16 accumulation at twice the fp32-accumulation rate; feature errors 1e-3 to
6e-3, below bf16 autocast's; no gain on A100 / H100).
fp16 holds the activations of [0, 1] images with a wide margin (VGG16's relu maps stay below ~10^3; fp16 overflows at
65504). The backward scales each image's gradients by a power of 2 that puts the largest incoming feature gradient of
the image at 2^8 .. 2^9 (they then stay within a few powers of 2 of it through the layers), and unscales the image
gradient (fp32).
conv1_1 (3 input channels) and its input gradient run on CUDA cores in fp32 (memory-bound: 64 channels written or read
per pixel). Each other convolution uses the fastest of three tile shapes for its problem size, measured once
(VGG16Features.tiles).
"""
import torch
import torchvision

try:
    from . import vgg_cuda
except ImportError:
    import vgg_cuda

LAYERS = (1, 6, 11, 18, 25)  # torchvision indices of relu1_1 ... relu5_1
_G_TYPE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _stream():
    return torch.cuda.current_stream().cuda_stream


class VGG16Features(torch.nn.Module):
    def __init__(self, accumulate='fp32'):
        super().__init__()
        assert accumulate in ('fp32', 'fp16')
        self.acc16 = accumulate == 'fp16'
        convs = [m for m in torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:max(LAYERS) + 1]
                 if isinstance(m, torch.nn.Conv2d)]  # conv1_1, 1_2, 2_1, 2_2, 3_1, 3_2, 3_3, 4_1, 4_2, 4_3, 5_1
        for i, c in enumerate(convs[1:], 1):
            self.register_buffer(f'w{i}', c.weight.detach().permute(0, 2, 3, 1).half().contiguous())  # (K, 3, 3, C)
            self.register_buffer(f'b{i}', c.bias.detach().half().contiguous())
        w = convs[0].weight.detach().permute(0, 2, 3, 1)  # conv1_1 (64, 3, 3, 3), on CUDA cores
        self.register_buffer('w_in', w.permute(1, 2, 3, 0).reshape(27, 64).contiguous())  # (tap, output channel) fp32
        self.register_buffer('b_in', convs[0].bias.detach().contiguous())
        self.register_buffer('w_in_grad', w.permute(1, 2, 0, 3).reshape(9, 64, 3).contiguous())  # (tap, k, c) fp32
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        self.register_buffer('ab_images', torch.cat([1 / self.std, -self.mean / self.std]))  # normalisation a x + b
        self.register_buffer('ab_normalized', torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
        self.tiles = {}  # (op, acc16, N, H, W, C, K) -> tile config

    def forward(self, x, positions=None, normalized=False, flat=False):
        positions = list(positions) if positions is not None else [None] * 5
        assert len(positions) == 5, 'one entry (or None) per feature layer'
        out = _VGG.apply(x, self, normalized, flat, *positions)
        maps = list(out[:5])
        if all(p is None for p in positions):
            return maps
        return maps, [s if p is not None else None for s, p in zip(out[5:], positions)]

    # ---------------------------------------------------------------- kernels with a tile choice
    def _conv(self, op, args, shape):
        key = (op, self.acc16) + shape
        fn = getattr(vgg_cuda, op)
        if key not in self.tiles:  # time the tile shapes once (the output is valid either way)
            times = []
            for cfg in (0, 1, 2):
                fn(cfg, *args, *shape, _stream())
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(3):
                    fn(cfg, *args, *shape, _stream())
                e1.record(); e1.synchronize()
                times.append(e0.elapsed_time(e1))
            self.tiles[key] = min(range(3), key=times.__getitem__)
        fn(self.tiles[key], *args, *shape, _stream())

    def fprop(self, x, i):
        N, H, W, C = x.shape
        K = getattr(self, f'w{i}').shape[0]
        y = torch.empty(N, H, W, K, device=x.device, dtype=torch.float16)
        self._conv('fprop', (self.acc16, x.data_ptr(), getattr(self, f'w{i}').data_ptr(), getattr(self, f'b{i}').data_ptr(),
                             y.data_ptr()), (N, H, W, C, K))
        return y

    def dgrad(self, dz, i, act):  # gradient at the pre-activation of the layer whose output is `act`
        N, H, W, C = act.shape
        dx = torch.empty(N, H, W, C, device=act.device, dtype=torch.float16)
        self._conv('dgrad_mask', (self.acc16, dz.data_ptr(), getattr(self, f'w{i}').data_ptr(), act.data_ptr(), dx.data_ptr()),
                   (N, H, W, C, dz.shape[-1]))
        return dx

    def conv_in(self, x, normalized):  # conv1_1 from the images (N, 3, H, W) or the normalised input (N, H, W, 3)
        x = x.float()
        if normalized:
            N, H, W, _ = x.shape
            strides, ab = (x.stride(0), x.stride(3), x.stride(1), x.stride(2)), self.ab_normalized
        else:
            N, _, H, W = x.shape
            strides, ab = x.stride(), self.ab_images
        y = torch.empty(N, H, W, 64, device=x.device, dtype=torch.float16)
        vgg_cuda.conv_in(x.data_ptr(), *strides, ab.data_ptr(), self.w_in.data_ptr(), self.b_in.data_ptr(), y.data_ptr(),
                         N, H, W, _stream())
        return y

    @staticmethod
    def pool(a, argmax=True):
        N, H, W, C = a.shape
        y = torch.empty(N, H // 2, W // 2, C, device=a.device, dtype=torch.float16)
        idx = torch.empty(N, H // 2, W // 2, C, device=a.device, dtype=torch.uint8) if argmax else None
        vgg_cuda.pool(a.data_ptr(), y.data_ptr(), idx.data_ptr() if argmax else 0, N, H, W, C, _stream())
        return y, idx

    @staticmethod
    def unpool(g, idx):
        N, h, w, C = g.shape
        out = torch.empty(N, 2 * h, 2 * w, C, device=g.device, dtype=torch.float16)
        vgg_cuda.unpool(g.data_ptr(), idx.data_ptr(), out.data_ptr(), N, 2 * h, 2 * w, C, _stream())
        return out


def _scatter(dz, act, rows, g, scale, add):
    """dz (N, h, w, C) fp16 [+]= (act > 0) * scale[image] * g at rows (R,) of the flattened positions, g (R, C); or g
    (N, h, w, C) (rows None). g fp32, fp16 or bf16; scale (N,) fp32."""
    N, h, w, C = act.shape
    R = rows.numel() if rows is not None else N * h * w
    vgg_cuda.scatter(dz.data_ptr(), act.data_ptr(), rows.data_ptr() if rows is not None else 0, g.data_ptr(),
                     _G_TYPE[g.dtype], scale.data_ptr(), h * w, R, C, add, _stream())


class _VGG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vgg, normalized, flat, *pos):
        assert x.is_cuda and x.dim() == 4 and x.shape[3 if normalized else 1] == 3
        H, W = x.shape[1:3] if normalized else x.shape[2:]
        assert H % 16 == 0 and W % 16 == 0, 'H and W must be multiples of 16'
        argmax = ctx.needs_input_grad[0]  # pooling keeps its argmax only for a backward
        a11 = vgg.conv_in(x, normalized)
        p1, i1 = vgg.pool(vgg.fprop(a11, 1), argmax)
        a21 = vgg.fprop(p1, 2)
        p2, i2 = vgg.pool(vgg.fprop(a21, 3), argmax)
        a31 = vgg.fprop(p2, 4)
        a32 = vgg.fprop(a31, 5)
        p3, i3 = vgg.pool(vgg.fprop(a32, 6), argmax)
        a41 = vgg.fprop(p3, 7)
        a42 = vgg.fprop(a41, 8)
        p4, i4 = vgg.pool(vgg.fprop(a42, 9), argmax)
        a51 = vgg.fprop(p4, 10)
        feats = (a11, a21, a31, a41, a51)
        samples, rows = [], []
        for f, p in zip(feats, pos):
            if p is None:
                samples.append(torch.empty(0, device=x.device))
                rows.append(None)
                continue
            N, h, w, C = f.shape
            assert p.dtype == torch.int64
            if not flat:  # per-image indices -> rows
                assert p.dim() == 2 and p.shape[0] == N
                p = p + torch.arange(N, device=p.device)[:, None] * (h * w)
            p = p.contiguous()
            s = torch.empty(*p.shape, C, device=x.device)
            vgg_cuda.gather(f.data_ptr(), p.data_ptr(), s.data_ptr(), p.numel(), C, _stream())
            samples.append(s)
            rows.append(p.view(-1))
        ctx.vgg, ctx.normalized = vgg, normalized
        ctx.save_for_backward(a11, p1, i1, a21, p2, i2, a31, a32, p3, i3, a41, a42, p4, i4, a51, *rows)
        ctx.set_materialize_grads(False)
        return (*[f.permute(0, 3, 1, 2) for f in feats], *samples)

    @staticmethod
    def backward(ctx, *grads):
        a11, p1, i1, a21, p2, i2, a31, a32, p3, i3, a41, a42, p4, i4, a51, *rows = ctx.saved_tensors
        vgg = ctx.vgg
        acts = (a11, a21, a31, a41, a51)
        gmap = [g.permute(0, 2, 3, 1) if g is not None else None for g in grads[:5]]  # (N, h, w, C)
        gmap = [g if g is None or g.dtype in _G_TYPE else g.float() for g in gmap]
        gsamp = [g.reshape(-1, a.shape[-1]) if g is not None and g.numel() else None for g, a in zip(grads[5:], acts)]
        gsamp = [g if g is None or g.dtype in _G_TYPE else g.float() for g in gsamp]
        # per image: 2^k that puts the largest incoming gradient at 2^8 .. 2^9
        N = a11.shape[0]
        amax = torch.zeros(N, device=a11.device)
        for act, gm, gs, r in zip(acts, gmap, gsamp, rows):
            if gm is not None:
                amax = torch.maximum(amax, torch.linalg.vector_norm(gm, float('inf'), (1, 2, 3)).float())
            if gs is not None:
                amax.scatter_reduce_(0, r // (act.shape[1] * act.shape[2]), torch.linalg.vector_norm(gs, float('inf'), 1).float(), 'amax')
        scale = torch.exp2(9 - torch.log2(amax).ceil()).clamp(2.0 ** -60, 2.0 ** 60)

        def feat(l, act, dz):  # add the gradients of feature layer l (dz: gradient at its pre-activation, None = 0)
            gm, gs = gmap[l], gsamp[l]
            if gm is not None:
                gm = gm.contiguous()
                if dz is None:
                    dz = torch.empty_like(act)
                    _scatter(dz, act, None, gm, scale, add=False)
                else:
                    _scatter(dz, act, None, gm, scale, add=True)
            if gs is not None:
                if dz is None:
                    dz = torch.zeros_like(act)
                _scatter(dz, act, rows[l], gs.contiguous(), scale, add=True)
            return dz

        dz = feat(4, a51, None)
        if dz is not None:
            dz = vgg.unpool(vgg.dgrad(dz, 10, p4), i4)  # conv5_1 -> pre-activation of relu4_3
            dz = vgg.dgrad(dz, 9, a42)
            dz = vgg.dgrad(dz, 8, a41)
        dz = feat(3, a41, dz)
        if dz is not None:
            dz = vgg.unpool(vgg.dgrad(dz, 7, p3), i3)
            dz = vgg.dgrad(dz, 6, a32)
            dz = vgg.dgrad(dz, 5, a31)
        dz = feat(2, a31, dz)
        if dz is not None:
            dz = vgg.unpool(vgg.dgrad(dz, 4, p2), i2)
            dz = vgg.dgrad(dz, 3, a21)
        dz = feat(1, a21, dz)
        if dz is not None:
            dz = vgg.unpool(vgg.dgrad(dz, 2, p1), i1)
            dz = vgg.dgrad(dz, 1, a11)
        dz = feat(0, a11, dz)
        nones = (None,) * (4 + len(rows))
        if dz is None or not ctx.needs_input_grad[0]:
            return nones
        N, H, W, _ = a11.shape  # the image gradient, unscaled: (N, H, W, 3) normalised, or (N, 3, H, W) / std
        if ctx.normalized:
            dx = torch.empty(N, H, W, 3, device=dz.device)
            strides, cmul = (dx.stride(0), dx.stride(3), dx.stride(1), dx.stride(2)), torch.ones_like(vgg.std)
        else:
            dx = torch.empty(N, 3, H, W, device=dz.device)
            strides, cmul = dx.stride(), 1 / vgg.std
        unscale = 1 / scale
        vgg_cuda.dgrad_in(dz.data_ptr(), vgg.w_in_grad.data_ptr(), unscale.data_ptr(), cmul.data_ptr(), dx.data_ptr(),
                          *strides, N, H, W, _stream())
        return (dx,) + nones[1:]
