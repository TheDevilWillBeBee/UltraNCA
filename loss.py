"""OT texture loss on VGG16 features: relaxed EMD (cosine) + mean / covariance matching, summed over 5 layers.
Same formulation as the OptimalTransportLoss of the HyperNCA project (independent random 1024-position subsets of
the target and generated features per image), with the target features computed once instead of every call.
VGG runs in channels_last (no layout conversions around the TF32 convolutions); bf16=True runs it under bf16
autocast (~2x faster VGG; the OT terms stay fp32). The relaxed EMD has its own backward through the minimising
pairs only (no dense (n, n) gradient)."""
import torch
import torchvision


class _Pool2d(torch.autograd.Function):
    """Non-overlapping 2x2 VGG pooling; save one byte per output for the backward."""

    @staticmethod
    def forward(ctx, x):
        y, idx = x[:, :, ::2, ::2], torch.zeros_like(x[:, :, ::2, ::2], dtype=torch.uint8)
        for i in range(1, 4):
            v = x[:, :, i // 2::2, i % 2::2]
            take = (v > y) | v.isnan()  # first maximum, last NaN, as in max_pool2d
            y, idx = torch.where(take, v, y), torch.where(take, i, idx)
        ctx.save_for_backward(idx)
        return y.contiguous(memory_format=torch.channels_last)

    @staticmethod
    def backward(ctx, g):
        idx, = ctx.saved_tensors
        N, C, H, W = g.shape
        v = torch.where(idx[..., None] == torch.arange(4, device=g.device), g[..., None], 0)
        dx = v.reshape(N, C, H, W, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(N, C, 2 * H, 2 * W)
        return dx.contiguous(memory_format=torch.channels_last)


class _REMD(torch.autograd.Function):
    """Relaxed EMD per item: max(mean_j min_i cos_ij, mean_i min_j cos_ij), cos_ij = 1 - x_i y_j / (|x_i| |y_j| + 1e-10),
    for x (b, n, c) and y (b, m, c) (no gradient to y). Returns (b,)."""

    @staticmethod
    def forward(ctx, x, y):
        nx, ny = x.norm(dim=2), y.norm(dim=2)
        if (torch.compiler.is_compiling() and x.is_cuda and x.dtype == torch.float32
                and hasattr(torch.library, 'triton_op')):
            from loss_triton import nearest
            near_y, j, near_x, i = nearest(x, y, nx, ny)
        else:
            sim = x @ y.transpose(1, 2) / (nx[:, :, None] * ny[:, None, :] + 1e-10)  # 1 - cos
            near_y, j = sim.max(2)  # per x_i: the nearest y_j
            near_x, i = sim.max(1)  # per y_j: the nearest x_i
        per_x, per_y = 1 - near_y.mean(1), 1 - near_x.mean(1)
        ctx.save_for_backward(x, y, nx, ny, i, j, per_x > per_y)
        return torch.maximum(per_x, per_y)

    @staticmethod
    def backward(ctx, g):
        x, y, nx, ny, i, j, x_side = ctx.saved_tensors
        b, n, c = x.shape

        def dsim(xp, nxp, yp, nyp):  # gradient of sim(x_p, y_p) w.r.t. x_p, for pairs p: (b, p, c)
            d = nxp * nyp + 1e-10
            dot = (xp * yp).sum(2)
            return yp / d[..., None] - (dot * nyp / (nxp * d * d)).nan_to_num(0, 0, 0)[..., None] * xp

        gx = -(g / n)[:, None, None] * dsim(x, nx, y.gather(1, j[..., None].expand(-1, -1, c)), ny.gather(1, j))
        gy_side = -(g / y.shape[1])[:, None, None] * dsim(x.gather(1, i[..., None].expand(-1, -1, c)), nx.gather(1, i), y, ny)
        gx = torch.where(x_side[:, None, None], gx, torch.zeros_like(gx).scatter_add_(1, i[..., None].expand(-1, -1, c), gy_side))
        return gx, None


def _moments(y):  # (b, n, c) -> mean (b, 1, c), covariance (b, c, c)
    m = y.mean(1, keepdim=True)
    return m, (y - m).transpose(1, 2) @ (y - m) / (y.shape[1] - 1)


class OTLoss(torch.nn.Module):
    LAYERS = (1, 6, 11, 18, 25)  # relu1_1 ... relu5_1

    def __init__(self, n_samples=1024, bf16=False):
        super().__init__()
        self.vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:26].eval().requires_grad_(False)
        self.vgg = self.vgg.to(memory_format=torch.channels_last)
        self.n, self.bf16 = n_samples, bf16
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def features(self, img):  # (b, 3, H, W) in [0, 1] -> list of (b, c, H*W)
        x, feats = ((img - self.mean) / self.std).contiguous(memory_format=torch.channels_last), []
        with torch.autocast('cuda', torch.bfloat16, enabled=self.bf16):
            for i, layer in enumerate(self.vgg):
                if (torch.compiler.is_compiling() and isinstance(layer, torch.nn.MaxPool2d)
                        and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0):
                    x = _Pool2d.apply(x)
                else:
                    x = layer(x)
                if i in self.LAYERS:
                    feats.append(x.flatten(2).float())
        return feats

    @torch.no_grad()
    def targets(self, img, cache=True):
        """The target side of the loss for target images (b0, 3, H, W): their features per layer. cache=True also
        computes, for the layers whose positions are all used (no sampling), the per-target terms once
        (features (b0, d, c), mean, covariance); with targets that change at every call, cache=False saves that work."""
        return [(f.transpose(1, 2).contiguous(), *_moments(f.transpose(1, 2))) if cache and f.shape[2] <= self.n else f
                for f in self.features(img)]

    def sample(self, f, rows):  # f (b0, c, d); rows (b,) picks a row of f per item -> (b, n, c)
        b, d = len(rows), f.shape[2]
        if d <= self.n:  # all positions (the loss is permutation invariant)
            return f[rows].transpose(1, 2)
        idx = torch.rand(b, d, device=f.device).argsort(-1)[:, None, :self.n]
        return f[rows[:, None, None], torch.arange(f.shape[1], device=f.device)[None, :, None], idx].transpose(1, 2)

    def forward(self, img, targets, target_rows):
        """img (b, 3, H, W); targets = self.targets(target images); target_rows (b,): target index of each image.
        Returns the per-image loss (b,)."""
        rows = torch.arange(len(img), device=img.device)
        loss = 0.0
        for fx, t in zip(self.features(img), targets):
            x = self.sample(fx, rows)
            if isinstance(t, tuple):  # cached
                y, my, cy = (v[target_rows] for v in t)
            else:
                y = self.sample(t, target_rows)
                my, cy = _moments(y)
            mx, cx = _moments(x)
            loss = loss + _REMD.apply(x, y) + (mx - my).abs().mean((1, 2)) + (cx - cy).abs().mean((1, 2))
        return loss
