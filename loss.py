"""OT texture loss on VGG16 features: relaxed EMD (cosine) + mean / covariance matching, summed over 5 layers.
Same formulation as the OptimalTransportLoss of the HyperNCA project (independent random 1024-position subsets of
the target and generated features per image), with the target features computed once instead of every call.
bf16=True runs VGG under bf16 autocast in channels_last (~2x faster VGG; the OT terms stay fp32)."""
import torch
import torchvision


class OTLoss(torch.nn.Module):
    LAYERS = (1, 6, 11, 18, 25)  # relu1_1 ... relu5_1

    def __init__(self, n_samples=1024, bf16=False):
        super().__init__()
        self.vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1').features[:26].eval().requires_grad_(False)
        self.n, self.bf16 = n_samples, bf16
        if bf16:
            self.vgg = self.vgg.to(memory_format=torch.channels_last)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def features(self, img):  # (b, 3, H, W) in [0, 1] -> list of (b, c, H*W)
        x, feats = (img - self.mean) / self.std, []
        if self.bf16:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', torch.bfloat16, enabled=self.bf16):
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in self.LAYERS:
                    feats.append(x.flatten(2).float())
        return feats

    def sample(self, f, rows):  # f (b0, c, d); rows (b,) picks a row of f per item -> (b, n, c)
        b, d = len(rows), f.shape[2]
        if d <= self.n:  # all positions (the loss is permutation invariant)
            return f[rows].transpose(1, 2)
        idx = torch.rand(b, d, device=f.device).argsort(-1)[:, None, :self.n]
        return f[rows[:, None, None], torch.arange(f.shape[1], device=f.device)[None, :, None], idx].transpose(1, 2)

    def forward(self, img, target_feats, target_rows):
        """img (b, 3, H, W); target_feats = self.features(targets); target_rows (b,): target index of each image.
        Returns the per-image loss (b,)."""
        rows = torch.arange(len(img), device=img.device)
        loss = 0.0
        for fx, fy in zip(self.features(img), target_feats):
            x, y = self.sample(fx, rows), self.sample(fy, target_rows)
            cos = 1 - x @ y.transpose(1, 2) / (x.norm(dim=2)[:, :, None] * y.norm(dim=2)[:, None, :] + 1e-10)
            remd = torch.maximum(cos.min(1)[0].mean(1), cos.min(2)[0].mean(1))
            mx, my = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
            cx = (x - mx).transpose(1, 2) @ (x - mx) / (x.shape[1] - 1)
            cy = (y - my).transpose(1, 2) @ (y - my) / (y.shape[1] - 1)
            loss = loss + remd + (mx - my).abs().mean((1, 2)) + (cx - cy).abs().mean((1, 2))
        return loss
