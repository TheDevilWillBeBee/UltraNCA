"""Train one NCA per target image, several NCAs at a time in a single forward / backward pass.

Tasks (config `task`):
- texture (config.yml): texture NCA, OT texture loss + overflow penalty, zero seed;
- growing (config_growing.yml): growing NCA (Distill "Growing Neural Cellular Automata"): alive mask on alpha =
  channel 3, L2 to the premultiplied RGBA target, single living cell as the seed.
The seed replaces the first sample of each NCA's batch every few iterations.
Each NCA has its own target, pool, batch, loss and per-tensor gradient normalisation, exactly as if trained alone
(Adam, step lr decay). The states of the group are stacked as (K, B, C, H, W) and the per-NCA 1x1 layers run as
batched matmuls (impl: eager), optionally torch.compile'd (impl: compile), as a fused Triton kernel (impl: triton,
nca_triton.py, texture only) or as fused CUDA kernels (impl: cuda, ncaops/). The only coupling: one random rollout
length per iteration for the group.
Weights are saved in the vanilla NCA format: w1.weight (hidden, 4C, 1, 1), w1.bias (hidden,), w2.weight (C, hidden, 1, 1).

python train.py [config.yml]
"""
import os, sys, glob, json, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

dev = 'cuda'
ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
sobel = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
lap = torch.tensor([[1., 2, 1], [2, -12, 2], [1, 2, 1]])
FILTERS = torch.stack([ident, sobel, sobel.T, lap])[:, None].to(dev)  # (4, 1, 3, 3)


def alive(x):  # (K, B, C, H, W) -> (K, B, 1, H, W): a living cell or a neighbour has alpha > 0.1 (no wrap-around)
    K, B, C, H, W = x.shape
    return (F.max_pool2d(x[:, :, 3].reshape(K * B, 1, H, W), 3, 1, 1) > 0.1).reshape(K, B, 1, H, W)


def step(x, w1, b1, w2, update_rate, growing=False):
    """x (K, B, C, H, W); w1 (K, hidden, 4C), b1 (K, hidden), w2 (K, C, hidden). update_rate >= 1: every cell updates."""
    K, B, C, H, W = x.shape
    p = F.conv2d(F.pad(x.reshape(-1, 1, H, W), (1, 1, 1, 1), mode='circular'), FILTERS)  # channel-major: 4c + f
    h = torch.relu(w1[:, None] @ p.reshape(K, B, 4 * C, H * W) + b1[:, None, :, None])
    u = (w2[:, None] @ h).reshape(K, B, C, H, W)
    y = x + (u if update_rate >= 1 else u * (torch.rand(K, B, 1, H, W, device=x.device) + update_rate).floor())
    return y * (alive(x) & alive(y)) if growing else y


class Texture:
    """Texture NCA: RGB = state[:3] + 0.5, OT texture loss + overflow penalty, zero seed."""

    def __init__(self, cfg):
        from loss import OTLoss
        self.cfg, self.size = cfg, cfg['img_size']
        self.ot = OTLoss(cfg['loss']['n_samples']).to(dev)
        if cfg['train']['compile_loss']:
            self.ot = torch.compile(self.ot)

    def load(self, file):
        img = Image.open(file).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255)

    def seed(self, K, n):
        return torch.zeros(K, n, self.cfg['nca']['channels'], self.size, self.size, device=dev)

    def prepare(self, targets):  # (K, 3, S, S)
        with torch.no_grad():
            self.target_feats = self.ot.features(targets)

    def loss(self, x):  # -> (objective, per-NCA loss (K,))
        K, B, C, S, _ = x.shape
        rows = torch.arange(K, device=dev).repeat_interleave(B)  # target of each image in the (K * B) batch
        tex = self.ot(x[:, :, :3].reshape(K * B, 3, S, S) + 0.5, self.target_feats, rows).reshape(K, B).mean(1)
        overflow = (x - x.clamp(-1, 1)).abs().sum()  # each term depends on one NCA only
        return tex.sum() + self.cfg['train']['overflow_weight'] * overflow, tex.detach()

    def preview(self, target, x):  # target | NCA, (3, S, 2S)
        return torch.cat([target, (x[:3] + 0.5).clamp(0, 1)], 2)


class Growing:
    """Growing NCA: RGBA = state[:4] (premultiplied), L2 loss, single living cell as the seed."""

    def __init__(self, cfg):
        self.cfg, self.size = cfg, cfg['img_size'] + 2 * cfg['pad']

    def load(self, file):  # RGBA, premultiplied, zero-padded
        img = Image.open(file).convert('RGBA').resize((self.cfg['img_size'],) * 2, Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255)
        return F.pad(torch.cat([t[:3] * t[3:], t[3:]]), [self.cfg['pad']] * 4)

    def seed(self, K, n):
        x = torch.zeros(K, n, self.cfg['nca']['channels'], self.size, self.size, device=dev)
        x[:, :, 3:, self.size // 2, self.size // 2] = 1
        return x

    def prepare(self, targets):  # (K, 4, S, S)
        self.targets = targets[:, None]

    def loss(self, x):
        loss = (x[:, :, :4] - self.targets).square().mean((1, 2, 3, 4))  # (K,)
        return loss.sum(), loss.detach()

    def preview(self, target, x):  # target | NCA, composited on white, (3, S, 2S)
        return torch.cat([1 - t[3:4] + t[:3] for t in (target, x.clamp(0, 1))], 2)


def rollout(w1, b1, w2, task, steps, step_fn):  # one sample per NCA from the seed
    x = task.seed(w1.shape[0], 1)
    for _ in range(steps):
        x = step_fn(x, w1, b1, w2)
    return x[:, 0]


def init_weights(init, names, cfg):
    C, Hd = cfg['nca']['channels'], cfg['nca']['hidden']
    if init is None:  # nn.Conv2d default for w1 / b1, zero w2
        bound = (4 * C) ** -0.5
        return [((torch.rand(Hd, 4 * C) * 2 - 1) * bound, (torch.rand(Hd) * 2 - 1) * bound, torch.zeros(C, Hd)) for _ in names]
    files = [os.path.join(init, os.path.splitext(n)[0] + '.pt') for n in names] if os.path.isdir(init) else [init] * len(names)
    sds = [torch.load(f, map_location='cpu') for f in files]
    return [(s['w1.weight'].reshape(Hd, 4 * C), s['w1.bias'], s['w2.weight'].reshape(C, Hd)) for s in sds]


def train_group(targets, init, cfg, task, step_fn):
    t = cfg['train']
    K, B, P = len(targets), t['batch_size'], t['pool_size']
    w1, b1, w2 = (torch.stack(ws).float().to(dev).requires_grad_() for ws in zip(*init))
    opt = torch.optim.Adam([w1, b1, w2], t['lr'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, t['milestones'], t['gamma'])
    pool = task.seed(K, P)
    task.prepare(targets)
    ks = torch.arange(K, device=dev)[:, None]
    losses = []
    t0 = time.time()
    for i in range(t['iterations']):
        idx = torch.rand(K, P, device=dev).argsort(1)[:, :B]  # B distinct pool entries per NCA
        x = pool[ks, idx]
        if i % t['inject_every'] == 0:  # the first sample of each NCA's batch restarts from the seed
            x[:, 0] = task.seed(K, 1)[:, 0]
        for _ in range(np.random.randint(*t['rollout'])):
            x = step_fn(x, w1, b1, w2)
        objective, loss = task.loss(x)
        objective.backward()
        with torch.no_grad():
            for p in (w1, b1, w2):  # per-NCA, per-tensor gradient normalisation
                p.grad /= p.grad.flatten(1).norm(dim=1).reshape(-1, *[1] * (p.dim() - 1)) + 1e-8
            opt.step(); opt.zero_grad(); sched.step()
            pool[ks, idx] = x
        losses.append(loss)
        if i == 0:  # the first iteration includes compilation (torch.compile / Triton JIT): timed separately
            torch.cuda.synchronize(); warmup = time.time() - t0
    torch.cuda.synchronize()
    return w1.detach(), b1.detach(), w2.detach(), torch.stack(losses), warmup, time.time() - t0 - warmup


def make_step(cfg):
    """step_fn(x, w1, b1, w2) of the configured implementation."""
    impl, tf32 = cfg['train']['impl'], cfg['train']['tf32']  # eager | compile | triton | cuda
    rate, growing = cfg['nca']['update_rate'], cfg['task'] == 'growing'
    if impl == 'triton':
        import nca_triton
        assert not growing and rate < 1, 'the Triton step implements the stochastic texture NCA only'
        nca_triton.PREC = 'tf32' if tf32 else 'ieee'
        return lambda x, w1, b1, w2: nca_triton.step(x, w1, b1, w2, rate)
    if impl == 'cuda':  # nanobind extension, build with ncaops/build.sh
        from ncaops import nca_ops
        return lambda x, w1, b1, w2: nca_ops.step(x, w1, b1, w2, rate, growing, fp32=not tf32)
    fn = torch.compile(step) if impl == 'compile' else step
    return lambda x, w1, b1, w2: fn(x, w1, b1, w2, rate, growing)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
    cfg = yaml.safe_load(open(cfg_path))
    here = os.path.dirname(os.path.abspath(cfg_path))
    path = lambda p: p if p is None or os.path.isabs(p) else os.path.join(here, p)
    out = path(cfg['out_dir']); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, 'config.yml'), 'w'), sort_keys=False)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = cfg['train']['tf32']

    files = sorted(glob.glob(os.path.join(path(cfg['targets']), '*.png')) + glob.glob(os.path.join(path(cfg['targets']), '*.jpg')))
    files = files[:cfg.get('num_targets')]  # null: all
    names = [os.path.basename(f) for f in files]
    todo = [i for i, n in enumerate(names) if not os.path.exists(os.path.join(out, os.path.splitext(n)[0] + '.pt'))]  # resumable
    log_path = os.path.join(out, 'log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    task = {'texture': Texture, 'growing': Growing}[cfg['task']](cfg)
    step_fn = make_step(cfg)
    G, C, Hd = cfg['train']['group_size'], cfg['nca']['channels'], cfg['nca']['hidden']
    print(f'{len(files)} targets, {len(todo)} to train, groups of {G}', flush=True)
    for g0 in range(0, len(todo), G):
        grp = todo[g0:g0 + G]
        targets = torch.stack([task.load(files[i]) for i in grp]).to(dev)
        w1, b1, w2, losses, warmup, sec = train_group(targets, init_weights(path(cfg['init']), [names[i] for i in grp], cfg),
                                                      cfg, task, step_fn)
        with torch.no_grad():
            final = rollout(w1, b1, w2, task, cfg['preview_steps'], step_fn)
        for k, i in enumerate(grp):
            stem = os.path.splitext(names[i])[0]
            torch.save({'w1.weight': w1[k].reshape(Hd, 4 * C, 1, 1).cpu(), 'w1.bias': b1[k].cpu(),
                        'w2.weight': w2[k].reshape(C, Hd, 1, 1).cpu()}, os.path.join(out, stem + '.pt'))
            preview = task.preview(targets[k], final[k]).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            Image.fromarray(preview).save(os.path.join(out, stem + '.png'))  # target | NCA after preview_steps steps
            log[names[i]] = {'final_loss': losses[-32:, k].mean().item(), 'group_sec': sec, 'warmup_sec': warmup, 'group_size': len(grp)}
        json.dump(log, open(log_path, 'w'), indent=1)
        print(f'group {g0 // G}: {len(grp)} NCAs in {sec:.0f}s ({sec / len(grp):.1f}s per NCA; first iteration incl. compilation '
              f'{warmup:.0f}s excluded), final loss ' +
              ' '.join(f'{losses[-32:, k].mean().item():.4g}' for k in range(len(grp))), flush=True)


if __name__ == '__main__':
    main()
