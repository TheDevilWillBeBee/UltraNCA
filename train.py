"""Train one texture NCA per target image, several NCAs at a time in a single forward / backward pass.

Each NCA has its own target, pool, batch, loss and per-tensor gradient normalisation, exactly as if trained alone
(OT texture loss + overflow penalty, Adam, step lr decay). The states of the group are stacked as (K, B, C, H, W) and
the per-NCA 1x1 layers run as batched matmuls (impl: eager), optionally torch.compile'd (impl: compile), as a fused
Triton kernel (impl: triton, nca_triton.py) or as fused CUDA kernels (impl: cuda, ncaops/). The only coupling: one random rollout length per iteration for the group.
Weights are saved in the vanilla NCA format: w1.weight (96, 48, 1, 1), w1.bias (96,), w2.weight (12, 96, 1, 1).

python train.py [config.yml]
"""
import os, sys, glob, json, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from loss import OTLoss

dev = 'cuda'
ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
sobel = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
lap = torch.tensor([[1., 2, 1], [2, -12, 2], [1, 2, 1]])
FILTERS = torch.stack([ident, sobel, sobel.T, lap])[:, None].to(dev)  # (4, 1, 3, 3)


def step(x, w1, b1, w2, update_rate):
    """x (K, B, C, H, W); w1 (K, hidden, 4C), b1 (K, hidden), w2 (K, C, hidden)."""
    K, B, C, H, W = x.shape
    p = F.conv2d(F.pad(x.reshape(-1, 1, H, W), (1, 1, 1, 1), mode='circular'), FILTERS)  # channel-major: 4c + f
    h = torch.relu(w1[:, None] @ p.reshape(K, B, 4 * C, H * W) + b1[:, None, :, None])
    u = (w2[:, None] @ h).reshape(K, B, C, H, W)
    return x + u * (torch.rand(K, B, 1, H, W, device=x.device) + update_rate).floor()


def rollout(w1, b1, w2, n, steps, cfg):  # n samples per NCA from the zero seed
    x = torch.zeros(w1.shape[0], n, cfg['nca']['channels'], cfg['img_size'], cfg['img_size'], device=dev)
    for _ in range(steps):
        x = step(x, w1, b1, w2, cfg['nca']['update_rate'])
    return x


def init_weights(init, names, cfg):
    C, Hd = cfg['nca']['channels'], cfg['nca']['hidden']
    if init is None:  # nn.Conv2d default for w1 / b1, zero w2
        bound = (4 * C) ** -0.5
        return [((torch.rand(Hd, 4 * C) * 2 - 1) * bound, (torch.rand(Hd) * 2 - 1) * bound, torch.zeros(C, Hd)) for _ in names]
    files = [os.path.join(init, os.path.splitext(n)[0] + '.pt') for n in names] if os.path.isdir(init) else [init] * len(names)
    sds = [torch.load(f, map_location='cpu') for f in files]
    return [(s['w1.weight'].reshape(Hd, 4 * C), s['w1.bias'], s['w2.weight'].reshape(C, Hd)) for s in sds]


def train_group(targets, init, cfg, ot, step_fn):
    t = cfg['train']
    K, B, P = len(targets), t['batch_size'], t['pool_size']
    C, S = cfg['nca']['channels'], cfg['img_size']
    w1, b1, w2 = (torch.stack(ws).float().to(dev).requires_grad_() for ws in zip(*init))
    opt = torch.optim.Adam([w1, b1, w2], t['lr'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, t['milestones'], t['gamma'])
    pool = torch.zeros(K, P, C, S, S, device=dev)
    with torch.no_grad():
        target_feats = ot.features(targets)
    rows = torch.arange(K, device=dev).repeat_interleave(B)  # target of each image in the (K * B) batch
    ks = torch.arange(K, device=dev)[:, None]
    losses = []
    t0 = time.time()
    for i in range(t['iterations']):
        idx = torch.rand(K, P, device=dev).argsort(1)[:, :B]  # B distinct pool entries per NCA
        x = pool[ks, idx]
        if i % t['inject_every'] == 0:
            x[:, 0] = 0
        for _ in range(np.random.randint(*t['rollout'])):
            x = step_fn(x, w1, b1, w2, cfg['nca']['update_rate'])
        tex = ot(x[:, :, :3].reshape(K * B, 3, S, S) + 0.5, target_feats, rows).reshape(K, B).mean(1)
        overflow = (x - x.clamp(-1, 1)).abs().sum()  # each term depends on one NCA only
        (tex.sum() + t['overflow_weight'] * overflow).backward()
        with torch.no_grad():
            for p in (w1, b1, w2):  # per-NCA, per-tensor gradient normalisation
                p.grad /= p.grad.flatten(1).norm(dim=1).reshape(-1, *[1] * (p.dim() - 1)) + 1e-8
            opt.step(); opt.zero_grad(); sched.step()
            pool[ks, idx] = x
        losses.append(tex.detach())
        if i == 0:  # the first iteration includes compilation (torch.compile / Triton JIT): timed separately
            torch.cuda.synchronize(); warmup = time.time() - t0
    torch.cuda.synchronize()
    return w1.detach(), b1.detach(), w2.detach(), torch.stack(losses), warmup, time.time() - t0 - warmup


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
    names = [os.path.basename(f) for f in files]
    todo = [i for i, n in enumerate(names) if not os.path.exists(os.path.join(out, os.path.splitext(n)[0] + '.pt'))]  # resumable
    S = cfg['img_size']
    log_path = os.path.join(out, 'log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    ot = OTLoss(cfg['loss']['n_samples']).to(dev)
    impl, step_fn = cfg['train']['impl'], step  # eager | compile | triton | cuda
    if impl == 'compile':
        step_fn = torch.compile(step)
    elif impl == 'triton':
        import nca_triton
        nca_triton.PREC = 'tf32' if cfg['train']['tf32'] else 'ieee'
        step_fn = nca_triton.step
    elif impl == 'cuda':  # nanobind extension, build with ncaops/build.sh
        from ncaops import nca_ops
        step_fn = lambda x, w1, b1, w2, rate: nca_ops.step(x, w1, b1, w2, rate, fp32=not cfg['train']['tf32'])
    if cfg['train']['compile_loss']:
        ot = torch.compile(ot)
    G = cfg['train']['group_size']
    print(f'{len(files)} targets, {len(todo)} to train, groups of {G}', flush=True)
    for g0 in range(0, len(todo), G):
        grp = todo[g0:g0 + G]
        targets = torch.stack([torch.from_numpy(np.array(Image.open(files[i]).convert('RGB').resize((S, S), Image.BICUBIC)))
                               for i in grp]).permute(0, 3, 1, 2).float().div(255).to(dev)
        w1, b1, w2, losses, warmup, sec = train_group(targets, init_weights(path(cfg['init']), [names[i] for i in grp], cfg), cfg, ot, step_fn)
        with torch.no_grad():
            rgb = (rollout(w1, b1, w2, 1, 256, cfg)[:, 0, :3] + 0.5).clamp(0, 1)
        C, Hd = cfg['nca']['channels'], cfg['nca']['hidden']
        for k, i in enumerate(grp):
            stem = os.path.splitext(names[i])[0]
            torch.save({'w1.weight': w1[k].reshape(Hd, 4 * C, 1, 1).cpu(), 'w1.bias': b1[k].cpu(),
                        'w2.weight': w2[k].reshape(C, Hd, 1, 1).cpu()}, os.path.join(out, stem + '.pt'))
            preview = torch.cat([targets[k], rgb[k]], 2).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            Image.fromarray(preview).save(os.path.join(out, stem + '.png'))  # target | NCA after 256 steps
            log[names[i]] = {'final_loss': losses[-32:, k].mean().item(), 'group_sec': sec, 'warmup_sec': warmup, 'group_size': len(grp)}
        json.dump(log, open(log_path, 'w'), indent=1)
        print(f'group {g0 // G}: {len(grp)} NCAs in {sec:.0f}s ({sec / len(grp):.1f}s per NCA; first iteration incl. compilation '
              f'{warmup:.0f}s excluded), final loss ' +
              ' '.join(f'{losses[-32:, k].mean().item():.2f}' for k in range(len(grp))), flush=True)


if __name__ == '__main__':
    main()
