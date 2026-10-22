"""Train hybrid NCA + LPPN texture models (Cells2Pixels): K targets, one NCA and one LPPN decoder per target, trained
together. Same semantics as Cells2Pixels' 2D texture task (configs/nca2d, output_type "s", RGB target, no pseudo
targets) run once per target:
- NCA (models/nca2d.NCA: 4 perception filters, circular padding, stochastic update; xavier_normal(0.2) w1, zero w2)
  stepped by the fused CUDA kernels (ncaops, K NCAs x B samples), decoder (lppn.py: the local ensemble by default, or
  the LPPN of Renderer2D) by the fused decoder kernels, with random sample offsets per image during training (jitter);
- own pool, batch, seed injection, the multi-scale OT appearance loss (loss_ms.py) and the overflow loss
  (|x - clamp(x, -1, 1)| mean), objective = appearance_weight * appearance + overflow_weight * overflow;
- per-tensor gradient normalisation of the NCA weights (training/common.normalize_model_grads; the decoder's are
  not normalised), one Adam over NCA and decoder with step decay.
The only coupling: one random rollout length per iteration for the group.
Checkpoints per target in the Cells2Pixels format: <out_dir>/<name>/model.pth (NCA), siren.pth (models/siren.Siren,
num_frequencies 0, outermost_linear; lppn.mode ensemble: batched/decoder.LocalEnsembleSiren), and <name>.png
(target | render of a rollout from the seed).

python train_lppn.py [config_lppn.yml]
"""
import os, sys, glob, json, time
import numpy as np
import torch
import yaml
from PIL import Image

from lppn import LPPN
from loss_ms import MultiScaleOTLoss
from ncaops import nca_ops

dev = 'cuda'


def load_target(path, size):
    """utils/misc.load_texture_image: centre crop to a square, PIL resize, (3, size, size) in [0, 1]."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if w != h:
        c = abs(w - h) // 2
        img = img.crop((c, 0, w - c, h) if w > h else (0, c, w, h - c))
    img = img.resize((size, size))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255)


def init_nca(K, C, hidden, generator=None):
    """models/nca2d.NCA: xavier_normal(gain 0.2) w1, Conv2d default bias, zero w2."""
    std = 0.2 * (2.0 / (4 * C + hidden)) ** 0.5
    w1 = torch.randn(K, hidden, 4 * C, generator=generator) * std
    b1 = (torch.rand(K, hidden, generator=generator) * 2 - 1) * (4 * C) ** -0.5
    return w1.to(dev), b1.to(dev), torch.zeros(K, C, hidden, device=dev)


class Group:
    """K NCAs + LPPN decoders on K targets (K, 3, R, R)."""

    def __init__(self, cfg, targets):
        self.cfg, n, t = cfg, cfg['nca'], cfg['train']
        self.K, self.B, self.C, self.grid = len(targets), t['batch_size'], n['channels'], n['grid']
        R = targets.shape[-1]
        assert R % self.grid == 0
        self.dec = LPPN(scale=R // self.grid, **{k: v for k, v in cfg['lppn'].items() if k != 'jitter'})
        self.jitter = cfg['lppn'].get('jitter', False)  # random sample offsets per image, training only
        lc = cfg['loss']
        self.loss = MultiScaleOTLoss(lc['patch'], lc['num_scales'], lc['n_samples'], lc['target_mode'], lc['pairing'],
                                     lc['vgg_bf16'], crops=lc.get('crops'), vgg=lc.get('vgg', 'torch'),
                                     vgg_acc16=lc.get('vgg_acc16', False)).to(dev)
        self.targets = targets
        self.cache = self.loss.targets(targets) if lc['target_mode'] == 'whole' and lc['cache_targets'] else None
        g = torch.Generator().manual_seed(cfg['seed'])
        self.w1, self.b1, self.w2 = (v.requires_grad_() for v in init_nca(self.K, self.C, n['hidden'], g))
        self.p = {k: v.requires_grad_() for k, v in self.dec.init(self.K, self.C, generator=torch.Generator(dev).manual_seed(cfg['seed'])).items()}
        self.opt = torch.optim.Adam([self.w1, self.b1, self.w2, *self.p.values()], t['lr'])
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.opt, t['milestones'], t['gamma'])
        self.pool = torch.zeros(self.K, t['pool_size'], self.C, self.grid, self.grid, device=dev)
        self.rows = torch.arange(self.K, device=dev).repeat_interleave(self.B)  # target / decoder of each image

    def step(self, x):
        return nca_ops.step(x, self.w1, self.b1, self.w2, self.cfg['nca']['update_rate'], fp32=not self.cfg['train']['tf32'])

    def objective(self, x):
        """x (K, B, C, h, w) -> (objective, per-NCA appearance loss (K,), per-NCA overflow (K,))"""
        t = self.cfg['train']
        offset = self.dec.jitter(len(self.rows)) if self.jitter else None
        img = self.dec.render(x.flatten(0, 1), {k: v[self.rows] for k, v in self.p.items()}, offset=offset)
        if self.cache is not None:
            app = self.loss(img, cache=self.cache, rows=self.rows)
        else:
            app = self.loss(img, self.targets, rows=self.rows)
        app = app.reshape(self.K, self.B).mean(1)
        over = (x - x.clamp(-1, 1)).abs().mean((1, 2, 3, 4))
        return (t['appearance_weight'] * app + t['overflow_weight'] * over).sum(), app.detach(), over.detach()

    def iteration(self, i):
        t = self.cfg['train']
        ks = torch.arange(self.K, device=dev)[:, None]
        idx = torch.rand(self.K, t['pool_size'], device=dev).argsort(1)[:, :self.B]  # B distinct pool entries per NCA
        x = self.pool[ks, idx]
        if i % t['inject_every'] == 0:
            x[:, 0] = 0  # NCA.seed: zeros
        for _ in range(np.random.randint(*t['rollout'])):
            x = self.step(x)
        objective, app, over = self.objective(x)
        objective.backward()
        with torch.no_grad():
            for w in (self.w1, self.b1, self.w2):  # per-NCA, per-tensor normalisation of the NCA gradients
                w.grad /= w.grad.flatten(1).norm(dim=1).reshape(-1, *[1] * (w.dim() - 1)) + 1e-8
            self.opt.step(); self.opt.zero_grad(); self.sched.step()
            self.pool[ks, idx] = x
        return app, over

    @torch.no_grad()
    def preview(self, steps):
        x = nca_ops.rollout(torch.zeros(self.K, 1, self.C, self.grid, self.grid, device=dev), self.w1, self.b1, self.w2,
                            self.cfg['nca']['update_rate'], steps, fp32=not self.cfg['train']['tf32'])
        return self.dec.render(x[:, 0], self.p)

    def save(self, k, path):
        C, Hd = self.C, self.cfg['nca']['hidden']
        os.makedirs(path, exist_ok=True)
        torch.save({'w1.weight': self.w1[k].detach().reshape(Hd, 4 * C, 1, 1).cpu(), 'w1.bias': self.b1[k].detach().cpu(),
                    'w2.weight': self.w2[k].detach().reshape(C, Hd, 1, 1).cpu()}, os.path.join(path, 'model.pth'))
        torch.save(self.dec.to_siren(self.p, k), os.path.join(path, 'siren.pth'))


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_lppn.yml')
    cfg = yaml.safe_load(open(cfg_path))
    here = os.path.dirname(os.path.abspath(cfg_path))
    path = lambda p: p if p is None or os.path.isabs(p) else os.path.join(here, p)
    out = path(cfg['out_dir']); os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, 'config.yml'), 'w'), sort_keys=False)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = cfg['train']['tf32']
    torch.backends.cudnn.allow_tf32 = True

    if cfg.get('names'):
        files = [os.path.join(path(cfg['targets']), n) for n in cfg['names']]
    else:
        files = sorted(glob.glob(os.path.join(path(cfg['targets']), '*.png')) + glob.glob(os.path.join(path(cfg['targets']), '*.jpg')))
        files = files[:cfg.get('num_targets')]
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    todo = [i for i, n in enumerate(names) if not os.path.exists(os.path.join(out, n, 'siren.pth'))]  # resumable
    log_path = os.path.join(out, 'log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    G, T = cfg['train']['group_size'], cfg['train']['iterations']
    print(f'{len(files)} targets, {len(todo)} to train, groups of {G}', flush=True)
    for g0 in range(0, len(todo), G):
        grp = todo[g0:g0 + G]
        targets = torch.stack([load_target(files[i], cfg['img_size']) for i in grp]).to(dev)
        group = Group(cfg, targets)
        apps, overs = [], []
        t0 = time.time()
        for i in range(T):
            app, over = group.iteration(i)
            apps.append(app); overs.append(over)
            if i == 0:  # includes compilation
                torch.cuda.synchronize(); warmup = time.time() - t0
            if (i + 1) % cfg['log_every'] == 0:
                a = torch.stack(apps[-cfg['log_every']:]).mean(0)
                print(f'group {g0 // G} iteration {i + 1}: appearance ' + ' '.join(f'{v:.3f}' for v in a.tolist()), flush=True)
        torch.cuda.synchronize()
        sec = time.time() - t0 - warmup
        apps, overs = torch.stack(apps), torch.stack(overs)
        renders = group.preview(cfg['preview_steps'])
        for k, i in enumerate(grp):
            group.save(k, os.path.join(out, names[i]))
            prev = torch.cat([targets[k], renders[k].clamp(0, 1)], 2).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            Image.fromarray(prev).save(os.path.join(out, names[i] + '.png'))
            log[names[i]] = {'final_appearance': apps[-32:, k].mean().item(), 'final_overflow': overs[-32:, k].mean().item(),
                             'appearance': apps[:, k].tolist(), 'group_sec': sec, 'warmup_sec': warmup, 'group_size': len(grp)}
        json.dump(log, open(log_path, 'w'))
        print(f'group {g0 // G}: {len(grp)} targets in {sec:.0f}s ({sec / len(grp):.1f}s per target; first iteration incl. '
              f'compilation {warmup:.0f}s excluded), final appearance ' +
              ' '.join(f'{apps[-32:, k].mean().item():.3f}' for k in range(len(grp))), flush=True)


if __name__ == '__main__':
    main()
