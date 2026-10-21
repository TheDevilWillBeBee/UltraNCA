"""Steady-state cost of one hybrid NCA + LPPN training iteration (train_lppn.py: a 48-step rollout forward + backward,
the fused decoder, the multi-scale OT loss, Adam) and of its components, on random targets and random NCA weights
(compilation and warm-up excluded).
Settings (render R / VGG patch / NCA grid): A 1024 / 256 / 128 (x8), B 512 / 128 / 128 (x4), C 256 / 64 / 64 (x4).
Target modes: whole (cached: fixed targets), whole-fresh (targets resent at every iteration), crops (pooled).
python benchmark_lppn.py [--settings A B C] [--vgg fp32 bf16] [--modes whole whole-fresh crops] [--KB 32x1 8x4]
    [--decoder ensemble lppn] [--crops 1 2 4]
"""
import argparse, copy, os, time
import numpy as np
import torch
import yaml

import train_lppn as TL

ap = argparse.ArgumentParser()
ap.add_argument('--settings', nargs='+', default=['A', 'B', 'C'])
ap.add_argument('--vgg', nargs='+', default=['fp32', 'bf16'])
ap.add_argument('--modes', nargs='+', default=['whole', 'whole-fresh', 'crops'])
ap.add_argument('--KB', nargs='+', default=['32x1', '8x4'], help='NCAs x samples per NCA')
ap.add_argument('--steps', type=int, default=48)
ap.add_argument('--crops', nargs='+', type=int, help='crops per scale (default: 1 2 4)')
ap.add_argument('--num_scales', type=int, default=3, help='loss scales (the settings are those of the 3-scale loss)')
ap.add_argument('--decoder', nargs='+', help='ensemble and / or lppn (default: config_lppn.yml)')
a = ap.parse_args()
SETTINGS = {'A': (1024, 256, 128), 'B': (512, 128, 128), 'C': (256, 64, 64)}
here = os.path.dirname(os.path.abspath(__file__))
torch.backends.cuda.matmul.allow_tf32 = True


def timed(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append(1000 * (time.time() - t))
    return float(np.median(ts))


def run(setting, vgg, mode, K, B, decoder_mode):
    R, P, grid = SETTINGS[setting]
    cfg = copy.deepcopy(yaml.safe_load(open(os.path.join(here, 'config_lppn.yml'))))
    cfg['lppn']['mode'] = decoder_mode
    cfg['nca']['grid'], cfg['img_size'] = grid, R
    cfg['train'].update(batch_size=B, rollout=[a.steps, a.steps + 1], pool_size=max(8, B))
    cfg['loss'].update(patch=P, vgg_bf16=vgg == 'bf16', target_mode='crops' if mode == 'crops' else 'whole',
                       cache_targets=mode == 'whole', crops=a.crops, num_scales=a.num_scales)
    torch._dynamo.reset()
    torch.manual_seed(0); np.random.seed(0)
    g = TL.Group(cfg, torch.rand(K, 3, R, R, device=TL.dev))
    with torch.no_grad():  # weights that grow patterns, states with structure
        g.w2.normal_(0, 0.01)
        g.pool.copy_(torch.rand_like(g.pool) * 0.2 - 0.1)
    x0 = g.pool[:, :B].clone()

    def nca():
        x = x0
        for _ in range(a.steps):
            x = g.step(x)
        x.backward(torch.randn_like(x) * 1e-3)

    xs = x0.flatten(0, 1).clone()

    def decoder():
        img = g.dec.render(xs, {k: v[g.rows] for k, v in g.p.items()})
        img.backward(torch.randn_like(img) * 1e-6)

    imgs = torch.rand(K * B, 3, R, R, device=TL.dev, requires_grad=True)

    def loss():
        if g.cache is not None:
            g.loss(imgs, cache=g.cache, rows=g.rows).sum().backward()
        else:
            g.loss(imgs, g.targets, rows=g.rows).sum().backward()

    res = {'nca': timed(nca), 'decoder': timed(decoder), 'loss': timed(loss), 'iteration': timed(lambda: g.iteration(1))}
    g.opt.zero_grad()
    torch.cuda.reset_peak_memory_stats(); g.iteration(1); torch.cuda.synchronize()
    N = K * B
    print(f'{setting} ({R}/{P}/{grid}) {decoder_mode:8s} {vgg} {mode:11s} crops {g.loss.crops} K={K:2d} B={B}: iteration {res["iteration"]:7.1f} ms '
          f'({res["iteration"] / N:5.2f} per image) = NCA {res["nca"]:6.1f} + decoder {res["decoder"]:5.1f} + loss {res["loss"]:6.1f} '
          f'(+ Adam, pool); peak {torch.cuda.max_memory_allocated() / 2 ** 30:4.1f} GB', flush=True)


for setting in a.settings:
    for vgg in a.vgg:
        for mode in a.modes:
            for kb in a.KB:
                K, B = map(int, kb.split('x'))
                if mode == 'whole' and setting == 'A' and K > 8:
                    continue  # 32 cached 1024^2 targets: 10 GB
                for decoder_mode in a.decoder or [yaml.safe_load(open(os.path.join(here, 'config_lppn.yml')))['lppn']['mode']]:
                    run(setting, vgg, mode, K, B, decoder_mode)
