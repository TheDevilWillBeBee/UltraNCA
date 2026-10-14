"""Benchmarks of the NCA step on the current GPU (TF32, warm-up / compilation excluded).
  train: steady-state cost of one training iteration of train.py (a rollout of the mean training length, forward +
         backward, plus the loss) for K NCAs: texture (config.yml: B = 2, 128 x 128, 64 steps, OT loss) and growing
         (config_growing.yml: B = 8, 72 x 72, 80 steps, L2 loss). Growing weights and states: trained NCAs from
         --growing_weights (runs/emoji, as written by train.py) if present, else random (no growth: unrepresentative
         alive areas).
  infer: forward-only rollouts (no grad) of 256 steps: K NCAs x 1 sample at 64^2 and 128^2, then one NCA at several
         grid sizes; texture (12 / 96, stochastic) and growing (16 / 128) NCAs.
python benchmark.py [train] [infer] [--impl cuda eager] [--K 32 64]
"""
import argparse, glob, os, time
import numpy as np
import torch
import yaml
import train as T

ap = argparse.ArgumentParser()
ap.add_argument('what', nargs='*', default=['train', 'infer'])
ap.add_argument('--impl', nargs='+', default=['cuda'])
ap.add_argument('--K', nargs='+', type=int, default=[32, 64])
ap.add_argument('--tasks', nargs='+', default=['texture', 'growing'])
ap.add_argument('--growing_weights', default='runs/emoji')
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
torch.backends.cuda.matmul.allow_tf32 = True


def config(task, impl):
    cfg = yaml.safe_load(open(os.path.join(here, 'config.yml' if task == 'texture' else 'config_growing.yml')))
    cfg['train']['impl'] = impl
    return cfg


def weights(cfg, K, trained=None):
    """(w1, b1, w2) of K NCAs: the trained NCAs in `trained` (cycled), else random with a small non-zero w2."""
    files = sorted(glob.glob(os.path.join(trained, '*.pt'))) if trained else []
    if files:
        ws = T.init_weights(os.path.dirname(files[0]), [os.path.basename(files[k % len(files)]) for k in range(K)], cfg)
    else:
        g = torch.Generator().manual_seed(0)
        C, Hd = cfg['nca']['channels'], cfg['nca']['hidden']
        ws = [(0.05 * torch.randn(Hd, 4 * C, generator=g), torch.zeros(Hd), 0.01 * torch.randn(C, Hd, generator=g))
              for _ in range(K)]
    return [torch.stack(w).to(T.dev) for w in zip(*ws)]


def timed(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.time() - t) / reps


def bench_train(task_name, impl, K):
    cfg = config(task_name, impl)
    t = cfg['train']
    task = {'texture': T.Texture, 'growing': T.Growing}[task_name](cfg)
    step = T.make_step(cfg)
    trained = os.path.join(here, a.growing_weights) if task_name == 'growing' else None
    w1, b1, w2 = (w.requires_grad_() for w in weights(cfg, K, trained))
    S = task.size
    if task_name == 'texture':
        task.prepare(torch.rand(K, 3, S, S, device=T.dev))
    else:
        files = sorted(glob.glob(os.path.join(here, cfg['targets'], '*.png')))
        task.prepare(torch.stack([task.load(files[k % len(files)]) for k in range(K)]).to(T.dev))
    steps = sum(t['rollout']) // 2
    with torch.no_grad():  # starting states: a grown pattern (growing) or noise (texture)
        x0 = task.seed(K, t['batch_size'])
        if task_name == 'growing':
            for _ in range(steps):
                x0 = step(x0, w1, b1, w2)
        else:
            x0 += 0.1 * torch.randn_like(x0)

    def iteration():
        x = x0
        for _ in range(steps):
            x = step(x, w1, b1, w2)
        task.loss(x)[0].backward()

    ms = timed(iteration)
    print(f'train {task_name:8s} {impl:6s} K={K:3d} B={t["batch_size"]} {S}^2 {steps} steps: {ms:7.1f} ms / iteration, '
          f'{t["iterations"] * ms / K / 1000:6.1f} s per NCA for {t["iterations"]} iterations, '
          f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB', flush=True)


def bench_infer(task_name, impl, K, size, steps=256):
    cfg = config(task_name, impl)
    step = T.make_step(cfg)
    trained = os.path.join(here, a.growing_weights) if task_name == 'growing' else None
    w1, b1, w2 = weights(cfg, K, trained)
    if task_name == 'growing':  # from the single-cell seed
        task = T.Growing(cfg)
        task.size = size
        x0 = task.seed(K, 1)
    else:
        x0 = 0.1 * torch.randn(K, 1, cfg['nca']['channels'], size, size, device=T.dev)

    @torch.no_grad()
    def run():
        x = x0
        for _ in range(steps):
            x = step(x, w1, b1, w2)

    ms = timed(run, 1, 3)
    print(f'infer {task_name:8s} {impl:6s} K={K:3d} {size:4d}^2 {steps} steps: {ms:8.2f} ms, '
          f'{1e6 * ms / (steps * K * size * size):6.2f} ns per cell-step', flush=True)


for impl in a.impl:
    if 'train' in a.what:
        for task_name in a.tasks:
            for K in a.K:
                bench_train(task_name, impl, K)
    if 'infer' in a.what:
        for task_name in a.tasks:
            for K, size in [(K, s) for K in a.K for s in (64, 128)] + [(1, s) for s in (64, 128, 256, 512, 1024, 2048)]:
                bench_infer(task_name, impl, K, size)
