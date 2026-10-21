"""Evaluate trained hybrid NCA + LPPN models (Cells2Pixels checkpoint format: <run>/<name>/{model.pth, siren.pth},
written by train_lppn.py or Cells2Pixels' batched/train.py) with common evaluators: per target, `rollouts` independent
stochastic rollouts of `steps` steps from the zero seed, rendered at R, scored by the multi-scale OT loss (fp32 VGG,
1024 samples) with the whole target and with target crops (pooled), each averaged over `seeds` sampling seeds; by two
single-scale judges on the full render (resized to --judge_res if that differs from R): the OT loss at one scale
(relaxed EMD + moments, 1024 samples, averaged over the seeds) and the Gram loss (Gatys et al.: sum over relu1_1 ...
relu5_1 of ||G_x - G_y||^2 / (4 c^2), G = F^T F / positions); by the cell-lattice imprint (grid: mean over the
lattice frequencies (a / S, b / S) of log10(power in a disc at the peak / power on the same frequency ring away from
the peaks), Hann window, power averaged over the rollouts; 0 = no lattice, the target is the control); and contrast /
structure statistics relative to the target (Cells2Pixels batched/analyze.py): luminance std ratio and the
absolute luminance power per frequency band (Hann window, mean removed; cycles / pixel) relative to the target's.
The decoders render at img_size / grid (any scale: the PyTorch decoder, e.g. 64 -> 1024 for models trained at x4);
--upsample_from S0 per run instead decodes at S0 and upsamples bilinearly to img_size (baseline).
Writes <out>.json and <out>.jpg (rows: targets; columns: target, then one render per run).
python eval_lppn.py --runs DIR [DIR ...] --names fire wood ... --targets DIR --img_size 1024 --grid 128 --patch 256 --out FILE
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lppn import LPPN
from loss_ms import MultiScaleOTLoss
from ncaops import nca_ops
from train_lppn import load_target

BANDS = [(0, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1.0)]


def luminance(img):  # (b, 3, H, W) -> (b, H, W)
    return 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]


def band_power(img):  # (b, 3, H, W) in [0, 1] -> (b, bands): luminance power per band
    y = luminance(img).double()
    H, W = y.shape[-2:]
    win = torch.outer(torch.hann_window(H, periodic=False, dtype=y.dtype, device=y.device),
                      torch.hann_window(W, periodic=False, dtype=y.dtype, device=y.device))
    p = (torch.fft.fft2((y - y.mean((1, 2), keepdim=True)) * win) / win.square().sum().sqrt()).abs().square()
    fy, fx = torch.fft.fftfreq(H, device=y.device, dtype=y.dtype), torch.fft.fftfreq(W, device=y.device, dtype=y.dtype)
    r = (fy[:, None] ** 2 + fx[None] ** 2).sqrt()
    return torch.stack([(p * ((r >= lo) & (r < hi) & (r > 0))).sum((1, 2)) for lo, hi in BANDS], 1)


def gram_loss(m, img, target):
    """Gram loss per image of img (b, 3, H, W) against target (1, 3, H, W), VGG of the MultiScaleOTLoss m."""
    def grams(x):
        x = ((x - m.mean[:, None, None]) * m.istd[:, None, None]).contiguous(memory_format=torch.channels_last)
        return [f.transpose(1, 2) @ f / f.shape[1] for f in m.features(x)]
    return sum(((gx - gy) ** 2).sum((1, 2)) / (4 * gx.shape[1] ** 2) for gx, gy in zip(grams(img), grams(target)))


def lattice_score(img, S):
    """Cell-lattice imprint of renders img (b, 3, H, W) of a decoder upsampling S times (see the docstring)."""
    y = luminance(img.clamp(0, 1)).double()
    b, H, W = y.shape
    win = torch.outer(torch.hann_window(H, periodic=False, dtype=y.dtype, device=y.device),
                      torch.hann_window(W, periodic=False, dtype=y.dtype, device=y.device))
    P = torch.fft.fftshift((torch.fft.fft2((y - y.mean((1, 2), keepdim=True)) * win)).abs().square().mean(0))
    yy, xx = torch.meshgrid(torch.arange(H, device=y.device, dtype=y.dtype) - H // 2,
                            torch.arange(W, device=y.device, dtype=y.dtype) - W // 2, indexing='ij')
    r = (yy ** 2 + xx ** 2).sqrt()
    pts = [(a * H / S, c * W / S) for a in range(-(S // 2), S // 2) for c in range(-(S // 2), S // 2) if (a, c) != (0, 0)]
    near = torch.zeros_like(r, dtype=torch.bool)
    for py, px in pts:
        near |= ((yy - py) ** 2 + (xx - px) ** 2).sqrt() <= 3
    scores = []
    for py, px in pts:
        disc = ((yy - py) ** 2 + (xx - px) ** 2).sqrt() <= 1.5
        ring = ((r - (py ** 2 + px ** 2) ** 0.5).abs() <= 2) & ~near
        scores.append(torch.log10(P[disc].mean() / P[ring].mean()))
    return torch.stack(scores).mean().item()


ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--labels', nargs='+')
ap.add_argument('--names', nargs='+', required=True)
ap.add_argument('--targets', required=True)
ap.add_argument('--img_size', type=int, default=1024)
ap.add_argument('--grid', type=int, default=128)
ap.add_argument('--patch', type=int, default=256)
ap.add_argument('--steps', type=int, default=1024)
ap.add_argument('--rollouts', type=int, default=3)
ap.add_argument('--seeds', type=int, default=4)
ap.add_argument('--judge_res', type=int, help='resolution of the single-scale judges (default: img_size)')
ap.add_argument('--tile_crop', action='store_true', help='figure tiles: the central 512^2 of each render (native pixels) '
                'instead of the whole render resized to 512^2')
ap.add_argument('--upsample_from', nargs='+', type=int, help='per run: 0 = decode at img_size / grid, S0 = decode at S0 '
                'and upsample bilinearly')
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
R, S = a.img_size, a.img_size // a.grid
losses = {mode: MultiScaleOTLoss(a.patch, 3, 1024, mode, 'pooled', compile=False).to(dev) for mode in ('whole', 'crops')}
J = a.judge_res or R
losses['ot1'] = MultiScaleOTLoss(J, 1, 1024, 'whole', compile=False).to(dev)  # single scale: the full render at J
judge = (lambda x: x) if J == R else (lambda x: F.interpolate(x, size=(J, J), mode='bilinear', antialias=True))
JUDGES = ('whole', 'crops', 'ot1', 'gram')
ext = lambda n: next(os.path.join(a.targets, n + e) for e in ('.jpg', '.png', '') if os.path.exists(os.path.join(a.targets, n + e)))
result, rows = {}, []
ups = a.upsample_from or [0] * len(a.runs)
for name in a.names:
    target = load_target(ext(name), R).to(dev)[None]
    tile = (lambda x: x[0, :, (R - 512) // 2:(R + 512) // 2, (R - 512) // 2:(R + 512) // 2]) if a.tile_crop and R >= 512 else \
        (lambda x: F.interpolate(x, size=(512, 512), mode='bilinear', antialias=True)[0])
    row = [tile(target)]
    result[name] = {}
    for run, label, up in zip(a.runs, labels, ups):
        sd = torch.load(os.path.join(run, name, 'model.pth'), map_location=dev)
        C = sd['w2.weight'].shape[0]
        w1, b1, w2 = sd['w1.weight'].flatten(1)[None], sd['w1.bias'][None], sd['w2.weight'].flatten(1)[None]
        dsd = torch.load(os.path.join(run, name, 'siren.pth'), map_location=dev)
        dec = LPPN(scale=up or S, mode='ensemble' if 'first_latent.weight' in dsd else 'lppn')  # LocalEnsembleSiren | Siren
        p = dec.from_state_dicts([dsd])
        torch.manual_seed(0)
        with torch.no_grad():
            x = nca_ops.rollout(torch.zeros(1, a.rollouts, C, a.grid, a.grid, device=dev), w1, b1, w2, 0.5, a.steps, fp32=True)[0]
            img = dec.render(x, {k: v.expand(a.rollouts, *v.shape[1:]) for k, v in p.items()}, impl='eager')
            if up:
                img = F.interpolate(img, size=(R, R), mode='bilinear', align_corners=False)
        scores = {}
        for mode, m in losses.items():
            x, t = (img, target) if mode != 'ot1' else (judge(img), judge(target))
            vals = []
            for s in range(a.seeds):
                torch.manual_seed(100 + s)
                with torch.no_grad():
                    vals.append(m(x, t.expand(a.rollouts, -1, -1, -1)).cpu())
            scores[mode] = torch.stack(vals).mean().item()
        with torch.no_grad():
            scores['gram'] = gram_loss(losses['ot1'], judge(img), judge(target)).mean().item()
        shown = img.clamp(0, 1)
        scores['lum_std_ratio'] = (luminance(shown).std((1, 2)).mean() / luminance(target).std()).item()
        scores['band_power_ratio'] = (band_power(shown).mean(0) / band_power(target)[0]).tolist()
        scores['grid'] = lattice_score(img, S)
        result[name].setdefault('target', {'grid': lattice_score(target, S)})
        result[name][label] = scores
        row.append(tile(img[:1].clamp(0, 1)))
        print(f'{name:20s} {label:20s} whole {scores["whole"]:.4f} crops {scores["crops"]:.4f} single-scale OT '
              f'{scores["ot1"]:.4f} Gram {scores["gram"]:.4g} grid {scores["grid"]:.2f} lum std ratio '
              f'{scores["lum_std_ratio"]:.2f} band power ratios ' + ' '.join(f'{v:.2f}' for v in scores['band_power_ratio']), flush=True)
    rows.append(torch.cat(row, 2))
summary = {label: {mode: float(np.mean([result[n][label][mode] for n in a.names])) for mode in (*JUDGES, 'lum_std_ratio', 'grid')}
           for label in labels}
summary['target'] = {'grid': float(np.mean([result[n]['target']['grid'] for n in a.names]))}
for label in labels:  # geometric mean over targets of the band power ratios
    summary[label]['band_power_ratio (geo. mean)'] = np.exp(np.mean([np.log(result[n][label]['band_power_ratio']) for n in a.names], 0)).tolist()
for label in labels[1:]:
    summary[label]['median ratio to ' + labels[0]] = {mode: float(np.median([result[n][label][mode] / result[n][labels[0]][mode] for n in a.names])) for mode in JUDGES}
print(json.dumps(summary, indent=1))
json.dump({'per_target': result, 'summary': summary, 'args': vars(a)}, open(a.out + '.json', 'w'), indent=1)
Image.fromarray(torch.cat(rows, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()).save(a.out + '.jpg', quality=90)
