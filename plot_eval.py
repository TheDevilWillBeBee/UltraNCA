"""Figure of an eval_lppn.py result: rows = targets, columns = target then one render per run, with column titles
(run and its means over the targets) and per-render scores.
python plot_eval.py EVAL (the --out of eval_lppn.py: reads EVAL.json, EVAL.jpg; writes EVAL_figure.jpg)
    [--titles "title 1" ...] (one per run; '\\n' for a line break) [--zoom] (the central quarter of each tile, nearest
    upsampled x4: writes EVAL_zoom.jpg)
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def plot(prefix, titles=None, tile=512, zoom=False, only=None, suffix=''):
    ev = json.load(open(prefix + '.json'))
    sheet = np.asarray(Image.open(prefix + '.jpg'))
    names = ev['args']['names']
    all_labels = ev['args']['labels'] or [r.rstrip('/').split('/')[-1] for r in ev['args']['runs']]
    labels = only or all_labels
    titles = titles or labels
    col = {l: all_labels.index(l) + 1 for l in labels}  # tile column of each shown run
    rows, cols = len(names), len(labels) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.6 * rows + 1.4), squeeze=False)
    for i, name in enumerate(names):
        for j in range(cols):
            ax = axes[i, j]
            jj = 0 if j == 0 else col[labels[j - 1]]
            im = sheet[i * tile:(i + 1) * tile, jj * tile:(jj + 1) * tile]
            if zoom:  # central quarter, every pixel 4 x 4: a render pixel is 8 x 8 screen pixels
                q = tile // 4
                im = np.asarray(Image.fromarray(im[3 * q // 2:5 * q // 2, 3 * q // 2:5 * q // 2]).resize((tile, tile), Image.NEAREST))
            ax.imshow(im, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
                if 'target' in ev['per_target'][name]:
                    ax.set_xlabel(f'grid {ev["per_target"][name]["target"]["grid"]:.2f}', fontsize=8.5)
            else:
                r = ev['per_target'][name][labels[j - 1]]
                ax.set_xlabel(f'W {r["whole"]:.2f}   C {r["crops"]:.2f}   σ {r["lum_std_ratio"]:.2f}' +
                              (f'\nOT {r["ot1"]:.2f}   Gram {r["gram"]:.3g}' if 'ot1' in r else '') +
                              (f'   grid {r["grid"]:.2f}' if 'grid' in r else ''), fontsize=8.5)
            if i == 0:
                if j == 0:
                    ax.set_title('Target', fontsize=10, fontweight='bold')
                else:
                    s = ev['summary'][labels[j - 1]]
                    ax.set_title(f'{titles[j - 1]}\nW {s["whole"]:.2f}  C {s["crops"]:.2f}  σ {s["lum_std_ratio"]:.2f}' +
                                 (f'\nOT {s["ot1"]:.2f}  Gram {s["gram"]:.3g}' if 'ot1' in s else '') +
                                 (f'\ngrid {s["grid"]:.2f}' if 'grid' in s else ''), fontsize=9.5, fontweight='bold')
    a = ev['args']
    fig.suptitle(('Zoom: central quarter of each render, nearest-neighbour upsampled. ' if zoom else '') + f'Renders after {a["steps"]} steps from the seed ({a["img_size"]}², NCA {a["grid"]}², VGG patch {a["patch"]}²). '
                 'W: multi-scale OT loss against the whole target, C: against target crops (generated and target crops pooled per '
                 'scale); lower is better, fp32 VGG, mean of '
                 f'{a["rollouts"]} rollouts x {a["seeds"]} sampling seeds. σ: luminance std / target\'s (1 = same contrast). '
                 + (f'Single-scale judges on the full {a.get("judge_res") or a["img_size"]}² render: OT (one scale, 1024 samples) and '
                    'Gram (Gatys, relu1_1 ... relu5_1). ' if 'ot1' in next(iter(ev['summary'].values())) else '')
                 + ('grid: cell-lattice imprint, log10 of the power at the lattice frequencies over the power on the same ring '
                    '(0 = none; the target is the control). ' if 'grid' in next(iter(ev['summary'].values())) else '')
                 + 'Column titles: means over the targets.',
                 fontsize=9, wrap=True, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = prefix + suffix + ('_zoom.jpg' if zoom else '_figure.jpg')
    fig.savefig(out, dpi=110, pil_kwargs={'quality': 88, 'optimize': True})
    plt.close(fig)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('--titles', nargs='+')
    ap.add_argument('--zoom', action='store_true')
    ap.add_argument('--only', nargs='+', help='labels of the runs to show (default: all)')
    ap.add_argument('--suffix', default='', help='appended to the output name')
    a = ap.parse_args()
    print(plot(a.prefix, [t.replace('\\n', '\n') for t in a.titles] if a.titles else None, zoom=a.zoom, only=a.only,
               suffix=a.suffix))
