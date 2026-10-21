"""Validate the fused LPPN decoder (ncaops/lppn_ops.py) against the fp32 PyTorch decoder (lppn.LPPN.reference: the
Cells2Pixels Renderer2D + Siren semantics, or LocalEnsembleSiren's for the local ensemble), for every instantiated
scale, both modes, odd shapes and tiny output gradients; the kernels use fp16 operands, so the reference's own error
under fp16 autocast (the Cells2Pixels training precision) is printed next to it as the yardstick. Also checks that the
backward is deterministic, the training-time jitter of the sample positions and, if ../Cells2Pixels (or $C2P) is there,
the ensemble reference against its batched/decoder.LocalEnsembleSiren; optionally times both modes.
python ncaops/testing/test_lppn.py [--bench]
"""
import os, sys, argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from lppn import LPPN  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def grads(dec, s, p, g, impl, autocast=False, offset=None):
    s = s.clone().requires_grad_()
    p = {k: v.clone().requires_grad_() for k, v in p.items()}
    with torch.autocast('cuda', torch.float16, enabled=autocast):
        out = dec.reference(s, p, offset) if impl == 'reference' else dec.render(s, p, impl, offset)
    (out.float() * g).sum().backward()
    return out.detach().float(), {'state': s.grad, **{k: v.grad for k, v in p.items()}}


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def check(S, N, C, h, w, gmag=1.0, seed=0, mode='lppn', jitter=False):
    torch.manual_seed(seed)
    dec = LPPN(scale=S, mode=mode)
    off = dec.jitter(N) if jitter else None
    p = dec.init(N, C)
    p['wo'] *= 10  # trained decoders have larger output weights than the init
    s = torch.rand(N, C, h, w, device='cuda') * 2 - 1
    g = torch.randn(N, 3, S * h, S * w, device='cuda') * gmag * torch.rand(N, 1, 1, 1, device='cuda')
    ref, gref = grads(dec, s, p, g, 'reference', offset=off)
    half, ghalf = grads(dec, s, p, g, 'reference', autocast=True, offset=off)
    out, gout = grads(dec, s, p, g, 'cuda', offset=off)
    _, gout2 = grads(dec, s, p, g, 'cuda', offset=off)
    if jitter:  # the decomposition itself (fp32 PyTorch) with offsets: exact
        eager, geager = grads(dec, s, p, g, 'eager', offset=off)
        e = max([rel(eager, ref)] + [rel(geager[key], gref[key]) for key in gref if key != 'bo'])
        assert e < 1e-4, f'decomposition with offsets: {e:.1e}'
    det = all(torch.equal(gout[k], gout2[k]) for k in gout)
    fwd, fwd_h = rel(out, ref), rel(half, ref)
    line = f'{mode}{" jitter" if jitter else ""} S={S} N={N} C={C} {h}x{w} |dout|~{gmag:.0e}: forward {fwd:.1e} (fp16 autocast {fwd_h:.1e}); grads'
    worst = 0.0
    for k in gref:
        e, eh = rel(gout[k], gref[k]), rel(ghalf[k], gref[k])
        worst = max(worst, e / max(eh, 1e-3))
        line += f' {k} {e:.1e} ({eh:.1e})'
    print(line + ('' if det else '  NOT DETERMINISTIC'), flush=True)
    assert det, 'backward not deterministic'
    assert fwd < max(2 * fwd_h, 2e-3), 'forward error above the fp16 reference error'
    assert worst < 2.0, 'gradient error above twice the fp16 reference error'


def local_ensemble_reference():
    """lppn.LPPN.reference (mode ensemble) against Cells2Pixels' batched/decoder.LocalEnsembleSiren."""
    c2p = os.environ.get('C2P', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Cells2Pixels')))
    if not os.path.isdir(os.path.join(c2p, 'batched')):
        print('Cells2Pixels not found: LocalEnsembleSiren comparison skipped')
        return
    sys.path.insert(1, c2p)
    from batched.decoder import LocalEnsembleSiren
    torch.manual_seed(0)
    for S in (8, 4):
        N, C, h, w = 2, 16, 6, 10
        les = [LocalEnsembleSiren(in_features=C, out_features=3, hidden_features=32, hidden_layers=2, outermost_linear=True,
                                  first_omega_0=10., hidden_omega_0=10., neighborhood=1, padding='circular',
                                  checkpoint=False).cuda() for _ in range(N)]
        dec = LPPN(scale=S, mode='ensemble')
        p = dec.from_state_dicts([m.state_dict() for m in les])
        s = torch.rand(N, C, h, w, device='cuda') * 2 - 1
        ref = torch.cat([les[n](s[n:n + 1], S) for n in range(N)]).permute(0, 3, 1, 2)
        ours = dec.reference(s, p)
        err = (ours - ref).abs().max().item()
        print(f'ensemble S={S}: lppn.LPPN.reference vs LocalEnsembleSiren max |diff| {err:.1e} (output scale {ref.abs().max():.2f})')
        assert err < 1e-5
        assert all(torch.equal(v, les[1].state_dict()[key].cpu()) for key, v in dec.to_state_dict(p, 1).items()), 'state_dict round trip'


def bench():
    import time
    for S, N, h, mode in ((8, 32, 128, 'lppn'), (8, 32, 128, 'ensemble'), (4, 32, 128, 'lppn'), (4, 32, 128, 'ensemble'),
                          (4, 32, 64, 'lppn'), (4, 32, 64, 'ensemble')):
        dec = LPPN(scale=S, mode=mode)
        p = {k: v.requires_grad_() for k, v in dec.init(N, 16).items()}
        s = (torch.rand(N, 16, h, h, device='cuda') * 2 - 1).requires_grad_()
        g = torch.randn(N, 3, S * h, S * h, device='cuda')
        for impl in ('cuda', 'eager'):
            if impl == 'eager':  # 1024^2 x 32 of fp32 activations do not fit: 4 images, times N / 4
                s, p, g = s[:4].detach().requires_grad_(), {k: v[:4].detach().requires_grad_() for k, v in p.items()}, g[:4]

            def fwd():
                with torch.no_grad():
                    dec.render(s, p, impl)

            def fwd_bwd():
                (dec.render(s, p, impl) * g).sum().backward()
            res = []
            for fn in (fwd, fwd_bwd):
                for _ in range(2):
                    fn()
                torch.cuda.synchronize(); t = time.time()
                for _ in range(5):
                    fn()
                torch.cuda.synchronize(); res.append(1000 * (time.time() - t) / 5)
            f = N / len(s)
            print(f'bench {mode:8s} S={S} N={N} {h}^2 -> {S * h}^2 {impl:5s}: forward {f * res[0]:7.2f} ms, forward + backward {f * res[1]:7.2f} ms'
                  + (' (4 images x N / 4)' if f > 1 else ''), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    for mode in ('lppn', 'ensemble'):
        for S in (8, 4):
            check(S, 3, 16, 8, 12, mode=mode)
            check(S, 2, 12, 4, 2, mode=mode)          # one band, two columns
            check(S, 2, 16, 12, 7, gmag=1e-7, mode=mode)  # tiny output gradients (fp16 subnormals without the scaling)
            check(S, 4, 16, 32, 32, seed=1, mode=mode)
    for mode in ('lppn', 'ensemble'):  # training-time jitter of the sample positions
        for S in (8, 4):
            check(S, 3, 16, 8, 12, mode=mode, jitter=True, seed=2)
    torch.manual_seed(3)  # the per-pixel reference at offset 0 is the Renderer2D one
    dec = LPPN(scale=4, mode='lppn')
    s, p = torch.rand(2, 16, 6, 10, device='cuda') * 2 - 1, dec.init(2, 16)
    e = (dec._reference_at(s, p, torch.zeros(2, device='cuda')) - dec.reference(s, p)).abs().max().item()
    print(f'lppn reference at offset 0 vs Renderer2D path: {e:.1e}')
    assert e < 1e-5
    local_ensemble_reference()
    print('all LPPN checks passed')
    if a.bench:
        bench()
