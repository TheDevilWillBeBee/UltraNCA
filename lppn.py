"""LPPN decoder of Cells2Pixels (models/siren.Siren rendered by utils/render.Renderer2D: sin activations, raw sub-cell
coordinates (num_frequencies = 0), outermost linear layer) for N images, each with its own SIREN weights.

Renderer2D upsamples the state (C, h, w) S times with bilinear interpolation (circular padding, align_corners=False)
and feeds [sub-cell coordinate (2), interpolated state (C)] of every pixel to the SIREN:
  h_0 = sin(w0 (W_0 [coord, s] + b_0)),  h_l = sin(w (W_l h_{l-1} + b_l)) (l = 1 .. L),  out = W_o h_L + b_o.
The bilinear weights of a pixel sum to one, so the state part of the first layer commutes with the interpolation:
  w0 (W_0 [coord, bilerp(s)] + b_0) = bilerp(G) + T(coord),  G = w0 (W_0s s + b_0) per cell,  T = w0 W_0c coord.
render() computes G (a matmul per cell) and T (a table over the S x S sub-cell positions) with PyTorch, so autograd
gives the gradients of the state, W_0 and b_0; the per-pixel rest runs in the fused CUDA kernels (ncaops/lppn_ops.py,
impl='cuda') or in PyTorch (impl='eager').

Pixels are grouped by dual cell: dual cell (i, j) holds the S x S pixels between the centres of cells (i, j) and
(i + 1, j + 1), which interpolate between these 4 corner cells (indices mod h, w). Its pixel (rho, kappa) is image pixel
((S i + S/2 + rho) mod Sh, (S j + S/2 + kappa) mod Sw), with interpolation weights lambda = (rho + 1/2) / S along y
(likewise x) and sub-cell coordinate c((rho + S/2) mod S), c(r) = (2 r + 1) / S - 1. T is indexed by (rho, kappa).

mode 'ensemble' (the default; Cells2Pixels batched/decoder.LocalEnsembleSiren with neighborhood 1, LIIF's local
ensemble): the SIREN runs at the 4 cells nearest to the pixel, each with the pixel's coordinate relative to that cell
(2 units per cell), and the 4 outputs are blended with the bilinear weights; the latent is never interpolated (no cell
seams). Its first layer has the same decomposition with one coordinate table per corner (T (N, 4, S, S, hidden)).

offset (N, 2), optional (training-time jitter): image n samples the continuous field at pixel positions shifted by
offset[n] cells (|offset| <= 1 / (2 S): every pixel stays in its cell and its dual cell); the bilinear weights, the
sub-cell / relative coordinates and so the tables shift with it. None: the S x S grid of the original.

Weights: w0 (N, hidden, 2 + C) (coordinate columns first, as Siren's torch.cat([coords, cell_states])), b0 (N, hidden), w (N, L, hidden, hidden), b (N, L, hidden), wo (N, out, hidden), bo (N, out).
"""
import math
import torch
import torch.nn.functional as F


class LPPN:
    def __init__(self, scale=8, hidden=32, layers=2, out=3, first_omega=10.0, hidden_omega=10.0, mode='ensemble'):
        assert mode in ('lppn', 'ensemble')
        self.scale, self.hidden, self.layers, self.out = scale, hidden, layers, out
        self.omega0, self.omega, self.mode = first_omega, hidden_omega, mode

    def init(self, N, C, device='cuda', generator=None):
        """Siren's / LocalEnsembleSiren's initialisation for N decoders (SineLayer, nn.Linear default biases)."""
        Hd, L, O = self.hidden, self.layers, self.out
        u = lambda shape, bound: (torch.rand(shape, device=device, generator=generator) * 2 - 1) * bound
        n0 = 2 + C
        return {'w0': u((N, Hd, n0), 1 / n0), 'b0': u((N, Hd), n0 ** -0.5),
                'w': u((N, L, Hd, Hd), math.sqrt(6 / Hd) / self.omega), 'b': u((N, L, Hd), Hd ** -0.5),
                'wo': u((N, O, Hd), math.sqrt(6 / Hd) / self.omega), 'bo': u((N, O), Hd ** -0.5)}

    def from_state_dicts(self, state_dicts, device='cuda'):
        """Stacked weights of Cells2Pixels Siren (outermost_linear=True, num_frequencies=0) or LocalEnsembleSiren
        state_dicts."""
        L = self.layers
        st = lambda key: torch.stack([sd[key] for sd in state_dicts]).float().to(device)
        if self.mode == 'ensemble':
            w0 = torch.cat([st('first_coord.weight'), st('first_latent.weight').flatten(2)], 2)
            return {'w0': w0, 'b0': st('first_latent.bias'),
                    'w': torch.stack([st(f'net.{l}.linear.weight') for l in range(L)], 1),
                    'b': torch.stack([st(f'net.{l}.linear.bias') for l in range(L)], 1),
                    'wo': st(f'net.{L}.weight'), 'bo': st(f'net.{L}.bias')}
        return {'w0': st('net.0.linear.weight'), 'b0': st('net.0.linear.bias'),
                'w': torch.stack([st(f'net.{l + 1}.linear.weight') for l in range(L)], 1),
                'b': torch.stack([st(f'net.{l + 1}.linear.bias') for l in range(L)], 1),
                'wo': st(f'net.{L + 1}.weight'), 'bo': st(f'net.{L + 1}.bias')}

    from_siren = from_state_dicts

    def to_state_dict(self, p, n):
        """The Cells2Pixels Siren (lppn) or LocalEnsembleSiren (ensemble) state_dict of decoder n."""
        L = self.layers
        if self.mode == 'ensemble':
            sd = {'first_latent.weight': p['w0'][n, :, 2:].reshape(self.hidden, -1, 1, 1), 'first_latent.bias': p['b0'][n],
                  'first_coord.weight': p['w0'][n, :, :2], f'net.{L}.weight': p['wo'][n], f'net.{L}.bias': p['bo'][n]}
            for l in range(L):
                sd[f'net.{l}.linear.weight'], sd[f'net.{l}.linear.bias'] = p['w'][n, l], p['b'][n, l]
        else:
            sd = {'net.0.linear.weight': p['w0'][n], 'net.0.linear.bias': p['b0'][n],
                  f'net.{L + 1}.weight': p['wo'][n], f'net.{L + 1}.bias': p['bo'][n]}
            for l in range(L):
                sd[f'net.{l + 1}.linear.weight'], sd[f'net.{l + 1}.linear.bias'] = p['w'][n, l], p['b'][n, l]
        return {key: v.detach().cpu().clone() for key, v in sd.items()}

    to_siren = to_state_dict

    def _mlp(self, z, p):  # first-layer pre-activation z (N, hidden, pixels) without omega -> (N, out, pixels)
        z = torch.sin(self.omega0 * z)
        for l in range(self.layers):
            z = torch.sin(self.omega * (p['w'][:, l] @ z + p['b'][:, l, :, None]))
        return p['wo'] @ z + p['bo'][..., None]

    def _latent(self, s, p):  # state part of the first layer per cell, without omega: (N, hidden, h w)
        return p['w0'][:, :, 2:] @ s.flatten(2) + p['b0'][..., None]

    def reference(self, s, p, offset=None):
        """state s (N, C, h, w) -> (N, out, S h, S w), straight from the definitions: lppn = Renderer2D.render +
        Siren.forward (bilinear state + sub-cell coordinate); ensemble = LocalEnsembleSiren.forward (per pixel, the
        4 nearest cells, relative coordinates, bilinear blend of the outputs). With an offset: the same per pixel at
        the shifted positions."""
        N, C, h, w = s.shape
        S = self.scale
        if offset is not None:
            return torch.cat([self._reference_at(s[n:n + 1], {k: v[n:n + 1] for k, v in p.items()}, offset[n]) for n in range(N)])
        if self.mode == 'lppn':
            x = F.interpolate(F.pad(s, [1, 1, 1, 1], mode='circular'), scale_factor=S, mode='bilinear')[:, :, S:-S, S:-S]
            r = (torch.arange(S, device=s.device) / S - 0.5 + 0.5 / S) * 2.0
            coords = torch.stack(torch.meshgrid(r, r, indexing='ij')).repeat(1, h, w)  # (2, S h, S w)
            z = torch.cat([coords.expand(N, -1, -1, -1), x], 1).flatten(2)  # (N, 2 + C, pixels)
            return self._mlp(p['w0'] @ z + p['b0'][..., None], p).reshape(N, self.out, S * h, S * w)
        return self._reference_at(s, p, torch.zeros(2, device=s.device))

    def _reference_at(self, s, p, off):  # reference() with the pixel positions shifted by off (2,) cells
        N, C, h, w = s.shape
        S = self.scale
        u = lambda n, o: (torch.arange(S * n, device=s.device) + 0.5) / S - 0.5 + o  # pixel centre in cell units
        uy, ux = u(h, off[0]), u(w, off[1])
        iy, ix = uy.floor().long(), ux.floor().long()
        ly, lx = uy - iy, ux - ix
        if self.mode == 'lppn':  # bilinear state; coordinate in the cell that contains the pixel centre, in [-1, 1)
            at = lambda dy, dx: s[:, :, (iy + dy) % h][:, :, :, (ix + dx) % w]
            x = ((1 - ly)[:, None] * (1 - lx)[None] * at(0, 0) + (1 - ly)[:, None] * lx[None] * at(0, 1)
                 + ly[:, None] * (1 - lx)[None] * at(1, 0) + ly[:, None] * lx[None] * at(1, 1))
            cy, cx = 2 * (uy + 0.5 - (uy + 0.5).floor()) - 1, 2 * (ux + 0.5 - (ux + 0.5).floor()) - 1
            coords = torch.stack(torch.broadcast_tensors(cy[:, None], cx[None]))  # (2, S h, S w)
            z = torch.cat([coords.expand(N, -1, -1, -1), x], 1).flatten(2)
            return self._mlp(p['w0'] @ z + p['b0'][..., None], p).reshape(N, self.out, S * h, S * w)
        g = self._latent(s, p).reshape(N, self.hidden, h, w)
        out = 0.0
        for oy in (0, 1):
            for ox in (0, 1):
                gc = g[:, :, (iy + oy) % h][:, :, :, (ix + ox) % w]  # (N, hidden, S h, S w): the corner cell's latent
                dy, dx = ly - oy, lx - ox  # pixel position relative to the cell, in cells
                pre = gc + p['w0'][:, :, 0, None, None] * (2 * dy)[:, None] + p['w0'][:, :, 1, None, None] * (2 * dx)[None]
                weight = (1 - dy.abs())[:, None] * (1 - dx.abs())[None]
                out = out + weight * self._mlp(pre.flatten(2), p).reshape(N, self.out, S * h, S * w)
        return out

    def tables(self, s, p, offset=None):
        """G (N, h, w, hidden): first-layer pre-activation of the state per cell; T: of the coordinate per dual-cell
        position (rho, kappa), (N, S, S, hidden), or ensemble (N, 4, S, S, hidden) per corner, for the sample
        offsets (N, 2) (None: 0); both scaled by w0."""
        N, C, h, w = s.shape
        S = self.scale
        off = torch.zeros(N, 2, device=s.device) if offset is None else offset
        # (cells, C) @ (C, hidden): 3x faster forward + backward than the einsum's layouts
        G = torch.baddbmm(p['b0'][:, None], s.flatten(2).transpose(1, 2), p['w0'][:, :, 2:].transpose(1, 2))
        G = (self.omega0 * G).view(N, h, w, -1)
        grid = lambda cy, cx: torch.stack(torch.broadcast_tensors(cy[:, :, None], cx[:, None, :]), -1)  # (N, S, S, 2)
        if self.mode == 'lppn':  # sub-cell coordinate of dual-cell position rho: 2 (((rho + S/2) mod S + 1/2) / S + o) - 1
            pos = (((torch.arange(S, device=s.device) + S // 2) % S) + 0.5) / S
            coord = grid(2 * (pos + off[:, :1]) - 1, 2 * (pos + off[:, 1:]) - 1)
            T = self.omega0 * torch.einsum('nkd,nabd->nabk', p['w0'][:, :, :2], coord)
        else:  # corner c = (c // 2, c % 2): the position relative to it, 2 (lambda + o - corner offset)
            lam = (torch.arange(S, device=s.device) + 0.5) / S
            coord = torch.stack([grid(2 * (lam + off[:, :1] - (c >> 1)), 2 * (lam + off[:, 1:] - (c & 1))) for c in range(4)], 1)
            T = self.omega0 * torch.einsum('nkd,ncabd->ncabk', p['w0'][:, :, :2], coord)
        return G.contiguous(), T.contiguous()

    def render(self, s, p, impl='cuda', offset=None):
        """state s (N, C, h, w) -> (N, out, S h, S w) fp32, differentiable w.r.t. s and the weights; offset (N, 2):
        per-image sample offsets in cells (training-time jitter), None for the regular grid."""
        G, T = self.tables(s, p, offset)
        V, c = self.omega * p['w'], self.omega * p['b']
        if impl == 'cuda':
            from ncaops import lppn_ops
            return lppn_ops.decode(G, T, V, c, p['wo'], p['bo'], offset)
        return decode_eager(G, T, V, c, p['wo'], p['bo'], offset)

    def jitter(self, N, device='cuda', generator=None):
        """Random sample offsets for training: uniform in [-1 / (2 S), 1 / (2 S)) cells per image and axis."""
        return (torch.rand(N, 2, device=device, generator=generator) - 0.5) / self.scale


def decode_eager(G, T, V, c, wo, bo, offset=None):
    """PyTorch version of lppn_ops.decode: G (N, h, w, H), T (N, S, S, H) or ensemble (N, 4, S, S, H), V (N, L, H, H),
    c (N, L, H), wo (N, O, H), bo (N, O), offset (N, 2) or None -> (N, O, S h, S w)."""
    N, h, w, Hd = G.shape
    S = T.shape[-2]
    off = torch.zeros(N, 2, device=G.device) if offset is None else offset
    ly = (torch.arange(S, device=G.device) + 0.5) / S + off[:, :1]  # (N, S)
    lx = (torch.arange(S, device=G.device) + 0.5) / S + off[:, 1:]
    wts = [((ly if cc >> 1 else 1 - ly)[:, :, None] * (lx if cc & 1 else 1 - lx)[:, None]) for cc in range(4)]  # (N, rho, kappa)
    corner = lambda cc: G.roll((-(cc >> 1), -(cc & 1)), (1, 2))[:, :, None, :, None]  # (N, h, 1, w, 1, H)

    def mlp(z):  # (N, h, rho, w, kappa, H) -> (N, h, w, rho, kappa, O)
        z = torch.sin(z.permute(0, 1, 3, 2, 4, 5).reshape(N, h * w * S * S, Hd))
        for l in range(V.shape[1]):
            z = torch.sin(z @ V[:, l].transpose(1, 2) + c[:, l, None])
        return (z @ wo.transpose(1, 2) + bo[:, None]).reshape(N, h, w, S, S, -1)

    if T.dim() == 4:
        out = mlp(sum(wts[cc][:, None, :, None, :, None] * corner(cc) for cc in range(4)) + T[:, None, :, None])
    else:
        out = sum(wts[cc][:, None, None, :, :, None] * mlp(corner(cc) + T[:, cc, None, :, None]) for cc in range(4))
    out = out.permute(0, 5, 1, 3, 2, 4).reshape(N, -1, h * S, w * S)
    return out.roll((S // 2, S // 2), (2, 3))  # dual-cell layout -> image pixels
