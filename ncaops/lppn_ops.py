"""PyTorch wrapper of the fused LPPN decoder kernels (lppn_cuda, built by build.sh; see lppn.py and lppn_decoder.cuh).
decode(G, T, V, c, wo, bo) has the signature of lppn.decode_eager: G (N, h, w, hidden), T (N, S, S, hidden) or, local
ensemble, (N, 4, S, S, hidden), V (N, layers, hidden, hidden), c (N, layers, hidden), wo (N, out, hidden), bo (N, out)
-> (N, out, S h, S w); offset (N, 2), optional: per-image sample offsets in cells (training-time jitter, |offset| <=
1 / (2 S); the tables must be computed for the same offsets: lppn.LPPN.tables)."""
import torch

try:
    from . import lppn_cuda
except ImportError:
    import lppn_cuda


def _stream(t):
    return torch.cuda.current_stream(t.device).cuda_stream


class _Decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G, T, V, c, wo, bo, offset):
        N, h, w, _ = G.shape
        S, O = T.shape[-2], wo.shape[1]
        out = torch.empty(N, O, S * h, S * w, device=G.device)
        lppn_cuda.forward(G, T, V, c, wo, bo, offset, out, _stream(G))
        ctx.save_for_backward(G, T, V, c, wo, offset)
        return out

    @staticmethod
    def backward(ctx, dout):
        G, T, V, c, wo, offset = ctx.saved_tensors
        N, h, w, Hd = G.shape
        S, O, L, ens = T.shape[-2], wo.shape[1], V.shape[1], T.dim() == 5
        dout = dout.contiguous()
        # per-image power of two with max |dout * gscale| in [1, 2): fp16 operands of the backward matmuls stay in range
        amax = torch.linalg.vector_norm(dout.view(N, -1), float('inf'), dim=1)  # one pass, no |dout| temporary
        gscale = torch.where(amax > 0, torch.exp2(-torch.floor(torch.log2(amax))), torch.ones_like(amax))
        size, bands = lppn_cuda.workspace(S, O, h, ens, Hd)
        partial = torch.empty(N, bands, size, device=G.device)
        edges = torch.empty((N, 4, h, w, Hd) if ens else (N, bands, 2, w, Hd), device=G.device)
        reduced = torch.empty(N, size, device=G.device)
        dG = torch.empty_like(G)
        lppn_cuda.backward(G, T, V, c, wo, dout, gscale, offset, dG, partial, edges, reduced, _stream(G))
        sizes = [L * Hd * Hd, L * Hd, O * Hd, 8, T[0].numel()]
        dV, dc, dwo, dbo, dT = reduced.split(sizes, 1)
        return (dG, dT.reshape(T.shape), dV.reshape(N, L, Hd, Hd), dc.reshape(N, L, Hd), dwo.reshape(N, O, Hd),
                dbo[:, :O], None)


def decode(G, T, V, c, wo, bo, offset=None):
    offset = torch.empty(0, device=G.device) if offset is None else offset.detach().float().contiguous()
    return _Decode.apply(G.contiguous(), T.contiguous(), V.contiguous(), c.contiguous(), wo.contiguous(), bo.contiguous(), offset)
