"""Check fused convolution input gradients + unpooling against a PyTorch unpool reference.
Run with python ncaops/testing/test_vgg_unpool.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ncaops import vgg_cuda
from ncaops.vgg_ops import VGG16Features


def unpool_reference(g, idx):
    N, H, W, C = g.shape
    out = torch.empty(N, 2 * H, 2 * W, C, device=g.device, dtype=g.dtype)
    for k in range(4):
        out[:, k // 2::2, k % 2::2] = torch.where(idx == k, g, 0)
    return out


class SeparateUnpool(VGG16Features):
    """Test-only reference; production always uses fused unpooling."""

    def dgrad_unpool(self, dz, i, act, idx):
        return unpool_reference(self.dgrad(dz, i, act), idx)


def check_kernel():
    torch.manual_seed(482)
    for acc16 in (False, True):
        for N, H, W, C, K in ((2, 3, 5, 64, 128), (1, 8, 12, 128, 256), (1, 16, 16, 256, 512), (1, 4, 4, 512, 512)):
            dy = torch.randn(N, H, W, K, device='cuda', dtype=torch.float16) * 0.1
            w = torch.randn(K, 3, 3, C, device='cuda', dtype=torch.float16) * 0.01
            act = torch.randn(N, H, W, C, device='cuda', dtype=torch.float16).relu()
            idx = torch.randint(4, act.shape, device='cuda', dtype=torch.uint8)
            pooled = torch.empty_like(act)
            actual = torch.empty(N, 2 * H, 2 * W, C, device='cuda', dtype=torch.float16)

            def reference(cfg):
                vgg_cuda.dgrad_mask(cfg, acc16, dy.data_ptr(), w.data_ptr(), act.data_ptr(), pooled.data_ptr(),
                                    N, H, W, C, K, torch.cuda.current_stream().cuda_stream)
                return unpool_reference(pooled, idx)

            def fused(cfg):
                vgg_cuda.dgrad_unpool(cfg, acc16, dy.data_ptr(), w.data_ptr(), act.data_ptr(), idx.data_ptr(),
                                     actual.data_ptr(), N, H, W, C, K, torch.cuda.current_stream().cuda_stream)

            for cfg in range(3):
                expected = reference(cfg)
                actual.fill_(float('nan'))
                fused(cfg)
                torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused(0)
            dy.mul_(2)
            graph.replay()
            torch.testing.assert_close(actual.view(torch.int16), reference(0).view(torch.int16), rtol=0, atol=0)
    print('fused dgrad + unpool: bitwise equal for both accumulations, every tile, rectangular/partial tiles, '
          'all argmax positions, zero ReLU masks, and graph replay', flush=True)


def check_wrapper():
    for accumulate in ('fp32', 'fp16'):
        reference = SeparateUnpool(accumulate).cuda()
        fused = VGG16Features(accumulate).cuda()
        fused.tiles = reference.tiles  # identical forward and ordinary backward tile choices
        image = torch.rand(2, 3, 32, 48, device='cuda')
        positions = [torch.rand(2, (32 >> l) * (48 >> l), device='cuda').topk(4, 1).indices for l in range(5)]
        for normalized in (False, True):
            source = ((image - fused.mean[None, :, None, None]) / fused.std[None, :, None, None]).permute(0, 2, 3, 1) \
                if normalized else image
            for highest in (0, 2, 4):
                results = []
                for m in (reference, fused):
                    # Compare identical convolution tiling; independent autotuning can change accumulation order.
                    for key, cfg in list(m.tiles.items()):
                        if key[0] == 'dgrad_mask':
                            m.tiles[('dgrad_unpool',) + key[1:]] = cfg
                    x = source.detach().requires_grad_()
                    maps, samples = m(x, positions=positions, normalized=normalized)
                    value = maps[highest].float().square().mean() + samples[highest].square().mean()
                    value.backward()
                    results.append(x.grad)
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    print('PyTorch wrapper: bitwise equal image gradients, both accumulations, RGB/normalized inputs, '
          'dense + sampled gradients and partial backward paths', flush=True)


if __name__ == '__main__':
    check_kernel()
    check_wrapper()
