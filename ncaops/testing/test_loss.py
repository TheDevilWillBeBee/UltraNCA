"""Check the compiled VGG pooling replacement against PyTorch, including ties and non-finite inputs.
python ncaops/testing/test_loss.py
"""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from loss import _Pool2d


def test():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        pool = torch.compile(_Pool2d.apply, fullgraph=True)
        for layout in (torch.contiguous_format, torch.channels_last):
            # ReLU activations include many ties; also check the first maximum and last NaN choices.
            x = torch.randn(2, 16, 12, 18, device='cuda', dtype=dtype).relu().contiguous(memory_format=layout)
            x[0, 0, :2, :2] = float('nan')
            x[0, 1, :2, :2] = -float('inf')
            x[0, 2, :2, :2] = float('inf')
            x[0, 3, :2, :2] = 1
            x.requires_grad_()
            ref = F.max_pool2d(x, 2)
            y = pool(x)
            torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)
            g = torch.randn_like(ref)
            dx, = torch.autograd.grad(y, x, g)
            dx_ref, = torch.autograd.grad(ref, x, g)
            torch.testing.assert_close(dx, dx_ref, rtol=0, atol=0)
            print(f'pool {dtype} {layout}: exact output and gradient')
    print('all passed')


if __name__ == '__main__':
    test()
