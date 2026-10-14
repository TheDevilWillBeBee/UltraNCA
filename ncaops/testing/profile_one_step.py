"""One forward + backward NCA step at K = 24, B = 2, 128 x 128, for Nsight Compute:
ncu --kernel-name regex:"backward_kernel|adjoint_kernel" --set full python testing/profile_one_step.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nca_ops
from test_nca import make
x, w1, b1, w2 = make(24, 2)
p = [t.requires_grad_() for t in (x, w1, b1, w2)]
nca_ops.step(*p, 0.5).square().mean().backward()
torch.cuda.synchronize()
