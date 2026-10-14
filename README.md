# UltraNCA: batched texture-NCA training

Trains one texture NCA per target image. K NCAs (one per target) train together in a single forward and backward pass, and each keeps exactly the semantics of an independent run:
- its own target, pool, batch, OT loss and per-tensor gradient normalisation;
- Adam (elementwise, so stacking the K weight sets changes nothing) with step learning-rate decay.

The only coupling is one random rollout length per iteration for the whole group.

The NCA is the standard texture NCA:
- 12 channels, 96 hidden units;
- perception = [identity, sobel_x, sobel_xᵀ, laplacian] with circular padding, feature index 4c + f;
- w1 (96×48) with bias, w2 (12×96) without bias;
- stochastic update with p = 0.5;
- RGB = state[:3] + 0.5.

Weights are saved in the vanilla format: `w1.weight` (96, 48, 1, 1), `w1.bias` (96,), `w2.weight` (12, 96, 1, 1).

## Quick start

```bash
pip install -r requirements.txt
python prepare/download_pbr.py      # 72 PBR albedo textures -> data/pbr_textures/*.jpg
bash ncaops/build.sh                # CUDA extension (nanobind + CMake, sm_80+); needed for impl: cuda
python ncaops/testing/test_nca.py   # CUDA kernels vs a PyTorch reference (add --bench for timings)
python train.py                     # config.yml -> runs/pbr/<texture>.pt, <texture>.png (target | NCA), log.json
python benchmark.py --impl eager triton cuda --K 1 24
```

## Files

| file | content |
|---|---|
| `config.yml` | data, init (`null` = random), NCA size, schedule, group size K, `impl` |
| `train.py` | batched training loop, eager step, saving / previews / log (resumable: finished targets are skipped) |
| `loss.py` | OT texture loss on VGG16 features: relaxed EMD (cosine) + mean / covariance matching over relu1_1 … relu5_1, 1024 random positions per layer; target features computed once per group |
| `nca_triton.py` | fused Triton step (`impl: triton`) |
| `ncaops/` | fused CUDA step (`impl: cuda`) |
| `ncaops/nca_step.cuh` | CUDA kernels, with tuning knobs at the top |
| `ncaops/nca_cuda.cu` | nanobind bindings |
| `ncaops/nca_ops.py` | autograd wrapper |
| `ncaops/build.sh`, `ncaops/CMakeLists.txt` | build |
| `ncaops/testing/test_nca.py` | correctness tests and benchmark |
| `ncaops/testing/profile_one_step.py` | one step, for profilers |
| `benchmark.py` | steady-state cost per training iteration vs K and `impl` |
| `prepare/download_pbr.py` | PBR dataset download (same source as Cells2Pixels), albedo maps only |

Default schedule (`config.yml`):
- batch 2 per NCA, pool 64, 768 iterations;
- lr 1e-3 halved at 256 and 512;
- rollout 32–95 steps;
- zero seed injected every 16 iterations;
- loss = OT + 10 × overflow (|x − clamp(x, −1, 1)| summed).

This is the batch-2 point of a batch-size study at a fixed sample budget. It came within about 3 % of the original 1500-iteration batch-4 schedule when started from a good (decoder-predicted) initialisation. From a **random** init (the default here), a longer schedule may be needed for full quality. That does not matter for speed work.

## What was done (measured on one A100 80 GB)

### 1. Batching NCAs on different targets

- One NCA alone leaves the GPU mostly idle: about 1 ms per step whatever the implementation, launch- and dispatch-bound.
- Training K NCAs in one pass amortises that overhead.
- Running separate processes in parallel instead gives only 1.24× (4 processes).

### 2. Why einsum / linear looked slower than conv2d

1. **Matmul precision default.** PyTorch runs matmuls (bmm, einsum, linear) in strict FP32 by default (`torch.backends.cuda.matmul.allow_tf32 = False`), but cuDNN convolutions in TF32. Batched matmuls are 2–8× slower in FP32 for these skinny shapes. Enable `torch.set_float32_matmul_precision('high')`.
2. **Host overhead for a single NCA.** At K = 1 the GPU work per step is tiny. einsum and linear do less GPU work than conv2d (0.24 vs 0.58 ms for the two layers' forward + backward), but more host dispatch per call. `torch.compile` of the step cuts a single-NCA rollout by 30 %.

### 3. Implementations

Measurements are on 24 DyNCA PBR textures, K = 24, batch 2, 768 iterations, starting from a transformer-decoder init. Compilation is excluded (the first iteration of each group is timed separately; it takes about 2 s with warm caches).

| `impl` | training s per NCA | vs eager | vs one NCA at a time (69.6 s) | 64-step rollout fwd+bwd | peak GPU memory | final OT vs independent training (median per-texture ratio) |
|---|---|---|---|---|---|---|
| `eager` (batched matmul, TF32) | 15.1 | 1× | 4.6× | 431 ms | 52 GB | 0.998 |
| `compile` (`torch.compile` step and loss) | 11.4 | 1.33× | 6.1× | 338 ms | 57 GB | 1.002 |
| `triton` | 5.6 | 2.7× | 12.5× | 153 ms | 8.6 GB | 1.005 |
| `cuda` (version of 13:30) | 3.9 | 3.9× | 17.9× | 99.5 ms | ~7 GB | 1.002 |
| `cuda` (current; measured on the 72 Cells2Pixels textures in the [standalone check](#standalone-check-current-code-this-folder)) | 3.4–3.5 | about 4.3× | about 20× | **86.1 ms** | 5.6 GB | – |

- **Quality is the same for every implementation.** An apparent 0.7 % Triton gap disappeared on re-runs. Exact-fp32 Triton was equal too, so TF32 is not a quality issue.
- **Group size.** Speed saturates around K ≈ 24–32: Triton gives 30.2 / 8.2 / 6.1 / 5.7 / 5.5 / 5.4 s per NCA at K = 1 / 4 / 12 / 24 / 32 / ≥48. Pick a K that divides the number of targets evenly; a smaller last group is slower per NCA.
- **Where the time goes** (eager, K = 24): the rollout is about 90 % of an iteration, and its backward costs about 2.4× its forward. With the CUDA step, the loss (compiled VGG + OT, about 23 ms per iteration) becomes about 20 % of an iteration.
- **What does not help:**
  - bf16 autocast in eager mode (slower);
  - `cudnn.benchmark`;
  - CUDA graphs: the GPU is already saturated at these K values;
  - K > 32.

### 4. CUDA kernels (`ncaops/`)

Design decisions (agreed with the user):
- TF32 tensor cores, with an FP32 mode via 3xTF32 (hi/lo operand split, three MMAs; matches fp32 to about 1e-6);
- deterministic weight-gradient reductions (bitwise reproducible);
- in-kernel Philox update mask;
- templates on channels and hidden size (12/96 instantiated; add a line in `dispatch()` for others);
- fixed 4 filters, circular padding, any H × W.

All matmuls use `mma.sync.m16n8k8` TF32. One warp handles 16 pixels of an image row, which is the MMA M dimension.

| kernel | what it does | time per step (K = 24, B = 2, 128², TF32) |
|---|---|---|
| `forward_kernel` | stages the state tile + circular halo in shared memory, and the weights once per block in fragment-major order (each lane reads its pre-converted B fragment: no bank conflicts, no conversions in the inner loops); perception → h → relu → u → y = x + u·m. The hidden activations stay in registers between the two matmuls: the accumulator columns (2t, 2t+1) are reused as A-operand columns (t, t+4) by permuting W2's hidden index. | 0.25 ms |
| `pack_kernel` | per-NCA weights → B fragments for the backward matmuls | 0.01 ms |
| `backward_kernel` | recomputes p and h, then dh = (g·m)W2·[h>0] and dp = dh·W1 (MMA, registers; dp → global). Weight gradients as [p \| 1]ᵀ·dh and (g·m)ᵀ·relu(h) over each 64-pixel block tile. The ones column gives db1; each warp owns 3 hidden n-tiles, so fragments are reused across output tiles. Shared-memory operands are stored already rounded to TF32. | 0.80 ms |
| `reduce_kernel` | sums the per-block partial weight gradients per NCA in a fixed order | 0.03 ms |
| `adjoint_kernel` | dx = g + Σ_f K_fᵀ ⋆ dp_f (transpose of the circular 3×3 filters); one thread per (pixel, channel), taps through L1 | 0.24 ms |

Correctness (`ncaops/testing/test_nca.py`, against PyTorch autograd with the same mask, including partial tiles):

| mode | forward | dx | dW1 / db1 / dW2 |
|---|---|---|---|
| FP32 | 9e-7 | 7e-7 | 2–7e-6 |
| TF32 | 3e-4 | 1e-2 | 1e-2 / 1e-2 / 4e-4 |

The TF32 errors are the same as eager PyTorch TF32 vs FP32 on the same step; they come from TF32 itself.

What moved the needle on the backward:

| change | backward kernel | 64-step rollout |
|---|---|---|
| first version | 1.83 ms | 171 ms |
| operands stored in shared memory pre-rounded to TF32 (the inner loops were bound by `cvt.rna.tf32`) | 1.38 ms | 135 ms |
| weight gradients reformulated with warp-owned hidden tiles (about 3× fewer shared-memory loads per MMA) | 0.83 ms | 99.5 ms |
| adjoint: shared-memory staging → direct L1 version (0.41 → 0.24 ms) | | 88 ms |
| tile tuning (`TILE_H` 16, 2 tiles per forward block, 8 per backward block) | 0.80 ms | 86 ms |

What did **not** help:
- k-outer loop reordering or split accumulators (more registers → spills);
- holding filter taps in registers;
- `NCA_BWD_MIN_BLOCKS` = 3 (spills; 1.14 ms).

Tuning knobs at the top of `nca_step.cuh` can be overridden at build time, e.g. `bash ncaops/build.sh "-DNCA_TILE_H=8 -DNCA_TILES_PER_BLOCK=8"`.

**Profiling note.** Nsight Compute had no performance-counter permission on the development pod (`ERR_NVGPUCTRPERM`). Kernels were analysed by ablation (the `NCA_ABLATE` macro skips the weight gradients and/or the dp stores) and with `torch.profiler`. Ablation of the current backward kernel: about 0.2 ms of its 0.80 ms are the dp stores.

### Standalone check (current code, this folder)

Run on 2026-10-14, A100 80 GB, with `prepare/download_pbr.py`, `ncaops/build.sh` and the default `config.yml` (72 PBR albedos, random init, `impl: cuda`, K = 24, 3 groups):
- training: 3.4–3.5 s per NCA (compilation excluded);
- the previews (`runs/pbr/*.png`) reproduce the textures;
- `ncaops/testing/test_nca.py` passes.

`python benchmark.py --impl eager triton cuda --K 1 24` (steady state, s per NCA for 768 iterations):

| impl | K = 1 | K = 24 | ms per iteration at K = 24 | peak GPU memory at K = 24 |
|---|---|---|---|---|
| eager | 84.2 | 14.6 | 456 | 32.9 GB |
| triton | 33.0 | 5.7 | 178 | 5.9 GB |
| **cuda** | **17.2** | **3.6** | **111** | **5.6 GB** |

## Planned next steps (in rough order of expected gain)

1. **Fuse the adjoint into the backward kernel**, removing the dp round trip through global memory: about 0.2 ms of stores plus the 0.24 ms adjoint kernel, roughly 30 % of the backward.
   - Recompute a 1-pixel halo: output tiles 14 columns wide, so a 16-pixel MMA row covers the halo on both sides, plus TH + 2 rows.
   - Stage dp for the tile and halo in shared memory, then apply the transposed stencil for the interior.
   - Weight gradients must count interior pixels only: zero g·m and dh for halo pixels in the weight-gradient staging, but not in the dp computation.
   - The recompute overhead is about (TH+2)/TH × 16/14: 1.43× at TH = 8.
2. **Backward register pressure.** It uses 255 registers with small spills at 2 blocks per SM (8 warps per SM, low occupancy). Options:
   - split dp and weight gradients into two kernels;
   - use 8 warps per block with half the per-warp accumulators;
   - move the weight-gradient accumulators to shared memory.
3. **Forward** is about 4× above a rough bound of 0.07 ms per step (MMA count, shared-memory loads, and about 90 MB of traffic). Ideas:
   - `cp.async` double-buffering of the next state tile;
   - computing the 4 filters per (pixel, channel) once instead of one filter per lane;
   - 8 warps.
4. **OT loss** (about 20 % of an iteration with the CUDA step):
   - `argsort` of a random vector over 16k positions per image and layer for the subsampling (use cheaper sampling);
   - VGG in bf16 or channels-last;
   - fusing the cosine-distance min-reductions.
5. **Philox mask:** currently 4 lanes compute the same pixel's mask (redundant). Compute it once per pixel and shuffle, or once per 4 pixels from the 4 Philox output words.
6. **Multi-step fusion** (several NCA steps per launch with halo growth) for the no-grad parts of a rollout, if truncated backpropagation is ever used.
7. **Larger targets / other GPUs:**
   - H100 `wgmma`;
   - other (C, HID) instantiations (`dispatch()` in `nca_cuda.cu`; HID must be a multiple of 32 (8 per n-tile × 4 warps) and C even).
