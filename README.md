# UltraNCA: batched NCA training

Trains one NCA per target image: texture NCAs (`config.yml`) or growing NCAs (`config_growing.yml`, Distill "Growing Neural Cellular Automata": alive mask on alpha = channel 3, L2 loss to the premultiplied RGBA target, single living cell as the seed). K NCAs (one per target) train together in a single forward and backward pass, and each keeps exactly the semantics of an independent run:
- its own target, pool, batch, OT loss and per-tensor gradient normalisation;
- Adam (elementwise, so stacking the K weight sets changes nothing) with step learning-rate decay.

The only coupling is one random rollout length per iteration for the whole group.

The NCA is the standard texture NCA:
- 12 channels, 96 hidden units;
- perception = [identity, sobel_x, sobel_xᵀ, laplacian] with circular padding, feature index 4c + f;
- w1 (96×48) with bias, w2 (12×96) without bias;
- stochastic update with p = 0.5;
- RGB = state[:3] + 0.5.

The growing NCA has 16 channels and 128 hidden units; the stochastic update can be switched off (`update_rate: 1`).

Weights are saved in the vanilla format: `w1.weight` (96, 48, 1, 1), `w1.bias` (96,), `w2.weight` (12, 96, 1, 1).

## Quick start

```bash
pip install -r requirements.txt
python prepare/download_pbr.py      # 72 PBR albedo textures -> data/pbr_textures/*.jpg
python prepare/download_emoji.py    # 64 Noto emojis -> data/emoji/*.png (growing NCAs)
bash ncaops/build.sh                # CUDA extension (nanobind + CMake, sm_80+); needed for impl: cuda
python ncaops/testing/test_nca.py   # CUDA kernels vs a PyTorch reference (add --bench for timings)
python train.py                     # config.yml -> runs/pbr/<texture>.pt, <texture>.png (target | NCA), log.json
python train.py config_growing.yml  # growing NCAs -> runs/emoji
python benchmark.py train infer --K 32
```

## Files

| file | content |
|---|---|
| `config.yml` | texture task: data, init (`null` = random), NCA size, schedule, group size K, `impl`, loss options |
| `config_growing.yml` | growing task (same schedule, 16 / 128 NCA) |
| `train.py` | batched training loop, eager step, saving / previews / log (resumable: finished targets are skipped) |
| `loss.py` | OT texture loss on VGG16 features: relaxed EMD (cosine) + mean / covariance matching over relu1_1 … relu5_1, 1024 random positions per layer; target features computed once per group (`cache_targets`: also the target terms of the unsampled layers), optional bf16 VGG (`vgg_bf16`) |
| `nca_triton.py` | fused Triton step (`impl: triton`) |
| `ncaops/` | fused CUDA step (`impl: cuda`) |
| `ncaops/nca_step.cuh` | CUDA kernels, with the per-GPU configuration and tuning knobs at the top |
| `ncaops/nca_cuda.cu` | nanobind bindings |
| `ncaops/nca_ops.py` | autograd wrapper |
| `ncaops/build.sh`, `ncaops/CMakeLists.txt` | build |
| `ncaops/testing/test_nca.py` | correctness tests and benchmark |
| `ncaops/testing/profile_one_step.py` | one step, for profilers |
| `benchmark.py` | steady-state cost per training iteration (`train`) and of no-grad rollouts (`infer`) vs K and `impl` |
| `prepare/download_pbr.py` | PBR dataset download (same source as Cells2Pixels), albedo maps only |
| `prepare/download_emoji.py` | 64 Noto emojis (v2.047) for the growing task |

Default schedule (`config.yml`):
- batch 2 per NCA, pool 64, 768 iterations;
- lr 1e-3 halved at 256 and 512;
- rollout 32–95 steps;
- zero seed injected every 16 iterations;
- loss = OT + 10 × overflow (|x − clamp(x, −1, 1)| summed).

This is the batch-2 point of a batch-size study at a fixed sample budget. It came within about 3 % of the original 1500-iteration batch-4 schedule when started from a good (decoder-predicted) initialisation. From a **random** init (the default here), a longer schedule may be needed for full quality. That does not matter for speed work.

## What was done (sections 1–3 and the A100 check: one A100 80 GB; section 5: RTX 4090)

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
| `cuda` (current; measured on the 72 Cells2Pixels textures in the [A100 check](#a100-check-kernels-of-that-time)) | 3.4–3.5 | about 4.3× | about 20× | **86.1 ms** | 5.6 GB | – |

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
- in-kernel Philox update mask (one Philox call per 4 pixels; `nca_ops.update_mask` gives the same mask);
- templates on channels, hidden size, FP32 mode, stochastic update and alive mask (12/96 and 16/128 instantiated; add a line in `dispatch()` for others);
- fixed 4 filters, circular padding, any H × W.

All matmuls use `mma.sync.m16n8k8` TF32; a warp computes 16 pixels at a time (the MMA M dimension). Only the **updating pixels** (m = 1 and, for a growing NCA, alive) go through the MLP, in the forward and the backward: about half the matmul work at update rate 0.5.

| kernel | what it does |
|---|---|
| `forward_kernel` | A block sweeps a 16-pixel-wide column strip (a segment of rows) of an image: state rows go into a ring in shared memory (loads of the next phase issued before the current matmuls), the pixels that do not update copy x, and the updating ones are queued; every 128 queued pixels each of the 8 warps runs perception → h → relu → u on 16 of them. Weights are staged once per block in fragment-major order; h stays in registers between the two matmuls (permuted hidden index). Growing NCAs: rows and segments without a living cell within 2 pixels are skipped. |
| `alive_kernel` | growing NCAs: alive mask from the old and new alpha (3×3 max-pools), y ×= mask, mask stored for the backward |
| `pack_kernel` | per-NCA weights → B fragments for the backward matmuls |
| `backward_kernel` | Per tile (6 rows × 16 when stochastic, else 4 × 16), the updating pixels in rounds of 4 groups of 16: recompute p and h; stage p, relu(h) and g·m in shared memory; dW2 += (g·m)ᵀ relu(h); then per hidden n-tile dh = (g·m)W2·[h>0] (the sign read back from the lane's own staged relu(h)) → dp += dh W1, dh staged; dW1 += pᵀ dh and db1 += column sums of the staged dh (fixed order). Weight-gradient MMAs cover only the occupied k-steps of 8 pixels. 16/128 on 100 KB GPUs stages relu(h) / dh in two halves (2 blocks per SM). dp is pixel-major (N, H, W, 4C) and written only at updating pixels (full 32-byte sectors), which an updating-pixel map marks. Weight B fragments are read in 16-byte pairs. A block without an updating pixel (dead region of a growing NCA) writes no partial (a flag marks the written ones). |
| `reduce_kernel` | sums the flagged per-block partial weight gradients per NCA in a fixed order |
| `adjoint_kernel` | dx = g(·a) + Σ_f K_fᵀ ⋆ dp_f (transpose of the circular 3×3 filters) over the updating pixels' dp: the 9 neighbours of a pixel (offset or "not updating") are found once per pixel, then a thread per (pixel, pair of channels) reads its channels' 4 features of each updating neighbour (16-byte loads, 18 in flight); results transposed through shared memory for row-wise dx stores |

The backward and adjoint run in chunks of images on two streams so that dp stays in L2: a chunk's backward is about two thirds of a wave of resident blocks (room for the other stream's adjoint); growing NCAs, whose blocks mostly have nothing to do, use at least N / 4 images per chunk. Inference (`nca_ops.rollout`) loops over steps in C++.

**Per-GPU configuration.** The top of `nca_step.cuh` has one hardware constant, `NCA_SMEM_PER_SM_KB` (shared memory per SM: A100 164, RTX 30xx / 40xx 100, H100 228), which `build.sh` sets from the compute capability of the machine's GPU. Blocks per SM (forward) and the backward staging layout (one or two parts) derive from it; the L2 size (backward chunks), SM count and occupancy (forward segment length) are queried at run time. The remaining knobs (`NCA_FWD_WARPS`, `NCA_FWD_RING`, `NCA_BWD_TILES_PER_BLOCK`, `NCA_BWD_UNROLL`, `NCA_BWD_MIN_BLOCKS`, `NCA_ADJ_CPT`) are tuning choices measured on sm_89; on another GPU, rebuild and optionally sweep them, e.g. `bash ncaops/build.sh "-DNCA_FWD_RING=20 -DNCA_BWD_TILES_PER_BLOCK=4"`.

Correctness (`ncaops/testing/test_nca.py`, against PyTorch autograd with the same mask, for both sizes, update rate 0.5 / 1, texture / growing, partial tiles; also determinism, chunked backward and rollout = repeated steps):

| mode | forward | gradients |
|---|---|---|
| FP32 | 9e-7 | 2–5e-6 |
| TF32 | 4e-4 | 1–2e-2 |

The TF32 errors are the same as eager PyTorch TF32 vs FP32 on the same step; they come from TF32 itself.

### A100 check (kernels of that time)

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

### 5. RTX 4090 results (2026-10-14)

Per step, K = 32, B = 2, 128², update rate 0.5 (µs; "before" = the CUDA kernels before the updating-pixel work, commit `ebbfd18`):

| | forward before | forward now | backward before | backward now |
|---|---|---|---|---|
| texture 12/96 | 259 | 146 | 1054 | ~760 |
| 16/128 (no alive mask) | 591 | 273 | 1880 | ~1360 |
| growing 16/128 | 384 | 248 | 1212 | ~850 |

The backward kernel alone runs at about 45 % of the TF32 tensor peak (82.6 TFLOPS); its stalls are mostly weight-fragment loads (the per-NCA fragments, 42 / 72 KB, do not fit the ~28 KB of L1 left next to 98 KB of shared memory), the tensor pipe and barriers.

Training, 32 targets, 768 iterations, `config.yml` / `config_growing.yml` schedule, K = 32 (s per NCA, compilation excluded): texture 3.05 → 2.6, growing 2.8 → 2.5 (before the pixel-major dp). `benchmark.py train`: 130.8 → 96.5 ms per texture iteration (OT loss about 40 ms of it), 85.8 → 57.1 ms per growing iteration.

OT loss (compiled, forward + backward, 64 images of 128²): 47.0 → 39.9 ms with TF32 VGG, 26.1 ms with `vgg_bf16: true`. The relaxed-EMD backward now goes through the minimising pairs only (no dense 1024² gradient), VGG runs in channels_last, and `cache_targets` computes the target terms of the unsampled layers once (turn it off when the targets change at every call). bf16 VGG was checked on 32 textures × 2 seeds: training 11 % faster, OT loss evaluated in fp32 within seed noise (+1–2 % on average), previews indistinguishable; it stays opt-in.

Inference, 256 steps without gradients (ms):

| workload | texture before | texture now | growing before | growing now |
|---|---|---|---|---|
| K = 32, 64² | 9.0 | 6.7 | 14.0 | 10.6 |
| K = 32, 128² | 29.9 | 17.7 | 26.2 | 21.6 |
| 1 NCA, 128² | 3.05 | 2.1 | 4.9 | 3.4 |
| 1 NCA, 512² | 14.9 | 10.5 | 9.8 | 8.2 |
| 1 NCA, 1024² | 59.5 | 37.1 | 29.5 | 28.8 |
| 1 NCA, 2048² | 231 | 143 | 116 | 107 |

Growing at 2048² is dominated by `alive_kernel` writing zeros to the dead cells (about 350 µs of 420 µs per step, near DRAM bandwidth).

### 6. Second RTX 4090 round (2026-10-15)

Per step, K = 32, B = 2, 128², update rate 0.5, forward + backward of a 64-step rollout (µs per step, interleaved runs of the two builds, median of 5; ±2 % run to run):

| | before (`6128814`) | now |
|---|---|---|
| texture 12/96 | 857–866 | 834–856 |
| growing 16/128, trained weights | 888–904 | 611–646 |

Kernels (single launch, texture): backward 696 → ~500 µs (115 M instead of 145 M instructions for the same 9.7 M MMAs: the relu mask is read back from the staged activations instead of a 64-bit bit mask, db1 is the column sum of the staged dh instead of warp shuffles, the weight-gradient MMAs cover only the occupied k-steps, the tile loads compute one address per pixel), adjoint 299 → ~220, reduce 46 → 39. Growing: backward 323 → ~270, adjoint 162 → ~70 (one neighbour table per pixel; most pixels are dead), reduce 82 → 18 (dead blocks write and sum nothing), chunks of N / 4 images (the 8-image chunks of the texture rule cost 100 µs of nearly empty launches).

The texture step gained less than its kernels: the adjoint used to run largely in the shadow of the slower backward launches (the two kernels interleave on the SMs; they cannot share one, since 2 backward blocks use the whole register file), and now the step is close to the plain sum of its kernels. `benchmark.py train`: texture 96.0 → 93.9 ms per iteration (the OT loss is 40 ms of it), growing 55.6 → 39.2 ms; the inference numbers of section 5 are unchanged (the forward kernel was not touched: pairing its weight-fragment loads into 16-byte loads gained nothing and spilled registers in the growing variant). Quality (32 textures / 32 emojis, 768 iterations, seed 0): median final loss ratio new / old 0.996 / 0.998.

**What did not help this round** (all measured):
- the adjoint fused into the backward kernel (each block owns a band of rows, also computes the dp of the two halo rows, and gathers gx for finished rows from L2): the gather is latency-bound at 8 warps per SM (+200 µs) and the extra live state spills;
- the weight-gradient reduction by the last block of each image / NCA (atomic counter): one block cannot keep enough loads in flight (75 µs vs 45 for the reduce kernel);
- one thread per pixel in the adjoint: 16-byte loads at a 192-byte stride are 32 L1 wavefronts per instruction (240 → 334 µs on texture; it is what helps growing, where the neighbour table does the same job);
- a 240-register cap on the backward so that small adjoint blocks can co-reside (one block per SM is far too little memory parallelism), streaming cache hints on the backward's x / gy loads (the halo rows are re-read by the next tile row), 16 tiles per block (backward +13, reduce −23 µs), 7-row tiles (all four warps busy in one round: −2–3 % on texture, within the noise, and +6 % on growing);
- OT loss: cuDNN 9.2 runs a `convertTensor` pass (TF32 rounding of the input) before every TF32 convolution and its gradient, 3.9 of the 40 ms; `cudnn.benchmark`, `allow_tf32 = False`, inductor's Triton convolutions (`max-autotune`) and the input gradient as a forward convolution with the transposed weights are no faster.

## Planned next steps (in rough order of expected gain)

1. **Backward occupancy / latency**: 255 registers and ~47 KB of shared memory per 4-warp block give 8 warps per SM, and the kernel issues every 9 cycles per warp (stalls: tensor pipe 2.4, weight-fragment loads 2.0, dependencies 1.7, barriers 1.6 cycles per instruction). On the A100 / H100 (164 / 228 KB), an 8-warp block of two 4-warp teams with the weight fragments resident in shared memory (~140 KB) would remove the fragment-load stalls; it does not fit 100 KB. Tried and dropped: two warps per pixel group (8-warp blocks at 128 registers, 16 warps per SM): twice the instructions per MMA, 15–30 % slower; a 168-register cap (spills).
2. **Fusing the adjoint** stays a loss on 100 KB GPUs (see above); with 164 KB or more, the gather could stage a tile's dp in shared memory instead of reading it 9 times from L2.
3. **OT loss** (40 ms of a 93 ms texture iteration): the max-pool backward reads int64 indices (2.5 ms; a backward that recomputes the argmax from the saved input would halve it), the gradient of the 1024 sampled feature positions is materialized as a dense scatter per layer (~1 ms), and the cuDNN TF32 conversion passes (3.9 ms) would only go away with a different convolution implementation; bf16 VGG (validated, opt-in) halves the whole loss.
4. **Sparse growing grids**: skip the zero writes of dead cells (e.g. track the live bounding box) for very large, mostly empty grids.
5. **Other GPUs**: H100 `wgmma`; other (C, HID) instantiations (`dispatch()` in `nca_cuda.cu`; C a multiple of 4, HID a multiple of 32).
