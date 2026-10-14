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

## What was done (sections 1–3 and the A100 check: one A100 80 GB; sections 5–6: RTX 4090; section 7: A100 tuning)

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

**Per-GPU configuration.** The top of `nca_step.cuh` has one hardware constant, `NCA_SMEM_PER_SM_KB` (shared memory per SM: A100 164, RTX 30xx / 40xx 100, H100 228), which `build.sh` sets from the compute capability of the machine's GPU. Blocks per SM (forward) and the backward staging layout (one or two parts) derive from it; the L2 size (backward chunks), SM count and occupancy (forward segment length) are queried at run time. The remaining knobs (`NCA_FWD_WARPS`, `NCA_FWD_RING`, `NCA_BWD_TILES_PER_BLOCK`, `NCA_BWD_UNROLL`, `NCA_BWD_MIN_BLOCKS`, `NCA_ADJ_CPT`, `NCA_FWD_MIN_BLOCKS`, `NCA_FWD_TAIL`, `NCA_BWD_TILE_ROWS`, `NCA_BWD_HPARTS`) are tuning choices measured on sm_89, with A100-specific values behind `NCA_A100_CLASS` (164 KB or more per SM; section 7); on another GPU, rebuild and optionally sweep them, e.g. `bash ncaops/build.sh "-DNCA_FWD_RING=20 -DNCA_BWD_TILES_PER_BLOCK=4"`.

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

### 7. A100 (2026-10-15)

One A100-SXM4 80 GB (108 SMs at 1.41 GHz, 164 KB of shared memory per SM, 40 MB L2), CUDA 12.6, PyTorch 2.14. `build.sh` sets `NCA_SMEM_PER_SM_KB=164`; everything below is conditional on that (`NCA_A100_CLASS` in `nca_step.cuh`) or on a run-time check of the shared memory per SM (`chunk_images()`), so the sm_89 configuration is unchanged. Nsight Compute could not read the hardware counters on this machine (`ERR_NVGPUCTRPERM`, even as root), so the kernels were taken apart by experiment: ablation builds (`NCA_ABLATE` bits: no weight gradients, no dp stores, weight fragments made up in registers, dp stores all to one record, no barriers around the dW2 phase), the torch profiler's kernel times, and A/B builds of every knob. One lesson: a build running on the CPU during a timing run inflates the step time by up to 15 % (the host enqueue slows down and the GPU idles); the numbers here were taken with an idle CPU, interleaving the two builds.

Per step, K = 32, B = 2, 128², update rate 0.5, forward + backward of a 32-step rollout (µs; growing NCAs with trained weights, states grown 64 steps from the seed = 5 % alive cells, or 256 steps = 30 %):

| | sm_89 configuration on the A100 | A100 configuration | RTX 4090 (section 6) |
|---|---|---|---|
| texture 12/96, step | 1296 | **1165** | 834–856 |
| texture kernels: forward / backward / adjoint / reduce + pack | 204 / 692 / 246 / 38 | 203 / 659 / 246 / 38 | 146 / ~500 / ~220 / ~50 |
| growing 16/128, 5 % alive, step | 939 | **629** | 611–646 |
| growing kernels: forward / alive / backward / adjoint / reduce + pack | 290 / 55 / 306 / 192 / 40 | 147 / 54 / 236 / 135 / 40 | 248 / – / ~270 / ~70 / 18 |
| growing 16/128, 30 % alive, step | 1374 | **1085** | – |
| forward only, texture / growing (5 %) | 205 / 290 | 204 / 202 | – |

The A100 has 2.1× fewer SM-cycles than the 4090 (108 × 1.41 vs 128 × 2.52 GHz) and its texture step is 1.4× slower; per SM-cycle it is the more efficient of the two. Both are register-file bound (64 K registers per SM): the backward runs at 8 warps per SM (255 registers), the forward at 16, and with 4× faster TF32 MMAs per SM-cycle the A100 kernels are even more latency-bound than on the 4090 (texture backward: 16 % of the TF32 peak).

**Configuration chosen** (all measured; the knob names are those of `nca_step.cuh` / `nca_cuda.cu`):
1. **Forward 16/128 at one block per SM.** On 164 KB two of its blocks fit, which capped the kernel at 128 registers with 264 bytes of spills; one block at 193 registers is twice as fast (growing forward 290 → 147 µs; growing inference 47 → 35 ms for K = 32 at 128² × 256 steps, 96 → 81 ms for one NCA at 2048²). `fwd_min_blocks()` now asks for two blocks only when a group's accumulators fit the 128-register budget (HID / 2 + 2 C ≤ 72: 12/96 yes, 16/128 no); on the 4090 nothing changes (16/128 was at one block there because two do not fit 100 KB). 12/96 at one block is 36 % slower.
2. **Forward split tail** (`NCA_FWD_TAIL`): 512 blocks on 216 slots are 2.4 waves; the items of the last partial wave run as two half-height blocks each (2.5 % faster forward on both tasks). Shorter segments for everyone were slower (texture: 64 rows +4 %, 32 rows +42 %).
3. **Backward `NCA_BWD_UNROLL` 6** (texture backward 688 → 659 µs; 3, 4 and 12 are not better; 7-row tiles alone gain 3 % but add nothing on top of unroll 6, `NCA_BWD_TILE_ROWS` stays 6).
4. **Growing backward: dead tiles load nothing.** The alive map (3 sectors per tile) is read before the x tile, which is fetched only if a pixel updates: 306 → 240 µs at 5 % alive, 555 → 523 at 30 % (on sm_89 the tile loads still overlap the mask computation; the change is probably good there too but was not measured).
5. **Growing adjoint: 4 channels per thread** (`adj_cpt()`: 192 → 130 µs at 5 % alive, 233 → 210 at 30 %; fewer threads per mostly dead pixel, more blocks in flight). Texture stays at 2: 4 gives 365 µs instead of 244, 1 gives 297.
6. **One launch of the backward and adjoint** (`chunk_images()` returns N): chunks of 12–24 texture images on the two streams give 1170–1245 µs per step against 1162 for one launch, chunks that keep dp in L2 run serially on one stream are all slower (4–16 images: 1400–1900), and growing chunks of 10 images cost 10 %. (On the 4090 the chunks gain 15–20 %; the A100 kernels stretch each other when they share the SMs.)
7. **16/128 backward with relu(h) / dh staged in one part** (automatic from the 164 KB; two parts, which leave more L1, are 10 % slower).

Texture backward, where the time goes (688 µs, sm_89 configuration; `NCA_ABLATE`): the weight gradients 205 µs, the dp stores 144 (102 of them the DRAM write traffic of 100 MB per step, the rest the instructions), the weight-fragment loads 132, the rest (perception, h, dh, dp MMAs, staging, tile loads, barriers) 410. The fragment loads are not L1-capacity misses: a 28 KB L1 instead of 92 KB (shared-memory carveout forced to 164 KB) costs only 3.5 %, so the 8-warp block with the fragments resident in shared memory ("Planned next steps" item 1) was not built; the barriers around the dW2 phase cost 10 µs, so merging the weight-gradient phases (a separate dh buffer fits on 164 KB) was dropped too.

**What did not help** (A100, all measured):
- 3 backward blocks per SM (`NCA_BWD_MIN_BLOCKS=3`, 168 registers with 20 bytes of spills): texture +12 %, growing +42 % (three blocks also leave 16 KB of L1);
- 4 or 16 backward tiles per block (16: backward +70 µs for a 10 µs cheaper reduce);
- a 42-row forward ring, a forward segment override (`NCA_FWD_SEG` env, removed);
- adjoint variants: 1 channel per thread (texture +22 %), 2-D patches of 2 or 4 rows per block (+8 to +36 %), the patch's dp records staged in shared memory with coalesced loads (texture equal at 2 rows, growing +14 %; the kernel is latency- and occupancy-bound, not L1-bound);
- evict-first (`__stcs`) dp stores (no change);
- the serial-chunk and carveout experiments above (env knobs, removed);
- `NCA_BWD_HPARTS=2` for 16/128 (+10 %).

`benchmark.py` and `train.py` (K = 32, the usual schedules), A100, sm_89 configuration → A100 configuration:

| | before | after |
|---|---|---|
| `benchmark.py train` texture, ms per iteration (OT loss about 27 ms of it) | 110.4 | 101.6 (K = 64: 193.9) |
| `benchmark.py train` growing, ms per iteration | 64.9 | 43.2 (K = 64: 80.4) |
| `train.py` texture / growing, s per NCA (32 targets, compilation excluded) | 2.6 / 2.5 | 2.4 / 2.1 |
| inference K = 32, 128², 256 steps, texture / growing (ms) | 30.0 / 47.4 | 29.9 / 35.0 |
| inference 1 NCA, 2048², 256 steps, texture / growing (ms) | 195.7 / 96.3 | 190.0 / 81.2 (1024²: 52.5 / 51.2 → 51.7 / 31.6) |
| `test_nca.py --bench` 64-step texture rollout fwd+bwd / 80-step growing (B = 8, 72²) | 82.8 / 192.8 | 74.4 / 174.4 |

Quality (final losses of `train.py` on the first 32 textures / 32 emojis, seed 0, A100 configuration vs the sm_89 one on the same GPU): the 32 growing NCAs end with bitwise identical losses (the kernels compute the same values in the same order; the changes only move work between blocks and launches); the textures have a median loss ratio of 1.0003 (mean 1.0003, per texture 0.945–1.047), which is the run-to-run noise of the OT loss (its `scatter_add_` and cuDNN are not bitwise deterministic: two runs of the unmodified kernels differ by the same amount, median ratio 0.997, per texture 0.934–1.049).

The OT loss was not touched: it is about 27 ms of a texture iteration on the A100 (`benchmark.py train` minus 64 steps: 25 % of the iteration; 40 ms and 40 % on the 4090).

`test_nca.py`: one 16/128 growing case at update rate 1 has a cell whose new alpha is 0.100131 in fp32 and falls below the alive threshold under TF32 rounding on the A100, taking its 8 neighbours with it (relative error 1.5e-2 against the 1e-2 bound). Such alive flips are now excluded from the forward error and counted instead (at most 32 per case; the error without them is 3.9e-4).

## Planned next steps (in rough order of expected gain)

1. **Backward occupancy / latency**: 255 registers and ~47 KB of shared memory per 4-warp block give 8 warps per SM, and the kernel issues every 9 cycles per warp (stalls: tensor pipe 2.4, weight-fragment loads 2.0, dependencies 1.7, barriers 1.6 cycles per instruction). On the A100 / H100 (164 / 228 KB), an 8-warp block of two 4-warp teams with the weight fragments resident in shared memory (~140 KB) would remove the fragment-load stalls; it does not fit 100 KB. Tried and dropped: two warps per pixel group (8-warp blocks at 128 registers, 16 warps per SM): twice the instructions per MMA, 15–30 % slower; a 168-register cap (spills). A100 (section 7): the fragment loads cost 132 of 688 µs but are not L1 misses (a 28 KB L1 costs 3.5 %), so shared-memory residency would save little; the 168-register cap with 3 blocks per SM is 12 % slower.
2. **Fusing the adjoint** stays a loss on 100 KB GPUs (see above); with 164 KB or more, the gather could stage a tile's dp in shared memory instead of reading it 9 times from L2. A100 (section 7): a standalone adjoint that stages the patch's records in shared memory was no faster (the kernel is latency- and occupancy-bound), and the fused version would also recompute the halo rows.
3. **OT loss** (40 ms of a 93 ms texture iteration): the max-pool backward reads int64 indices (2.5 ms; a backward that recomputes the argmax from the saved input would halve it), the gradient of the 1024 sampled feature positions is materialized as a dense scatter per layer (~1 ms), and the cuDNN TF32 conversion passes (3.9 ms) would only go away with a different convolution implementation; bf16 VGG (validated, opt-in) halves the whole loss.
4. **Sparse growing grids**: skip the zero writes of dead cells (e.g. track the live bounding box) for very large, mostly empty grids.
5. **Other GPUs**: H100 `wgmma`; other (C, HID) instantiations (`dispatch()` in `nca_cuda.cu`; C a multiple of 4, HID a multiple of 32).
