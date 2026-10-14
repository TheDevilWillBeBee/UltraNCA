# Task: tune the batched NCA kernels for the A100

## Context

This repo trains many Neural Cellular Automata at once on one GPU (one NCA per target image): texture NCAs
(`config.yml`, 12 channels / 96 hidden, OT loss on VGG16 features) and growing NCAs (`config_growing.yml`, 16 / 128,
alive mask, L2 loss). K NCAs (typically 32) are stacked as (K, B, C, H, W) and stepped by fused CUDA kernels in
`ncaops/` (forward, backward with weight gradients, adjoint, alive mask, reduce). `README.md` describes the design,
the per-GPU configuration, the RTX 4090 results and, importantly, what has already been tried and did not help.

The kernels were developed and tuned on an RTX 4090 (sm_89, 100 KB of shared memory per SM, 72 MB L2, 128 SMs).
Your machine has an A100 (sm_80, 164 KB of shared memory per SM, 40 MB L2, 108 SMs, TF32 tensor peak 156 TFLOPS,
about 1.5–2 TB/s of DRAM bandwidth). The goal is to make training and inference as fast as possible on the A100
**without changing the behaviour on the 4090**, which stays in use.

## How the per-GPU configuration works

- `ncaops/build.sh` sets `NCA_SMEM_PER_SM_KB` from the compute capability of the machine's GPU (8.0 → 164). The
  block counts per SM and the backward's shared-memory layout (one or two parts for relu(h) / dh) derive from it; the
  L2 size and SM count are queried at run time (`nca_cuda.cu`: `chunk_images`, `launch_forward`).
- The remaining knobs at the top of `ncaops/nca_step.cuh` (`NCA_FWD_WARPS`, `NCA_FWD_RING`,
  `NCA_BWD_TILES_PER_BLOCK`, `NCA_BWD_UNROLL`, `NCA_BWD_MIN_BLOCKS`, `NCA_ADJ_CPT`) and the host-side heuristics in
  `nca_cuda.cu` (chunk size: two thirds of a wave of backward blocks, growing NCAs at least N / 4 images; forward
  segment length) are tuning choices measured on sm_89. They can be overridden per build:
  `bash ncaops/build.sh "-DNCA_FWD_RING=20 -DNCA_BWD_TILES_PER_BLOCK=4"`.
- **Rule for this task:** keep every sm_89 default as it is. Make A100-specific choices conditional, e.g.
  `#if NCA_SMEM_PER_SM_KB >= 164` blocks for the defaults in `nca_step.cuh`, or a compute-capability check at run
  time in `nca_cuda.cu`, so that one build of the same source is right on both GPUs.

## Setup

1. A Python environment with `requirements.txt` (PyTorch with CUDA, nanobind, CMake 3.24+), the CUDA toolkit with
   `nvcc` on `PATH` (`CUDA_HOME` if not `/usr/local/cuda`), and Nsight Compute (`ncu`) if available.
2. `python prepare/download_pbr.py` and `python prepare/download_emoji.py` for the targets.
3. `bash ncaops/build.sh` builds the extension and prints registers and spills per kernel.
4. `python ncaops/testing/test_nca.py` must keep passing (both NCA sizes, update rates 0.5 and 1, texture and
   growing, FP32 and TF32, determinism, chunked backward, rollout = repeated steps). `--bench` adds timings.
5. Baselines before any change: `python benchmark.py train infer --K 32` (per-iteration training cost and inference
   rollouts; growing NCAs need trained weights, `python train.py config_growing.yml` with `num_targets: 32`, or pass
   `--growing_weights`), and `python train.py` with `num_targets: 32` for the real training time per NCA.

## Where to look first (expected to matter on 164 KB of shared memory)

- Backward: with 164 KB, 16/128 no longer needs the two-part relu(h) / dh staging, and 3 blocks per SM may fit
  (check `BWD_MIN_BLOCKS`, registers: 255 per thread now). The kernel is latency-bound at 8 warps per SM on the
  4090; its stalls are the weight-fragment loads from L2, the tensor pipe and the barriers. README "Planned next
  steps" item 1 describes an 8-warp block with the weight fragments resident in shared memory (about 140 KB): it
  does not fit the 4090 but fits the A100.
- Forward: `fwd_min_blocks` and `NCA_FWD_RING` (ring of state rows) against 164 KB; the sm_89 ring is 26 rows.
- Chunking of the backward / adjoint: the A100 L2 is 40 MB (vs 72 MB), so the chunk sizes that keep dp in L2 differ;
  sweep them (`nca_ops.CHUNK` overrides the rule for experiments).
- The OT loss (about 40 % of a texture iteration on the 4090) runs through cuDNN; measure before touching it.

## Deliverables

- Work on a branch from `perf`; commit at milestones with measured before / after numbers (kernel times from
  the profiler or `ncu`, `benchmark.py`, and `train.py` seconds per NCA).
- Tests pass on the A100; the sm_89 configuration is untouched (no change in `nca_step.cuh` defaults for
  `NCA_SMEM_PER_SM_KB < 164`).
- For any change that could affect training numerics, check quality on a subset of targets (final losses of
  `train.py` on 32 textures / 32 emojis against a run with the unmodified kernels).
- An "A100" section in `README.md` with the configuration chosen, the numbers and what did not help.
