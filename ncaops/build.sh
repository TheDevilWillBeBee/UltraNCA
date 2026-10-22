#!/bin/bash
# Build the nanobind CUDA extension and copy it next to nca_ops.py. Optional $1: tuning defines, e.g. "-DNCA_FWD_RING=20".
set -e
cd "$(dirname "$0")"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
rm -rf build && mkdir build && cd build
# shared memory per SM of this machine's GPU (KB, by compute capability), unless $1 sets NCA_SMEM_PER_SM_KB
case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" in
  8.0|8.7) SMEM=164 ;; 9.0|10.0) SMEM=228 ;; *) SMEM=100 ;;
esac
DEFS="$1"; [[ "$DEFS" == *NCA_SMEM_PER_SM_KB* ]] || DEFS="-DNCA_SMEM_PER_SM_KB=$SMEM $DEFS"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DNCA_DEFS="$DEFS" .. > /dev/null
# errors / warnings, then registers and spills per kernel (from ptxas -v)
cmake --build . --config Release -j"$(nproc)" 2>&1 | tee log | grep -E "error:|warning:" || true
grep -E "Compiling entry function|registers|spill" log | sed -E "s/.*function '([^']+)'.*/\1/" | c++filt \
  | awk '/nca_step::/{sub(/\(.*/, ""); sub(/^void nca_step::/, ""); name=$0} /spill stores/{spill=$5" "$9} /Used/{print $5" regs, spills "spill"  "name}' \
  | grep -E "forward|backward|alive|adjoint" || true
grep -E "Compiling entry function|registers|spill" log | sed -E "s/.*function '([^']+)'.*/\1/" | c++filt \
  | awk '/lppn::/{sub(/\(.*/, ""); sub(/^void lppn::/, ""); name=$0} /spill stores/{spill=$5" "$9} /Used/{print $5" regs, spills "spill"  lppn::"name}' \
  | grep -E "lppn::(forward|backward)" || true
if grep -q "error:" log; then exit 1; fi
for f in nca_cuda.cpython-3*.so lppn_cuda.cpython-3*.so vgg_cuda.cpython-3*.so; do cp "$f" "../$f.tmp" && mv -f "../$f.tmp" "../$f"; done  # new inode: running processes keep theirs
cd .. && rm -rf build
