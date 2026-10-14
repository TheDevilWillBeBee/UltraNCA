#!/bin/bash
# Build the nanobind CUDA extension and copy it next to nca_ops.py. Optional $1: tuning defines, e.g. "-DNCA_TILE_H=4".
set -e
cd "$(dirname "$0")"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DNCA_DEFS="$1" .. > /dev/null
# errors / warnings, then registers and spills per kernel (from ptxas -v)
cmake --build . --config Release -j"$(nproc)" 2>&1 | tee log | grep -E "error:|warning:" || true
grep -E "Compiling entry function|registers|spill" log | sed -E "s/.*function '([^']+)'.*/\1/" | c++filt \
  | awk '/nca_step::/{sub(/\(.*/, ""); sub(/^void nca_step::/, ""); name=$0} /spill stores/{spill=$5" "$9} /Used/{print $5" regs, spills "spill"  "name}' \
  | grep -E "forward|backward|alive|adjoint" || true
if grep -q "error:" log; then exit 1; fi
for f in nca_cuda.cpython-3*.so; do cp "$f" "../$f.tmp" && mv -f "../$f.tmp" "../$f"; done  # new inode: running processes keep theirs
cd .. && rm -rf build
