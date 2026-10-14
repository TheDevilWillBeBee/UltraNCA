#!/bin/bash
# Build the nanobind CUDA extension and copy it next to nca_ops.py. Optional $1: tuning defines, e.g. "-DNCA_TILE_H=4".
set -e
cd "$(dirname "$0")"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DNCA_DEFS="$1" .. > /dev/null
cmake --build . --config Release -j"$(nproc)" 2>&1 | grep -E "error|warning|registers|spill|bytes stack" || true
for f in nca_cuda.cpython-3*.so; do cp "$f" "../$f.tmp" && mv -f "../$f.tmp" "../$f"; done  # new inode: running processes keep theirs
cd .. && rm -rf build
