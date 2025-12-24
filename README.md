# CUDA Neural Network with Per-Kernel Benchmarking

## Overview
- 1-hidden-layer fully connected neural network
- CUDA implementation (no cuBLAS)
- Per-kernel benchmarking for forward/backward/SGD
- Kernel times saved to `kernel_times.csv`
- Plotting script generates charts for GitHub

## Build & Run

```bash
mkdir build && cd build
cmake .. #cmake --build build -j 4
make -j 
./cuda_nn
## CPU vs GPU
The program now records a simple CPU baseline (naive single-threaded) and GPU timings. Use  to generate comparison plots in .

## CPU vs GPU
The program now records a simple CPU baseline (naive single-threaded) and GPU timings. Use scripts/plot_kernel_times.py to generate comparison plots in the plots/ directory.
./build/cuda_nn