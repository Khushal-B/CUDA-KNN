# CUDA KNN – GPU Optimized K-Nearest-Neighbors Search (first draft)

Brute-force KNN with a **fused CUDA kernel**: distance computation, bitonic
sorting and top-k selection run in one kernel using shared memory, with no
global-memory round-trips between stages.

## How it works
- One thread block per query; block size = smallest power of 2 >= k (min 32).
- The database is processed in tiles of that size. Each thread computes one distance.
- Each tile is bitonic-sorted in shared memory, then merged with the running top-k
  (`min(top[i], tile[K2-1-i])` followed by a bitonic merge).

## Build & run
```
make
./knn [N] [D] [Q] [k]     # defaults: 100000 128 256 10
```
Prints latency, QPS, and a CPU correctness check. Needs the CUDA toolkit
(if `-arch=native` fails on an old nvcc, run `make ARCH=sm_80`, etc.).

## TODO
- [ ] Naive multi-kernel baseline + benchmark script
- [ ] O(log k) binary-search insertion for top-k overflow
- [ ] Coalesced database loads (tile the dimension)
- [ ] Block size / occupancy tuning, Nsight Compute profiling
- [ ] 100M+ vector benchmarks (multi-batch)
