// Simple first draft: fused CUDA kernel for brute-force KNN.
// One block per query. Distance -> bitonic sort -> top-k merge all happen
// in shared memory inside a single kernel.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ void cmp_swap(float* d, int* id, int a, int b) {
    if (d[a] > d[b]) {
        float t = d[a]; d[a] = d[b]; d[b] = t;
        int ti = id[a]; id[a] = id[b]; id[b] = ti;
    }
}

// Bitonic sort (ascending) of n = blockDim.x elements in shared memory.
__device__ void bitonic_sort(float* d, int* id, int n) {
    int t = threadIdx.x;
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int p = t ^ stride;
            if (p > t) {
                if ((t & size) == 0) cmp_swap(d, id, t, p);   // ascending
                else                 cmp_swap(d, id, p, t);   // descending
            }
            __syncthreads();
        }
    }
}

// blockDim.x = K2 = smallest power of 2 >= k (min 32).
__global__ void knn_fused(const float* db, const float* queries, int N, int D,
                          int k, float* out_d, int* out_i) {
    extern __shared__ float smem[];
    const int K2 = blockDim.x, t = threadIdx.x;
    float* sq   = smem;            // query vector (D)
    float* topd = sq + D;          // current best K2 distances (sorted)
    float* tmpd = topd + K2;       // new tile distances
    int*   topi = (int*)(tmpd + K2);
    int*   tmpi = topi + K2;

    const float* q = queries + (size_t)blockIdx.x * D;
    for (int i = t; i < D; i += K2) sq[i] = q[i];
    topd[t] = INFINITY; topi[t] = -1;
    __syncthreads();

    for (int base = 0; base < N; base += K2) {
        int j = base + t;
        float dist = INFINITY;
        if (j < N) {
            dist = 0.f;
            const float* v = db + (size_t)j * D;
            for (int c = 0; c < D; ++c) { float df = v[c] - sq[c]; dist += df * df; }
        }
        tmpd[t] = dist; tmpi[t] = (j < N) ? j : -1;
        __syncthreads();

        bitonic_sort(tmpd, tmpi, K2);                 // sort the tile

        // merge: min(top[i], tile[K2-1-i]) gives a bitonic seq of the K2 best
        float b = tmpd[K2 - 1 - t];
        if (b < topd[t]) { topd[t] = b; topi[t] = tmpi[K2 - 1 - t]; }
        __syncthreads();

        for (int stride = K2 >> 1; stride > 0; stride >>= 1) {   // bitonic merge
            int p = t ^ stride;
            if (p > t) cmp_swap(topd, topi, t, p);
            __syncthreads();
        }
    }
    if (t < k) {
        out_d[(size_t)blockIdx.x * k + t] = topd[t];
        out_i[(size_t)blockIdx.x * k + t] = topi[t];
    }
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100000;
    int D = argc > 2 ? atoi(argv[2]) : 128;
    int Q = argc > 3 ? atoi(argv[3]) : 256;
    int k = argc > 4 ? atoi(argv[4]) : 10;
    if (k < 1 || k > 1024) { printf("k must be in [1,1024]\n"); return 1; }
    int K2 = 32; while (K2 < k) K2 <<= 1;

    std::vector<float> hdb((size_t)N * D), hq((size_t)Q * D);
    srand(0);
    for (auto& x : hdb) x = rand() / (float)RAND_MAX;
    for (auto& x : hq)  x = rand() / (float)RAND_MAX;

    float *ddb, *dq, *dod; int* doi;
    CUDA_CHECK(cudaMalloc(&ddb, hdb.size() * 4));
    CUDA_CHECK(cudaMalloc(&dq, hq.size() * 4));
    CUDA_CHECK(cudaMalloc(&dod, (size_t)Q * k * 4));
    CUDA_CHECK(cudaMalloc(&doi, (size_t)Q * k * 4));
    CUDA_CHECK(cudaMemcpy(ddb, hdb.data(), hdb.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));

    size_t smem = ((size_t)D + 2 * K2) * 4 + 2 * K2 * 4;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    knn_fused<<<Q, K2, smem>>>(ddb, dq, N, D, k, dod, doi);
    cudaEventRecord(e);
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms, s, e);
    printf("N=%d D=%d Q=%d k=%d | %.3f ms | %.0f QPS\n", N, D, Q, k, ms, Q / (ms / 1e3));

    std::vector<float> od((size_t)Q * k);
    CUDA_CHECK(cudaMemcpy(od.data(), dod, od.size() * 4, cudaMemcpyDeviceToHost));

    // CPU check on a few queries
    int bad = 0;
    for (int qi = 0; qi < std::min(Q, 4); ++qi) {
        std::vector<float> ds(N);
        for (int j = 0; j < N; ++j) {
            float a = 0; for (int c = 0; c < D; ++c) { float df = hdb[(size_t)j*D+c] - hq[(size_t)qi*D+c]; a += df*df; }
            ds[j] = a;
        }
        std::partial_sort(ds.begin(), ds.begin() + k, ds.end());
        for (int i = 0; i < k; ++i)
            if (fabsf(ds[i] - od[(size_t)qi * k + i]) > 1e-3f) bad++;
    }
    printf("CPU check: %s\n", bad ? "MISMATCH" : "OK");
    return bad != 0;
}
