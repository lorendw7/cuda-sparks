#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA_CHECK  —  abort on any CUDA API error
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
//
// The macro prints the file, line, and human-readable error string, then
// exits the program.  Use it around every CUDA API call.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr,                                                 \
                    "\n[CUDA ERROR] %s  (line %d)\n  %s\n\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// gpu_info  —  print a one-line summary of device 0
// ---------------------------------------------------------------------------
inline void gpu_info() {
    int dev = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("[GPU] %s  |  SM %d.%d  |  %.0f MB VRAM  |  %d SMs\n\n",
           p.name, p.major, p.minor,
           p.totalGlobalMem / 1e6,
           p.multiProcessorCount);
}
