# CUDA Quick Reference

A concise cheat sheet covering every concept used in this course.

---

## Function Space Qualifiers

| Qualifier | Runs on | Called from | Notes |
|-----------|---------|-------------|-------|
| `__global__` | GPU | CPU (host) | Entry point for a kernel |
| `__device__` | GPU | GPU only | Helper function for kernels |
| `__host__`   | CPU | CPU only | Default; same as plain C++ |
| `__host__ __device__` | both | both | Compiled for both targets |

---

## Kernel Launch Syntax

```cuda
kernel_name<<<gridDim, blockDim>>>(arg1, arg2, ...);
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `gridDim`  | `dim3` or `int` | Number of blocks in the grid |
| `blockDim` | `dim3` or `int` | Number of threads per block |

**Rule of thumb:** `blockDim.x = 256` (multiple of 32, fits in one SM).  
**Grid sizing:** `gridDim.x = (N + blockDim.x - 1) / blockDim.x`  (ceiling division)

---

## Built-in Thread Variables

```
Grid
└── Block [blockIdx.x, blockIdx.y, blockIdx.z]
    └── Thread [threadIdx.x, threadIdx.y, threadIdx.z]
```

| Variable | Type | Meaning |
|----------|------|---------|
| `threadIdx.{x,y,z}` | `uint3` | Thread index within its block |
| `blockIdx.{x,y,z}`  | `uint3` | Block index within the grid |
| `blockDim.{x,y,z}`  | `dim3`  | Block dimensions (threads per block) |
| `gridDim.{x,y,z}`   | `dim3`  | Grid dimensions (blocks) |

**1-D global index formula:**
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;   // ← always bounds-check!
```

---

## Memory API

```cuda
// Allocate device memory
cudaMalloc(void** ptr, size_t bytes);

// Copy between host and device
cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
// kinds: cudaMemcpyHostToDevice  |  cudaMemcpyDeviceToHost  |  cudaMemcpyDeviceToDevice

// Free device memory
cudaFree(void* ptr);

// Block host until all GPU work is done
cudaDeviceSynchronize();
```

---

## Memory Hierarchy (slowest → fastest)

| Memory | Scope | Lifetime | Latency | Declared with |
|--------|-------|----------|---------|----------------|
| Global | All threads | App | ~200 cycles | `cudaMalloc` |
| Shared | One block | Kernel | ~4 cycles | `__shared__` |
| Registers | One thread | Kernel | 1 cycle | automatic |
| Constant | All threads (read-only) | App | ~4 cycles (cached) | `__constant__` |

---

## Shared Memory Pattern

```cuda
#define BLOCK_SIZE 256

__global__ void example(float* data, int n) {
    __shared__ float s_data[BLOCK_SIZE];       // shared buffer

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element into shared memory
    if (i < n) s_data[threadIdx.x] = data[i];
    __syncthreads();   // wait for all threads in block to finish loading

    // Now every thread can read any element in s_data cheaply
    float neighbor = s_data[(threadIdx.x + 1) % BLOCK_SIZE];

    __syncthreads();   // sync again before writing back
    if (i < n) data[i] = neighbor;
}
```

**Rule:** always call `__syncthreads()` after writing to and before reading from shared memory.

---

## CUDA Event Timing

```cuda
cudaEvent_t t0, t1;
cudaEventCreate(&t0);
cudaEventCreate(&t1);

cudaEventRecord(t0);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(t1);
cudaEventSynchronize(t1);

float ms = 0.0f;
cudaEventElapsedTime(&ms, t0, t1);
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(t0);
cudaEventDestroy(t1);
```

---

## Common Pitfalls

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Missing bounds check | Out-of-bounds write, silent corruption | `if (i >= n) return;` |
| Missing `__syncthreads` | Race condition on shared memory | Sync after every shared write phase |
| Accessing host pointer in kernel | Segfault / illegal memory access | Use `cudaMalloc`-ed pointer only |
| Forgetting `cudaDeviceSynchronize` | Results not ready when you read them | Sync before `cudaMemcpy D→H` |
| Block size not a multiple of 32 | Wasted warp lanes (performance) | Use 32, 64, 128, 256, 512 |
