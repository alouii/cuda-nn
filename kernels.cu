#include "kernels.cuh"
#include <cuda_runtime.h>

// Matrix Multiply: C = A(MxK) * B(KxN)
// Matrix Multiply: C = A(MxK) * B(KxN)
// Simple tiled/shared-memory implementation for better performance on GPUs.
#define TILE 16
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int Arow = row;
        int Acol = t * TILE + threadIdx.x;
        if (Arow < M && Acol < K)
            sA[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int Brow = t * TILE + threadIdx.y;
        int Bcol = col;
        if (Brow < K && Bcol < N)
            sB[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Tiled matmul with runtime transpose flags
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_k = t * TILE + threadIdx.x; // k index for this tile (x)
        int a_m = row;                     // m index (row)
        if (!transA) {
            // A is M x K
            if (a_m < M && a_k < K)
                sA[threadIdx.y][threadIdx.x] = A[a_m * K + a_k];
            else
                sA[threadIdx.y][threadIdx.x] = 0.0f;
        } else {
            // A is K x M stored row-major (we want A^T)
            if (a_k < K && a_m < M)
                sA[threadIdx.y][threadIdx.x] = A[a_k * M + a_m];
            else
                sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_k = t * TILE + threadIdx.y; // k index for this tile (y)
        int b_n = col;                     // n index (col)
        if (!transB) {
            // B is K x N
            if (b_k < K && b_n < N)
                sB[threadIdx.y][threadIdx.x] = B[b_k * N + b_n];
            else
                sB[threadIdx.y][threadIdx.x] = 0.0f;
        } else {
            // B is N x K stored row-major (we want B^T)
            if (b_n < N && b_k < K)
                sB[threadIdx.y][threadIdx.x] = B[b_n * K + b_k];
            else
                sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// C = A^T * B where A is K x M (stored row-major), B is K x N -> C is M x N
__global__ void matmul_at_b(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: K x M, B: K x N -> want C = A^T * B (M x N)
    // Use tiled kernel with transA = true, transB = false
    matmul_tiled(A, B, C, M, N, K, /*transA=*/true, /*transB=*/false);
}

// C = A * B^T where A is M x K, B is N x K -> C is M x N
__global__ void matmul_bt(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: M x K, B: N x K -> want C = A * B^T (M x N)
    // Use tiled kernel with transA = false, transB = true
    matmul_tiled(A, B, C, M, N, K, /*transA=*/false, /*transB=*/true);
}

// Add bias
__global__ void add_bias(float* A, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        A[idx] += b[col];
    }
}

// ReLU activation
__global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// fused add_bias + relu
__global__ void add_bias_relu(float* A, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float v = A[idx] + b[col];
        A[idx] = v > 0.0f ? v : 0.0f;
    }
}

// ReLU backward
__global__ void relu_backward(const float* z, float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        grad[i] = z[i] > 0.0f ? grad[i] : 0.0f;
}

// Mean Squared Error gradient
__global__ void mse_grad(const float* pred, const float* y, float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        grad[i] = 2.0f * (pred[i] - y[i]) / n;
}

// SGD update
__global__ void sgd_update(float* w, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        w[i] -= lr * grad[i];
}
