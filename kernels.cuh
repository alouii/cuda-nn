#pragma once

__global__ void matmul(const float*, const float*, float*, int, int, int);
// Tiled/shared-memory matmul with optional transposes (faster)
// transA: if true, A is provided as K x M (and treated as A^T)
// transB: if true, B is provided as N x K (and treated as B^T)
__global__ void matmul_tiled(const float*, const float*, float*, int M, int N, int K, bool transA, bool transB);
// C = A^T * B where A is K x M (stored row-major), B is K x N -> C is M x N
__global__ void matmul_at_b(const float*, const float*, float*, int M, int N, int K);
// C = A * B^T where A is M x K, B is N x K -> C is M x N
__global__ void matmul_bt(const float*, const float*, float*, int M, int N, int K);

// Fused add bias and ReLU (A += b[col]; A = max(A,0))
__global__ void add_bias_relu(float* A, const float* b, int M, int N);
__global__ void add_bias(float*, const float*, int, int);
__global__ void relu(float*, int);
__global__ void relu_backward(const float*, float*, int);
__global__ void mse_grad(const float*, const float*, float*, int);
__global__ void sgd_update(float*, const float*, float, int);