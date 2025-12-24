#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include "kernels.cuh"

#define CHECK(cmd) cudaAssert((cmd), __FILE__, __LINE__)
inline void cudaAssert(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        std::cerr << cudaGetErrorString(e)
                  << " at " << f << ":" << l << std::endl;
        exit(1);
    }
}

// check kernel launches
#define CHECK_KERNEL() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        std::cerr << "Kernel launch error: " << cudaGetErrorString(_e) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {

    const int N = 8192; // dataset size
    const int D = 2;
    const int H = 512;
    const int O = 1;
    const float lr = 0.01f;

    std::vector<float> h_X(N*D), h_y(N);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0,1);
    for(int i=0;i<N;i++){
        h_X[i*D] = dist(gen);
        h_X[i*D+1] = dist(gen);
        h_y[i] = h_X[i*D]+h_X[i*D+1];
    }

    float *X, *y, *W1, *b1, *Z1, *W2, *b2, *Z2;
    float *dZ2, *dZ1, *dW1, *dW2;

    CHECK(cudaMalloc(&X, N*D*sizeof(float)));
    CHECK(cudaMalloc(&y, N*sizeof(float)));
    CHECK(cudaMalloc(&W1, D*H*sizeof(float)));
    CHECK(cudaMalloc(&b1, H*sizeof(float)));
    CHECK(cudaMalloc(&Z1, N*H*sizeof(float)));
    CHECK(cudaMalloc(&W2, H*O*sizeof(float)));
    CHECK(cudaMalloc(&b2, O*sizeof(float)));
    CHECK(cudaMalloc(&Z2, N*O*sizeof(float)));
    CHECK(cudaMalloc(&dZ2, N*O*sizeof(float)));
    CHECK(cudaMalloc(&dZ1, N*H*sizeof(float)));
    CHECK(cudaMalloc(&dW1, D*H*sizeof(float)));
    CHECK(cudaMalloc(&dW2, H*O*sizeof(float)));

    CHECK(cudaMemcpy(X, h_X.data(), N*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, h_y.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    // initialize weights and biases on host and copy to device
    std::vector<float> h_W1(D*H), h_b1(H), h_W2(H*O), h_b2(O);
    std::normal_distribution<float> ndist(0.0f, 0.1f);
    for (int i = 0; i < D*H; ++i) h_W1[i] = ndist(gen);
    for (int i = 0; i < H; ++i) h_b1[i] = 0.0f;
    for (int i = 0; i < H*O; ++i) h_W2[i] = ndist(gen);
    for (int i = 0; i < O; ++i) h_b2[i] = 0.0f;

    CHECK(cudaMemcpy(W1, h_W1.data(), D*H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b1, h_b1.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(W2, h_W2.data(), H*O*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b2, h_b2.data(), O*sizeof(float), cudaMemcpyHostToDevice));

    // Host-side intermediate buffers for CPU baseline
    std::vector<float> h_Z1(N*H), h_Z2(N*O);
    // host buffer to receive GPU predictions for logging mse
    std::vector<float> h_Z2_gpu(N*O);
    std::vector<float> h_dZ2(N*O), h_dZ1(N*H), h_dW1(D*H), h_dW2(H*O);

    dim3 block2D(16,16);
    dim3 gridZ1((H+15)/16, (N+15)/16);
    dim3 gridZ2((O+15)/16, (N+15)/16);

    // GPU events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // simple CPU helper lambdas
    auto cpu_matmul = [&](const float* A, const float* B, float* C, int M, int N, int K){
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                float s = 0.0f;
                for(int k=0;k<K;k++) s += A[i*K + k] * B[k*N + j];
                C[i*N + j] = s;
            }
        }
    };
    auto cpu_add_bias = [&](float* A, const float* b, int M, int N){
        for(int i=0;i<M;i++) for(int j=0;j<N;j++) A[i*N + j] += b[j];
    };
    auto cpu_relu = [&](float* x, int n){ for(int i=0;i<n;i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f; };
    auto cpu_relu_backward = [&](const float* z, float* grad, int n){ for(int i=0;i<n;i++) grad[i] = z[i] > 0.0f ? grad[i] : 0.0f; };
    auto cpu_mse_grad = [&](const float* pred, const float* y_, float* grad, int n){ for(int i=0;i<n;i++) grad[i] = 2.0f * (pred[i] - y_[i]) / n; };
    auto cpu_sgd_update = [&](float* w, const float* grad, float lr_, int n){ for(int i=0;i<n;i++) w[i] -= lr_ * grad[i]; };

    // Warm up GPU (driver init + first kernels)
    matmul<<<gridZ1, block2D>>>(X, W1, Z1, N, H, D);
    CHECK_KERNEL();
    cudaDeviceSynchronize();

    // Warm up CPU (cache)
    cpu_matmul(h_X.data(), h_W1.data(), h_Z1.data(), N, H, D);

    const int epochs=50;
    std::ofstream log("kernel_times.csv");
    log << "Epoch,matmul1,bias1,relu1,matmul2,bias2,mse,dW2,dZ1,relu_b,dW1,sgd1,sgd2,mse_value,cpu_mse_value,";
    log << "cpu_matmul1,cpu_bias1,cpu_relu1,cpu_matmul2,cpu_bias2,cpu_mse,cpu_dW2,cpu_dZ1,cpu_relu_b,cpu_dW1,cpu_sgd1,cpu_sgd2\n";

    float t_matmul1,t_bias1,t_relu1,t_matmul2,t_bias2,t_mse;
    float t_matmul_dW2,t_matmul_dZ1,t_relu_b,t_matmul_dW1,t_sgd1,t_sgd2;

    for(int epoch=0;epoch<epochs;epoch++){
        // --- CPU baseline timings ---
        double cpu_t_matmul1=0, cpu_t_bias1=0, cpu_t_relu1=0, cpu_t_matmul2=0, cpu_t_bias2=0, cpu_t_mse=0;
        double cpu_t_matmul_dW2=0, cpu_t_matmul_dZ1=0, cpu_t_relu_b=0, cpu_t_matmul_dW1=0, cpu_t_sgd1=0, cpu_t_sgd2=0;

        // CPU forward layer1
        auto tc0 = std::chrono::high_resolution_clock::now();
        cpu_matmul(h_X.data(), h_W1.data(), h_Z1.data(), N, H, D);
        auto tc1 = std::chrono::high_resolution_clock::now(); cpu_t_matmul1 = std::chrono::duration<double,std::milli>(tc1-tc0).count();

        auto tc2 = std::chrono::high_resolution_clock::now();
        cpu_add_bias(h_Z1.data(), h_b1.data(), N, H);
        auto tc3 = std::chrono::high_resolution_clock::now(); cpu_t_bias1 = std::chrono::duration<double,std::milli>(tc3-tc2).count();

        auto tc4 = std::chrono::high_resolution_clock::now();
        cpu_relu(h_Z1.data(), N*H);
        auto tc5 = std::chrono::high_resolution_clock::now(); cpu_t_relu1 = std::chrono::duration<double,std::milli>(tc5-tc4).count();

        // CPU forward layer2
        auto tc6 = std::chrono::high_resolution_clock::now();
        cpu_matmul(h_Z1.data(), h_W2.data(), h_Z2.data(), N, O, H);
        auto tc7 = std::chrono::high_resolution_clock::now(); cpu_t_matmul2 = std::chrono::duration<double,std::milli>(tc7-tc6).count();

        auto tc8 = std::chrono::high_resolution_clock::now();
        cpu_add_bias(h_Z2.data(), h_b2.data(), N, O);
        auto tc9 = std::chrono::high_resolution_clock::now(); cpu_t_bias2 = std::chrono::duration<double,std::milli>(tc9-tc8).count();

        auto tc10 = std::chrono::high_resolution_clock::now();
        cpu_mse_grad(h_Z2.data(), h_y.data(), h_dZ2.data(), N);
        // compute CPU MSE value for logging
        float cpu_mse_value = 0.0f;
        for (int i = 0; i < N; ++i) {
            float diff = h_Z2[i*O + 0] - h_y[i];
            cpu_mse_value += diff * diff;
        }
        cpu_mse_value /= N;
        auto tc11 = std::chrono::high_resolution_clock::now(); cpu_t_mse = std::chrono::duration<double,std::milli>(tc11-tc10).count();

        // CPU backprop
        auto tc12 = std::chrono::high_resolution_clock::now();
        // dW2 = Z1^T * dZ2  (H x O)
        for(int h=0; h<H; ++h){ for(int o=0;o<O;++o){ float s=0; for(int n=0;n<N;++n) s += h_Z1[n*H + h] * h_dZ2[n*O + o]; h_dW2[h*O + o] = s; }}
        auto tc13 = std::chrono::high_resolution_clock::now(); cpu_t_matmul_dW2 = std::chrono::duration<double,std::milli>(tc13-tc12).count();

        auto tc14 = std::chrono::high_resolution_clock::now();
        // dZ1 = dZ2 * W2^T  (N x H)
        for(int n=0;n<N;++n){ for(int h=0; h<H; ++h){ float s=0; for(int o=0;o<O;++o) s += h_dZ2[n*O + o] * h_W2[h*O + o]; h_dZ1[n*H + h] = s; }}
        auto tc15 = std::chrono::high_resolution_clock::now(); cpu_t_matmul_dZ1 = std::chrono::duration<double,std::milli>(tc15-tc14).count();

        auto tc16 = std::chrono::high_resolution_clock::now();
        cpu_relu_backward(h_Z1.data(), h_dZ1.data(), N*H);
        auto tc17 = std::chrono::high_resolution_clock::now(); cpu_t_relu_b = std::chrono::duration<double,std::milli>(tc17-tc16).count();

        auto tc18 = std::chrono::high_resolution_clock::now();
        // dW1 = X^T * dZ1 (D x H)
        for(int d=0; d<D; ++d){ for(int h=0; h<H; ++h){ float s=0; for(int n=0;n<N;++n) s += h_X[n*D + d] * h_dZ1[n*H + h]; h_dW1[d*H + h] = s; }}
        auto tc19 = std::chrono::high_resolution_clock::now(); cpu_t_matmul_dW1 = std::chrono::duration<double,std::milli>(tc19-tc18).count();

        auto tc20 = std::chrono::high_resolution_clock::now();
        cpu_sgd_update(h_W1.data(), h_dW1.data(), lr, D*H);
        auto tc21 = std::chrono::high_resolution_clock::now(); cpu_t_sgd1 = std::chrono::duration<double,std::milli>(tc21-tc20).count();

        auto tc22 = std::chrono::high_resolution_clock::now();
        cpu_sgd_update(h_W2.data(), h_dW2.data(), lr, H*O);
        auto tc23 = std::chrono::high_resolution_clock::now(); cpu_t_sgd2 = std::chrono::duration<double,std::milli>(tc23-tc22).count();
        // Forward layer1
        CHECK(cudaEventRecord(start));
        matmul<<<gridZ1, block2D>>>(X, W1, Z1, N, H, D);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_matmul1,start,stop));

        CHECK(cudaEventRecord(start));
        add_bias<<<(N*H+255)/256,256>>>(Z1,b1,N,H);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_bias1,start,stop));

        CHECK(cudaEventRecord(start));
        relu<<<(N*H+255)/256,256>>>(Z1,N*H);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_relu1,start,stop));

        // Forward layer2
        CHECK(cudaEventRecord(start));
        matmul<<<gridZ2, block2D>>>(Z1,W2,Z2,N,O,H);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_matmul2,start,stop));

        CHECK(cudaEventRecord(start));
        add_bias<<<(N*O+255)/256,256>>>(Z2,b2,N,O);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_bias2,start,stop));

        CHECK(cudaEventRecord(start));
        mse_grad<<<(N+255)/256,256>>>(Z2,y,dZ2,N);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_mse,start,stop));

        // copy GPU predictions back to host and compute MSE value for logging
        CHECK(cudaMemcpy(h_Z2_gpu.data(), Z2, N*O*sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_mse_value = 0.0f;
        for (int i = 0; i < N; ++i) {
            float diff = h_Z2_gpu[i*O + 0] - h_y[i];
            gpu_mse_value += diff * diff;
        }
        gpu_mse_value /= N;

        // Backpropagation
        CHECK(cudaEventRecord(start));
        // dW2 = Z1^T * dZ2  -> use matmul_at_b where A is K x M (K=N, M=H), B is K x N (K=N, N=O)
        matmul_at_b<<<dim3((O+15)/16,(H+15)/16),block2D>>>(Z1,dZ2,dW2,H,O,N);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_matmul_dW2,start,stop));

        CHECK(cudaEventRecord(start));
        // dZ1 = dZ2 * W2^T -> use matmul_bt where A is M x K (N x O), B is N x K (H x O) -> C N x H
        matmul_bt<<<gridZ1, block2D>>>(dZ2,W2,dZ1,N,H,O);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_matmul_dZ1,start,stop));

        CHECK(cudaEventRecord(start));
        relu_backward<<<(N*H+255)/256,256>>>(Z1,dZ1,N*H);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_relu_b,start,stop));

        CHECK(cudaEventRecord(start));
        // dW1 = X^T * dZ1 -> matmul_at_b with A = X (K=N, M=D), B = dZ1 (K=N, N=H)
        matmul_at_b<<<dim3((H+15)/16,(D+15)/16),block2D>>>(X,dZ1,dW1,D,H,N);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_matmul_dW1,start,stop));

        CHECK(cudaEventRecord(start));
        sgd_update<<<(D*H+255)/256,256>>>(W1,dW1,lr,D*H);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_sgd1,start,stop));

        CHECK(cudaEventRecord(start));
        sgd_update<<<(H*O+255)/256,256>>>(W2,dW2,lr,H*O);
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&t_sgd2,start,stop));

        log << epoch << ","
            << t_matmul1 << "," << t_bias1 << "," << t_relu1 << ","
            << t_matmul2 << "," << t_bias2 << "," << t_mse << ","
            << t_matmul_dW2 << "," << t_matmul_dZ1 << "," << t_relu_b << ","
            << t_matmul_dW1 << "," << t_sgd1 << "," << t_sgd2 << ","
            << gpu_mse_value << "," << cpu_mse_value << ","
            << cpu_t_matmul1 << "," << cpu_t_bias1 << "," << cpu_t_relu1 << ","
            << cpu_t_matmul2 << "," << cpu_t_bias2 << "," << cpu_t_mse << ","
            << cpu_t_matmul_dW2 << "," << cpu_t_matmul_dZ1 << "," << cpu_t_relu_b << ","
            << cpu_t_matmul_dW1 << "," << cpu_t_sgd1 << "," << cpu_t_sgd2 << "\n";

        if(epoch%10==0)
            std::cout << "Epoch " << epoch << " logged\n";
    }

    log.close();
    std::cout << "All kernel timings saved to kernel_times.csv\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free GPU memory
    cudaFree(X); cudaFree(y); cudaFree(W1); cudaFree(b1); cudaFree(Z1);
    cudaFree(W2); cudaFree(b2); cudaFree(Z2);
    cudaFree(dZ2); cudaFree(dZ1); cudaFree(dW1); cudaFree(dW2);

    return 0;
}
