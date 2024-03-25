#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Declaration of the CUDA kernel remains unchanged

extern "C" void gemm_lowbit_cuda(
    fp8 *a, fp8 *b, fp8 *c, int M, int N, int K);

// This is the corrected wrapper function
void gemm_lowbit(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    float w_scale,
    float x_scale) {

    // Ensure tensors are on the correct device
    a = a.to(at::kCUDA);
    b = b.to(at::kCUDA);
    c = c.to(at::kCUDA);

    // Ensure tensors are of type half (FP16), as fp8 is typedef'd to half for demonstration
    a = a.to(at::kHalf);
    b = b.to(at::kHalf);
    c = c.to(at::kHalf);

    const auto M = a.size(0);
    const auto K = a.size(1);
    const auto N = b.size(1);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // Directly call the CUDA kernel using tensor data pointers
    gemm_lowbit_kernel<<<blocks, threads>>>(
        a.data_ptr<at::Half>(),
        b.data_ptr<at::Half>(),
        c.data_ptr<at::Half>(),
        M, N, K
    );

    // Synchronize to ensure CUDA operations have completed
    cudaDeviceSynchronize();

    // Apply scaling after ensuring the operation has completed
    c.mul_(1.0 / (w_scale * x_scale));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_lowbit", &gemm_lowbit, "Low precision GEMM operation with scaling factors");
}
