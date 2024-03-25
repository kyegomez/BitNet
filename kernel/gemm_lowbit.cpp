#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h> // Ensure CUDA context for PyTorch is included

// CUDA forward declarations
void gemm_lowbit_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int M, int N, int K);

// The wrapper function to be called from Python
void gemm_lowbit(at::Tensor a, at::Tensor b, at::Tensor c, float w_scale, float x_scale) {
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);

    // Ensure inputs are on the correct device and are of half precision
    a = a.to(at::device(at::kCUDA).dtype(at::kHalf));
    b = b.to(at::device(at::kCUDA).dtype(at::kHalf));
    c = c.to(at::device(at::kCUDA).dtype(at::kHalf));

    // Call the CUDA kernel wrapper
    gemm_lowbit_cuda(a, b, c, M, N, K);

    // Apply scale factors
    c.div_(w_scale * x_scale);
}

// The PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_lowbit", &gemm_lowbit, "A low precision GEMM operation with scaling");
}
