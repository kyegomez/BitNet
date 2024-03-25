#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void gemm_lowbit_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int M, int N, int K);

// This function wraps the CUDA call
void gemm_lowbit(at::Tensor a, at::Tensor b, at::Tensor c, float w_scale, float x_scale) {
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);

    // Ensure tensors are on the correct device and have the expected types
    // Adjust as necessary for your specific use-case
    AT_ASSERTM(a.device().is_cuda(), "a must be a CUDA tensor");
    AT_ASSERTM(b.device().is_cuda(), "b must be a CUDA tensor");
    AT_ASSERTM(c.device().is_cuda(), "c must be a CUDA tensor");
    AT_ASSERTM(a.scalar_type() == at::ScalarType::Half, "a must be of type Half");
    AT_ASSERTM(b.scalar_type() == at::ScalarType::Half, "b must be of type Half");
    AT_ASSERTM(c.scalar_type() == at::ScalarType::Half, "c must be of type Half");

    // Call the CUDA kernel
    gemm_lowbit_cuda(a, b, c, M, N, K);

    // Apply scaling factors. Note: This operation is done in higher precision.
    c.mul_(1.0 / (w_scale * x_scale));
}

// PyBind11 Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_lowbit", &gemm_lowbit, "Low precision GEMM operation");
}
