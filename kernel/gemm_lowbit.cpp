#include <torch/extension.h>

void gemm_lowbit_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_lowbit", &gemm_lowbit_cuda, "Low precision GEMM operation (CUDA)");
}
