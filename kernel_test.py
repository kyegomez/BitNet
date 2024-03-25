import torch
import gemm_lowbit_ext  # This imports the compiled module

# Example usage
a = torch.randn(10, 20, dtype=torch.half, device='cuda')  # Example tensor
b = torch.randn(20, 30, dtype=torch.half, device='cuda')  # Example tensor
c = torch.empty(10, 30, dtype=torch.half, device='cuda')  # Output tensor

w_scale = 1.0  # Example scale factor
x_scale = 1.0  # Example scale factor

# Call the custom CUDA GEMM operation
gemm_lowbit_ext.gemm_lowbit(a, b, c, w_scale, x_scale)

print(c)  # View the result
