from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_lowbit_ext',
    ext_modules=[
        CUDAExtension(
            name='gemm_lowbit_ext',
            sources=['kernel/gemm_lowbit.cpp', 'kernel/gemm_lowbit_kernel.cu'],  # Adjusted paths
            extra_compile_args={'cxx': [], 'nvcc': []}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
