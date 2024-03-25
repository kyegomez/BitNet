from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_lowbit_ext',  # Name of your extension
    ext_modules=[
        CUDAExtension(
            name='gemm_lowbit_ext',  # Extension name, used in Python import
            sources=['kernel/gemm_lowbit.cpp'],  # Source files
            extra_compile_args={
                'cxx': [],
                'nvcc': [],  # You can specify extra args for nvcc here
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
