from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slayerCuda',
    ext_modules=[
        CUDAExtension(
            name='slayerCuda',
            sources=[
                'cuda/slayerKernels.cu'
            ],
            depends=[
                'cuda/spikeKernels.h',
                'cuda/convKernels.h',
                'cuda/shiftKernels.h'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O2', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='slayerLoihiCuda',
    ext_modules=[
        CUDAExtension(
            name='slayerLoihiCuda',
            sources=[
                'cuda/slayerLoihiKernels.cu'
            ],
            depends=[
                'cuda/spikeLoihiKernels.h'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O2', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='slayerSNN',
    packages=["slayerSNN"],
)
