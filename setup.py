from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slayer_kernels',
    ext_modules=[
        CUDAExtension('slayer_cuda', [
            'src/cuda/slayer_kernels.cpp',
            'src/cuda/spikeKernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )