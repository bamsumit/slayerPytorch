from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='slayer_kernels',
#     ext_modules=[
#         CUDAExtension('slayer_cuda', [
#             'src/cuda/slayer_kernels.cpp',
#             'src/cuda/spikeKernels.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
#     )

setup(
    name='slayerCuda',
    ext_modules=[
        CUDAExtension(
            name='slayerCuda',
            sources=[
                'src/cuda/slayerKernels.cu',
                # 'src/cuda/spikeKernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O2', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)