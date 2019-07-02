import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
#from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

headers = os.path.join(this_file, 'src')
sources = []
headers = [headers]
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Build with GPU mode.')
    sources += ['src/slice_unpool_layer_cuda.cpp']
    headers += ['/usr/local/cuda/include']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

    extra_objects = ['src/cuda/slice_unpool_layer_cuda_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]


    setup(
        name='_ext.slice_unpool_layer',
        ext_modules=[
            CppExtension(
                name='_ext.slice_unpool_layer',
                sources=sources,
                include_dirs = headers,
                define_macros=defines,
                extra_objects=extra_objects)
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

    #ffi = create_extension(
    #                       '_ext.slice_unpool_layer',
    #                       headers=headers,
    #                       sources=sources,
    #                       define_macros=defines,
    #                       relative_to=__file__,
    #                       with_cuda=with_cuda,
    #                       extra_objects=extra_objects
    #                       )

else:
    print('Build with CPU mode.')
    sources += ['src/slice_unpool_layer.cpp']
    setup(
        name='_ext.slice_unpool_layer',
        ext_modules=[
            CppExtension(
                name='_ext.slice_unpool_layer',
                sources=sources,
                include_dirs = headers,
                define_macros=defines,
                extra_objects=extra_objects
                )
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
    #ffi = create_extension(
    #                       '_ext.slice_unpool_layer',
    #                       headers=headers,
    #                       sources=sources,
    #                       define_macros=defines,
    #                       relative_to=__file__,
    #                       with_cuda=with_cuda,
    #                       #extra_objects=extra_objects
    #                       )

#if __name__ == '__main__':
#    ffi.build()
