#include <THC/THC.h>
#include <math.h>
#include "cuda/slice_unpool_layer_cuda_kernel.h"
#include "slice_unpool_layer_cuda.h"

extern THCState *state;

// Max Unpooling Forward CUDA
int slice_unpool_forward_cuda
( at::Tensor * data_tensor, 
  at::Tensor * slice_idx_tensor, 
  int num_slice, 
  int num_batch, 
  int channels, 
  int num_points, 
  at::Tensor * output_tensor )
{
    // Grab the input tensor
    float * data = data_tensor->data<float>();
    float * output = output_tensor->data<float>();
    int * slice_idx = slice_idx_tensor->data<int>();
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_unpool_forward_gpu_laucher( data, slice_idx, num_slice, num_batch, channels, num_points, output, stream );

    return 1;
    
}



// Max Unpooling Backward CUDA
int slice_unpool_backward_cuda
( at::Tensor * top_grad_tensor, 
  at::Tensor * slice_idx_tensor, 
  int num_slice, 
  int num_batch, 
  int channels, 
  int num_points, 
  at::Tensor * output_tensor )
{
    // Grab the input tensor
    float * top_grad = top_grad_tensor->data<float>();
    int * slice_idx = slice_idx_tensor->data<int>();
    float * output = output_tensor->data<float>();
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_unpool_backward_gpu_laucher( top_grad, slice_idx, num_slice, num_batch, channels, num_points, output, stream );
    
    return 1;
    
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("slice_unpool_forward_cuda", &slice_unpool_forward_cuda, "slice_unpool_forward_cuda");
  m.def("slice_unpool_backward_cuda", &slice_unpool_backward_cuda, "slice_unpool_backward_cuda");
}