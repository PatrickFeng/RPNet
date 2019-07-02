#include <torch/extension.h>

int slice_unpool_forward_cuda( at::Tensor * data_tensor, at::Tensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, at::Tensor * output_tensor );

int slice_unpool_backward_cuda( at::Tensor * top_grad_tensor, at::Tensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, at::Tensor * output_tensor );

