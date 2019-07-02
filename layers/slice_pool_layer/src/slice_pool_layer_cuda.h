#include <torch/extension.h>

int slice_pool_max_forward_cuda( at::Tensor * data_tensor, at::Tensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, at::Tensor * output_tensor, at::Tensor * pool_mask_tensor );

int slice_pool_max_backward_cuda( at::Tensor * top_grad_tensor, at::Tensor * pool_mask_tensor, int num_slice, int num_batch, int channels, at::Tensor * output_tensor );

int slice_pool_avg_forward_cuda( at::Tensor * data_tensor, at::Tensor * slice_idx_tensor, at::Tensor * slice_counts_tensor, int num_slice, int num_batch, int channels, int num_points, at::Tensor * output_tensor);

int slice_pool_avg_backward_cuda( at::Tensor * top_grad_tensor, at::Tensor * slice_idx_tensor, at::Tensor * slice_counts_tensor, int num_slice, int num_batch, int channels, int num_points, at::Tensor * output_tensor );

