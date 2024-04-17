// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "phi/core/distributed/utils.h"

inline void ExtractNCWHD(const phi::DDim& dims,
                         const phi::DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* H,
                         int* W,
                         int* D) {
  *N = dims[0];
  if (dims.size() == 2) {
    *C = dims[1];
    *H = 1;
    *W = 1;
    *D = 1;
  } else {
    *C =
        data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == phi::DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == phi::DataLayout::kNCHW ? dims[4] : dims[3])
             : 1;
  }
}

namespace custom_kernel {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out);

template <typename T, typename Context>
<<<<<<< HEAD
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y);

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs);

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out);

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     const paddle::optional<phi::DenseTensor>& scale,
                     const paddle::optional<phi::DenseTensor>& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool use_global_stats,
                     bool trainable_stats,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space);

template <typename Context>
void custom_all_gather(const Context& dev_ctx,
                       phi::DenseTensor* out_tensor,
                       const phi::DenseTensor& in_tensor,
                       bool sync_op) {
  int global_gid = 0;
  int offset = 0;
  int numel = -1;
  bool use_calc_stream = false;

  // get comm_context
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::XCCLCommContext*>(
      comm_context_manager.Get(std::to_string(global_gid)));

  auto tensor_tmp =
      paddle::experimental::CheckAndTrans2NewContiguousTensor(in_tensor);
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor_tmp, offset, numel) : tensor_tmp;
  auto task = RunFnInXCCLEnv(
      [&](const phi::stream::Stream& stream) {
        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
  task->UpdateWaitChain(*dev_ctx);
}

template <typename T, typename Context>
void SyncBatchNormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& mean,
                         const phi::DenseTensor& variance,
                         const phi::DenseTensor& scale,
                         const phi::DenseTensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon_f,
                         const std::string& data_layout_str,
                         bool use_global_stats,
                         bool trainable_statistics,
                         phi::DenseTensor* y,
                         phi::DenseTensor* mean_out,
                         phi::DenseTensor* variance_out,
                         phi::DenseTensor* saved_mean,
                         phi::DenseTensor* saved_variance,
                         phi::DenseTensor* reserve_space) {
  const DataLayout layout = StringToDataLayout(data_layout_str);
  PADDLE_ENFORCE_EQ(use_global_stats,
                    false,
                    phi::errors::InvalidArgument(
                        "sync_batch_norm doesn't support "
                        "to set use_global_stats True. Please use batch_norm "
                        "in this case."));
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be larger than 1."));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be less than 6."));
  int N, C, H, W, D;
  ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

  bool test_mode = is_test && (!trainable_statistics);
  bool training = !test_mode && !use_global_stats;

  if (!training) {  // inference
    custom_kernel::BatchNormInferKernel<T, Context>(dev_ctx,
                                                    x,
                                                    mean,
                                                    variance,
                                                    scale,
                                                    bias,
                                                    momentum,
                                                    epsilon_f,
                                                    data_layout_str,
                                                    y,
                                                    mean_out,
                                                    variance_out);
  } else {  // training

    // ------------------ calculate global mean/variance --------------------

    // ===================================================
    // step 1: calculate sum/square_sum of current input.
    // ===================================================
    phi::DenseTensor x_tensor(x)
        // transform 3d tensor to 4d tensor to satisfy the format
        if (x.dims().size() == 3) {
      auto x_shape_vec = phi::vectorize(x.dims());
      if (channel_last) {
        x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
      } else {
        x_shape_vec.push_back(1);  // expand NCL -> NCL1
      }
      auto x_new_shape = phi::make_ddim(x_shape_vec);
      x_tensor.Resize(x_new_shape);
    }
    if (x.dims().size() == 5) {
      phi::DenseTensorMeta x_meta, y_meta;
      if (channel_last) {
        x_meta = {x.dtype(), x_tensor.dims(), phi::DataLayout::kNDHWC};
      } else {
        x_meta = {x.dtype(), x_tensor.dims(), phi::DataLayout::kNCDHW};
      }
      x_tensor.set_meta(x_meta);
    } else {
      if (channel_last) {
        phi::DenseTensorMeta x_meta = {
            x.dtype(), x_tensor.dims(), phi::DataLayout::kNHWC};
        x_tensor.set_meta(x_meta);
      }
    }

    auto stream = dev_ctx.stream();

    phi::DenseTensorMeta sum_meta = {
        phi::DataType::FLOAT32, {C}, x_tensor.layout()};
    phi::DenseTensor sum, square_sum;
    sum.set_meta(sum_meta);
    square_sum.set_meta(sum_meta);
    dev_ctx.template Alloc<float>(&sum);
    dev_ctx.template Alloc<float>(&square_sum);

    std::string reduce_name =
        (x.dims().size() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
    NpuOpRunner runner_reduce;
    runner_reduce.SetType(reduce_name)
        .AddInput(x_tensor)
        .AddOutput(sum)
        .AddOutput(square_sum)
        .AddAttrs({{"epsilon", epsilon}})
        .Run(stream);

    // ======================================================
    // step 2: gather sum_all/square_sum_all of global input
    // ======================================================
    phi::DenseTensorMeta count_meta = {phi::DataType::FLOAT32, {1}};
    phi::DenseTensor count;
    count.set_meta(count_meta);
    dev_ctx.template Alloc<float>(count);
    FillNpuTensorWithConstant<float>(
        &count, dev_ctx, static_cast<float>(N * H * W * D));

    // C, C, 1 -> (2C + 1)
    phi::DenseTensor combined;
    custom_kernel::ConcatKernel<float, Context>(
        dev_ctx, {&sum, &square_sum, &count}, 0, &combined);
    // world_size * (2C + 1)
    std::vector<phi::DenseTensor> combined_list(
        world_size, phi::EmptyLike<float, Context>(dev_ctx, combined));
    std::vector<phi::DenseTensor*> combined_list_ptr;
    for (int i = 0; i < combined_list.size(); ++i) {
      combined_list_ptr.push_back(&combined_list[i]);
    }

    phi::DenseTensor stacked_combined;
    custom_kernel::StackKernel<float, Context>(
        dev_ctx, combined_list_ptr, 0, &stacked_combined);
    custom_kernel::custom_all_gather<Context>(
        dev_ctx, &stacked_combined, combined);
    std::vector<phi::DenseTensor*> split_stacked_combined(3);
    phi::IntArray num_secs = {world_size * C, world_size * C, world_size * 1};
    custom_kernel::SplitKernel<float, Context>(
        dev_ctx, stacked_combined, num_secs, 1, split_stacked_combined);

    phi::DenseTensor mean_all, invstd_all;
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {C}};
    mean_all.set_meta(meta);
    invstd_all.set_meta(meta);
    dev_ctx.template Alloc<float>(&mean_all);
    dev_ctx.template Alloc<float>(&invstd_all);

    phi::DenseTensor running_mean, running_variance;
    running_mean.Resize(mean_out->dims());
    running_variance.Resize(variance_out->dims());

    dev_ctx.template Alloc<float>(&running_mean);
    dev_ctx.template Alloc<float>(&running_variance);
    TensorCopy(dev_ctx, mean, false, &running_mean);
    TensorCopy(dev_ctx, variance, false, &running_variance);

    NpuOpRunner runner_gather;
    runner_gather.SetType("SyncBatchNormGatherStats")
        .AddInput(*split_stacked_combined[0])  // sum all
        .AddInput(*split_stacked_combined[1])  // square_sum_all
        .AddInput(*split_stacked_combined[2])  // count_all
        .Input(running_mean)
        .Input(running_variance)
        .AddOutput(mean_all)
        .AddOutput(invstd_all)
        .AddOutput(running_mean)
        .AddOutput(running_variance)
        .AddAttrs({{"momentum", static_cast<float>(1 - momentum)}})
        .AddAttrs({{"eps", static_cast<float>(epsilon)}})
        .Run(dev_ctx.stream());

    // ======================================================
    // step 3: compute global mean/variance
    // ======================================================
    phi::DenseTensor global_mean(mean_all), global_variance;
    phi::DenseTensorMeta one_meta = {phi::DataType::FLOAT32, {C}};
    phi::DenseTensor one;
    one.set_meta(one_meta);
    dev_ctx.template Alloc<float>(one);
    FillNpuTensorWithConstant<float>(&one, dev_ctx, static_cast<float>(1.0));

    phi::DenseTensor invstd_square;
    custom_kernel::MultiplyKernel<float, Context>(
        dev_ctx, invstd_all, invstd_all, &invstd_square);
    custom_kernel::DivideKernel<float, Context>(
        dev_ctx, one, invstd_pow, &global_variance);

    // ======================================================
    // step 4: normalize input and update mean_out/variance_out
    // ======================================================
    custom_kernel::BatchNormKernel<T, Context>(
        dev_ctx,
        x,
        global_mean,      // use global mean
        global_variance,  // use global variance
        scale,
        bias,
        is_test,
        momentum,
        epsilon,
        data_layout_str,
        use_global_stats,
        trainable_stats,
        y,
        mean_out,
        variance_out,
        saved_mean,
        saved_variance,
        reserve_space);
  }
}

template <typename T, typename Context>
void SyncBatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& scale,
    const phi::DenseTensor& bias,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& y_grad,
    float momentum,
    float epsilon_f,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sync_batch_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SyncBatchNormKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);   // mean
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);   // variance
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);   // scale
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);   // bias
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);  // saved_mean
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);  // saved_variance
  }
}

PD_REGISTER_PLUGIN_KERNEL(sync_batch_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SyncBatchNormGradKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
