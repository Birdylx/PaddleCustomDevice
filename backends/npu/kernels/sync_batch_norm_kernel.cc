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

inline void ExtractNCWHD(const phi::DDim& dims,
                         const DataLayout& data_layout,
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
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == DataLayout::kNCHW ? dims[4] : dims[3])
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

    phi::DenseTensor x_tensor(x), y_tensor(*y);
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
        y_meta = {y->dtype(), y->dims(), phi::DataLayout::kNDHWC};
      } else {
        x_meta = {x.dtype(), x_tensor.dims(), phi::DataLayout::kNCDHW};
        y_meta = {y->dtype(), y->dims(), phi::DataLayout::kNCDHW};
      }
      x_tensor.set_meta(x_meta);
      y_tensor.set_meta(y_meta);
    } else {
      if (channel_last) {
        phi::DenseTensorMeta x_meta = {
            x.dtype(), x_tensor.dims(), phi::DataLayout::kNHWC};
        phi::DenseTensorMeta y_meta = {
            y->dtype(), y->dims(), phi::DataLayout::kNHWC};
        x_tensor.set_meta(x_meta);
        y_tensor.set_meta(y_meta);
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

    phi::DenseTensorMeta tmp_meta = {phi::DataType::FLOAT32, {1}};
    phi::DenseTensor count;
    cout.set_meta(count_meta);
    dev_ctx.template Alloc<float>(count);
    FillNpuTensorWithConstant<float>(
        &count, dev_ctx, static_cast<float>(N * H * W * D));

    phi::DenseTensor combined;
    custom_kernel::ConcatKernel<float, Context>(
        dev_ctx, {&sum, &square_sum, &count}, 0, &combined);

    dev_ctx.template Alloc<float>(mean_out);
    dev_ctx.template Alloc<float>(variance_out);
    dev_ctx.template Alloc<float>(saved_mean);
    dev_ctx.template Alloc<float>(saved_variance);
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
