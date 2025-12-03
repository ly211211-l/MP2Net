#include <vector>
#include "cuda/dcn_v2_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
// Removed in PyTorch 2.0+: THC API has been removed
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>
#include <cublas_v2.h>
#include <ATen/cuda/CUDABlas.h>

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    AT_ASSERTM(status == CUBLAS_STATUS_SUCCESS, "CUBLAS error: %d", status); \
  } while (0)

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu

__global__ void createBatchGemmBuffer(const float **input_b, float **output_b,
                                      float **columns_b, const float **ones_b,
                                      const float **weight_b, const float **bias_b,
                                      float *input, float *output,
                                      float *columns, float *ones,
                                      float *weight, float *bias,
                                      const int input_stride, const int output_stride,
                                      const int columns_stride, const int ones_stride,
                                      const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}

at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int deformable_group)
{
    using scalar_t = float;
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({batch, height_out, width_out}, input.options());
    auto columns = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    // prepare for batch-wise computing, which is significantly faster than instance-wise computing
    // when batch size is large.
    // launch batch threads
    int matrices_size = batch * sizeof(float *);
    const float **input_b;
    float **output_b;
    float **columns_b;
    const float **ones_b;
    const float **weight_b;
    const float **bias_b;
    
    cudaError_t err;
    err = cudaMalloc(&input_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for input_b");
    err = cudaMalloc(&output_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for output_b");
    err = cudaMalloc(&columns_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for columns_b");
    err = cudaMalloc(&ones_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for ones_b");
    err = cudaMalloc(&weight_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for weight_b");
    err = cudaMalloc(&bias_b, matrices_size);
    AT_ASSERTM(err == cudaSuccess, "cudaMalloc failed for bias_b");

    const int block = 128;
    const int grid = (batch + block - 1) / block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    createBatchGemmBuffer<<<grid, block, 0, stream>>>(
        input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        columns.data<scalar_t>(),
        ones.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        channels * width * height,
        channels_out * width_out * height_out,
        channels * kernel_h * kernel_w * height_out * width_out,
        height_out * width_out,
        batch);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta_bias = 0.0f;
    
    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            n_,
                            m_,
                            k_,
                            &alpha,
                            ones_b, k_,
                            bias_b, k_,
                            &beta_bias,
                            output_b, n_,
                            batch));

    modulated_deformable_im2col_cuda(stream,
                                     input.data<scalar_t>(),
                                     offset.data<scalar_t>(),
                                     mask.data<scalar_t>(),
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                     deformable_group,
                                     columns.data<scalar_t>());

    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;
    const float beta = 1.0f;
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            (const float **)columns_b, n,
                            weight_b, k,
                            &beta,
                            output_b, n,
                            batch));

    cudaFree(input_b);
    cudaFree(output_b);
    cudaFree(columns_b);
    cudaFree(ones_b);
    cudaFree(weight_b);
    cudaFree(bias_b);
    return output;
}

__global__ void createBatchGemmBufferBackward(
    float **grad_output_b,
    float **columns_b,
    float **ones_b,
    float **weight_b,
    float **grad_weight_b,
    float **grad_bias_b,
    float *grad_output,
    float *columns,
    float *ones,
    float *weight,
    float *grad_weight,
    float *grad_bias,
    const int grad_output_stride,
    const int columns_stride,
    const int ones_stride,
    const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        grad_output_b[idx] = grad_output + idx * grad_output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;

        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        grad_weight_b[idx] = grad_weight;
        grad_bias_b[idx] = grad_bias;
    }
}

std::vector<at::Tensor> dcn_v2_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &bias,
                                             const at::Tensor &offset,
                                             const at::Tensor &mask,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w,
                                             int dilation_h, int dilation_w,
                                             int deformable_group)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({height_out, width_out}, input.options());
    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float beta_zero = 0.0f;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);

        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha,
                         grad_output_n.data<scalar_t>(), n,
                         weight.data<scalar_t>(), m, &beta_zero,
                         columns.data<scalar_t>(), n));

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(stream,
                                               columns.data<scalar_t>(),
                                               input_n.data<scalar_t>(),
                                               offset_n.data<scalar_t>(),
                                               mask_n.data<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data<scalar_t>(),
                                               grad_mask_n.data<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(stream,
                                         columns.data<scalar_t>(),
                                         offset_n.data<scalar_t>(),
                                         mask_n.data<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(stream,
                                         input_n.data<scalar_t>(),
                                         offset_n.data<scalar_t>(),
                                         mask_n.data<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data<scalar_t>());

        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_, m_, k_, &alpha,
                         columns.data<scalar_t>(), k_,
                         grad_output_n.data<scalar_t>(), k_, &beta,
                         grad_weight.data<scalar_t>(), n_));

        // gradient w.r.t. bias
        // Use gemv: grad_bias = grad_output^T * ones
        // grad_output_n is k_ x m_ matrix, ones is k_ x 1 vector
        // After transpose, we compute: y = alpha * A^T * x + beta * y
        // where A is k_ x m_, x is k_ x 1, y is m_ x 1
        CUBLAS_CHECK(cublasSgemv(handle,
                         CUBLAS_OP_T,
                         k_,  // m: number of rows of A (before transpose)
                         m_,  // n: number of columns of A (before transpose)
                         &alpha,
                         grad_output_n.data<scalar_t>(), k_,  // A: k_ x m_ matrix, lda = k_
                         ones.data<scalar_t>(), 1,  // x: k_ x 1 vector, incx = 1
                         &beta,
                         grad_bias.data<scalar_t>(), 1));  // y: m_ x 1 vector, incy = 1
    }

    return {
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias
    };
}
