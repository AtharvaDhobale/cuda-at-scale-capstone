// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "filters_cuda.cuh"

#include <cmath>
#include <iostream>

namespace cuda_imgproc {

// Constant memory allocations for high-speed broadcast access.
__constant__ float c_gaussian_weights[25];
__constant__ int c_sobel_x[9];
__constant__ int c_sobel_y[9];

namespace {

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;
constexpr int kRadius = 2;
constexpr int kTileDimX = kBlockDimX + 2 * kRadius;  // 20
constexpr int kTileDimY = kBlockDimY + 2 * kRadius;  // 20

// Clamps coordinate to valid range [0, max_val - 1].
__device__ inline int DeviceClamp(int val, int max_val) {
  return max(0, min(val, max_val - 1));
}

// 2D Gaussian convolution kernel utilizing shared memory tiling and constant memory weights.
__global__ void GaussianBlurSharedKernel(const uint8_t* __restrict__ src,
                                         uint8_t* __restrict__ dst,
                                         int width, int height, int channels) {
  __shared__ uint8_t shared_tile[kTileDimY][kTileDimX];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int out_x = blockIdx.x * blockDim.x + tx;
  int out_y = blockIdx.y * blockDim.y + ty;

  for (int c = 0; c < channels; ++c) {
    // Cooperatively load tile into shared memory (including halo/apron).
    // Each 16x16 thread block loads 20x20 elements.
    int linear_tid = ty * kBlockDimX + tx;
    int total_elements = kTileDimX * kTileDimY;  // 400 elements

    for (int idx = linear_tid; idx < total_elements; idx += (kBlockDimX * kBlockDimY)) {
      int sm_y = idx / kTileDimX;
      int sm_x = idx % kTileDimX;
      int global_x = DeviceClamp(blockIdx.x * blockDim.x - kRadius + sm_x, width);
      int global_y = DeviceClamp(blockIdx.y * blockDim.y - kRadius + sm_y, height);
      shared_tile[sm_y][sm_x] = src[(global_y * width + global_x) * channels + c];
    }

    __syncthreads();

    if (out_x < width && out_y < height) {
      float accum = 0.0f;
      #pragma unroll
      for (int ky = -kRadius; ky <= kRadius; ++ky) {
        #pragma unroll
        for (int kx = -kRadius; kx <= kRadius; ++kx) {
          float weight = c_gaussian_weights[(ky + kRadius) * (2 * kRadius + 1) + (kx + kRadius)];
          uint8_t pixel = shared_tile[ty + kRadius + ky][tx + kRadius + kx];
          accum += pixel * weight;
        }
      }
      dst[(out_y * width + out_x) * channels + c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, accum)));
    }

    __syncthreads();
  }
}

// 3x3 Sobel edge detection kernel.
__global__ void SobelEdgeKernel(const uint8_t* __restrict__ src_gray,
                                uint8_t* __restrict__ dst_edge,
                                int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int gx = 0;
  int gy = 0;

  #pragma unroll
  for (int ky = -1; ky <= 1; ++ky) {
    #pragma unroll
    for (int kx = -1; kx <= 1; ++kx) {
      int px = DeviceClamp(x + kx, width);
      int py = DeviceClamp(y + ky, height);
      int val = src_gray[py * width + px];
      int kidx = (ky + 1) * 3 + (kx + 1);
      gx += val * c_sobel_x[kidx];
      gy += val * c_sobel_y[kidx];
    }
  }

  int mag = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
  dst_edge[y * width + x] = static_cast<uint8_t>(min(255, max(0, mag)));
}

// Fast parallel color inversion kernel.
__global__ void ColorInvertKernel(const uint8_t* __restrict__ src,
                                  uint8_t* __restrict__ dst,
                                  size_t total_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_bytes) {
    dst[idx] = 255 - src[idx];
  }
}

// 2D Rotation kernel using inverse coordinate transformation.
__global__ void RotationKernel(const uint8_t* __restrict__ src,
                               uint8_t* __restrict__ dst,
                               int width, int height, int channels,
                               float cos_a, float sin_a,
                               float center_x, float center_y) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  float dx = x - center_x;
  float dy = y - center_y;

  // Inverse rotation mapping
  float src_xf = center_x + (dx * cos_a + dy * sin_a);
  float src_yf = center_y + (-dx * sin_a + dy * cos_a);

  int src_x = __float2int_rn(src_xf);
  int src_y = __float2int_rn(src_yf);

  for (int c = 0; c < channels; ++c) {
    if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
      dst[(y * width + x) * channels + c] = src[(src_y * width + src_x) * channels + c];
    } else {
      dst[(y * width + x) * channels + c] = 0;  // Black background
    }
  }
}

}  // namespace

void CudaImageFilter::Initialize() {
  static bool initialized = false;
  if (initialized) return;

  // Setup 5x5 Gaussian kernel weights (sigma = 1.0)
  float h_gaussian[25];
  float sum = 0.0f;
  const float sigma = 1.0f;
  for (int y = -2; y <= 2; ++y) {
    for (int x = -2; x <= 2; ++x) {
      float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
      h_gaussian[(y + 2) * 5 + (x + 2)] = val;
      sum += val;
    }
  }
  for (int i = 0; i < 25; ++i) {
    h_gaussian[i] /= sum;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian_weights, h_gaussian, sizeof(h_gaussian)));

  // Setup Sobel operators
  const int h_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const int h_sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  CUDA_CHECK(cudaMemcpyToSymbol(c_sobel_x, h_sobel_x, sizeof(h_sobel_x)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sobel_y, h_sobel_y, sizeof(h_sobel_y)));

  initialized = true;
}

bool CudaImageFilter::ApplyGaussianBlur(const ImageBuffer& input, ImageBuffer* output,
                                        PerformanceMetrics* metrics,
                                        cudaStream_t stream) {
  Initialize();
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  size_t bytes = input.TotalBytes();

  *output = ImageBuffer(width, height, channels);

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  cudaEvent_t ev_start, ev_htod_done, ev_kernel_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_src, input.data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(ev_htod_done, stream));

  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GaussianBlurSharedKernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height, channels);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream));

  CUDA_CHECK(cudaMemcpyAsync(output->data(), d_dst, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, kernel_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&kernel_ms, ev_htod_done, ev_kernel_done);
    cudaEventElapsedTime(&dtoh_ms, ev_kernel_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = kernel_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_kernel_done);
  cudaEventDestroy(ev_dtoh_done);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));

  return true;
}

bool CudaImageFilter::ApplySobelFilter(const ImageBuffer& input, ImageBuffer* output,
                                      PerformanceMetrics* metrics,
                                      cudaStream_t stream) {
  Initialize();
  ImageBuffer gray_input = (input.channels() == 3) ? input.ToGrayscale() : input;
  int width = gray_input.width();
  int height = gray_input.height();
  size_t bytes = width * height;

  *output = ImageBuffer(width, height, 1);

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  cudaEvent_t ev_start, ev_htod_done, ev_kernel_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_src, gray_input.data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(ev_htod_done, stream));

  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  SobelEdgeKernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream));

  CUDA_CHECK(cudaMemcpyAsync(output->data(), d_dst, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, kernel_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&kernel_ms, ev_htod_done, ev_kernel_done);
    cudaEventElapsedTime(&dtoh_ms, ev_kernel_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = kernel_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_kernel_done);
  cudaEventDestroy(ev_dtoh_done);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));

  return true;
}

bool CudaImageFilter::ApplyColorInvert(const ImageBuffer& input, ImageBuffer* output,
                                       PerformanceMetrics* metrics,
                                       cudaStream_t stream) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  size_t bytes = input.TotalBytes();

  *output = ImageBuffer(width, height, channels);

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  cudaEvent_t ev_start, ev_htod_done, ev_kernel_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_src, input.data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(ev_htod_done, stream));

  int threads = 256;
  int blocks = (bytes + threads - 1) / threads;

  ColorInvertKernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, bytes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream));

  CUDA_CHECK(cudaMemcpyAsync(output->data(), d_dst, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, kernel_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&kernel_ms, ev_htod_done, ev_kernel_done);
    cudaEventElapsedTime(&dtoh_ms, ev_kernel_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = kernel_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_kernel_done);
  cudaEventDestroy(ev_dtoh_done);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));

  return true;
}

bool CudaImageFilter::ApplyRotation(const ImageBuffer& input, ImageBuffer* output,
                                    double angle_deg, PerformanceMetrics* metrics,
                                    cudaStream_t stream) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  size_t bytes = input.TotalBytes();

  *output = ImageBuffer(width, height, channels);

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  cudaEvent_t ev_start, ev_htod_done, ev_kernel_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_kernel_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_src, input.data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(ev_htod_done, stream));

  float rad = static_cast<float>(angle_deg * M_PI / 180.0);
  float cos_a = std::cos(rad);
  float sin_a = std::sin(rad);
  float cx = width / 2.0f;
  float cy = height / 2.0f;

  dim3 block(kBlockDimX, kBlockDimY);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  RotationKernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height, channels,
                                             cos_a, sin_a, cx, cy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_kernel_done, stream));

  CUDA_CHECK(cudaMemcpyAsync(output->data(), d_dst, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, kernel_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&kernel_ms, ev_htod_done, ev_kernel_done);
    cudaEventElapsedTime(&dtoh_ms, ev_kernel_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = kernel_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_kernel_done);
  cudaEventDestroy(ev_dtoh_done);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));

  return true;
}

}  // namespace cuda_imgproc
