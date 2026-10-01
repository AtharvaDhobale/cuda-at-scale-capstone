// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_FILTERS_CUDA_CUH_
#define CUDA_IMGPROC_INCLUDE_FILTERS_CUDA_CUH_

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "image_types.h"

namespace cuda_imgproc {

// High-performance custom CUDA kernel wrappers featuring shared memory
// tiling, constant memory caching, and asynchronous execution across CUDA streams.
class CudaImageFilter {
 public:
  // Initializes constant memory weights and GPU buffers.
  static void Initialize();

  // Applies 5x5 Gaussian blur using shared memory tiled 2D convolution.
  static bool ApplyGaussianBlur(const ImageBuffer& input, ImageBuffer* output,
                                PerformanceMetrics* metrics = nullptr,
                                cudaStream_t stream = 0);

  // Applies 3x3 Sobel edge detection using shared memory tiling.
  static bool ApplySobelFilter(const ImageBuffer& input, ImageBuffer* output,
                               PerformanceMetrics* metrics = nullptr,
                               cudaStream_t stream = 0);

  // Applies color inversion kernel with coalesced global memory access.
  static bool ApplyColorInvert(const ImageBuffer& input, ImageBuffer* output,
                               PerformanceMetrics* metrics = nullptr,
                               cudaStream_t stream = 0);

  // Applies GPU-accelerated 2D affine rotation.
  static bool ApplyRotation(const ImageBuffer& input, ImageBuffer* output,
                            double angle_deg, PerformanceMetrics* metrics = nullptr,
                            cudaStream_t stream = 0);
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_FILTERS_CUDA_CUH_
