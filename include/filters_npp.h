// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_FILTERS_NPP_H_
#define CUDA_IMGPROC_INCLUDE_FILTERS_NPP_H_

#include "cuda_utils.h"
#include "image_types.h"

namespace cuda_imgproc {

// NVIDIA Performance Primitives (NPP) accelerated filter wrappers.
class NppImageFilter {
 public:
  // Applies Gaussian filter using NVIDIA NPP.
  static bool ApplyGaussianBlur(const ImageBuffer& input, ImageBuffer* output,
                                PerformanceMetrics* metrics = nullptr);

  // Applies Sobel edge filter using NVIDIA NPP.
  static bool ApplySobelFilter(const ImageBuffer& input, ImageBuffer* output,
                               PerformanceMetrics* metrics = nullptr);

  // Applies Rotation transform using NVIDIA NPP.
  static bool ApplyRotation(const ImageBuffer& input, ImageBuffer* output,
                            double angle_deg, PerformanceMetrics* metrics = nullptr);

  // Applies Threshold filter using NVIDIA NPP.
  static bool ApplyThreshold(const ImageBuffer& input, ImageBuffer* output,
                             uint8_t threshold_val, PerformanceMetrics* metrics = nullptr);
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_FILTERS_NPP_H_
