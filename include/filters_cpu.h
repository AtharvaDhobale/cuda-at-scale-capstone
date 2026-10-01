// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_FILTERS_CPU_H_
#define CUDA_IMGPROC_INCLUDE_FILTERS_CPU_H_

#include "image_types.h"

namespace cuda_imgproc {

// CPU Baseline implementation of image processing filters for verification
// and benchmarking against GPU-accelerated implementations.
class CpuImageFilter {
 public:
  // Applies 3x3 or 5x5 Gaussian convolution blur on CPU.
  static ImageBuffer ApplyGaussianBlur(const ImageBuffer& input, int kernel_size = 5,
                                      double sigma = 1.0);

  // Applies Sobel edge detection on CPU.
  static ImageBuffer ApplySobelFilter(const ImageBuffer& input);

  // Inverts pixel colors on CPU.
  static ImageBuffer ApplyColorInvert(const ImageBuffer& input);

  // Rotates an image around center on CPU.
  static ImageBuffer ApplyRotation(const ImageBuffer& input, double angle_deg);
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_FILTERS_CPU_H_
