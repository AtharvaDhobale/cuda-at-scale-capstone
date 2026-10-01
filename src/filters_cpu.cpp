// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "filters_cpu.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuda_imgproc {

namespace {

// Clamps value between [0, max_val].
inline int Clamp(int val, int max_val) {
  return std::max(0, std::min(val, max_val));
}

}  // namespace

ImageBuffer CpuImageFilter::ApplyGaussianBlur(const ImageBuffer& input, int kernel_size,
                                             double sigma) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  ImageBuffer output(width, height, channels);

  int radius = kernel_size / 2;
  std::vector<float> kernel(kernel_size * kernel_size);
  float sum = 0.0f;

  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
      kernel[(y + radius) * kernel_size + (x + radius)] = val;
      sum += val;
    }
  }

  for (size_t i = 0; i < kernel.size(); ++i) {
    kernel[i] /= sum;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        float accum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
          for (int kx = -radius; kx <= radius; ++kx) {
            int px = Clamp(x + kx, width - 1);
            int py = Clamp(y + ky, height - 1);
            float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
            accum += input.at(px, py, c) * weight;
          }
        }
        output.at(x, y, c) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, accum)));
      }
    }
  }

  return output;
}

ImageBuffer CpuImageFilter::ApplySobelFilter(const ImageBuffer& input) {
  ImageBuffer gray = (input.channels() == 3) ? input.ToGrayscale() : input;
  int width = gray.width();
  int height = gray.height();
  ImageBuffer output(width, height, 1);

  const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int gx = 0;
      int gy = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int px = Clamp(x + kx, width - 1);
          int py = Clamp(y + ky, height - 1);
          int val = gray.at(px, py, 0);
          gx += val * sobel_x[ky + 1][kx + 1];
          gy += val * sobel_y[ky + 1][kx + 1];
        }
      }
      int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
      output.at(x, y, 0) = static_cast<uint8_t>(std::min(255, std::max(0, magnitude)));
    }
  }

  return output;
}

ImageBuffer CpuImageFilter::ApplyColorInvert(const ImageBuffer& input) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  ImageBuffer output(width, height, channels);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        output.at(x, y, c) = 255 - input.at(x, y, c);
      }
    }
  }

  return output;
}

ImageBuffer CpuImageFilter::ApplyRotation(const ImageBuffer& input, double angle_deg) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();
  ImageBuffer output(width, height, channels);

  double rad = angle_deg * M_PI / 180.0;
  double cos_a = std::cos(rad);
  double sin_a = std::sin(rad);

  double cx = width / 2.0;
  double cy = height / 2.0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double dx = x - cx;
      double dy = y - cy;

      // Inverse mapping
      double src_x = cx + (dx * cos_a + dy * sin_a);
      double src_y = cy + (-dx * sin_a + dy * cos_a);

      int sx = static_cast<int>(std::round(src_x));
      int sy = static_cast<int>(std::round(src_y));

      for (int c = 0; c < channels; ++c) {
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
          output.at(x, y, c) = input.at(sx, sy, c);
        } else {
          output.at(x, y, c) = 0;  // Black background
        }
      }
    }
  }

  return output;
}

}  // namespace cuda_imgproc
