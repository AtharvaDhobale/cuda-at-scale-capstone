// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_IMAGE_TYPES_H_
#define CUDA_IMGPROC_INCLUDE_IMAGE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuda_imgproc {

// Pixel representation for 3-channel 24-bit RGB images.
struct PixelRGB {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

// Image channel format.
enum class ChannelFormat {
  kGrayscale = 1,
  kRGB = 3,
};

// Supported image processing filter operations.
enum class FilterType {
  kGaussian,
  kSobel,
  kRotate,
  kInvert,
  kAll,
};

// Supported execution processing backends.
enum class ExecutionMode {
  kNpp,
  kCustomCuda,
  kCpu,
  kCompare,
};

// Command-line execution configuration.
struct CommandLineConfig {
  std::string input_dir = "data/inputs";
  std::string output_dir = "data/outputs";
  FilterType filter = FilterType::kGaussian;
  ExecutionMode mode = ExecutionMode::kCustomCuda;
  int batch_size = 16;
  int num_streams = 4;
  double rotation_angle = 45.0;
  bool verbose = false;
  bool run_benchmark = false;
};

// Performance metrics for a single image or batch run.
struct PerformanceMetrics {
  std::string filename;
  int width = 0;
  int height = 0;
  int channels = 0;
  double host_to_device_ms = 0.0;
  double kernel_execution_ms = 0.0;
  double device_to_host_ms = 0.0;
  double total_gpu_ms = 0.0;
  double cpu_baseline_ms = 0.0;
  double speedup_vs_cpu = 0.0;
  double throughput_megapixels_per_sec = 0.0;
};

// In-memory 8-bit image representation supporting Grayscale and RGB.
class ImageBuffer {
 public:
  ImageBuffer() : width_(0), height_(0), channels_(1), pitch_(0) {}

  ImageBuffer(int width, int height, int channels = 1)
      : width_(width), height_(height), channels_(channels),
        pitch_(width * channels), data_(width * height * channels, 0) {}

  ImageBuffer(int width, int height, int channels, const uint8_t* raw_data)
      : width_(width), height_(height), channels_(channels),
        pitch_(width * channels),
        data_(raw_data, raw_data + (width * height * channels)) {}

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  size_t pitch() const { return pitch_; }
  size_t TotalBytes() const { return data_.size(); }
  bool IsEmpty() const { return data_.empty(); }

  uint8_t* data() { return data_.data(); }
  const uint8_t* data() const { return data_.data(); }

  uint8_t& at(int x, int y, int c = 0) {
    return data_[(y * width_ + x) * channels_ + c];
  }

  const uint8_t& at(int x, int y, int c = 0) const {
    return data_[(y * width_ + x) * channels_ + c];
  }

  // Convert an RGB image to single-channel Grayscale using ITU-R BT.601 standard.
  ImageBuffer ToGrayscale() const {
    if (channels_ == 1) {
      return *this;
    }
    ImageBuffer gray(width_, height_, 1);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        int idx = (y * width_ + x) * channels_;
        double lum = 0.299 * data_[idx] + 0.587 * data_[idx + 1] + 0.114 * data_[idx + 2];
        gray.at(x, y, 0) = static_cast<uint8_t>(lum > 255.0 ? 255 : (lum < 0.0 ? 0 : lum));
      }
    }
    return gray;
  }

 private:
  int width_;
  int height_;
  int channels_;
  size_t pitch_;
  std::vector<uint8_t> data_;
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_IMAGE_TYPES_H_
