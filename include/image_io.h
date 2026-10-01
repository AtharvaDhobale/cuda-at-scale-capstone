// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_IMAGE_IO_H_
#define CUDA_IMGPROC_INCLUDE_IMAGE_IO_H_

#include <string>
#include <vector>

#include "image_types.h"

namespace cuda_imgproc {

// Portable Netpbm image I/O handler for PGM (grayscale) and PPM (RGB) formats.
class ImageIO {
 public:
  // Loads an image from disk (.pgm or .ppm).
  // Automatically detects P5 (Grayscale binary) and P6 (RGB binary).
  static bool ReadImage(const std::string& filepath, ImageBuffer* image_out);

  // Writes an image to disk as either PGM (if channels == 1) or PPM (if channels == 3).
  static bool WriteImage(const std::string& filepath, const ImageBuffer& image);

  // Scans a directory for all .ppm and .pgm files.
  static std::vector<std::string> ListImageFiles(const std::string& directory_path);

  // Extracts filename without directory path.
  static std::string GetBaseFilename(const std::string& filepath);

  // Generates an output filename with filter suffix, e.g. "image_gaussian.pgm".
  static std::string FormatOutputFilename(const std::string& input_filepath,
                                          const std::string& filter_name,
                                          const std::string& output_dir);
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_IMAGE_IO_H_
