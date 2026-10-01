// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_PIPELINE_H_
#define CUDA_IMGPROC_INCLUDE_PIPELINE_H_

#include <string>
#include <vector>

#include "image_types.h"

namespace cuda_imgproc {

// Batch image processing pipeline orchestrator.
class BatchPipeline {
 public:
  explicit BatchPipeline(const CommandLineConfig& config);
  ~BatchPipeline();

  // Executes the batch image processing pipeline across all input images.
  bool Run();

  // Returns recorded performance metrics for all processed images.
  const std::vector<PerformanceMetrics>& GetMetrics() const { return metrics_; }

  // Exports metrics to a CSV benchmark file.
  bool ExportMetricsCsv(const std::string& csv_filepath) const;

  // Prints formatted summary statistics table to stdout.
  void PrintSummary() const;

 private:
  // Processes a single image through the selected engine.
  bool ProcessImage(const std::string& filepath, int stream_idx = 0);

  CommandLineConfig config_;
  std::vector<PerformanceMetrics> metrics_;
  std::vector<cudaStream_t> streams_;
  double total_wall_time_ms_ = 0.0;
  int total_images_processed_ = 0;
};

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_PIPELINE_H_
