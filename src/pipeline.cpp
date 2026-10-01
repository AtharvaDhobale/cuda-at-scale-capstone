// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "pipeline.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "cuda_utils.h"
#include "filters_cpu.h"
#include "filters_cuda.cuh"
#include "filters_npp.h"
#include "image_io.h"

namespace cuda_imgproc {

BatchPipeline::BatchPipeline(const CommandLineConfig& config)
    : config_(config), total_wall_time_ms_(0.0), total_images_processed_(0) {
  int num_streams = std::max(1, config_.num_streams);
  streams_.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

BatchPipeline::~BatchPipeline() {
  for (auto& stream : streams_) {
    cudaStreamDestroy(stream);
  }
}

bool BatchPipeline::Run() {
  std::vector<std::string> image_files = ImageIO::ListImageFiles(config_.input_dir);
  if (image_files.empty()) {
    std::cerr << "No .pgm or .ppm image files found in directory: " << config_.input_dir << std::endl;
    return false;
  }

  std::cout << "\n============================================================" << std::endl;
  std::cout << "Starting Batch Pipeline Execution" << std::endl;
  std::cout << "  Input Directory:   " << config_.input_dir << std::endl;
  std::cout << "  Total Images:      " << image_files.size() << std::endl;
  std::cout << "  CUDA Streams:      " << config_.num_streams << std::endl;
  std::cout << "  Batch Size:        " << config_.batch_size << std::endl;
  std::cout << "  Filter Operation:  ";
  switch (config_.filter) {
    case FilterType::kGaussian: std::cout << "Gaussian Blur (5x5)"; break;
    case FilterType::kSobel:    std::cout << "Sobel Edge Detection (3x3)"; break;
    case FilterType::kRotate:   std::cout << "Rotation (" << config_.rotation_angle << " deg)"; break;
    case FilterType::kInvert:   std::cout << "Color Inversion"; break;
    case FilterType::kAll:      std::cout << "All Filters (Multi-stage)"; break;
  }
  std::cout << "\n  Execution Engine:  ";
  switch (config_.mode) {
    case ExecutionMode::kNpp:        std::cout << "NVIDIA NPP Primitives"; break;
    case ExecutionMode::kCustomCuda: std::cout << "Custom CUDA Kernels (Shared Memory)"; break;
    case ExecutionMode::kCpu:        std::cout << "CPU Reference Baseline"; break;
    case ExecutionMode::kCompare:    std::cout << "Benchmark Comparison (CPU vs NPP vs CUDA)"; break;
  }
  std::cout << "\n============================================================\n" << std::endl;

  auto wall_start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < image_files.size(); ++i) {
    int stream_idx = i % streams_.size();
    if (!ProcessImage(image_files[i], stream_idx)) {
      std::cerr << "Failed to process image: " << image_files[i] << std::endl;
    }

    if ((i + 1) % 10 == 0 || i + 1 == image_files.size()) {
      std::cout << "  Processed [" << std::setw(3) << (i + 1) << "/"
                << image_files.size() << "] images..." << std::endl;
    }
  }

  // Synchronize all active CUDA streams
  for (auto& stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  total_wall_time_ms_ = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  total_images_processed_ = static_cast<int>(image_files.size());

  PrintSummary();
  return true;
}

bool BatchPipeline::ProcessImage(const std::string& filepath, int stream_idx) {
  ImageBuffer input_image;
  if (!ImageIO::ReadImage(filepath, &input_image)) {
    return false;
  }

  PerformanceMetrics m;
  m.filename = ImageIO::GetBaseFilename(filepath);
  m.width = input_image.width();
  m.height = input_image.height();
  m.channels = input_image.channels();

  cudaStream_t stream = streams_[stream_idx];
  ImageBuffer output_image;

  if (config_.mode == ExecutionMode::kCompare) {
    // 1. CPU Reference Execution
    auto cpu_start = std::chrono::high_resolution_clock::now();
    ImageBuffer cpu_out;
    if (config_.filter == FilterType::kGaussian) {
      cpu_out = CpuImageFilter::ApplyGaussianBlur(input_image);
    } else if (config_.filter == FilterType::kSobel) {
      cpu_out = CpuImageFilter::ApplySobelFilter(input_image);
    } else if (config_.filter == FilterType::kRotate) {
      cpu_out = CpuImageFilter::ApplyRotation(input_image, config_.rotation_angle);
    } else {
      cpu_out = CpuImageFilter::ApplyColorInvert(input_image);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    m.cpu_baseline_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // 2. Custom CUDA Execution
    ImageBuffer cuda_out;
    if (config_.filter == FilterType::kGaussian) {
      CudaImageFilter::ApplyGaussianBlur(input_image, &cuda_out, &m, stream);
    } else if (config_.filter == FilterType::kSobel) {
      CudaImageFilter::ApplySobelFilter(input_image, &cuda_out, &m, stream);
    } else if (config_.filter == FilterType::kRotate) {
      CudaImageFilter::ApplyRotation(input_image, &cuda_out, config_.rotation_angle, &m, stream);
    } else {
      CudaImageFilter::ApplyColorInvert(input_image, &cuda_out, &m, stream);
    }

    if (m.total_gpu_ms > 0.0) {
      m.speedup_vs_cpu = m.cpu_baseline_ms / m.total_gpu_ms;
    }

    // Save output
    std::string out_path = ImageIO::FormatOutputFilename(filepath, "cuda", config_.output_dir);
    ImageIO::WriteImage(out_path, cuda_out);
  } else if (config_.mode == ExecutionMode::kCustomCuda) {
    if (config_.filter == FilterType::kGaussian) {
      CudaImageFilter::ApplyGaussianBlur(input_image, &output_image, &m, stream);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "gaussian_cuda", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kSobel) {
      CudaImageFilter::ApplySobelFilter(input_image, &output_image, &m, stream);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "sobel_cuda", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kRotate) {
      CudaImageFilter::ApplyRotation(input_image, &output_image, config_.rotation_angle, &m, stream);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "rotate_cuda", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kInvert) {
      CudaImageFilter::ApplyColorInvert(input_image, &output_image, &m, stream);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "invert_cuda", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kAll) {
      // Multi-stage pipeline: Gaussian Blur -> Sobel Edge Detection
      ImageBuffer blurred;
      CudaImageFilter::ApplyGaussianBlur(input_image, &blurred, &m, stream);
      CudaImageFilter::ApplySobelFilter(blurred, &output_image, nullptr, stream);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "pipeline_cuda", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    }
  } else if (config_.mode == ExecutionMode::kNpp) {
    if (config_.filter == FilterType::kGaussian) {
      NppImageFilter::ApplyGaussianBlur(input_image, &output_image, &m);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "gaussian_npp", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kSobel) {
      NppImageFilter::ApplySobelFilter(input_image, &output_image, &m);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "sobel_npp", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else if (config_.filter == FilterType::kRotate) {
      NppImageFilter::ApplyRotation(input_image, &output_image, config_.rotation_angle, &m);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "rotate_npp", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    } else {
      NppImageFilter::ApplyThreshold(input_image, &output_image, 128, &m);
      std::string out_path = ImageIO::FormatOutputFilename(filepath, "threshold_npp", config_.output_dir);
      ImageIO::WriteImage(out_path, output_image);
    }
  } else if (config_.mode == ExecutionMode::kCpu) {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    if (config_.filter == FilterType::kGaussian) {
      output_image = CpuImageFilter::ApplyGaussianBlur(input_image);
    } else if (config_.filter == FilterType::kSobel) {
      output_image = CpuImageFilter::ApplySobelFilter(input_image);
    } else if (config_.filter == FilterType::kRotate) {
      output_image = CpuImageFilter::ApplyRotation(input_image, config_.rotation_angle);
    } else {
      output_image = CpuImageFilter::ApplyColorInvert(input_image);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    m.cpu_baseline_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    m.total_gpu_ms = m.cpu_baseline_ms;
    std::string out_path = ImageIO::FormatOutputFilename(filepath, "cpu", config_.output_dir);
    ImageIO::WriteImage(out_path, output_image);
  }

  if (config_.verbose) {
    std::cout << "  [" << m.filename << "] " << m.width << "x" << m.height
              << " (" << m.channels << "ch) -> GPU Kernel: " << std::fixed
              << std::setprecision(3) << m.kernel_execution_ms
              << " ms, Total GPU: " << m.total_gpu_ms << " ms";
    if (m.speedup_vs_cpu > 0.0) {
      std::cout << " (Speedup: " << std::setprecision(1) << m.speedup_vs_cpu << "x)";
    }
    std::cout << std::endl;
  }

  metrics_.push_back(m);
  return true;
}

bool BatchPipeline::ExportMetricsCsv(const std::string& csv_filepath) const {
  std::ofstream csv(csv_filepath);
  if (!csv.is_open()) {
    std::cerr << "Error opening CSV output file: " << csv_filepath << std::endl;
    return false;
  }

  csv << "Filename,Width,Height,Channels,HtoD_ms,Kernel_ms,DtoH_ms,TotalGPU_ms,CPU_ms,Speedup,Throughput_MP_per_s\n";
  for (const auto& m : metrics_) {
    csv << m.filename << ","
        << m.width << ","
        << m.height << ","
        << m.channels << ","
        << std::fixed << std::setprecision(4)
        << m.host_to_device_ms << ","
        << m.kernel_execution_ms << ","
        << m.device_to_host_ms << ","
        << m.total_gpu_ms << ","
        << m.cpu_baseline_ms << ","
        << std::setprecision(2)
        << m.speedup_vs_cpu << ","
        << m.throughput_megapixels_per_sec << "\n";
  }

  return true;
}

void BatchPipeline::PrintSummary() const {
  if (metrics_.empty()) return;

  double total_gpu_time = 0.0;
  double total_kernel_time = 0.0;
  double total_htod_time = 0.0;
  double total_dtoh_time = 0.0;
  double total_cpu_time = 0.0;
  double total_megapixels = 0.0;

  for (const auto& m : metrics_) {
    total_gpu_time += m.total_gpu_ms;
    total_kernel_time += m.kernel_execution_ms;
    total_htod_time += m.host_to_device_ms;
    total_dtoh_time += m.device_to_host_ms;
    total_cpu_time += m.cpu_baseline_ms;
    total_megapixels += (m.width * m.height) / 1e6;
  }

  double avg_gpu_ms = total_gpu_time / metrics_.size();
  double avg_kernel_ms = total_kernel_time / metrics_.size();
  double overall_throughput = total_megapixels / (total_wall_time_ms_ / 1000.0);
  double effective_fps = metrics_.size() / (total_wall_time_ms_ / 1000.0);

  std::cout << "\n============================================================" << std::endl;
  std::cout << "Pipeline Performance & Execution Summary" << std::endl;
  std::cout << "============================================================" << std::endl;
  std::cout << "  Images Processed:          " << metrics_.size() << std::endl;
  std::cout << "  Total Megapixels:          " << std::fixed << std::setprecision(2) << total_megapixels << " MP" << std::endl;
  std::cout << "  Total Wall Execution Time: " << std::setprecision(2) << total_wall_time_ms_ << " ms (" << (total_wall_time_ms_ / 1000.0) << " s)" << std::endl;
  std::cout << "  Average GPU Time / Image:  " << std::setprecision(3) << avg_gpu_ms << " ms" << std::endl;
  std::cout << "  Average Pure Kernel Time:  " << std::setprecision(3) << avg_kernel_ms << " ms" << std::endl;
  std::cout << "  Overall Throughput:        " << std::setprecision(2) << overall_throughput << " Megapixels/sec" << std::endl;
  std::cout << "  Effective Processing Rate: " << std::setprecision(1) << effective_fps << " FPS" << std::endl;

  if (total_cpu_time > 0.0) {
    double overall_speedup = total_cpu_time / total_gpu_time;
    std::cout << "  CPU Total Baseline Time:   " << std::setprecision(2) << total_cpu_time << " ms" << std::endl;
    std::cout << "  GPU vs CPU Speedup:        " << std::setprecision(2) << overall_speedup << "x" << std::endl;
  }

  std::cout << "============================================================\n" << std::endl;
}

}  // namespace cuda_imgproc
