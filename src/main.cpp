// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "cuda_utils.h"
#include "image_types.h"
#include "pipeline.h"

namespace cuda_imgproc {

// Displays comprehensive CLI usage and options.
void PrintUsage(const char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n\n"
            << "High-Throughput GPU Image Processing Pipeline (CUDA & NPP)\n\n"
            << "Options:\n"
            << "  -i, --input_dir <path>     Directory containing input .pgm / .ppm images (default: data/inputs)\n"
            << "  -o, --output_dir <path>    Directory where processed images are saved (default: data/outputs)\n"
            << "  -f, --filter <type>        Filter operation: gaussian | sobel | rotate | invert | all (default: gaussian)\n"
            << "  -m, --mode <engine>        Execution engine: custom | npp | cpu | compare (default: custom)\n"
            << "  -b, --batch_size <N>       Batch size for pipelined processing (default: 16)\n"
            << "  -s, --streams <N>          Number of concurrent CUDA streams (default: 4)\n"
            << "  -a, --angle <deg>          Rotation angle in degrees for rotate filter (default: 45.0)\n"
            << "  -c, --csv <path>           CSV metrics output file path (default: artifacts/benchmark_results.csv)\n"
            << "  -v, --verbose              Enable detailed per-image latency and throughput logs\n"
            << "  -h, --help                 Display this help message and exit\n\n"
            << "Examples:\n"
            << "  " << prog_name << " --input_dir data/inputs --filter gaussian --mode custom\n"
            << "  " << prog_name << " --input_dir data/inputs --filter sobel --mode npp --verbose\n"
            << "  " << prog_name << " --input_dir data/inputs --filter gaussian --mode compare\n"
            << std::endl;
}

// Parses command line arguments into CommandLineConfig struct.
bool ParseCommandLineArguments(int argc, char* argv[], CommandLineConfig* config,
                               std::string* csv_path) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if ((arg == "-i" || arg == "--input_dir") && i + 1 < argc) {
      config->input_dir = argv[++i];
    } else if ((arg == "-o" || arg == "--output_dir") && i + 1 < argc) {
      config->output_dir = argv[++i];
    } else if ((arg == "-f" || arg == "--filter") && i + 1 < argc) {
      std::string filter_str = argv[++i];
      if (filter_str == "gaussian") {
        config->filter = FilterType::kGaussian;
      } else if (filter_str == "sobel") {
        config->filter = FilterType::kSobel;
      } else if (filter_str == "rotate") {
        config->filter = FilterType::kRotate;
      } else if (filter_str == "invert") {
        config->filter = FilterType::kInvert;
      } else if (filter_str == "all") {
        config->filter = FilterType::kAll;
      } else {
        std::cerr << "Unknown filter type: " << filter_str << std::endl;
        return false;
      }
    } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
      std::string mode_str = argv[++i];
      if (mode_str == "custom") {
        config->mode = ExecutionMode::kCustomCuda;
      } else if (mode_str == "npp") {
        config->mode = ExecutionMode::kNpp;
      } else if (mode_str == "cpu") {
        config->mode = ExecutionMode::kCpu;
      } else if (mode_str == "compare") {
        config->mode = ExecutionMode::kCompare;
      } else {
        std::cerr << "Unknown execution mode: " << mode_str << std::endl;
        return false;
      }
    } else if ((arg == "-b" || arg == "--batch_size") && i + 1 < argc) {
      config->batch_size = std::atoi(argv[++i]);
    } else if ((arg == "-s" || arg == "--streams") && i + 1 < argc) {
      config->num_streams = std::atoi(argv[++i]);
    } else if ((arg == "-a" || arg == "--angle") && i + 1 < argc) {
      config->rotation_angle = std::atof(argv[++i]);
    } else if ((arg == "-c" || arg == "--csv") && i + 1 < argc) {
      *csv_path = argv[++i];
    } else if (arg == "-v" || arg == "--verbose") {
      config->verbose = true;
    } else {
      std::cerr << "Unrecognized option: " << arg << std::endl;
      PrintUsage(argv[0]);
      return false;
    }
  }

  return true;
}

}  // namespace cuda_imgproc

int main(int argc, char* argv[]) {
  cuda_imgproc::CommandLineConfig config;
  std::string csv_path = "artifacts/benchmark_results.csv";

  if (!cuda_imgproc::ParseCommandLineArguments(argc, argv, &config, &csv_path)) {
    return EXIT_FAILURE;
  }

  std::cout << "============================================================" << std::endl;
  std::cout << "  CUDA AT SCALE FOR THE ENTERPRISE - CAPSTONE PROJECT       " << std::endl;
  std::cout << "  High-Throughput GPU Image Processing Pipeline             " << std::endl;
  std::cout << "============================================================" << std::endl;

  // Print GPU device hardware info and verify CUDA availability
  if (!cuda_imgproc::PrintGpuDeviceInfo()) {
    std::cerr << "Initialization failed: No valid CUDA GPU detected." << std::endl;
    return EXIT_FAILURE;
  }

  // Initialize and run the batch processing pipeline
  cuda_imgproc::BatchPipeline pipeline(config);
  if (!pipeline.Run()) {
    std::cerr << "Pipeline execution completed with errors." << std::endl;
    return EXIT_FAILURE;
  }

  // Export benchmark metrics
  if (pipeline.ExportMetricsCsv(csv_path)) {
    std::cout << "Successfully exported benchmark metrics to: " << csv_path << std::endl;
  }

  std::cout << "All operations finished successfully." << std::endl;
  return EXIT_SUCCESS;
}
