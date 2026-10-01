// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef CUDA_IMGPROC_INCLUDE_CUDA_UTILS_H_
#define CUDA_IMGPROC_INCLUDE_CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <string>

namespace cuda_imgproc {

// Macro to check CUDA runtime API return status.
#define CUDA_CHECK(val) cuda_imgproc::CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t result, const char* const func,
                      const char* const file, int const line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " code="
              << static_cast<unsigned int>(result) << " ("
              << cudaGetErrorString(result) << ") \"" << func << "\""
              << std::endl;
    // Don't call exit immediately to allow caller exception/cleanup if needed.
  }
}

// Macro to check NPP status.
#define NPP_CHECK(val) cuda_imgproc::CheckNpp((val), #val, __FILE__, __LINE__)

inline void CheckNpp(NppStatus result, const char* const func,
                     const char* const file, int const line) {
  if (result != NPP_SUCCESS) {
    std::cerr << "NPP error at " << file << ":" << line << " code="
              << static_cast<int>(result) << " \"" << func << "\""
              << std::endl;
  }
}

// High-precision GPU timer using CUDA Events.
class GpuTimer {
 public:
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Start(cudaStream_t stream = 0) {
    cudaEventRecord(start_, stream);
  }

  void Stop(cudaStream_t stream = 0) {
    cudaEventRecord(stop_, stream);
  }

  // Returns elapsed time in milliseconds.
  float ElapsedTimeMs() {
    cudaEventSynchronize(stop_);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_, stop_);
    return elapsed_ms;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

// Prints comprehensive GPU architecture and CUDA runtime information.
inline bool PrintGpuDeviceInfo() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA-capable GPU detected or CUDA driver not installed." << std::endl;
    return false;
  }

  int current_device = 0;
  cudaGetDevice(&current_device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  std::cout << "============================================================" << std::endl;
  std::cout << "CUDA GPU Hardware Information:" << std::endl;
  std::cout << "  Device ID:                " << current_device << std::endl;
  std::cout << "  Device Name:              " << prop.name << std::endl;
  std::cout << "  Compute Capability:       " << prop.major << "." << prop.minor << std::endl;
  std::cout << "  Streaming Multiprocessors:" << prop.multiProcessorCount << std::endl;
  std::cout << "  Total Global Memory:      " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
  std::cout << "  Shared Memory per Block:  " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
  std::cout << "  Warp Size:                " << prop.warpSize << std::endl;
  std::cout << "  Max Threads per Block:    " << prop.maxThreadsPerBlock << std::endl;

  int driver_version = 0;
  int runtime_version = 0;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  std::cout << "  CUDA Driver Version:      " << driver_version / 1000 << "." << (driver_version % 100) / 10 << std::endl;
  std::cout << "  CUDA Runtime Version:     " << runtime_version / 1000 << "." << (runtime_version % 100) / 10 << std::endl;

  const NppLibraryVersion* lib_ver = nppGetLibVersion();
  std::cout << "  NPP Library Version:      " << lib_ver->major << "." << lib_ver->minor << "." << lib_ver->build << std::endl;
  std::cout << "============================================================" << std::endl;

  return true;
}

}  // namespace cuda_imgproc

#endif  // CUDA_IMGPROC_INCLUDE_CUDA_UTILS_H_
