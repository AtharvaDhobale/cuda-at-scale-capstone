// Copyright (c) 2026 Atharva Dhobale. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "filters_npp.h"

#include <iostream>
#include <npp.h>

namespace cuda_imgproc {

bool NppImageFilter::ApplyGaussianBlur(const ImageBuffer& input, ImageBuffer* output,
                                       PerformanceMetrics* metrics) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();

  *output = ImageBuffer(width, height, channels);

  int src_step = 0;
  int dst_step = 0;
  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;

  if (channels == 1) {
    d_src = nppiMalloc_8u_C1(width, height, &src_step);
    d_dst = nppiMalloc_8u_C1(width, height, &dst_step);
  } else {
    d_src = nppiMalloc_8u_C3(width, height, &src_step);
    d_dst = nppiMalloc_8u_C3(width, height, &dst_step);
  }

  if (d_src == nullptr || d_dst == nullptr) {
    std::cerr << "Failed to allocate NPP device memory." << std::endl;
    if (d_src) nppiFree(d_src);
    if (d_dst) nppiFree(d_dst);
    return false;
  }

  cudaEvent_t ev_start, ev_htod_done, ev_npp_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_npp_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start));
  CUDA_CHECK(cudaMemcpy2D(d_src, src_step, input.data(), width * channels,
                          width * channels, height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(ev_htod_done));

  NppiSize roi_size = {width, height};
  NppStatus status = NPP_SUCCESS;

  if (channels == 1) {
    status = nppiFilterGauss_8u_C1R(d_src, src_step, d_dst, dst_step, roi_size, NPP_MASK_SIZE_5_X_5);
  } else {
    status = nppiFilterGauss_8u_C3R(d_src, src_step, d_dst, dst_step, roi_size, NPP_MASK_SIZE_5_X_5);
  }

  if (status != NPP_SUCCESS) {
    std::cerr << "NPP Gaussian filter failed with code: " << status << std::endl;
  }
  CUDA_CHECK(cudaEventRecord(ev_npp_done));

  CUDA_CHECK(cudaMemcpy2D(output->data(), width * channels, d_dst, dst_step,
                          width * channels, height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, npp_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&npp_ms, ev_htod_done, ev_npp_done);
    cudaEventElapsedTime(&dtoh_ms, ev_npp_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = npp_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_npp_done);
  cudaEventDestroy(ev_dtoh_done);
  nppiFree(d_src);
  nppiFree(d_dst);

  return (status == NPP_SUCCESS);
}

bool NppImageFilter::ApplySobelFilter(const ImageBuffer& input, ImageBuffer* output,
                                      PerformanceMetrics* metrics) {
  ImageBuffer gray_input = (input.channels() == 3) ? input.ToGrayscale() : input;
  int width = gray_input.width();
  int height = gray_input.height();

  *output = ImageBuffer(width, height, 1);

  int src_step = 0;
  int dst_step = 0;
  Npp8u* d_src = nppiMalloc_8u_C1(width, height, &src_step);
  Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dst_step);

  if (d_src == nullptr || d_dst == nullptr) {
    if (d_src) nppiFree(d_src);
    if (d_dst) nppiFree(d_dst);
    return false;
  }

  cudaEvent_t ev_start, ev_htod_done, ev_npp_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_npp_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start));
  CUDA_CHECK(cudaMemcpy2D(d_src, src_step, gray_input.data(), width,
                          width, height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(ev_htod_done));

  NppiSize roi_size = {width, height};
  NppStatus status = nppiFilterSobelHoriz_8u_C1R(d_src, src_step, d_dst, dst_step, roi_size);

  CUDA_CHECK(cudaEventRecord(ev_npp_done));

  CUDA_CHECK(cudaMemcpy2D(output->data(), width, d_dst, dst_step,
                          width, height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, npp_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&npp_ms, ev_htod_done, ev_npp_done);
    cudaEventElapsedTime(&dtoh_ms, ev_npp_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = npp_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_npp_done);
  cudaEventDestroy(ev_dtoh_done);
  nppiFree(d_src);
  nppiFree(d_dst);

  return (status == NPP_SUCCESS);
}

bool NppImageFilter::ApplyRotation(const ImageBuffer& input, ImageBuffer* output,
                                   double angle_deg, PerformanceMetrics* metrics) {
  int width = input.width();
  int height = input.height();
  int channels = input.channels();

  NppiSize src_size = {width, height};
  NppiRect bounding_box;
  NPP_CHECK(nppiGetRotateBound(src_size, angle_deg, &bounding_box));

  int out_width = bounding_box.width;
  int out_height = bounding_box.height;
  *output = ImageBuffer(out_width, out_height, channels);

  int src_step = 0;
  int dst_step = 0;
  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;

  if (channels == 1) {
    d_src = nppiMalloc_8u_C1(width, height, &src_step);
    d_dst = nppiMalloc_8u_C1(out_width, out_height, &dst_step);
  } else {
    d_src = nppiMalloc_8u_C3(width, height, &src_step);
    d_dst = nppiMalloc_8u_C3(out_width, out_height, &dst_step);
  }

  if (d_src == nullptr || d_dst == nullptr) {
    if (d_src) nppiFree(d_src);
    if (d_dst) nppiFree(d_dst);
    return false;
  }

  cudaEvent_t ev_start, ev_htod_done, ev_npp_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_npp_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start));
  CUDA_CHECK(cudaMemcpy2D(d_src, src_step, input.data(), width * channels,
                          width * channels, height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(ev_htod_done));

  NppiPoint src_offset = {0, 0};
  NppiPoint rotation_center = {width / 2, height / 2};
  NppStatus status = NPP_SUCCESS;

  if (channels == 1) {
    status = nppiRotate_8u_C1R(d_src, src_size, src_step, src_offset,
                              d_dst, dst_step, bounding_box, angle_deg,
                              rotation_center, NPPI_INTER_LINEAR);
  } else {
    status = nppiRotate_8u_C3R(d_src, src_size, src_step, src_offset,
                              d_dst, dst_step, bounding_box, angle_deg,
                              rotation_center, NPPI_INTER_LINEAR);
  }

  CUDA_CHECK(cudaEventRecord(ev_npp_done));

  CUDA_CHECK(cudaMemcpy2D(output->data(), out_width * channels, d_dst, dst_step,
                          out_width * channels, out_height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, npp_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&npp_ms, ev_htod_done, ev_npp_done);
    cudaEventElapsedTime(&dtoh_ms, ev_npp_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = npp_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_npp_done);
  cudaEventDestroy(ev_dtoh_done);
  nppiFree(d_src);
  nppiFree(d_dst);

  return (status == NPP_SUCCESS);
}

bool NppImageFilter::ApplyThreshold(const ImageBuffer& input, ImageBuffer* output,
                                    uint8_t threshold_val, PerformanceMetrics* metrics) {
  ImageBuffer gray_input = (input.channels() == 3) ? input.ToGrayscale() : input;
  int width = gray_input.width();
  int height = gray_input.height();

  *output = ImageBuffer(width, height, 1);

  int src_step = 0;
  int dst_step = 0;
  Npp8u* d_src = nppiMalloc_8u_C1(width, height, &src_step);
  Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dst_step);

  if (d_src == nullptr || d_dst == nullptr) {
    if (d_src) nppiFree(d_src);
    if (d_dst) nppiFree(d_dst);
    return false;
  }

  cudaEvent_t ev_start, ev_htod_done, ev_npp_done, ev_dtoh_done;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_htod_done));
  CUDA_CHECK(cudaEventCreate(&ev_npp_done));
  CUDA_CHECK(cudaEventCreate(&ev_dtoh_done));

  CUDA_CHECK(cudaEventRecord(ev_start));
  CUDA_CHECK(cudaMemcpy2D(d_src, src_step, gray_input.data(), width,
                          width, height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(ev_htod_done));

  NppiSize roi_size = {width, height};
  NppStatus status = nppiThreshold_LTValGTVal_8u_C1R(d_src, src_step, d_dst, dst_step,
                                                     roi_size, threshold_val, 0, 255);

  CUDA_CHECK(cudaEventRecord(ev_npp_done));

  CUDA_CHECK(cudaMemcpy2D(output->data(), width, d_dst, dst_step,
                          width, height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev_dtoh_done));
  CUDA_CHECK(cudaEventSynchronize(ev_dtoh_done));

  if (metrics != nullptr) {
    float htod_ms = 0.0f, npp_ms = 0.0f, dtoh_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&htod_ms, ev_start, ev_htod_done);
    cudaEventElapsedTime(&npp_ms, ev_htod_done, ev_npp_done);
    cudaEventElapsedTime(&dtoh_ms, ev_npp_done, ev_dtoh_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_dtoh_done);

    metrics->host_to_device_ms = htod_ms;
    metrics->kernel_execution_ms = npp_ms;
    metrics->device_to_host_ms = dtoh_ms;
    metrics->total_gpu_ms = total_ms;
    metrics->throughput_megapixels_per_sec = (width * height / 1e6) / (total_ms / 1000.0);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_htod_done);
  cudaEventDestroy(ev_npp_done);
  cudaEventDestroy(ev_dtoh_done);
  nppiFree(d_src);
  nppiFree(d_dst);

  return (status == NPP_SUCCESS);
}

}  // namespace cuda_imgproc
