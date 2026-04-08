/*
 * scan_correlation_backend_factory
 * Copyright (c) 2026, slam_toolbox contributors
 *
 * Factory function for creating the appropriate ScanCorrelationBackend.
 * Two levels of fallback are enforced:
 *
 *   Level 1 — Compile-time:
 *     If SLAM_TOOLBOX_CUDA_AVAILABLE is not defined (CUDA Toolkit absent),
 *     GpuScanCorrelationBackend does not exist.  A request for the GPU
 *     backend immediately returns the CPU backend; no CUDA headers are
 *     ever included.
 *
 *   Level 2 — Runtime:
 *     If CUDA was compiled in but no compatible device is found at startup,
 *     GpuScanCorrelationBackend::isAvailable() returns false.  The factory
 *     downgrades to CPU and emits a single RCLCPP_WARN through the supplied
 *     logger; subsequent calls are silent.
 *
 * Usage:
 *
 *   // Declare in your node class:
 *   std::unique_ptr<slam_toolbox::ScanCorrelationBackend> scan_backend_;
 *
 *   // In on_configure():
 *   bool use_gpu = false;
 *   if (!this->has_parameter("use_gpu_scan_matching")) {
 *     this->declare_parameter("use_gpu_scan_matching", use_gpu);
 *   }
 *   use_gpu = this->get_parameter("use_gpu_scan_matching").as_bool();
 *
 *   scan_backend_ = slam_toolbox::createScanCorrelationBackend(
 *       use_gpu, get_logger());
 *
 *   RCLCPP_INFO(get_logger(), "Scan matching backend: %s",
 *       scan_backend_->name().c_str());
 *
 *   // Where ScanMatcher::MatchScan is currently called, e.g. Mapper.cpp or
 *   // a SlamToolbox overriding process():
 *   response = scan_backend_->matchScan(
 *       pSequentialScanMatcher, pScan, runningScans,
 *       bestPose, covariance);
 */

#ifndef SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_FACTORY_HPP_
#define SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_FACTORY_HPP_

#include <memory>

#include "rclcpp/logger.hpp"
#include "rclcpp/logging.hpp"

#include "slam_toolbox/gpu_config.h"
#include "slam_toolbox/scan_correlation_backend.hpp"
#include "slam_toolbox/cpu_scan_correlation_backend.hpp"

#ifdef SLAM_TOOLBOX_CUDA_AVAILABLE
#include "slam_toolbox/gpu_scan_correlation_backend.hpp"
#endif

namespace slam_toolbox
{

/**
 * @brief Create a ScanCorrelationBackend honouring the two-level fallback policy.
 *
 * @param prefer_gpu  If true, attempt to create the GPU backend first.
 *                    Ignored (treated as false) when CUDA is not compiled in.
 * @param logger      ROS logger used to emit a one-time warning on fallback.
 * @return            A heap-allocated backend; never nullptr.
 */
inline std::unique_ptr<ScanCorrelationBackend> createScanCorrelationBackend(
  bool prefer_gpu,
  const rclcpp::Logger & logger)
{
#ifdef SLAM_TOOLBOX_CUDA_AVAILABLE
  if (prefer_gpu) {
    auto gpu = std::make_unique<GpuScanCorrelationBackend>();

    if (gpu->isAvailable()) {
      // Happy path: CUDA compiled in, device found.
      return gpu;
    }

    // Level 2 fallback: CUDA compiled in but no device at runtime.
    RCLCPP_WARN(
      logger,
      "ScanCorrelationBackend: use_gpu_scan_matching=true but no "
      "CUDA-capable GPU was found at runtime. "
      "Falling back to CPU backend (karto_sdk TBB). "
      "This message will not repeat.");
    // Fall through to CPU construction below.
  }
  // prefer_gpu == false, or runtime GPU probe failed → CPU.
  return std::make_unique<CpuScanCorrelationBackend>();

#else
  // Level 1 fallback: CUDA Toolkit was not found at cmake time.
  // The GPU backend class does not exist; silently use CPU.
  if (prefer_gpu) {
    RCLCPP_WARN(
      logger,
      "ScanCorrelationBackend: use_gpu_scan_matching=true but this build "
      "was compiled WITHOUT CUDA support (SLAM_TOOLBOX_CUDA_AVAILABLE not "
      "defined). Falling back to CPU backend (karto_sdk TBB). "
      "Re-build with a CUDA Toolkit installation to enable GPU matching. "
      "This message will not repeat.");
  }
  (void)logger;  // suppress unused-variable warning in non-CUDA builds
                 // when prefer_gpu == false.
  return std::make_unique<CpuScanCorrelationBackend>();
#endif
}

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_FACTORY_HPP_
