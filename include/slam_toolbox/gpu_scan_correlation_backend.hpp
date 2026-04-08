/*
 * gpu_scan_correlation_backend
 * Copyright (c) 2026, slam_toolbox contributors
 *
 * PIMPL class implementing karto::GpuCorrelationHandle.
 * Compiled only when SLAM_TOOLBOX_CUDA_AVAILABLE is defined.
 * Mapper.cpp and all non-CUDA TUs see only the pure-C++ interface defined
 * in karto_sdk/GpuCorrelation.h — no CUDA types ever leak into host code.
 */

#ifndef SLAM_TOOLBOX__GPU_SCAN_CORRELATION_BACKEND_HPP_
#define SLAM_TOOLBOX__GPU_SCAN_CORRELATION_BACKEND_HPP_

#include "slam_toolbox/gpu_config.h"

#ifdef SLAM_TOOLBOX_CUDA_AVAILABLE

#include "karto_sdk/GpuCorrelation.h"   // karto::GpuCorrelationHandle + Params

namespace slam_toolbox
{

// Forward declaration at namespace scope so that std::unique_ptr<> can see
// the type without a nested-class forward declaration inside the class body.
// The full definition lives in gpu_scan_correlation_backend.cu.
// A nested forward declaration ("struct Impl;") inside the class confuses
// nvcc 11.x which cannot resolve the outer class at the point where it
// later tries to define GpuScanCorrelationBackend::Impl.
struct GpuScanCorrelationBackendImpl;

/**
 * @brief CUDA implementation of karto::GpuCorrelationHandle.
 *
 * Lifecycle:
 *   1. Constructor picks the GPU with the highest compute capability,
 *      creates a CUDA stream, and pre-allocates the fixed-size device
 *      accumulation buffers (sumX, sumY, …).  All variable-size buffers
 *      (grid, offsets, responses) grow on demand inside correlateScan().
 *   2. isAvailable() returns false if cudaGetDeviceCount() found nothing,
 *      or if any subsequent CUDA call throws; the flag is sticky — once
 *      false it stays false so CorrelateScan always falls through to CPU.
 *   3. correlateScan() runs the full GPU pipeline:
 *        H→D memcpy  — upload pre-smeared grid (from CPU AddScans) + local pts
 *        Kernel A    — ComputeOffsets: rotate pts × angles, build index table
 *        Texture     — rebind d_corrGrid read-only for Kernel B
 *        Kernel B    — 3-D correlate: one thread per (xi,yi,ai) pose candidate
 *        Thrust      — max_element for bestResponse
 *        Kernel C2   — atomic-double accumulation of best-pose mean
 *        Host stats  — download offset table + SSP grid; compute covariance
 *                      in double precision (matches CPU output <= 1e-6)
 *      On any CUDA failure: logs to stderr once, sets available_=false,
 *      returns -1.0 so CorrelateScan falls through to the CPU path.
 */
class GpuScanCorrelationBackend final : public karto::GpuCorrelationHandle
{
public:
  GpuScanCorrelationBackend();
  ~GpuScanCorrelationBackend() override;

  GpuScanCorrelationBackend(const GpuScanCorrelationBackend &) = delete;
  GpuScanCorrelationBackend & operator=(const GpuScanCorrelationBackend &) = delete;
  GpuScanCorrelationBackend(GpuScanCorrelationBackend &&) noexcept;
  GpuScanCorrelationBackend & operator=(GpuScanCorrelationBackend &&) noexcept;

  // ---- karto::GpuCorrelationHandle interface ------------------------------

  bool isAvailable() const override;

  double correlateScan(
    const karto::GpuCorrelationCallParams & p,
    double * outMeanX,
    double * outMeanY,
    double * outMeanTheta,
    double   outCovData[9]) override;

  // ---- Extra query --------------------------------------------------------
  const char * name() const;

private:
  // PIMPL: all CUDA types (device pointers, stream, texture) live in the
  // namespace-scope struct GpuScanCorrelationBackendImpl (defined in the .cu).
  // cuda_runtime.h is never included by this header.
  GpuScanCorrelationBackendImpl * impl_;
  bool available_;
};

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX_CUDA_AVAILABLE
#endif  // SLAM_TOOLBOX__GPU_SCAN_CORRELATION_BACKEND_HPP_
