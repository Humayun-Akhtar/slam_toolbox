/*
 * scan_correlation_backend
 * Copyright (c) 2026, slam_toolbox contributors
 *
 * Abstract interface for the scan-matching correlation pipeline.
 */

#ifndef SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_HPP_
#define SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_HPP_

#include <string>
#include "karto_sdk/Mapper.h"

namespace slam_toolbox
{

/**
 * @brief Abstract interface that decouples the scan-matching correlation
 *        algorithm from its compute backend (CPU or GPU).
 *
 * Two concrete implementations exist:
 *
 *   CpuScanCorrelationBackend
 *       Delegates every call to the existing karto::ScanMatcher public API.
 *       The original TBB-parallelized code runs unmodified.
 *
 *   GpuScanCorrelationBackend   (compiled only when SLAM_TOOLBOX_CUDA_AVAILABLE)
 *       Reimplements the full pipeline in CUDA:
 *         1. Grid population + Gaussian smear         (AddScans / SmearPoint)
 *         2. Rotated lookup-offset precomputation      (ComputeOffsets)
 *         3. Exhaustive 3-D pose-space search + reduce (operator() / GetResponse)
 *         4. Covariance extraction
 *
 * Both overloads of matchScan() mirror the two template instantiations of
 * karto::ScanMatcher::MatchScan<T> (T = LocalizedRangeScanVector or
 * LocalizedRangeScanMap) so that any call-site that currently invokes
 * ScanMatcher::MatchScan can be trivially routed through this interface.
 *
 * ---------------------------------------------------------------------------
 * Integration sketch (slam_toolbox_common.cpp or Mapper call-sites):
 *
 *   // At node startup:
 *   bool use_gpu = node->get_parameter("use_gpu_scan_matching").as_bool();
 *   backend_ = slam_toolbox::createScanCorrelationBackend(use_gpu, logger);
 *
 *   // Where scan matching is invoked:
 *   response = backend_->matchScan(
 *       pSequentialScanMatcher, pScan, runningScans,
 *       bestPose, covariance);
 * ---------------------------------------------------------------------------
 */
class ScanCorrelationBackend
{
public:
  virtual ~ScanCorrelationBackend() = default;

  /**
   * Match @p pScan against a vector of base scans.
   * Semantically identical to ScanMatcher::MatchScan<LocalizedRangeScanVector>.
   *
   * @param pScanMatcher  The ScanMatcher instance that owns the correlation
   *                      grid, search-space grids, and lookup tables.
   * @param pScan         The incoming scan to be matched.
   * @param rBaseScans    Reference scans that define the occupied cells.
   * @param rMean         [out] Best-match pose.
   * @param rCovariance   [out] Match covariance.
   * @param doPenalize    Penalise poses far from the search centre.
   * @param doRefineMatch Run a fine-grained pass after the coarse match.
   * @return Response strength in [0, 1].
   */
  virtual kt_double matchScan(
    karto::ScanMatcher * pScanMatcher,
    karto::LocalizedRangeScan * pScan,
    const karto::LocalizedRangeScanVector & rBaseScans,
    karto::Pose2 & rMean,
    karto::Matrix3 & rCovariance,
    kt_bool doPenalize = true,
    kt_bool doRefineMatch = true) = 0;

  /**
   * Match @p pScan against a map of base scans (keyed by state-id).
   * Semantically identical to ScanMatcher::MatchScan<LocalizedRangeScanMap>.
   */
  virtual kt_double matchScan(
    karto::ScanMatcher * pScanMatcher,
    karto::LocalizedRangeScan * pScan,
    const karto::LocalizedRangeScanMap & rBaseScans,
    karto::Pose2 & rMean,
    karto::Matrix3 & rCovariance,
    kt_bool doPenalize = true,
    kt_bool doRefineMatch = true) = 0;

  /** @return true when the backend's compute resources are operational. */
  virtual bool isAvailable() const = 0;

  /** @return Human-readable backend name for log messages. */
  virtual std::string name() const = 0;
};

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX__SCAN_CORRELATION_BACKEND_HPP_
