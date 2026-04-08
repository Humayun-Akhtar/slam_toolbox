/*
 * cpu_scan_correlation_backend
 * Copyright (c) 2026, slam_toolbox contributors
 *
 * CPU backend: zero-overhead wrapper around the existing karto_sdk
 * ScanMatcher.  Every call delegates directly to
 * ScanMatcher::MatchScan<T>(), preserving the original TBB-parallelized
 * code path without any modification, move, or refactor.
 */

#ifndef SLAM_TOOLBOX__CPU_SCAN_CORRELATION_BACKEND_HPP_
#define SLAM_TOOLBOX__CPU_SCAN_CORRELATION_BACKEND_HPP_

#include "slam_toolbox/scan_correlation_backend.hpp"

namespace slam_toolbox
{

/**
 * @brief CPU backend — pure pass-through to karto_sdk.
 *
 * How this wraps without modifying karto_sdk:
 *
 *   ScanMatcher::MatchScan<T>() is a public template that orchestrates the
 *   entire correlative scan-matching pipeline:
 *
 *     1. Grid setup & coordinate transforms
 *     2. AddScans()          — populate correlation grid + Gaussian smear
 *     3. CorrelateScan()     — ComputeOffsets, TBB parallel pose search,
 *                              GetResponse, best-pose extraction, covariance
 *     4. Optional fine match — second CorrelateScan at higher resolution
 *
 *   All of these remain in their original files (Mapper.cpp, Karto.h,
 *   Mapper.h) compiled into the kartoSlamToolbox shared library.  This
 *   class simply calls the public MatchScan entry point.
 */
class CpuScanCorrelationBackend final : public ScanCorrelationBackend
{
public:
  // -- LocalizedRangeScanVector overload -----------------------------------

  kt_double matchScan(
    karto::ScanMatcher * pScanMatcher,
    karto::LocalizedRangeScan * pScan,
    const karto::LocalizedRangeScanVector & rBaseScans,
    karto::Pose2 & rMean,
    karto::Matrix3 & rCovariance,
    kt_bool doPenalize,
    kt_bool doRefineMatch) override
  {
    //  Direct delegation.
    //  ScanMatcher::MatchScan<LocalizedRangeScanVector> runs the full
    //  original pipeline: AddScans → CorrelateScan (coarse) →
    //  CorrelateScan (fine).  Nothing is changed.
    return pScanMatcher->MatchScan(
      pScan, rBaseScans, rMean, rCovariance, doPenalize, doRefineMatch);
  }

  // -- LocalizedRangeScanMap overload --------------------------------------

  kt_double matchScan(
    karto::ScanMatcher * pScanMatcher,
    karto::LocalizedRangeScan * pScan,
    const karto::LocalizedRangeScanMap & rBaseScans,
    karto::Pose2 & rMean,
    karto::Matrix3 & rCovariance,
    kt_bool doPenalize,
    kt_bool doRefineMatch) override
  {
    //  Direct delegation.
    //  ScanMatcher::MatchScan<LocalizedRangeScanMap> runs the same
    //  pipeline, only differing in which AddScans overload populates
    //  the correlation grid.
    return pScanMatcher->MatchScan(
      pScan, rBaseScans, rMean, rCovariance, doPenalize, doRefineMatch);
  }

  // -- Metadata ------------------------------------------------------------

  bool isAvailable() const override
  {
    return true;  // CPU is always available
  }

  std::string name() const override
  {
    return "CPU (karto_sdk TBB)";
  }
};

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX__CPU_SCAN_CORRELATION_BACKEND_HPP_
