/*
 * GpuCorrelation.h
 * Abstract interface for GPU-accelerated correlative scan matching.
 *
 * This header is intentionally free of CUDA types, karto types, and
 * slam_toolbox types so that it can be safely included from Mapper.h
 * without creating circular dependencies.  All parameters use C++
 * standard-library types only.
 *
 * The concrete implementation (GpuScanCorrelationBackend) lives in
 * slam_toolbox/include/slam_toolbox/gpu_scan_correlation_backend.hpp
 * and slam_toolbox/src/gpu_scan_correlation_backend.cu.
 */

#ifndef KARTO_SDK__GPU_CORRELATION_H_
#define KARTO_SDK__GPU_CORRELATION_H_

#include <cstdint>   // uint8_t
#include <cstddef>   // size_t

namespace karto
{

/**
 * @brief Parameters for a single GpuCorrelationHandle::correlateScan() call.
 *
 * All geometric parameters are in float to match the precision used inside
 * the CUDA kernels.  The output covariance is accumulated in double (the
 * same precision as the CPU ComputePositionalCovariance /
 * ComputeAngularCovariance functions) so loop-closure thresholds are
 * unaffected.
 */
struct GpuCorrelationCallParams
{
  // ---- Scan points (pScan, local frame after InverseTransformPose) --------
  // Length: nPoints * 2, interleaved [x0,y0, x1,y1, ...].
  // NaN pair marks an invalid scan reading (mirrors INVALID_SCAN sentinel).
  const float * localPointsXY = nullptr;
  int           nPoints       = 0;

  // ---- Correlation grid (already populated + smeared by CPU AddScans) -----
  const uint8_t * corrGridData  = nullptr;  // host pointer, const read-only
  int             gridWidth     = 0;        // Grid::GetWidth()
  int             gridHeight    = 0;        // Grid::GetHeight()
  int             gridWidthStep = 0;        // Grid::GetWidthStep()  (8-aligned)
  int             gridDataSize  = 0;        // Grid::GetDataSize()   (widthStep*height)

  // CoordinateConverter parameters of the CorrelationGrid:
  float gridScale   = 0.0f;   // GetCoordinateConverter()->GetScale()  (= 1/resolution)
  float gridOffsetX = 0.0f;   // GetCoordinateConverter()->GetOffset().GetX()
  float gridOffsetY = 0.0f;   // GetCoordinateConverter()->GetOffset().GetY()

  // CorrelationGrid ROI origin (CorrelationGrid::GetROI().GetX()/GetY()):
  int roiOffsetX = 0;
  int roiOffsetY = 0;

  // ---- Search space geometry ----------------------------------------------
  float searchCenterX     = 0.0f;
  float searchCenterY     = 0.0f;
  float searchCenterTheta = 0.0f;

  float searchAngleOffset     = 0.0f;
  float searchAngleResolution = 0.0f;

  float searchSpaceOffsetX      = 0.0f;   // rSearchSpaceOffset.GetX()
  float searchSpaceOffsetY      = 0.0f;
  float searchSpaceResolutionX  = 0.0f;   // rSearchSpaceResolution.GetX()
  float searchSpaceResolutionY  = 0.0f;

  // ---- Odometry penalty ---------------------------------------------------
  bool  doPenalize             = false;
  float distVariancePenalty    = 1.0f;   // m_pMapper->m_pDistanceVariancePenalty
  float angleVariancePenalty   = 1.0f;   // m_pMapper->m_pAngleVariancePenalty
  float minDistPenalty         = 0.5f;   // m_pMapper->m_pMinimumDistancePenalty
  float minAnglePenalty        = 0.9f;   // m_pMapper->m_pMinimumAnglePenalty

  // ---- Which covariance to compute (mirrors doingFineMatch flag) ----------
  bool doingFineMatch = false;
};

/**
 * @brief Pure-C++ abstract interface for the GPU correlation pipeline.
 *
 * ScanMatcher holds a nullable pointer of this type.  When the pointer is
 * non-null and isAvailable() returns true, CorrelateScan forwards to
 * correlateScan() instead of executing the TBB-parallel CPU path.
 *
 * Contract:
 *   - correlateScan() must return a value in [0.0, 1.0] on success.
 *   - correlateScan() must return -1.0 on any GPU failure; the caller
 *     (ScanMatcher::CorrelateScan) will then fall through to the CPU path
 *     transparently.
 *   - After a failure that sets isAvailable()=false, every subsequent call
 *     to correlateScan() should immediately return -1.0 so the CPU path
 *     takes over permanently without logging again.
 */
class GpuCorrelationHandle
{
public:
  virtual ~GpuCorrelationHandle() = default;

  /** @return true when a CUDA device was found and no fatal error has occurred. */
  virtual bool isAvailable() const = 0;

  /**
   * @brief Run the GPU correlation pipeline for one CorrelateScan call.
   *
   * @param p       Packed call parameters (see GpuCorrelationCallParams).
   * @param outMeanX      Best-match pose X (metres).
   * @param outMeanY      Best-match pose Y (metres).
   * @param outMeanTheta  Best-match heading (radians, in [-π, π]).
   * @param outCovData    9-element row-major 3×3 covariance matrix,
   *                      laid out as [c00,c01,c02, c10,c11,c12, c20,c21,c22].
   * @return Response strength in [0.0, 1.0], or -1.0 on failure.
   */
  virtual double correlateScan(
    const GpuCorrelationCallParams & p,
    double * outMeanX,
    double * outMeanY,
    double * outMeanTheta,
    double   outCovData[9]) = 0;
};

}  // namespace karto

#endif  // KARTO_SDK__GPU_CORRELATION_H_
