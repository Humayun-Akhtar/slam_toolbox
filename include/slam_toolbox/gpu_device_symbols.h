#ifndef SLAM_TOOLBOX_GPU_DEVICE_SYMBOLS_H_
#define SLAM_TOOLBOX_GPU_DEVICE_SYMBOLS_H_

#ifdef SLAM_TOOLBOX_CUDA_AVAILABLE

// ============================================================================
// CorrelateParams — packed into __constant__ memory.
// Struct definition shared across all TUs.
// ============================================================================
struct CorrelateParams {
  float gridScale, gridOffsetX, gridOffsetY;
  int   gridWidthStep, gridDataSize;
  int   roiOffsetX, roiOffsetY;
  float searchCenterX, searchCenterY, searchCenterTheta;
  float startX, xRes;
  float startY, yRes;
  float startAngle, angleRes;
  int   nX, nY, nAngles, nPoints;
  bool  doPenalize;
  float distVariancePenalty, angleVariancePenalty;
  float minDistPenalty, minAnglePenalty;
};

// Define the __constant__ symbol at header level to ensure visibility in all TUs.
// This works with unified compilation and ensures all kernels can access it.
__constant__ CorrelateParams d_params;

#endif  // SLAM_TOOLBOX_CUDA_AVAILABLE

#endif  // SLAM_TOOLBOX_GPU_DEVICE_SYMBOLS_H_
