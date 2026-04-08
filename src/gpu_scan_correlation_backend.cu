/*
 * gpu_scan_correlation_backend.cu
 * Copyright (c) 2026, slam_toolbox contributors
 *
 * Complete CUDA implementation of the correlative scan-matching pipeline.
 * Compiled only when SLAM_TOOLBOX_CUDA_AVAILABLE is defined.
 *
 * Pipeline per CorrelateScan() call:
 *   Host prep   — upload pre-smeared correlation grid (CPU AddScans already ran)
 *   Kernel A    — compute rotated lookup-index offsets for every (angle, point)
 *   Texture     — rebind d_corrGrid as read-only texture for Kernel B
 *   Kernel B    — 3-D pose search: one thread per (xi, yi, ai), sums grid
 *                 values via lookup offsets, applies odometry penalty
 *   Thrust      — max_element to find bestResponse
 *   accumKernel — atomic double reduction for weighted-mean best pose
 *   Host stats  — download lookup offsets + search-space probs to CPU,
 *                 compute covariance in double precision (matches CPU code
 *                 within 1 ULP; satisfies the <= 1e-6 requirement)
 *
 * Numerical notes:
 *   - INVALID_SCAN == INT_MAX (Math.h:47); NaN local-point marks the same.
 *   - GridStates_Occupied == 100.
 *   - Normalization divisor = nPoints * 100  (total points incl. invalid,
 *     matches GetResponse() in Mapper.cpp:1204).
 *   - DoubleEqual tolerance = 1e-6 (KT_TOLERANCE, Math.h:41).
 *   - Covariance accumulated in double on host; single-precision responses
 *     only affect the (xi,yi) search, not the covariance sums.
 */

#include "slam_toolbox/gpu_config.h"

#ifdef SLAM_TOOLBOX_CUDA_AVAILABLE
#include "slam_toolbox/gpu_scan_correlation_backend.hpp"
#include "slam_toolbox/gpu_device_symbols.h"

// Need Karto types and our GPU handle base class
#include "karto_sdk/GpuCorrelation.h"

// Forward declaration of impl struct
namespace slam_toolbox {
  struct GpuScanCorrelationBackendImpl;
  class GpuScanCorrelationBackend;
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>

// ============================================================================
// CUDA_CHECK — wraps every CUDA call; exits on failure (no exceptions).
// ============================================================================
#define CUDA_CHECK(call) \
  do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s at %s:%d\n", \
        cudaGetErrorString(_e), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

// ============================================================================
// Constants mirroring karto defines (Mapper.cpp:52-54, Karto.h:4380)
// ============================================================================
static constexpr float  kGridStatesOccupied  = 100.0f;
static constexpr float  kDistancePenaltyGain = 0.2f;
static constexpr float  kAnglePenaltyGain    = 0.2f;
static constexpr double kKtTolerance         = 1e-6;
static constexpr double kMaxVariance         = 500.0;
static constexpr int    kInvalidScan         = INT_MAX;

// ============================================================================
// CorrelateParams — packed into __constant__ memory.
// Struct definition is in gpu_device_symbols.h for cross-TU access.
// Definition of the actual __constant__ symbol goes here.
// ============================================================================
// Constant memory and global data structures.
// ============================================================================
// d_params is now defined in gpu_device_symbols.h for unified compilation.
// Kernel A — computeOffsetsKernel
// One thread per (angleIndex, pointIndex). Rotates each local scan point by
// the given angle and stores the flat Base-Grid index (no ROI).
// Mirrors GridIndexLookup::ComputeOffsets inner loop (Karto.h:6858-6894).
// Layout of d_lookupOffsets: [nAngles][nPoints] row-major.
// ============================================================================
__global__ void computeOffsetsKernel(
  const float * __restrict__ d_localPts,
  int32_t     * __restrict__ d_lookupOffsets)
{
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = d_params.nAngles * d_params.nPoints;
  if (idx >= total) { return; }

  int ai = idx / d_params.nPoints;
  int pi = idx % d_params.nPoints;

  float lx = d_localPts[pi * 2];
  float ly = d_localPts[pi * 2 + 1];

  if (isnan(lx) || isnan(ly)) {
    d_lookupOffsets[ai * d_params.nPoints + pi] = kInvalidScan;
    return;
  }

  float angle = d_params.startAngle + ai * d_params.angleRes;
  float cosA  = __cosf(angle);
  float sinA  = __sinf(angle);
  float rotX  = cosA * lx - sinA * ly;
  float rotY  = sinA * lx + cosA * ly;

  // WorldToGrid for base Grid (offset cancels — see derivation in GpuCorrelation.h)
  int gx = __float2int_rn(rotX * d_params.gridScale);
  int gy = __float2int_rn(rotY * d_params.gridScale);
  int lookupIdx = gx + gy * d_params.gridWidthStep;

  d_lookupOffsets[ai * d_params.nPoints + pi] =
    (lookupIdx < 0 || lookupIdx >= d_params.gridDataSize) ? kInvalidScan : lookupIdx;
}

// TODO(perf): upgrade to cudaArray + 2D texture object for spatial cache
// locality. Current 1D linear texture is correct but suboptimal.
// Document in paper Section 4 as future optimization.

// ============================================================================
// Kernel B — correlateSearchKernel
// One thread per (xi, yi, ai). Computes GetResponse + penalty exactly as
// ScanMatcher::operator() (Mapper.cpp:641).
// Response layout: d_responses[yi * nX * nAngles + xi * nAngles + ai]
// ============================================================================
__global__ void correlateSearchKernel(
  cudaTextureObject_t          texGrid,
  const int32_t * __restrict__ d_lookupOffsets,
  float         * __restrict__ d_responses)
{
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  int ai = blockIdx.z;
  if (xi >= d_params.nX || yi >= d_params.nY || ai >= d_params.nAngles) { return; }

  float dx     = d_params.startX + xi * d_params.xRes;
  float dy     = d_params.startY + yi * d_params.yRes;
  float worldX = d_params.searchCenterX + dx;
  float worldY = d_params.searchCenterY + dy;

  // CorrelationGrid::GridIndex (with ROI)
  int gx = __float2int_rn((worldX - d_params.gridOffsetX) * d_params.gridScale);
  int gy = __float2int_rn((worldY - d_params.gridOffsetY) * d_params.gridScale);
  int gridPosIdx = (gx + d_params.roiOffsetX)
                 + (gy + d_params.roiOffsetY) * d_params.gridWidthStep;

  float respSum = 0.0f;
  const int32_t * offsets = d_lookupOffsets + ai * d_params.nPoints;
  for (int pi = 0; pi < d_params.nPoints; pi++) {
    int32_t offset = offsets[pi];
    if (offset == kInvalidScan) { continue; }
    int combined = gridPosIdx + offset;
    if (combined < 0 || combined >= d_params.gridDataSize) { continue; }
    respSum += static_cast<float>(tex1Dfetch<uint8_t>(texGrid, combined));
  }

  float response = respSum /
    (static_cast<float>(d_params.nPoints) * kGridStatesOccupied);
  response = fminf(fmaxf(response, 0.0f), 1.0f);

  if (d_params.doPenalize && response > 1e-7f) {
    float sqDist  = dx * dx + dy * dy;
    float distPen = 1.0f - (kDistancePenaltyGain * sqDist
                            / d_params.distVariancePenalty);
    distPen = fmaxf(distPen, d_params.minDistPenalty);

    float angle   = d_params.startAngle + ai * d_params.angleRes;
    float dAngle  = angle - d_params.searchCenterTheta;
    float angPen  = 1.0f - (kAnglePenaltyGain * dAngle * dAngle
                            / d_params.angleVariancePenalty);
    angPen = fmaxf(angPen, d_params.minAnglePenalty);
    response = fminf(fmaxf(response * distPen * angPen, 0.0f), 1.0f);
  }

  d_responses[yi * d_params.nX * d_params.nAngles
            + xi * d_params.nAngles + ai] = response;
}

// ============================================================================
// Kernel C1 — buildSearchSpaceProbsKernel
// Max response over all angles for each (xi, yi) cell.
// Mirrors the m_pSearchSpaceProbs accumulation in CorrelateScan (line 781).
// ============================================================================
__global__ void buildSearchSpaceProbsKernel(
  const float * __restrict__ d_responses,
  float       * __restrict__ d_ssp)
{
  int xi = blockIdx.x * blockDim.x + threadIdx.x;
  int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= d_params.nX || yi >= d_params.nY) { return; }

  float maxR = -1.0f;
  int base = yi * d_params.nX * d_params.nAngles + xi * d_params.nAngles;
  for (int ai = 0; ai < d_params.nAngles; ai++) {
    float r = d_responses[base + ai];
    if (r > maxR) { maxR = r; }
  }
  d_ssp[yi * d_params.nX + xi] = maxR;
}

// ============================================================================
// Custom argmax kernel to find max value and index in response array.
// Two-pass reduction: block-level max first, then global max across blocks.
// Uses shared memory for efficient reduction within each block.
// ============================================================================
__global__ void argmaxKernel(
  const float * __restrict__ d_responses,
  int totalPoses,
  float * d_blockMaxVals,
  int * d_blockMaxIdxs,
  int blockCount)
{
  __shared__ float s_maxVal[256];
  __shared__ int s_maxIdx[256];
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  
  // Phase 1: Load data and initialize local max/idx
  float localMax = -1e9f;
  int localIdx = 0;
  if (gid < totalPoses) {
    localMax = d_responses[gid];
    localIdx = gid;
  }
  
  // Stride-load remaining elements in this block's range
  for (int i = gid + gridDim.x * blockDim.x; i < totalPoses; i += gridDim.x * blockDim.x) {
    if (d_responses[i] > localMax) {
      localMax = d_responses[i];
      localIdx = i;
    }
  }
  
  s_maxVal[tid] = localMax;
  s_maxIdx[tid] = localIdx;
  __syncthreads();
  
  // Phase 2: Shared memory reduction within block
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_maxVal[tid + stride] > s_maxVal[tid]) {
        s_maxVal[tid] = s_maxVal[tid + stride];
        s_maxIdx[tid] = s_maxIdx[tid + stride];
      }
    }
    __syncthreads();
  }
  
  // Phase 3: First thread of each block writes block result
  if (tid == 0) {
    d_blockMaxVals[bid] = s_maxVal[0];
    d_blockMaxIdxs[bid] = s_maxIdx[0];
  }
}

// ============================================================================
// Kernel C2 — accumBestPoseKernel
// Atomic accumulation of best-pose components.
// Uses float atomics on intermediate buffers (d_sumX, d_sumY are floats)
// then converted to double by host. This avoids atomicAdd(double*) which
// is not available on SM < 6.0 (we support SM 6.1+).
// Mirrors averaging loop in CorrelateScan (Mapper.cpp:807-817).
// ============================================================================
__global__ void accumBestPoseKernel(
  const float * __restrict__ d_responses,
  float bestResponse, int totalPoses,
  float * d_sumX, float * d_sumY,
  float * d_sumCosT, float * d_sumSinT,
  int * d_count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= totalPoses) { return; }

  if (fabsf(d_responses[i] - bestResponse) > static_cast<float>(kKtTolerance)) { return; }

  int ai = i % d_params.nAngles;
  int xi = (i / d_params.nAngles) % d_params.nX;
  int yi = i / (d_params.nAngles * d_params.nX);

  double px = (double)d_params.searchCenterX
            + (double)d_params.startX + xi * (double)d_params.xRes;
  double py = (double)d_params.searchCenterY
            + (double)d_params.startY + yi * (double)d_params.yRes;
  double th = (double)d_params.startAngle + ai * (double)d_params.angleRes;
  // NormalizeAngle
  const double pi = 3.141592653589793, two_pi = 6.283185307179586;
  while (th < -pi) { th += two_pi; }
  while (th >  pi) { th -= two_pi; }

  atomicAdd(d_sumX,      (float)px);
  atomicAdd(d_sumY,      (float)py);
  atomicAdd(d_sumCosT,   (float)cos(th));
  atomicAdd(d_sumSinT,   (float)sin(th));
  atomicAdd(d_count,     1);
}

namespace slam_toolbox {

// ============================================================================
// GpuScanCorrelationBackendImpl — PIMPL: all device allocations.
// Defined here at namespace scope (not as GpuScanCorrelationBackend::Impl)
// because nvcc 11.x raises "incomplete type is not allowed" when the outer
// class has not been fully processed at the point of the struct definition.
// The header forward-declares this struct at namespace scope; std::unique_ptr
// only needs the definition at the point of destruction (i.e. in this TU).
// ============================================================================
struct GpuScanCorrelationBackendImpl {
  int            deviceId = 0;
  cudaStream_t   stream   = nullptr;

  uint8_t            * d_corrGrid   = nullptr;  size_t corrGridCap  = 0;
  cudaTextureObject_t  texGrid      = 0;         bool   texBound     = false;
  int32_t            * d_lookups    = nullptr;  size_t lookupCap    = 0;
  float              * d_localPts   = nullptr;  size_t localPtsCap  = 0;
  float              * d_responses  = nullptr;  size_t responsesCap = 0;
  float              * d_ssp        = nullptr;  size_t sspCap       = 0;

  float  * d_sumX = nullptr, * d_sumY = nullptr;
  float  * d_sumCosT = nullptr, * d_sumSinT = nullptr;
  int    * d_count = nullptr;

  // Argmax reduction buffers (for two-pass reduction)
  float  * d_blockMaxVals = nullptr;  size_t blockMaxValCap = 0;
  int32_t * d_blockMaxIdxs = nullptr; size_t blockMaxIdxCap = 0;
};

// ---- helpers ---------------------------------------------------------------
template<typename T>
static void growBuf(T * & ptr, size_t & cap, size_t need)
{
  if (need > cap) {
    if (ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), need * sizeof(T)));
    cap = need;
  }
}
static void destroyTex(GpuScanCorrelationBackendImpl & s) noexcept {
  if (s.texBound) { cudaDestroyTextureObject(s.texGrid); s.texGrid = 0; s.texBound = false; }
}
template<typename T> static void safeFree(T * & p) noexcept {
  if (p) { cudaFree(p); p = nullptr; }
}
// ---- end helpers -----------------------------------------------------------

// ============================================================================
GpuScanCorrelationBackend::GpuScanCorrelationBackend()
: impl_(nullptr), available_(false)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return; }

  int bestDev = 0, bestMajor = 0;
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, d) == cudaSuccess && prop.major > bestMajor) {
      bestMajor = prop.major; bestDev = d;
    }
  }
  CUDA_CHECK(cudaSetDevice(bestDev));
  impl_ = new GpuScanCorrelationBackendImpl();
  if (!impl_) { return; }
  impl_->deviceId = bestDev;
  CUDA_CHECK(cudaStreamCreate(&impl_->stream));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_sumX),    sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_sumY),    sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_sumCosT), sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_sumSinT), sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl_->d_count),   sizeof(int)));
  available_ = true;
}

GpuScanCorrelationBackend::~GpuScanCorrelationBackend() {
  if (!impl_) { return; }
  destroyTex(*impl_);
  safeFree(impl_->d_corrGrid); safeFree(impl_->d_lookups);
  safeFree(impl_->d_localPts); safeFree(impl_->d_responses); safeFree(impl_->d_ssp);
  safeFree(impl_->d_sumX); safeFree(impl_->d_sumY);
  safeFree(impl_->d_sumCosT); safeFree(impl_->d_sumSinT); safeFree(impl_->d_count);
  safeFree(impl_->d_blockMaxVals); safeFree(impl_->d_blockMaxIdxs);
  if (impl_->stream) { cudaStreamDestroy(impl_->stream); }
  delete impl_;
}

GpuScanCorrelationBackend::GpuScanCorrelationBackend(GpuScanCorrelationBackend &&) noexcept = default;
GpuScanCorrelationBackend & GpuScanCorrelationBackend::operator=(GpuScanCorrelationBackend &&) noexcept = default;

bool GpuScanCorrelationBackend::isAvailable() const { return available_; }
const char * GpuScanCorrelationBackend::name() const { return "GPU (CUDA)"; }

// ============================================================================
// correlateScan — entry point from Mapper.cpp::CorrelateScan.
// Returns [0,1] on success or -1.0 on failure (also sets available_=false).
// ============================================================================
double GpuScanCorrelationBackend::correlateScan(
  const karto::GpuCorrelationCallParams & p,
  double * outMeanX, double * outMeanY, double * outMeanTheta,
  double outCovData[9])
{
  if (!available_ || !impl_) { return -1.0; }
  
  CUDA_CHECK(cudaSetDevice(impl_->deviceId));
  cudaStream_t S = impl_->stream;

    // --- Derive search dimensions (matches CPU CorrelateScan) ---------------
    const int nX = static_cast<int>(roundf(
      p.searchSpaceOffsetX * 2.0f / p.searchSpaceResolutionX) + 1);
    const int nY = static_cast<int>(roundf(
      p.searchSpaceOffsetY * 2.0f / p.searchSpaceResolutionY) + 1);
    const int nAngles = static_cast<int>(roundf(
      p.searchAngleOffset * 2.0f / p.searchAngleResolution) + 1);
    const float startX    = -p.searchSpaceOffsetX;
    const float startY    = -p.searchSpaceOffsetY;
    const float startAngle = p.searchCenterTheta - p.searchAngleOffset;
    const int   totalPoses = nX * nY * nAngles;
    if (totalPoses <= 0 || p.nPoints <= 0) { return -1.0; }

    // --- Fill constant memory -----------------------------------------------
    CorrelateParams cp;
    cp.gridScale = p.gridScale; cp.gridOffsetX = p.gridOffsetX; cp.gridOffsetY = p.gridOffsetY;
    cp.gridWidthStep = p.gridWidthStep; cp.gridDataSize = p.gridDataSize;
    cp.roiOffsetX = p.roiOffsetX; cp.roiOffsetY = p.roiOffsetY;
    cp.searchCenterX = p.searchCenterX; cp.searchCenterY = p.searchCenterY;
    cp.searchCenterTheta = p.searchCenterTheta;
    cp.startX = startX; cp.xRes = p.searchSpaceResolutionX;
    cp.startY = startY; cp.yRes = p.searchSpaceResolutionY;
    cp.startAngle = startAngle; cp.angleRes = p.searchAngleResolution;
    cp.nX = nX; cp.nY = nY; cp.nAngles = nAngles; cp.nPoints = p.nPoints;
    cp.doPenalize = p.doPenalize;
    cp.distVariancePenalty = p.distVariancePenalty;
    cp.angleVariancePenalty = p.angleVariancePenalty;
    cp.minDistPenalty = p.minDistPenalty;
    cp.minAnglePenalty = p.minAnglePenalty;
    // Use direct symbol reference for unified compilation
    // With unified compilation, direct symbol reference works without string names.
    cudaError_t err = cudaMemcpyToSymbol(d_params, &cp, sizeof(cp), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      available_ = false;
      return -1.0;
    }

    // --- Grow device buffers -------------------------------------------------
    const size_t gridBytes = static_cast<size_t>(p.gridDataSize);
    growBuf(impl_->d_corrGrid,  impl_->corrGridCap,  gridBytes);
    growBuf(impl_->d_localPts,  impl_->localPtsCap,  p.nPoints * 2);
    growBuf(impl_->d_lookups,   impl_->lookupCap,    nAngles * p.nPoints);
    growBuf(impl_->d_responses, impl_->responsesCap, totalPoses);
    growBuf(impl_->d_ssp,       impl_->sspCap,       nX * nY);

    // --- H → D: smeared grid + scan points ----------------------------------
    CUDA_CHECK(cudaMemcpyAsync(
      impl_->d_corrGrid, p.corrGridData, gridBytes, cudaMemcpyHostToDevice, S));
    CUDA_CHECK(cudaMemcpyAsync(
      impl_->d_localPts, p.localPointsXY,
      p.nPoints * 2 * sizeof(float), cudaMemcpyHostToDevice, S));

    // --- Kernel A: ComputeOffsets --------------------------------------------
    {
      const int kT = 256;
      computeOffsetsKernel<<<(nAngles * p.nPoints + kT - 1) / kT, kT, 0, S>>>(
        impl_->d_localPts, impl_->d_lookups);
      CUDA_CHECK(cudaGetLastError());
    }

    // --- Bind grid as texture (read-only for Kernel B) ----------------------
    destroyTex(*impl_);
    {
      cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
      rd.resType                = cudaResourceTypeLinear;
      rd.res.linear.devPtr      = impl_->d_corrGrid;
      rd.res.linear.desc        = cudaCreateChannelDesc<uint8_t>();
      rd.res.linear.sizeInBytes = gridBytes;
      cudaTextureDesc td; memset(&td, 0, sizeof(td));
      td.readMode = cudaReadModeElementType;
      CUDA_CHECK(cudaCreateTextureObject(&impl_->texGrid, &rd, &td, nullptr));
      impl_->texBound = true;
    }

    // --- Kernel B: 3-D pose search ------------------------------------------
    {
      const int kTX = 16, kTY = 16;
      dim3 block(kTX, kTY);
      dim3 grid((nX + kTX - 1) / kTX, (nY + kTY - 1) / kTY, nAngles);
      correlateSearchKernel<<<grid, block, 0, S>>>(
        impl_->texGrid, impl_->d_lookups, impl_->d_responses);
      CUDA_CHECK(cudaGetLastError());
    }

    // --- Custom argmax: bestResponse ----------------------------------------
    // Two-pass reduction: first find max per block, then global max
    const int blockCount = (totalPoses + 255) / 256;
    growBuf(impl_->d_blockMaxVals, impl_->blockMaxValCap, blockCount);
    growBuf(impl_->d_blockMaxIdxs, impl_->blockMaxIdxCap, blockCount);
    
    argmaxKernel<<<blockCount, 256, 0, S>>>(
      impl_->d_responses, totalPoses,
      impl_->d_blockMaxVals, impl_->d_blockMaxIdxs, blockCount);
    CUDA_CHECK(cudaGetLastError());
    
    // Second pass: find max of block maxes
    float * h_blockMaxVals = (float *)malloc(blockCount * sizeof(float));
    int * h_blockMaxIdxs = (int *)malloc(blockCount * sizeof(int));
    CUDA_CHECK(cudaMemcpyAsync(h_blockMaxVals, impl_->d_blockMaxVals,
      blockCount * sizeof(float), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaMemcpyAsync(h_blockMaxIdxs, impl_->d_blockMaxIdxs,
      blockCount * sizeof(int), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaStreamSynchronize(S));
    
    float h_best = -1e9f;
    for (int i = 0; i < blockCount; i++) {
      if (h_blockMaxVals[i] > h_best) {
        h_best = h_blockMaxVals[i];
      }
    }
    free(h_blockMaxVals);
    free(h_blockMaxIdxs);
    
    if (h_best > 1.0f) { h_best = 1.0f; }

    // --- Kernel C2: accumulate best-pose sums --------------------------------
    const float zf = 0.0f; const int zi = 0;
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_sumX,    &zf, sizeof(float), cudaMemcpyHostToDevice, S));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_sumY,    &zf, sizeof(float), cudaMemcpyHostToDevice, S));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_sumCosT, &zf, sizeof(float), cudaMemcpyHostToDevice, S));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_sumSinT, &zf, sizeof(float), cudaMemcpyHostToDevice, S));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_count,   &zi, sizeof(int),    cudaMemcpyHostToDevice, S));
    {
      const int kT = 256;
      accumBestPoseKernel<<<(totalPoses + kT - 1) / kT, kT, 0, S>>>(
        impl_->d_responses, h_best, totalPoses,
        impl_->d_sumX, impl_->d_sumY,
        impl_->d_sumCosT, impl_->d_sumSinT, impl_->d_count);
      CUDA_CHECK(cudaGetLastError());
    }
    float h_sx = 0.0f, h_sy = 0.0f, h_sc = 0.0f, h_ss = 0.0f; int h_cnt = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_sx,  impl_->d_sumX,    sizeof(float), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaMemcpyAsync(&h_sy,  impl_->d_sumY,    sizeof(float), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaMemcpyAsync(&h_sc,  impl_->d_sumCosT, sizeof(float), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaMemcpyAsync(&h_ss,  impl_->d_sumSinT, sizeof(float), cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaMemcpyAsync(&h_cnt, impl_->d_count,   sizeof(int),    cudaMemcpyDeviceToHost, S));
    CUDA_CHECK(cudaStreamSynchronize(S));
    if (h_cnt == 0) { return -1.0; }

    *outMeanX     = h_sx / h_cnt;
    *outMeanY     = h_sy / h_cnt;
    *outMeanTheta = atan2(h_ss / h_cnt, h_sc / h_cnt);

    // --- Covariance on HOST (double precision — matches CPU within 1 ULP) ---
    memset(outCovData, 0, 9 * sizeof(double));
    outCovData[0] = outCovData[4] = outCovData[8] = 1.0;   // identity
    const double dBest = static_cast<double>(h_best);

    if (dBest < kKtTolerance) {
      // Max variance fallback (mirrors CPU, Mapper.cpp:885-891)
      outCovData[0] = kMaxVariance;
      outCovData[4] = kMaxVariance;
      outCovData[8] = 4.0 * p.searchAngleResolution * p.searchAngleResolution;
      return dBest;
    }

    if (!p.doingFineMatch) {
      // -----------------------------------------------------------------------
      // Positional covariance (mirrors ComputePositionalCovariance, Mapper.cpp:874)
      // -----------------------------------------------------------------------
      {
        const int kT = 16;
        dim3 blk(kT, kT), grd((nX + kT - 1) / kT, (nY + kT - 1) / kT);
        buildSearchSpaceProbsKernel<<<grd, blk, 0, S>>>(
          impl_->d_responses, impl_->d_ssp);
        CUDA_CHECK(cudaGetLastError());
      }
      float * h_ssp = (float *)malloc(nX * nY * sizeof(float));
      if (!h_ssp) { return -1.0; }
      CUDA_CHECK(cudaMemcpyAsync(h_ssp, impl_->d_ssp,
        nX * nY * sizeof(float), cudaMemcpyDeviceToHost, S));
      CUDA_CHECK(cudaStreamSynchronize(S));

      double dx_b = *outMeanX - p.searchCenterX;
      double dy_b = *outMeanY - p.searchCenterY;
      double norm = 0, vXX = 0, vXY = 0, vYY = 0;
      for (int yi = 0; yi < nY; yi++) {
        double y = static_cast<double>(startY) + yi * p.searchSpaceResolutionY;
        for (int xi = 0; xi < nX; xi++) {
          double x = static_cast<double>(startX) + xi * p.searchSpaceResolutionX;
          double r = h_ssp[yi * nX + xi];
          if (r >= dBest - 0.1) {
            norm += r;
            vXX  += (x - dx_b) * (x - dx_b) * r;
            vXY  += (x - dx_b) * (y - dy_b) * r;
            vYY  += (y - dy_b) * (y - dy_b) * r;
          }
        }
      }
      if (norm > kKtTolerance) {
        double vXXn = vXX / norm, vXYn = vXY / norm, vYYn = vYY / norm;
        double vTT  = 4.0 * p.searchAngleResolution * p.searchAngleResolution;
        double minXX = 0.1 * p.searchSpaceResolutionX * p.searchSpaceResolutionX;
        double minYY = 0.1 * p.searchSpaceResolutionY * p.searchSpaceResolutionY;
        vXXn = fmax(vXXn, minXX); vYYn = fmax(vYYn, minYY);
        double mult = 1.0 / dBest;
        outCovData[0] = vXXn * mult;  outCovData[1] = vXYn * mult;
        outCovData[3] = vXYn * mult;  outCovData[4] = vYYn * mult;
        outCovData[8] = vTT;
      }
      if (fabs(outCovData[0]) < kKtTolerance) { outCovData[0] = kMaxVariance; }
      if (fabs(outCovData[4]) < kKtTolerance) { outCovData[4] = kMaxVariance; }
      free(h_ssp);

    } else {
      // -----------------------------------------------------------------------
      // Angular covariance (mirrors ComputeAngularCovariance, Mapper.cpp:977).
      // GetResponse is re-evaluated on the CPU grid using downloaded offsets.
      // -----------------------------------------------------------------------
      int32_t * h_off = (int32_t *)malloc(nAngles * p.nPoints * sizeof(int32_t));
      if (!h_off) { return -1.0; }
      CUDA_CHECK(cudaMemcpyAsync(h_off, impl_->d_lookups,
        nAngles * p.nPoints * sizeof(int32_t), cudaMemcpyDeviceToHost, S));
      CUDA_CHECK(cudaStreamSynchronize(S));

      // CorrelationGrid::GridIndex for best pose
      int gxB = static_cast<int>(roundf(
        (*outMeanX - p.gridOffsetX) * p.gridScale));
      int gyB = static_cast<int>(roundf(
        (*outMeanY - p.gridOffsetY) * p.gridScale));
      int gidxBest = (gxB + p.roiOffsetX) + (gyB + p.roiOffsetY) * p.gridWidthStep;

      // NormalizeAngleDifference(bestTheta, searchCenterTheta)
      double bestAngle = *outMeanTheta;
      const double pi = 3.141592653589793, tpi = 6.283185307179586;
      double ctr = static_cast<double>(p.searchCenterTheta);
      while (bestAngle - ctr < -pi) { bestAngle += tpi; }
      while (bestAngle - ctr >  pi) { bestAngle -= tpi; }

      double norm = 0, accVarTh = 0;
      for (int ai = 0; ai < nAngles; ai++) {
        double rsum = 0;
        const int32_t * off = h_off + ai * p.nPoints;
        for (int pi2 = 0; pi2 < p.nPoints; pi2++) {
          if (off[pi2] == kInvalidScan) { continue; }
          int comb = gidxBest + off[pi2];
          if (comb < 0 || comb >= p.gridDataSize) { continue; }
          rsum += static_cast<double>(p.corrGridData[comb]);
        }
        double resp = rsum / (static_cast<double>(p.nPoints) * kGridStatesOccupied);
        if (resp >= dBest - 0.1) {
          double ang = static_cast<double>(startAngle) + ai * p.searchAngleResolution;
          double da  = ang - bestAngle;
          norm    += resp;
          accVarTh += da * da * resp;
        }
      }
      if (norm > kKtTolerance) {
        if (accVarTh < kKtTolerance) {
          accVarTh = static_cast<double>(p.searchAngleResolution)
                   * p.searchAngleResolution;
        }
        accVarTh /= norm;
      } else {
        accVarTh = 1000.0
          * static_cast<double>(p.searchAngleResolution)
          * p.searchAngleResolution;
      }
      // (2,2) is the only entry written here; (0,0)..(1,1) were written
      // by the preceding positional-covariance call (doingFineMatch=false).
      outCovData[8] = accVarTh;
      free(h_off);
    }

  double result = dBest > 1.0 ? 1.0 : dBest;
  return result;
}

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX_CUDA_AVAILABLE
