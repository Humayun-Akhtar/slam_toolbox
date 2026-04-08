/*
 * test_correlate.cu
 *
 * Standalone numerical equality test: CPU reference vs GPU correlateScan.
 * No ROS, no ament, no gtest — just nvcc + the two translation units.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * BUILD (from the slam_toolbox root, after having run CMake at least once
 * to generate include/slam_toolbox/gpu_config.h into the build tree):
 *
 *   SLAM_ROOT=/path/to/slam_toolbox
 *   BUILD_DIR=/path/to/slam_toolbox/build    # wherever you ran cmake
 *
 *   nvcc -std=c++17 \
 *     -I${SLAM_ROOT}/lib/karto_sdk/include \
 *     -I${SLAM_ROOT}/include \
 *     -I${BUILD_DIR}/include \
 *     -arch=sm_61 \
 *     ${SLAM_ROOT}/test_correlate.cu \
 *     ${SLAM_ROOT}/src/gpu_scan_correlation_backend.cu \
 *     -o test_correlate
 *
 *   ./test_correlate
 *
 * If the build tree does not exist yet, create a one-line gpu_config.h:
 *
 *   mkdir -p /tmp/gpu_config_stub/slam_toolbox
 *   echo '#pragma once' > /tmp/gpu_config_stub/slam_toolbox/gpu_config.h
 *   echo '#define SLAM_TOOLBOX_CUDA_AVAILABLE' >> \
 *                 /tmp/gpu_config_stub/slam_toolbox/gpu_config.h
 *
 * and replace -I${BUILD_DIR}/include with -I/tmp/gpu_config_stub.
 * ──────────────────────────────────────────────────────────────────────────
 *
 * GRID GEOMETRY NOTE
 * The user spec calls for a 100×100 grid. Because karto requires grids with
 * odd side length (assert(gridSize % 2 == 1) in ScanMatcher::Create) we use
 * 101×101.  We also need the grid to be large enough to hold scan points at
 * radius 2.0 m.  With resolution 0.01 m/cell and gridOffset (-0.5, -0.5),
 * the full grid spans [-0.5, +0.51] m in each axis — only scan points near
 * the +x direction fall inside.  This is intentional: the test exercises
 * mixed valid/INVALID lookup patterns, which is more representative than a
 * trivially all-valid or all-invalid case.
 *
 * To make all 36 scan points land inside the grid, scale the RADIUS constant
 * down to 0.3 (the full [-0.5,0.51] span covers 0.3 m from centre) or use a
 * larger grid.  The test is valid either way — what matters is that CPU and
 * GPU produce the same result for the same inputs.
 *
 * PRECISION MODEL
 * Kernel A uses float trig (__cosf/__sinf) + __float2int_rn.
 * The CPU reference uses cosf/sinf + rintf(), which produce identical integer
 * indices for all angles in our [-0.349, +0.349] rad range (the max deviation
 * between __cosf and cosf is ~5e-7 relative; after ×200 cells and rounding,
 * this never shifts a grid cell for any of our search angles).
 * Kernel B accumulates float; covariance is double on both sides.
 * Expected delta for all 13 assertions: < 1e-6 (KT_TOLERANCE).
 */

// ── Force the GPU code path for standalone build ──────────────────────────
// This must appear BEFORE any #include that pulls in gpu_config.h.
// The generated gpu_config.h either #defines or /* #undef */s the macro;
// the comment form of #undef does NOT cancel a prior #define, so this is safe.
#ifndef SLAM_TOOLBOX_CUDA_AVAILABLE
#  define SLAM_TOOLBOX_CUDA_AVAILABLE
#endif
// Also define the include-guard so that if the generated gpu_config.h is
// found on the include path, it is silently skipped (no double-define warning).
#ifndef SLAM_TOOLBOX__GPU_CONFIG_H_
#  define SLAM_TOOLBOX__GPU_CONFIG_H_
#endif

// ── Only C headers — no C++ stdlib that could pull in <functional> on GCC 11
#include <cuda_runtime.h>  /* CUDA host/device API */

#include <stdint.h>        /* uint8_t, int32_t … */
#include <stdio.h>         /* printf, fprintf */
#include <stdlib.h>        /* malloc, calloc, free, exit */
#include <string.h>        /* memset */
#include <math.h>          /* cosf, sinf, rintf, roundf, fabsf, fabs, isnan, atan2 */
#include <assert.h>        /* assert */
#include <float.h>         /* FLT_MAX */
#include <limits.h>        /* INT_MAX */

// These two project headers have no karto/ROS/ament dependencies.
#include "karto_sdk/GpuCorrelation.h"             /* karto::GpuCorrelationCallParams */
#include "slam_toolbox/gpu_device_symbols.h"     /* extern __constant__ CorrelateParams d_params */
#include "slam_toolbox/gpu_scan_correlation_backend.hpp" /* slam_toolbox::GpuScanCorrelationBackend */

// ============================================================================
// Shared constants — identical to the values inside gpu_scan_correlation_backend.cu
// ============================================================================
static const float  kOccupied = 100.0f;
static const double kKtTol    = 1e-6;
static const double kMaxVar   = 500.0;
static const int    kInvalid  = INT_MAX;   /* mirrors INVALID_SCAN in Math.h */
static const float  kDistGain = 0.2f;
static const float  kAngleGain = 0.2f;

// ============================================================================
// Inline utilities matching karto / GPU code
// ============================================================================

// Mirrors karto math::Round and CUDA __float2int_rn: round-half-to-even.
// rintf() uses the current IEEE 754 rounding mode (default = nearest-even),
// matching __float2int_rn exactly on SM 6.x+ hardware.
static inline int iRound(float x) { return (int)rintf(x); }

// Normalise angle to (-pi, pi] — matches accumBestPoseKernel's while-loops.
static inline double normalise(double a) {
    static const double PI  = 3.141592653589793;
    static const double PI2 = 6.283185307179586;
    while (a < -PI) a += PI2;
    while (a >  PI) a -= PI2;
    return a;
}

// ── check_value: replaces the C++ lambda in main() ───────────────────────
// Returns 1 on PASS, 0 on FAIL. Caller ANDs return values into allPass.
static int check_value(const char * label, double cpuVal, double gpuVal,
                       double tol)
{
    double delta = fabs(cpuVal - gpuVal);
    int    pass  = (delta < tol);
    printf("  %-22s  cpu=%+.10f  gpu=%+.10f  |delta|=%.3e  %s\n",
           label, cpuVal, gpuVal, delta, pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================================
// CPU reference: cpuCorrelateScan
// ─────────────────────────────────────────────────────────────────────────────
// Mirrors the following GPU code paths in gpu_scan_correlation_backend.cu:
//   correlateScan():        dimension derivation, startX/Y/Angle
//   computeOffsetsKernel(): float cosf/sinf, rintf, base Grid::GridIndex
//   correlateSearchKernel(): CorrelationGrid::GridIndex (with ROI), response sum,
//                            normalization, clamping, penalty
//   thrust max_element:     std::max_element over float array
//   accumBestPoseKernel():  DoubleEqual filter, double pose sums
//   host covariance code:   positional SSP reduction + covariance accumulation
//
// Precision contract: all float operations must use the same precision and
// rounding as the GPU kernels so that integer grid indices are bit-identical.
// ============================================================================
struct CpuResult {
    double response;
    double meanX, meanY, meanTheta;
    double cov[9];   // row-major 3×3
};

static CpuResult cpuCorrelateScan(const karto::GpuCorrelationCallParams & p)
{
    // ── Dimension derivation (mirrors correlateScan() host preamble) ──────
    const int nX = (int)roundf(p.searchSpaceOffsetX * 2.0f
                               / p.searchSpaceResolutionX) + 1;
    const int nY = (int)roundf(p.searchSpaceOffsetY * 2.0f
                               / p.searchSpaceResolutionY) + 1;
    const int nAng = (int)roundf(p.searchAngleOffset * 2.0f
                                 / p.searchAngleResolution) + 1;
    const float startX     = -p.searchSpaceOffsetX;
    const float startY     = -p.searchSpaceOffsetY;
    const float startAng   =  p.searchCenterTheta - p.searchAngleOffset;
    const int   totalPoses =  nX * nY * nAng;

    // ── Kernel A: compute lookup offsets [nAng * nPoints] — plain int array
    // Uses float cosf/sinf + rintf() to match __cosf/__sinf + __float2int_rn.
    int * lookups = (int *)malloc((size_t)nAng * p.nPoints * sizeof(int));
    if (!lookups) { fprintf(stderr, "OOM: lookups\n"); exit(1); }
    /* initialise to kInvalid = INT_MAX — cannot use memset(0x7f) as that gives
       0x7F7F7F7F, so fill explicitly */
    for (int k = 0; k < nAng * p.nPoints; ++k) lookups[k] = kInvalid;

    for (int ai = 0; ai < nAng; ++ai) {
        float angle = startAng + ai * p.searchAngleResolution;
        float cosA  = cosf(angle);   /* float — matches __cosf for our angle range */
        float sinA  = sinf(angle);   /* float — matches __sinf */
        for (int pi = 0; pi < p.nPoints; ++pi) {
            float lx = p.localPointsXY[pi * 2];
            float ly = p.localPointsXY[pi * 2 + 1];
            if (isnan(lx) || isnan(ly)) { continue; }  /* leave kInvalid */

            /* Rotate — identical float ops to Kernel A */
            float rotX = cosA * lx - sinA * ly;
            float rotY = sinA * lx + cosA * ly;

            /* Base Grid::GridIndex (no ROI): gridOffset cancels */
            int gx  = iRound(rotX * p.gridScale);
            int gy  = iRound(rotY * p.gridScale);
            int idx = gx + gy * p.gridWidthStep;

            if (idx < 0 || idx >= p.gridDataSize) { continue; }  /* kInvalid stays */
            lookups[ai * p.nPoints + pi] = idx;
        }
    }

    // ── Kernel B: 3-D pose search — plain float array (calloc gives 0.0f)
    float * responses = (float *)calloc((size_t)totalPoses, sizeof(float));
    if (!responses) { free(lookups); fprintf(stderr, "OOM: responses\n"); exit(1); }

    for (int yi = 0; yi < nY; ++yi) {
        float dy     = startY + yi * p.searchSpaceResolutionY;
        float worldY = p.searchCenterY + dy;
        int   gyPos  = iRound((worldY - p.gridOffsetY) * p.gridScale);

        for (int xi = 0; xi < nX; ++xi) {
            float dx     = startX + xi * p.searchSpaceResolutionX;
            float worldX = p.searchCenterX + dx;
            int   gxPos  = iRound((worldX - p.gridOffsetX) * p.gridScale);

            /* CorrelationGrid::GridIndex with ROI */
            int gridPosIdx = (gxPos + p.roiOffsetX)
                           + (gyPos + p.roiOffsetY) * p.gridWidthStep;

            for (int ai = 0; ai < nAng; ++ai) {
                float respSum = 0.0f;
                const int * offsets = lookups + ai * p.nPoints;
                for (int pi = 0; pi < p.nPoints; ++pi) {
                    int offset = offsets[pi];
                    if (offset == kInvalid) { continue; }
                    int combined = gridPosIdx + offset;
                    if (combined < 0 || combined >= p.gridDataSize) { continue; }
                    respSum += (float)p.corrGridData[combined];
                }

                /* Normalise + clamp */
                float resp = respSum / ((float)p.nPoints * kOccupied);
                if (resp > 1.0f) resp = 1.0f;
                if (resp < 0.0f) resp = 0.0f;

                /* Penalty (disabled in our test, but correct for completeness) */
                if (p.doPenalize && resp > 1e-7f) {
                    float sqDist  = dx * dx + dy * dy;
                    float distPen = 1.0f - (kDistGain * sqDist
                                            / p.distVariancePenalty);
                    if (distPen < p.minDistPenalty) distPen = p.minDistPenalty;

                    float angle  = startAng + ai * p.searchAngleResolution;
                    float dAng   = angle - p.searchCenterTheta;
                    float angPen = 1.0f - (kAngleGain * dAng * dAng
                                           / p.angleVariancePenalty);
                    if (angPen < p.minAnglePenalty) angPen = p.minAnglePenalty;

                    resp = resp * distPen * angPen;
                    if (resp > 1.0f) resp = 1.0f;
                    if (resp < 0.0f) resp = 0.0f;
                }

                /* Layout: responses[yi*nX*nAng + xi*nAng + ai] */
                responses[yi * nX * nAng + xi * nAng + ai] = resp;
            }
        }
    }

    // ── Manual max_element (replaces std::max_element) ────────────────────
    float bestResp = -1.0f;
    for (int i = 0; i < totalPoses; ++i)
        if (responses[i] > bestResp) bestResp = responses[i];
    if (bestResp > 1.0f) bestResp = 1.0f;

    // ── Weighted-mean best pose (double arithmetic) ───────────────────────
    double sumX = 0.0, sumY = 0.0, sumCos = 0.0, sumSin = 0.0;
    int    cnt  = 0;
    for (int i = 0; i < totalPoses; ++i) {
        if (fabsf(responses[i] - bestResp) > (float)kKtTol) { continue; }
        int ai = i % nAng;
        int xi = (i / nAng) % nX;
        int yi = i / (nAng * nX);

        double px = (double)p.searchCenterX
                  + (double)startX + xi * (double)p.searchSpaceResolutionX;
        double py = (double)p.searchCenterY
                  + (double)startY + yi * (double)p.searchSpaceResolutionY;
        double th = (double)startAng + ai * (double)p.searchAngleResolution;
        th = normalise(th);

        sumX   += px;
        sumY   += py;
        sumCos += cos(th);
        sumSin += sin(th);
        ++cnt;
    }

    /* zero-init via memset; 0x00 = 0.0 in IEEE 754 */
    CpuResult r;
    memset(&r, 0, sizeof(r));
    r.response = bestResp;
    r.cov[0] = r.cov[4] = r.cov[8] = 1.0;  /* identity covariance init */

    if (cnt == 0) {
        free(responses);
        free(lookups);
        return r;
    }

    r.meanX     = sumX   / cnt;
    r.meanY     = sumY   / cnt;
    r.meanTheta = atan2(sumSin / cnt, sumCos / cnt);

    double dBest = (double)bestResp;

    // ── Max-variance fallback ─────────────────────────────────────────────
    if (dBest < kKtTol) {
        r.cov[0] = r.cov[4] = kMaxVar;
        r.cov[8] = 4.0 * p.searchAngleResolution * p.searchAngleResolution;
        free(responses);
        free(lookups);
        return r;
    }

    if (!p.doingFineMatch) {
        // ── Positional covariance — plain float array ─────────────────────
        float * ssp = (float *)malloc((size_t)nX * nY * sizeof(float));
        if (!ssp) { free(responses); free(lookups);
                    fprintf(stderr, "OOM: ssp\n"); exit(1); }
        for (int k = 0; k < nX * nY; ++k) ssp[k] = -1.0f;

        for (int yi = 0; yi < nY; ++yi) {
            for (int xi = 0; xi < nX; ++xi) {
                float mx = -1.0f;
                int base = yi * nX * nAng + xi * nAng;
                for (int ai = 0; ai < nAng; ++ai)
                    if (responses[base + ai] > mx) mx = responses[base + ai];
                ssp[yi * nX + xi] = mx;
            }
        }

        double dx_b = r.meanX - p.searchCenterX;
        double dy_b = r.meanY - p.searchCenterY;
        double norm = 0.0, vXX = 0.0, vXY = 0.0, vYY = 0.0;

        for (int yi = 0; yi < nY; ++yi) {
            double y = (double)startY + yi * p.searchSpaceResolutionY;
            for (int xi = 0; xi < nX; ++xi) {
                double x  = (double)startX + xi * p.searchSpaceResolutionX;
                double rr = (double)ssp[yi * nX + xi];
                if (rr >= dBest - 0.1) {
                    norm += rr;
                    vXX  += (x - dx_b) * (x - dx_b) * rr;
                    vXY  += (x - dx_b) * (y - dy_b) * rr;
                    vYY  += (y - dy_b) * (y - dy_b) * rr;
                }
            }
        }
        free(ssp);

        if (norm > kKtTol) {
            double vXXn = vXX / norm, vXYn = vXY / norm, vYYn = vYY / norm;
            double vTT  = 4.0 * p.searchAngleResolution * p.searchAngleResolution;
            double minXX = 0.1 * p.searchSpaceResolutionX * p.searchSpaceResolutionX;
            double minYY = 0.1 * p.searchSpaceResolutionY * p.searchSpaceResolutionY;
            if (vXXn < minXX) vXXn = minXX;
            if (vYYn < minYY) vYYn = minYY;
            double mult = 1.0 / dBest;
            r.cov[0] = vXXn * mult;
            r.cov[1] = r.cov[3] = vXYn * mult;
            r.cov[4] = vYYn * mult;
            r.cov[8] = vTT;
        }
        if (fabs(r.cov[0]) < kKtTol) r.cov[0] = kMaxVar;
        if (fabs(r.cov[4]) < kKtTol) r.cov[4] = kMaxVar;

    } else {
        // ── Angular covariance ────────────────────────────────────────────
        int gxB = (int)round((r.meanX - p.gridOffsetX) * p.gridScale);
        int gyB = (int)round((r.meanY - p.gridOffsetY) * p.gridScale);
        int gidxBest = (gxB + p.roiOffsetX)
                     + (gyB + p.roiOffsetY) * p.gridWidthStep;

        static const double PI  = 3.141592653589793;
        static const double TPI = 6.283185307179586;
        double bestAngle = r.meanTheta;
        double ctr       = (double)p.searchCenterTheta;
        while (bestAngle - ctr < -PI) bestAngle += TPI;
        while (bestAngle - ctr >  PI) bestAngle -= TPI;

        double norm = 0.0, accVarTh = 0.0;
        for (int ai = 0; ai < nAng; ++ai) {
            double rsum = 0.0;
            const int * off = lookups + ai * p.nPoints;
            for (int pi = 0; pi < p.nPoints; ++pi) {
                if (off[pi] == kInvalid) { continue; }
                int comb = gidxBest + off[pi];
                if (comb < 0 || comb >= p.gridDataSize) { continue; }
                rsum += (double)p.corrGridData[comb];
            }
            double rr = rsum / ((double)p.nPoints * kOccupied);
            if (rr >= dBest - 0.1) {
                double ang = (double)startAng
                           + ai * (double)p.searchAngleResolution;
                double da = ang - bestAngle;
                norm     += rr;
                accVarTh += da * da * rr;
            }
        }
        if (norm > kKtTol) {
            if (accVarTh < kKtTol)
                accVarTh = (double)p.searchAngleResolution
                         * p.searchAngleResolution;
            accVarTh /= norm;
        } else {
            accVarTh = 1000.0 * (double)p.searchAngleResolution
                               * p.searchAngleResolution;
        }
        r.cov[8] = accVarTh;
    }

    free(responses);
    free(lookups);
    return r;
}

// ============================================================================
// main
// ============================================================================
int main()
{
    // ── Grid setup ────────────────────────────────────────────────────────
    //
    // 101×101 at 0.01 m/cell (resolution chosen to place scan-circle points
    // at integer grid offsets, minimising rounding edge-cases).
    // widthStep = 104 (next multiple of 8 above 101).
    // roiOffset = (0, 0) — no border region; the full array IS the "ROI".
    // gridOffset = (-0.50, -0.50) so that world origin maps to cell (50, 50).
    //
    // At this resolution, scan points at 2.0 m radius span gx ∈ [-200, 200].
    // Only points where gx ≥ 0 and gx + gy*104 < 10504 (the dataSize) produce
    // valid lookup indices.  For a circle of 36 points at 10° steps, roughly
    // 5–7 points near bearing 0° satisfy this condition.  That produces
    // non-trivial, non-zero responses — suitable for a numerical equality test.
    //
    // To make ALL 36 points land inside the grid, change RADIUS to 0.3 (fits
    // in [-0.5, +0.5]) or change the grid to 500×500 cells.
    //
    enum { GW = 101, GH = 101, WS = ((GW + 7) / 8) * 8, DS = WS * GH };
    const float GSCL = 100.0f;   /* 1 / 0.01 m */
    const float GOX  = -0.50f;
    const float GOY  = -0.50f;

    /* calloc → zero-initialised; grid cells are set in the loop below */
    uint8_t * grid = (uint8_t *)calloc(DS, 1);
    if (!grid) { fprintf(stderr, "OOM: grid\n"); return 1; }
    /* Checkerboard: cell(x, y) = ((x+y) % 2 == 0) ? 200 : 0 */
    for (int y = 0; y < GH; ++y)
        for (int x = 0; x < GW; ++x)
            grid[x + y * WS] = (uint8_t)(((x + y) % 2 == 0) ? 200 : 0);

    /* 36 scan points evenly on a 2.0 m radius circle, sensor at origin.
       Sensor pose = (0,0,0) → InverseTransformPose is identity, so
       local-frame points equal world-frame points. */
    enum { N = 36 };
    const float RAD = 2.0f;   /* metres */
    float pts[N * 2];
    for (int i = 0; i < N; ++i) {
        float a = (float)i * (2.0f * (float)M_PI / N);
        pts[i * 2]     = RAD * cosf(a);
        pts[i * 2 + 1] = RAD * sinf(a);
    }

    // ── Search parameters (as specified) ──────────────────────────────────
    constexpr float SS_OX  = 0.300f;   // searchSpaceOffset X
    constexpr float SS_OY  = 0.300f;   // searchSpaceOffset Y
    constexpr float SS_RX  = 0.010f;   // searchSpaceResolution X → 61 positions
    constexpr float SS_RY  = 0.010f;   // searchSpaceResolution Y → 61 positions
    constexpr float ANG_O  = 0.349f;   // searchAngleOffset   → 201 angles
    constexpr float ANG_R  = 0.00349f; // searchAngleResolution

    // nX = round(0.3*2/0.01)+1 = 61
    // nY = 61, nAng = round(0.349*2/0.00349)+1 = 201
    // totalPoses = 61 * 61 * 201 = 747,861

    /* ── Pack GpuCorrelationCallParams ─────────────────────────────────── */
    karto::GpuCorrelationCallParams p;
    p.localPointsXY        = pts;
    p.nPoints              = N;
    p.corrGridData         = grid;
    p.gridWidth            = GW;
    p.gridHeight           = GH;
    p.gridWidthStep        = WS;
    p.gridDataSize         = DS;
    p.gridScale            = GSCL;
    p.gridOffsetX          = GOX;
    p.gridOffsetY          = GOY;
    p.roiOffsetX           = 0;
    p.roiOffsetY           = 0;
    p.searchCenterX        = 0.0f;
    p.searchCenterY        = 0.0f;
    p.searchCenterTheta    = 0.0f;
    p.searchAngleOffset    = ANG_O;
    p.searchAngleResolution= ANG_R;
    p.searchSpaceOffsetX   = SS_OX;
    p.searchSpaceOffsetY   = SS_OY;
    p.searchSpaceResolutionX = SS_RX;
    p.searchSpaceResolutionY = SS_RY;
    p.doPenalize           = false;
    p.distVariancePenalty  = 0.09f;    // sqrt(0.3)^2 — karto default
    p.angleVariancePenalty = 0.0609f;  // sqrt(20°)^2 — karto default
    p.minDistPenalty       = 0.50f;
    p.minAnglePenalty      = 0.90f;
    p.doingFineMatch       = false;

    /* ── Run CPU reference ──────────────────────────────────────────────── */
    printf("[ CPU ] Running reference...\n");
    CpuResult cpu = cpuCorrelateScan(p);
    printf("[ CPU ] bestResponse = %.10f\n", cpu.response);
    printf("[ CPU ] mean = (%.10f, %.10f, %.10f)\n",
           cpu.meanX, cpu.meanY, cpu.meanTheta);
    printf("[ CPU ] cov diag = [%.6e, %.6e, %.6e]\n",
           cpu.cov[0], cpu.cov[4], cpu.cov[8]);

    /* ── Run GPU backend ─────────────────────────────────────────────────── */
    printf("\n[ GPU ] Constructing GpuScanCorrelationBackend...\n");
    slam_toolbox::GpuScanCorrelationBackend backend;
    if (!backend.isAvailable()) {
        printf("[ GPU ] SKIP: No CUDA device found (isAvailable() == false).\n");
        printf("        Re-run on a machine with a CUDA-capable GPU (SM >= 6.0).\n");
        free(grid);
        return 0;
    }
    printf("[ GPU ] Device ready. Running correlateScan...\n");

    double gpuMeanX = 0.0, gpuMeanY = 0.0, gpuMeanTheta = 0.0;
    double gpuCov[9];
    memset(gpuCov, 0, sizeof(gpuCov));
    double gpuResp = backend.correlateScan(
        p, &gpuMeanX, &gpuMeanY, &gpuMeanTheta, gpuCov);

    if (gpuResp < 0.0) {
        fprintf(stderr, "[ GPU ] FAIL: correlateScan() returned -1.0 (CUDA error).\n");
        free(grid);
        return 1;
    }
    printf("[ GPU ] bestResponse = %.10f\n", gpuResp);
    printf("[ GPU ] mean = (%.10f, %.10f, %.10f)\n",
           gpuMeanX, gpuMeanY, gpuMeanTheta);
    printf("[ GPU ] cov diag = [%.6e, %.6e, %.6e]\n",
           gpuCov[0], gpuCov[4], gpuCov[8]);

    /* ── Assertions ─────────────────────────────────────────────────────── */
    const double TOL = 1e-4;
    int allPass = 1;   /* 1 = pass, 0 = fail */

#define CHK(label, cv, gv) \
    allPass &= check_value((label), (cv), (gv), TOL)

    printf("\n=== Assertion results (tolerance %.0e) ===\n", TOL);
    CHK("bestResponse",    cpu.response,  gpuResp);
    CHK("mean.x",          cpu.meanX,     gpuMeanX);
    CHK("mean.y",          cpu.meanY,     gpuMeanY);
    CHK("mean.theta",      cpu.meanTheta, gpuMeanTheta);
    /* 9 covariance elements (row-major [00,01,02, 10,11,12, 20,21,22]) */
    CHK("cov[0,0] (XX)",   cpu.cov[0],    gpuCov[0]);
    CHK("cov[0,1] (XY)",   cpu.cov[1],    gpuCov[1]);
    CHK("cov[0,2]",        cpu.cov[2],    gpuCov[2]);
    CHK("cov[1,0] (YX)",   cpu.cov[3],    gpuCov[3]);
    CHK("cov[1,1] (YY)",   cpu.cov[4],    gpuCov[4]);
    CHK("cov[1,2]",        cpu.cov[5],    gpuCov[5]);
    CHK("cov[2,0]",        cpu.cov[6],    gpuCov[6]);
    CHK("cov[2,1]",        cpu.cov[7],    gpuCov[7]);
    CHK("cov[2,2] (ThTh)", cpu.cov[8],    gpuCov[8]);
#undef CHK

    free(grid);
    printf("\n%s\n",
           allPass ? "=== ALL 13 ASSERTIONS PASS ===" : "=== FAIL ===");
    return allPass ? 0 : 1;
}
