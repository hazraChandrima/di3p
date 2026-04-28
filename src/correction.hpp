#pragma once

#include <vector>
#include <cstdint>
#include "dft.hpp"

using RGBBuf = std::vector<uint8_t>;  // [R0,G0,B0, R1,G1,B1, ...]

// ── Channel helpers ───────────────────────────────────────────────────────────
std::vector<double> extractChannel(const RGBBuf& buf, int ch);
void                writeChannel(RGBBuf& buf, const std::vector<double>& ch, int idx);
std::vector<double> blendCh(const std::vector<double>& orig,
                             const std::vector<double>& corr, double strength);

// ── Histogram stats / stretch ─────────────────────────────────────────────────
struct HistStats {
    double mean, stddev, p2, p98;
    bool underExposed;  // mean < 60  && p98 < 180
    bool overExposed;   // mean > 195 && p2  > 80
};

HistStats computeHistStats(const RGBBuf& pixels);
RGBBuf    histogramStretchLAB(const RGBBuf& pixels, double strength = 0.9);

// ── Sharpening ────────────────────────────────────────────────────────────────
RGBBuf sharpenRGB(const RGBBuf& pixels, int rows, int cols,
                   double blurCutoff = 0.15, double amount = 0.5, double strength = 0.5);

// ── DFT denoising ─────────────────────────────────────────────────────────────
RGBBuf denoiseRGB(const RGBBuf& pixels, int rows, int cols,
                   double cutoffRatio = 0.35, double strength = 0.5);

// ── DCT blockiness removal ────────────────────────────────────────────────────
RGBBuf removeBlockinessRGB(const RGBBuf& pixels, int rows, int cols,
                             double sf = 0.6, double strength = 0.6);

// ── Diagnosis + auto-correction ───────────────────────────────────────────────
struct RegionDiagnosis {
    bool   blurry, noisy, underExposed, overExposed, blocky;
    double blurScore, noiseScore, histMean, histStddev, boundaryDiff;
};

RegionDiagnosis diagnoseRegion(const RGBBuf& regionBuf, int rows, int cols, bool runDCT);
RGBBuf          correctRegionRGB(const RGBBuf& pixels, int rows, int cols,
                                  const RegionDiagnosis& diag);
