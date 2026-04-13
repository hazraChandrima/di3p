#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — DFT / IDFT from scratch
//   • 1D DFT (O(N²)), applied row-wise then column-wise for 2D
//   • Zero-pads input to next power-of-2 for better spectral resolution
//   • Outputs magnitude spectrum, blur score, noise score
//   • Applies spectral mask (low-pass or high-pass) and reconstructs via IDFT
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Complex   = std::complex<double>;
using CRow      = std::vector<Complex>;
using CMatrix   = std::vector<CRow>;
using RealMatrix = std::vector<std::vector<double>>;

constexpr double PI = 3.14159265358979323846;

// ── helpers ──────────────────────────────────────────────────────────────────

// Next power of 2 >= n
inline int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// RGB → grayscale  (ITU-R BT.601)
inline double toGray(uint8_t r, uint8_t g, uint8_t b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

// ── 1-D DFT ──────────────────────────────────────────────────────────────────
// forward: X[k] = Σ x[n] · e^(−j·2π·k·n/N)
// inverse: x[n] = (1/N) Σ X[k] · e^(+j·2π·k·n/N)
CRow dft1d(const CRow& x, bool inverse = false) {
    int N = (int)x.size();
    CRow X(N, {0.0, 0.0});
    double sign = inverse ? +1.0 : -1.0;
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            double angle = sign * 2.0 * PI * k * n / N;
            X[k] += x[n] * Complex(std::cos(angle), std::sin(angle));
        }
        if (inverse) X[k] /= N;
    }
    return X;
}

// ── 2-D DFT via row-column decomposition ─────────────────────────────────────
CMatrix dft2d(const CMatrix& in, bool inverse = false) {
    int rows = (int)in.size();
    int cols = (int)in[0].size();
    CMatrix out(rows, CRow(cols));

    // Transform each row
    for (int r = 0; r < rows; ++r)
        out[r] = dft1d(in[r], inverse);

    // Transform each column of the row-transformed result
    for (int c = 0; c < cols; ++c) {
        CRow col(rows);
        for (int r = 0; r < rows; ++r) col[r] = out[r][c];
        CRow colT = dft1d(col, inverse);
        for (int r = 0; r < rows; ++r) out[r][c] = colT[r];
    }
    return out;
}

// ── FFT shift: move DC component to centre ───────────────────────────────────
CMatrix fftShift(const CMatrix& in) {
    int rows = (int)in.size();
    int cols = (int)in[0].size();
    CMatrix out(rows, CRow(cols));
    int hr = rows / 2, hc = cols / 2;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[(r + hr) % rows][(c + hc) % cols] = in[r][c];
    return out;
}

CMatrix ifftShift(const CMatrix& in) { return fftShift(in); } // shift is its own inverse

// ── Build padded complex matrix from grayscale image ─────────────────────────
struct ImageGray {
    int rows, cols;
    std::vector<double> data; // row-major, values [0,255]
    double& at(int r, int c)       { return data[r * cols + c]; }
    double  at(int r, int c) const { return data[r * cols + c]; }
};

CMatrix toCMatrix(const ImageGray& img) {
    int pr = nextPow2(img.rows);
    int pc = nextPow2(img.cols);
    CMatrix m(pr, CRow(pc, {0.0, 0.0}));
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            m[r][c] = {img.at(r, c) / 255.0, 0.0};
    return m;
}

// Extract real part back to image (crop to original size, re-scale to [0,255])
ImageGray toImageGray(const CMatrix& m, int origRows, int origCols) {
    ImageGray img;
    img.rows = origRows;
    img.cols = origCols;
    img.data.resize(origRows * origCols);
    double minV = 1e18, maxV = -1e18;
    for (int r = 0; r < origRows; ++r)
        for (int c = 0; c < origCols; ++c) {
            double v = m[r][c].real();
            minV = std::min(minV, v);
            maxV = std::max(maxV, v);
        }
    double range = (maxV - minV) < 1e-9 ? 1.0 : (maxV - minV);
    for (int r = 0; r < origRows; ++r)
        for (int c = 0; c < origCols; ++c)
            img.at(r, c) = std::clamp((m[r][c].real() - minV) / range * 255.0, 0.0, 255.0);
    return img;
}

// ── Magnitude spectrum (log-scaled for display) ───────────────────────────────
RealMatrix magnitudeSpectrum(const CMatrix& dft) {
    int rows = (int)dft.size(), cols = (int)dft[0].size();
    RealMatrix mag(rows, std::vector<double>(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mag[r][c] = std::log(1.0 + std::abs(dft[r][c]));
    return mag;
}

// ── Diagnosis scores ──────────────────────────────────────────────────────────
struct DFTScores {
    double blurScore;   // highFreqEnergy / totalEnergy  — low = blurry
    double noiseScore;  // variance of |F(u,v)| in high-freq ring — low = noise
};

DFTScores computeScores(const CMatrix& shifted, double cutoffRatio = 0.3) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double maxR = std::sqrt((double)(cr * cr + cc * cc));
    double cutoff = cutoffRatio * maxR;

    double totalEnergy = 0, highEnergy = 0;
    std::vector<double> highMags;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double mag2 = std::norm(shifted[r][c]); // |.|^2
            totalEnergy += mag2;
            double dist = std::sqrt((double)((r - cr) * (r - cr) + (c - cc) * (c - cc)));
            if (dist > cutoff) {
                highEnergy += mag2;
                highMags.push_back(std::sqrt(mag2));
            }
        }
    }

    double blurScore = (totalEnergy < 1e-12) ? 0.0 : highEnergy / totalEnergy;

    // Variance of high-freq magnitudes
    double mean = 0;
    for (double v : highMags) mean += v;
    mean /= std::max((int)highMags.size(), 1);
    double var = 0;
    for (double v : highMags) var += (v - mean) * (v - mean);
    var /= std::max((int)highMags.size(), 1);

    return {blurScore, var};
}

// ── Spectral masks ─────────────────────────────────────────────────────────────
// Low-pass: keep frequencies inside cutoff  (denoising)
// High-pass: boost frequencies outside cutoff  (sharpening)

CMatrix applyLowPassMask(const CMatrix& shifted, double cutoffRatio = 0.3) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double maxR = std::sqrt((double)(cr * cr + cc * cc));
    double cutoff = cutoffRatio * maxR;
    CMatrix out = shifted;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double d = std::sqrt((double)((r - cr) * (r - cr) + (c - cc) * (c - cc)));
            if (d > cutoff) out[r][c] = {0.0, 0.0};
        }
    return out;
}

CMatrix applyHighBoostMask(const CMatrix& shifted, double cutoffRatio = 0.3,
                            double boostFactor = 1.8) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double maxR = std::sqrt((double)(cr * cr + cc * cc));
    double cutoff = cutoffRatio * maxR;
    CMatrix out = shifted;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double d = std::sqrt((double)((r - cr) * (r - cr) + (c - cc) * (c - cc)));
            if (d > cutoff) out[r][c] *= boostFactor;
        }
    return out;
}

// ── Full pipeline: image → DFT → diagnose → mask → IDFT → image ─────────────

enum class DFTMode { DENOISE, SHARPEN, DIAGNOSE_ONLY };

struct DFTResult {
    ImageGray processed;
    DFTScores scores;
    bool      isBlurry;   // blurScore < blurThreshold
    bool      isNoisy;    // noiseScore < noiseThreshold
};

DFTResult processDFT(const ImageGray& img,
                     DFTMode mode           = DFTMode::DIAGNOSE_ONLY,
                     double cutoffRatio     = 0.3,
                     double boostFactor     = 1.8,
                     double blurThreshold   = 0.05,
                     double noiseThreshold  = 0.01) {
    // Build padded complex matrix
    CMatrix padded = toCMatrix(img);

    // Forward 2D DFT
    CMatrix dft = dft2d(padded, false);

    // Shift DC to centre
    CMatrix shifted = fftShift(dft);

    // Compute diagnosis scores
    DFTScores scores = computeScores(shifted, cutoffRatio);

    // Apply spectral mask based on mode
    CMatrix masked = shifted;
    if (mode == DFTMode::DENOISE)
        masked = applyLowPassMask(shifted, cutoffRatio);
    else if (mode == DFTMode::SHARPEN)
        masked = applyHighBoostMask(shifted, cutoffRatio, boostFactor);

    // Shift back, inverse DFT
    CMatrix unshifted = ifftShift(masked);
    CMatrix reconstructed = dft2d(unshifted, true);

    return {
        toImageGray(reconstructed, img.rows, img.cols),
        scores,
        scores.blurScore  < blurThreshold,
        scores.noiseScore < noiseThreshold
    };
}
