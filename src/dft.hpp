#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — FFT / IFFT from scratch  (Cooley-Tukey radix-2 DIT)
//
// Replaces the O(N²) naive DFT with O(N log N) FFT.
// Everything else (2D decomposition, shift, masks, scores) is unchanged.
//
// Speedup on a 256-point signal: 256² = 65536 ops  →  256×8 = 2048 ops  (32×)
// On a full 512×512 region:     O(N²) ≈ 68 billion  →  O(N logN) ≈ 2.4 million
//
// Algorithm: Cooley-Tukey in-place radix-2 DIT (decimation-in-time)
//   1. Bit-reverse permutation
//   2. Butterfly passes: log2(N) stages, N/2 butterflies per stage
//   Twiddle factors: W_N^k = e^(−j·2π·k/N)  (forward)
//                             e^(+j·2π·k/N)  (inverse, then divide by N)
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <numeric>

using Complex    = std::complex<double>;
using CRow       = std::vector<Complex>;
using CMatrix    = std::vector<CRow>;
using RealMatrix = std::vector<std::vector<double>>;

constexpr double PI = 3.14159265358979323846;

// ── helpers ───────────────────────────────────────────────────────────────────

inline int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline double toGray(uint8_t r, uint8_t g, uint8_t b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

// ── Bit-reverse permutation ───────────────────────────────────────────────────
// Reorders x so that x[i] moves to x[bitrev(i, log2(N))].
// Required before the butterfly passes so butterflies access contiguous pairs.
void bitReversePermute(CRow& x) {
    int N = (int)x.size();
    int bits = 0;
    while ((1 << bits) < N) ++bits;

    for (int i = 0; i < N; ++i) {
        // Compute bit-reverse of i
        int rev = 0;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b)) rev |= (1 << (bits - 1 - b));
        if (i < rev) std::swap(x[i], x[rev]);
    }
}

// ── 1-D Cooley-Tukey radix-2 FFT (in-place) ─────────────────────────────────
// forward=true  → X[k] = Σ x[n] e^(−j2πkn/N)
// forward=false → x[n] = (1/N) Σ X[k] e^(+j2πkn/N)
// N must be a power of 2.
void fft1d_inplace(CRow& x, bool forward) {
    int N = (int)x.size();

    // Bit-reverse permutation
    bitReversePermute(x);

    // Butterfly passes — log2(N) stages
    for (int len = 2; len <= N; len <<= 1) {
        // Twiddle factor for this stage: W = e^(±j·2π/len)
        double angle = (forward ? -1.0 : +1.0) * 2.0 * PI / len;
        Complex W(std::cos(angle), std::sin(angle));

        // Process each group of `len` elements
        for (int i = 0; i < N; i += len) {
            Complex w(1.0, 0.0); // current twiddle, starts at W^0 = 1
            for (int j = 0; j < len / 2; ++j) {
                // Butterfly:
                //   even = x[i+j]        + w * x[i+j+len/2]
                //   odd  = x[i+j]        − w * x[i+j+len/2]
                Complex u = x[i + j];
                Complex t = w * x[i + j + len/2];
                x[i + j]          = u + t;
                x[i + j + len/2]  = u - t;
                w *= W;  // advance twiddle: W^(j+1) = W^j * W
            }
        }
    }

    // Inverse: divide by N
    if (!forward) {
        for (auto& v : x) v /= double(N);
    }
}

// Public wrapper — same signature as old dft1d for drop-in replacement
CRow dft1d(const CRow& x, bool inverse = false) {
    CRow y = x;
    // Pad to power of 2 if needed (shouldn't happen in practice since
    // toCMatrix already pads, but defensive)
    int N = (int)y.size();
    int P = nextPow2(N);
    if (P != N) y.resize(P, {0.0, 0.0});
    fft1d_inplace(y, !inverse);
    return y;
}

// ── 2-D FFT via row-column decomposition ─────────────────────────────────────
// Identical logic to before — just calls the O(N log N) fft1d now
CMatrix dft2d(const CMatrix& in, bool inverse = false) {
    int rows = (int)in.size();
    int cols = (int)in[0].size();
    CMatrix out(rows, CRow(cols));

    // Transform each row
    for (int r = 0; r < rows; ++r)
        out[r] = dft1d(in[r], inverse);

    // Transform each column
    for (int c = 0; c < cols; ++c) {
        CRow col(rows);
        for (int r = 0; r < rows; ++r) col[r] = out[r][c];
        CRow colT = dft1d(col, inverse);
        for (int r = 0; r < rows; ++r) out[r][c] = colT[r];
    }
    return out;
}

// ── FFT shift ─────────────────────────────────────────────────────────────────
CMatrix fftShift(const CMatrix& in) {
    int rows = (int)in.size(), cols = (int)in[0].size();
    CMatrix out(rows, CRow(cols));
    int hr = rows / 2, hc = cols / 2;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[(r + hr) % rows][(c + hc) % cols] = in[r][c];
    return out;
}

CMatrix ifftShift(const CMatrix& in) { return fftShift(in); }

// ── ImageGray ─────────────────────────────────────────────────────────────────
struct ImageGray {
    int rows, cols;
    std::vector<double> data; // row-major, [0,255]
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

ImageGray toImageGray(const CMatrix& m, int origRows, int origCols) {
    ImageGray img;
    img.rows = origRows; img.cols = origCols;
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

// ── Magnitude spectrum ────────────────────────────────────────────────────────
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
    double blurScore;   // highFreqEnergy / totalEnergy — low = blurry
    double noiseScore;  // variance of high-freq magnitudes — low = uniform noise
};

DFTScores computeScores(const CMatrix& shifted, double cutoffRatio = 0.3) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double maxR = std::sqrt((double)(cr * cr + cc * cc));
    double cutoff = cutoffRatio * maxR;

    double totalEnergy = 0, highEnergy = 0;
    std::vector<double> highMags;
    highMags.reserve(rows * cols / 2);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double mag2 = std::norm(shifted[r][c]);
            totalEnergy += mag2;
            double dist = std::sqrt((double)((r-cr)*(r-cr) + (c-cc)*(c-cc)));
            if (dist > cutoff) {
                highEnergy += mag2;
                highMags.push_back(std::sqrt(mag2));
            }
        }
    }

    double blurScore = (totalEnergy < 1e-12) ? 0.0 : highEnergy / totalEnergy;

    double mean = 0;
    for (double v : highMags) mean += v;
    mean /= std::max((int)highMags.size(), 1);
    double var = 0;
    for (double v : highMags) var += (v - mean) * (v - mean);
    var /= std::max((int)highMags.size(), 1);

    return {blurScore, var};
}

// ── Spectral masks ────────────────────────────────────────────────────────────
CMatrix applyLowPassMask(const CMatrix& shifted, double cutoffRatio = 0.3) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double cutoff = cutoffRatio * std::sqrt((double)(cr*cr + cc*cc));
    CMatrix out = shifted;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double d = std::sqrt((double)((r-cr)*(r-cr) + (c-cc)*(c-cc)));
            if (d > cutoff) out[r][c] = {0.0, 0.0};
        }
    return out;
}

CMatrix applyHighBoostMask(const CMatrix& shifted,
                            double cutoffRatio = 0.3, double boostFactor = 1.8) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double cutoff = cutoffRatio * std::sqrt((double)(cr*cr + cc*cc));
    CMatrix out = shifted;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double d = std::sqrt((double)((r-cr)*(r-cr) + (c-cc)*(c-cc)));
            if (d > cutoff) out[r][c] *= boostFactor;
        }
    return out;
}

// ── Full pipeline ─────────────────────────────────────────────────────────────
enum class DFTMode { DENOISE, SHARPEN, DIAGNOSE_ONLY };

struct DFTResult {
    ImageGray processed;
    DFTScores scores;
    bool      isBlurry;
    bool      isNoisy;
};

DFTResult processDFT(const ImageGray& img,
                     DFTMode mode          = DFTMode::DIAGNOSE_ONLY,
                     double cutoffRatio    = 0.3,
                     double boostFactor    = 1.8,
                     double blurThreshold  = 0.05,
                     double noiseThreshold = 0.01) {
    CMatrix padded       = toCMatrix(img);
    CMatrix spectrum     = dft2d(padded, false);
    CMatrix shifted      = fftShift(spectrum);
    DFTScores scores     = computeScores(shifted, cutoffRatio);

    CMatrix masked = shifted;
    if      (mode == DFTMode::DENOISE) masked = applyLowPassMask(shifted, cutoffRatio);
    else if (mode == DFTMode::SHARPEN) masked = applyHighBoostMask(shifted, cutoffRatio, boostFactor);

    CMatrix unshifted    = ifftShift(masked);
    CMatrix reconstructed = dft2d(unshifted, true);

    return {
        toImageGray(reconstructed, img.rows, img.cols),
        scores,
        scores.blurScore  < blurThreshold,
        scores.noiseScore < noiseThreshold
    };
}
