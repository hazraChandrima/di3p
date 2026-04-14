#pragma once

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>

constexpr double MAX_PIXEL = 255.0;

// ─────────────────────────────────────────────────────────────────────────────
// PSNR
//   MSE  = (1/MN) Σ_i Σ_j (I(i,j) - K(i,j))²
//   PSNR = 10 · log10(MAX² / MSE)
//   Unit: dB.  Higher = better.  >40 dB = excellent, <20 dB = poor.
// ─────────────────────────────────────────────────────────────────────────────
double computeMSE(const std::vector<double>& original,
                  const std::vector<double>& processed) {
    if (original.size() != processed.size() || original.empty())
        return 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = original[i] - processed[i];
        mse += diff * diff;
    }
    return mse / original.size();
}

double computePSNR(const std::vector<double>& original,
                   const std::vector<double>& processed) {
    double mse = computeMSE(original, processed);
    if (mse < 1e-10) return 100.0; // identical images
    return 10.0 * std::log10((MAX_PIXEL * MAX_PIXEL) / mse);
}

// ─────────────────────────────────────────────────────────────────────────────
// SSIM
//   SSIM(x, y) = [2μ_x μ_y + C1][2σ_xy + C2]
//                / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]
//
//   Computed over the entire signal (global) or over an 11×11 Gaussian
//   window.  Here we compute a global single-window version for simplicity,
//   which is accurate enough for region-level comparison.
// ─────────────────────────────────────────────────────────────────────────────

// Constants (from paper: K1=0.01, K2=0.03, L=255)
constexpr double C1 = (0.01 * MAX_PIXEL) * (0.01 * MAX_PIXEL); // (K1·L)²
constexpr double C2 = (0.03 * MAX_PIXEL) * (0.03 * MAX_PIXEL); // (K2·L)²

double computeSSIM(const std::vector<double>& x,
                   const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    int N = (int)x.size();

    double muX = 0, muY = 0;
    for (int i = 0; i < N; ++i) { muX += x[i]; muY += y[i]; }
    muX /= N; muY /= N;

    double sigX2 = 0, sigY2 = 0, sigXY = 0;
    for (int i = 0; i < N; ++i) {
        double dx = x[i] - muX, dy = y[i] - muY;
        sigX2 += dx * dx;
        sigY2 += dy * dy;
        sigXY += dx * dy;
    }
    sigX2 /= (N - 1); sigY2 /= (N - 1); sigXY /= (N - 1);

    double num = (2.0 * muX * muY + C1) * (2.0 * sigXY + C2);
    double den = (muX*muX + muY*muY + C1) * (sigX2 + sigY2 + C2);
    return den < 1e-12 ? 1.0 : num / den;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shannon Entropy
//   H = −Σ_{i=0}^{L−1} p(i) · log2(p(i))
//   where p(i) is the normalised histogram of pixel intensities.
//   Higher entropy = more information content / detail.
// ─────────────────────────────────────────────────────────────────────────────
double computeEntropy(const std::vector<double>& pixels) {
    if (pixels.empty()) return 0.0;

    // Build 256-bin histogram (values assumed in [0,255])
    std::vector<int> hist(256, 0);
    for (double v : pixels) {
        int bin = std::clamp((int)std::round(v), 0, 255);
        ++hist[bin];
    }
    double N = (double)pixels.size();
    double H = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] == 0) continue;
        double p = hist[i] / N;
        H -= p * std::log2(p);
    }
    return H; // bits per pixel, max ~8 bits for uniform distribution
}

struct RegionMetrics {
    double psnr_before, psnr_after;
    double ssim_before, ssim_after;
    double entropy_before, entropy_after;

    double psnrGain()    const { return psnr_after    - psnr_before;    }
    double ssimGain()    const { return ssim_after     - ssim_before;    }
    double entropyGain() const { return entropy_after  - entropy_before; }
};

// Evaluate a single region: needs original, degraded-before, and corrected-after
// All three must be the same size.
RegionMetrics evaluateRegion(const std::vector<double>& original,
                              const std::vector<double>& before,
                              const std::vector<double>& after) {
    return {
        computePSNR(original, before),    computePSNR(original, after),
        computeSSIM(original, before),    computeSSIM(original, after),
        computeEntropy(before),            computeEntropy(after)
    };
}

std::string formatReport(int regionId,
                         const std::string& diagnosis,
                         const RegionMetrics& m) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "┌─ Region " << regionId << " ─────────────────────────\n";
    ss << "│  Diagnosis : " << diagnosis << "\n";
    ss << "│  PSNR   before: " << m.psnr_before    << " dB"
       << "  →  after: "        << m.psnr_after     << " dB"
       << "  (Δ " << (m.psnrGain() >= 0 ? "+" : "") << m.psnrGain() << ")\n";
    ss << "│  SSIM   before: " << m.ssim_before
       << "  →  after: "        << m.ssim_after
       << "  (Δ " << (m.ssimGain() >= 0 ? "+" : "") << m.ssimGain() << ")\n";
    ss << "│  Entropy before: " << m.entropy_before
       << "  →  after: "         << m.entropy_after
       << "  (Δ " << (m.entropyGain() >= 0 ? "+" : "") << m.entropyGain() << ")\n";
    ss << "└────────────────────────────────────────\n";
    return ss.str();
}

std::string formatSummary(const std::vector<double>& origFlat,
                           const std::vector<double>& beforeFlat,
                           const std::vector<double>& afterFlat) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    double psnr_b = computePSNR(origFlat, beforeFlat);
    double psnr_a = computePSNR(origFlat, afterFlat);
    double ssim_b = computeSSIM(origFlat, beforeFlat);
    double ssim_a = computeSSIM(origFlat, afterFlat);
    double ent_b  = computeEntropy(beforeFlat);
    double ent_a  = computeEntropy(afterFlat);
    ss << "═══ Whole-image summary ════════════════\n";
    ss << "  PSNR:    " << psnr_b << " → " << psnr_a << " dB\n";
    ss << "  SSIM:    " << ssim_b << " → " << ssim_a << "\n";
    ss << "  Entropy: " << ent_b  << " → " << ent_a  << " bits/px\n";
    ss << "════════════════════════════════════════\n";
    return ss.str();
}
