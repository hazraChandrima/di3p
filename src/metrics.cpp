#include "metrics.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <numeric>

constexpr double MAX_PIXEL = 255.0;
constexpr double C1 = (0.01 * MAX_PIXEL) * (0.01 * MAX_PIXEL);
constexpr double C2 = (0.03 * MAX_PIXEL) * (0.03 * MAX_PIXEL);

// ── Scalar metrics ────────────────────────────────────────────────────────────

double computeMSE(const std::vector<double>& original,
                  const std::vector<double>& processed) {
    if (original.size() != processed.size() || original.empty()) return 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double d = original[i] - processed[i];
        mse += d * d;
    }
    return mse / original.size();
}

double computePSNR(const std::vector<double>& original,
                   const std::vector<double>& processed) {
    double mse = computeMSE(original, processed);
    if (mse < 1e-10) return 100.0;
    return 10.0 * std::log10((MAX_PIXEL * MAX_PIXEL) / mse);
}

double computeSSIM(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    int N = (int)x.size();
    double muX = 0, muY = 0;
    for (int i = 0; i < N; ++i) { muX += x[i]; muY += y[i]; }
    muX /= N; muY /= N;
    double sigX2 = 0, sigY2 = 0, sigXY = 0;
    for (int i = 0; i < N; ++i) {
        double dx = x[i]-muX, dy = y[i]-muY;
        sigX2 += dx*dx; sigY2 += dy*dy; sigXY += dx*dy;
    }
    sigX2 /= (N-1); sigY2 /= (N-1); sigXY /= (N-1);
    double num = (2.0*muX*muY + C1) * (2.0*sigXY + C2);
    double den = (muX*muX + muY*muY + C1) * (sigX2 + sigY2 + C2);
    return den < 1e-12 ? 1.0 : num / den;
}

double computeEntropy(const std::vector<double>& pixels) {
    if (pixels.empty()) return 0.0;
    std::vector<int> hist(256, 0);
    for (double v : pixels) ++hist[std::clamp((int)std::round(v), 0, 255)];
    double N = (double)pixels.size(), H = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (!hist[i]) continue;
        double p = hist[i] / N;
        H -= p * std::log2(p);
    }
    return H;
}

// ── Per-channel helpers ───────────────────────────────────────────────────────

// Extract pixel values for one channel from a flat full-image RGBBuf,
// using only the pixels at the given indices.
std::vector<double> extractIndexedChannel(const std::vector<uint8_t>& rgbBuf,
                                           const std::vector<int>& indices,
                                           int ch) {
    std::vector<double> out(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        out[i] = rgbBuf[indices[i]*3 + ch];
    return out;
}

// Compute per-channel + grayscale PSNR and SSIM between two full-image RGBBufs,
// restricted to the pixels at indices.
ChannelMetrics computeChannelMetrics(const std::vector<uint8_t>& origRGB,
                                      const std::vector<uint8_t>& otherRGB,
                                      const std::vector<int>&     indices) {
    ChannelMetrics cm{};

    // R, G, B
    for (int ch = 0; ch < 3; ++ch) {
        auto a = extractIndexedChannel(origRGB,  indices, ch);
        auto b = extractIndexedChannel(otherRGB, indices, ch);
        cm.psnr[ch] = computePSNR(a, b);
        cm.ssim[ch] = computeSSIM(a, b);
    }

    // Grayscale (index 3): 0.299R + 0.587G + 0.114B
    std::vector<double> ga(indices.size()), gb(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        ga[i] = 0.299*origRGB[idx*3]  + 0.587*origRGB[idx*3+1]  + 0.114*origRGB[idx*3+2];
        gb[i] = 0.299*otherRGB[idx*3] + 0.587*otherRGB[idx*3+1] + 0.114*otherRGB[idx*3+2];
    }
    cm.psnr[3] = computePSNR(ga, gb);
    cm.ssim[3] = computeSSIM(ga, gb);

    return cm;
}

// ── Region evaluation ─────────────────────────────────────────────────────────

RegionMetrics evaluateRegion(const std::vector<uint8_t>& origRGB,
                              const std::vector<uint8_t>& afterRGB,
                              const std::vector<int>&     indices) {
    RegionMetrics m{};
    // "before" is orig vs orig — all perfect scores, used as baseline reference
    m.before = computeChannelMetrics(origRGB, origRGB, indices);
    m.after  = computeChannelMetrics(origRGB, afterRGB, indices);

    // Entropy on grayscale after-pixels
    std::vector<double> grayAfter(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        grayAfter[i] = 0.299*afterRGB[idx*3] + 0.587*afterRGB[idx*3+1] + 0.114*afterRGB[idx*3+2];
    }
    std::vector<double> grayOrig(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        grayOrig[i] = 0.299*origRGB[idx*3] + 0.587*origRGB[idx*3+1] + 0.114*origRGB[idx*3+2];
    }
    m.entropy_before = computeEntropy(grayOrig);
    m.entropy_after  = computeEntropy(grayAfter);
    return m;
}

// ── Formatting ────────────────────────────────────────────────────────────────

static const char* CH_NAME[4] = { "R", "G", "B", "gray" };

std::string formatReport(int regionId, const std::string& diagnosis,
                          const RegionMetrics& m) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "  region " << regionId << " metrics (" << diagnosis << ")\n";

    // PSNR row header
    ss << "    PSNR  ";
    for (int ch = 0; ch < 4; ++ch)
        ss << "  " << CH_NAME[ch] << ": " << m.after.psnr[ch] << " dB";
    ss << "\n";

    // SSIM row
    ss << "    SSIM  ";
    for (int ch = 0; ch < 4; ++ch)
        ss << "  " << CH_NAME[ch] << ": " << m.after.ssim[ch];
    ss << "\n";

    // Entropy
    ss << "    entropy  before: " << m.entropy_before
       << "  after: "              << m.entropy_after
       << "  (";
    double eg = m.entropyGain();
    ss << (eg >= 0 ? "+" : "") << eg << ")\n";

    return ss.str();
}

std::string formatSummary(const std::vector<uint8_t>& origRGB,
                           const std::vector<uint8_t>& afterRGB) {
    int N = (int)(origRGB.size() / 3);
    std::vector<int> allIdx(N);
    std::iota(allIdx.begin(), allIdx.end(), 0);

    ChannelMetrics cm = computeChannelMetrics(origRGB, afterRGB, allIdx);

    std::vector<double> gOrig(N), gAfter(N);
    for (int i = 0; i < N; ++i) {
        gOrig[i]  = 0.299*origRGB[i*3]  + 0.587*origRGB[i*3+1]  + 0.114*origRGB[i*3+2];
        gAfter[i] = 0.299*afterRGB[i*3] + 0.587*afterRGB[i*3+1] + 0.114*afterRGB[i*3+2];
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "summary\n";
    ss << "  PSNR  ";
    for (int ch = 0; ch < 4; ++ch)
        ss << "  " << CH_NAME[ch] << ": " << cm.psnr[ch] << " dB";
    ss << "\n";
    ss << "  SSIM  ";
    for (int ch = 0; ch < 4; ++ch)
        ss << "  " << CH_NAME[ch] << ": " << cm.ssim[ch];
    ss << "\n";
    ss << "  entropy  before: " << computeEntropy(gOrig)
       << "  after: "            << computeEntropy(gAfter) << "\n";
    return ss.str();
}
