#pragma once

#include <vector>
#include <string>
#include <cstdint>

double computeMSE    (const std::vector<double>& original, const std::vector<double>& processed);
double computePSNR   (const std::vector<double>& original, const std::vector<double>& processed);
double computeSSIM   (const std::vector<double>& x,        const std::vector<double>& y);
double computeEntropy(const std::vector<double>& pixels);

// ── Per-channel metrics ───────────────────────────────────────────────────────
// Holds one value per channel: [0]=R, [1]=G, [2]=B, [3]=gray/overall

struct ChannelMetrics {
    double psnr[4];   // R, G, B, gray
    double ssim[4];
};

// Extract a single channel (0=R,1=G,2=B) from a packed RGB pixel list
// (indices are pixel indices into the full image, buf is the full RGBBuf)
std::vector<double> extractIndexedChannel(const std::vector<uint8_t>& rgbBuf,
                                           const std::vector<int>& indices,
                                           int ch);

ChannelMetrics computeChannelMetrics(
    const std::vector<uint8_t>& origRGB,
    const std::vector<uint8_t>& otherRGB,
    const std::vector<int>&     indices);   // pixel indices into the full flat buf

// ── Region metrics (before/after, per-channel) ───────────────────────────────

struct RegionMetrics {
    ChannelMetrics before;
    ChannelMetrics after;
    double entropy_before;
    double entropy_after;

    double psnrGain(int ch = 3) const { return after.psnr[ch] - before.psnr[ch]; }
    double ssimGain(int ch = 3) const { return after.ssim[ch] - before.ssim[ch]; }
    double entropyGain()        const { return entropy_after  - entropy_before;   }
};

RegionMetrics evaluateRegion(
    const std::vector<uint8_t>& origRGB,
    const std::vector<uint8_t>& afterRGB,
    const std::vector<int>&     indices);

std::string formatReport  (int regionId, const std::string& diagnosis, const RegionMetrics& m);
std::string formatSummary (const std::vector<uint8_t>& origRGB,
                            const std::vector<uint8_t>& afterRGB);
