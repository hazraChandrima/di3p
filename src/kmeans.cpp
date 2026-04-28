#include "kmeans.hpp"
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <algorithm>

constexpr double Xn = 95.047, Yn = 100.0, Zn = 108.883;

static double srgbLinear(double c) {
    c /= 255.0;
    return (c <= 0.04045) ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

static double labF(double t) {
    constexpr double delta = 6.0 / 29.0;
    return (t > delta * delta * delta)
        ? std::cbrt(t)
        : t / (3.0 * delta * delta) + 4.0 / 29.0;
}

PixelLAB rgbToLab(PixelRGB p) {
    double R  = srgbLinear(p.r);
    double G  = srgbLinear(p.g);
    double B_ = srgbLinear(p.b);

    double X = (0.4124564*R + 0.3575761*G + 0.1804375*B_) * 100.0;
    double Y = (0.2126729*R + 0.7151522*G + 0.0721750*B_) * 100.0;
    double Z = (0.0193339*R + 0.1191920*G + 0.9503041*B_) * 100.0;

    double fx = labF(X / Xn);
    double fy = labF(Y / Yn);
    double fz = labF(Z / Zn);

    return { 116.0*fy - 16.0, 500.0*(fx - fy), 200.0*(fy - fz) };
}

static inline double labDist2(const PixelLAB& p, const Centroid& c) {
    double dL = p.L-c.L, dA = p.A-c.A, dB = p.B-c.B;
    return dL*dL + dA*dA + dB*dB;
}

KMeansResult kmeansSegment(const std::vector<PixelRGB>& pixels,
                            int rows, int cols, int K, int maxIter, unsigned seed) {
    if (K <= 0 || K > (int)pixels.size())
        throw std::invalid_argument("K out of range");

    int N = rows * cols;
    std::vector<PixelLAB> lab(N);
    for (int i = 0; i < N; ++i) lab[i] = rgbToLab(pixels[i]);

    std::mt19937 rng(seed);
    std::vector<Centroid> centroids(K);

    {
        std::uniform_int_distribution<int> pick(0, N-1);
        int idx = pick(rng);
        centroids[0] = {lab[idx].L, lab[idx].A, lab[idx].B};
    }
    for (int k = 1; k < K; ++k) {
        std::vector<double> dists(N);
        for (int i = 0; i < N; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int j = 0; j < k; ++j)
                best = std::min(best, labDist2(lab[i], centroids[j]));
            dists[i] = best;
        }
        std::discrete_distribution<int> weighted(dists.begin(), dists.end());
        int idx = weighted(rng);
        centroids[k] = {lab[idx].L, lab[idx].A, lab[idx].B};
    }

    std::vector<int> labels(N, 0);
    for (int iter = 0; iter < maxIter; ++iter) {
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            double bestDist = std::numeric_limits<double>::max();
            int    bestK    = 0;
            for (int k = 0; k < K; ++k) {
                double d = labDist2(lab[i], centroids[k]);
                if (d < bestDist) { bestDist = d; bestK = k; }
            }
            if (bestK != labels[i]) { labels[i] = bestK; changed = true; }
        }
        if (!changed) break;

        std::vector<double> sumL(K,0), sumA(K,0), sumB(K,0);
        std::vector<int>    cnt(K, 0);
        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            sumL[k] += lab[i].L; sumA[k] += lab[i].A; sumB[k] += lab[i].B;
            ++cnt[k];
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[k] > 0)
                centroids[k] = {sumL[k]/cnt[k], sumA[k]/cnt[k], sumB[k]/cnt[k]};
            else {
                std::uniform_int_distribution<int> pick(0, N-1);
                int idx = pick(rng);
                centroids[k] = {lab[idx].L, lab[idx].A, lab[idx].B};
            }
        }
    }
    return {labels, centroids, rows, cols};
}

std::vector<bool> regionMask(const KMeansResult& seg, int k) {
    int N = seg.rows * seg.cols;
    std::vector<bool> mask(N);
    for (int i = 0; i < N; ++i) mask[i] = (seg.labels[i] == k);
    return mask;
}

std::vector<int> regionIndices(const KMeansResult& seg, int k) {
    std::vector<int> idx;
    for (int i = 0; i < (int)seg.labels.size(); ++i)
        if (seg.labels[i] == k) idx.push_back(i);
    return idx;
}
