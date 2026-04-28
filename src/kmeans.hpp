#pragma once

#include <vector>
#include <cstdint>

struct PixelRGB  { uint8_t r, g, b; };
struct PixelLAB  { double L, A, B;  };
struct Centroid  { double L, A, B;  };

struct KMeansResult {
    std::vector<int>      labels;
    std::vector<Centroid> centroids;
    int                   rows, cols;
};

PixelLAB    rgbToLab(PixelRGB p);
KMeansResult kmeansSegment(const std::vector<PixelRGB>& pixels,
                            int rows, int cols,
                            int K = 4, int maxIter = 30, unsigned seed = 42);

std::vector<bool> regionMask(const KMeansResult& seg, int k);
std::vector<int>  regionIndices(const KMeansResult& seg, int k);
