#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "src/dft.hpp"
#include "src/kmeans.hpp"
#include "src/correction.hpp"
#include "src/metrics.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

struct Image {
    int rows = 0, cols = 0;
    std::vector<PixelRGB> pixels;
    PixelRGB& at(int r, int c)       { return pixels[r*cols+c]; }
    PixelRGB  at(int r, int c) const { return pixels[r*cols+c]; }
};

Image loadImage(const std::string& path) {
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (m.empty()) throw std::runtime_error("Cannot open: " + path);
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
    Image img; img.rows = m.rows; img.cols = m.cols;
    img.pixels.resize(img.rows * img.cols);
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c) {
            auto px = m.at<cv::Vec3b>(r, c);
            img.at(r, c) = {px[0], px[1], px[2]};
        }
    return img;
}

void saveImage(const Image& img, const std::string& path) {
    cv::Mat m(img.rows, img.cols, CV_8UC3);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c) {
            auto p = img.at(r, c);
            m.at<cv::Vec3b>(r, c) = {p.b, p.g, p.r};
        }
    cv::imwrite(path, m);
}

void showImage(const std::string& title, const Image& img) {
    cv::Mat m(img.rows, img.cols, CV_8UC3);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c) {
            auto p = img.at(r, c);
            m.at<cv::Vec3b>(r, c) = {p.b, p.g, p.r};
        }
    cv::imshow(title, m);
}

ImageGray toGrayImage(const Image& img) {
    ImageGray g; g.rows = img.rows; g.cols = img.cols;
    g.data.resize(img.rows * img.cols);
    for (int i = 0; i < img.rows*img.cols; ++i)
        g.data[i] = toGray(img.pixels[i].r, img.pixels[i].g, img.pixels[i].b);
    return g;
}

// Segmentation visualiser
Image visualiseSegmentation(const Image& img, const KMeansResult& seg) {
    const std::vector<PixelRGB> pal = {
        {100,200,220},{240,180,100},{180,230,130},{230,140,160},{150,150,240},{200,200,100}
    };
    Image vis = img;
    int N = img.rows * img.cols;
    for (int i = 0; i < N; ++i) {
        int k = seg.labels[i] % (int)pal.size();
        vis.pixels[i].r = (uint8_t)((img.pixels[i].r + pal[k].r) / 2);
        vis.pixels[i].g = (uint8_t)((img.pixels[i].g + pal[k].g) / 2);
        vis.pixels[i].b = (uint8_t)((img.pixels[i].b + pal[k].b) / 2);
    }
    return vis;
}

// Bounding box of a region
struct BBox { int rMin, rMax, cMin, cMax; };

BBox boundingBox(const std::vector<int>& indices, int cols) {
    BBox bb{INT_MAX, INT_MIN, INT_MAX, INT_MIN};
    for (int idx : indices) {
        int r = idx / cols, c = idx % cols;
        bb.rMin = std::min(bb.rMin, r); bb.rMax = std::max(bb.rMax, r);
        bb.cMin = std::min(bb.cMin, c); bb.cMax = std::max(bb.cMax, c);
    }
    return bb;
}

// Extract bounding-box RGB patch as a contiguous rows×cols RGB buffer
// Non-region pixels within the bbox are filled with region's mean colour
RGBBuf extractBBoxRGB(const Image& img, const BBox& bb,
                       const std::vector<int>& indices) {
    int rows = bb.rMax - bb.rMin + 1;
    int cols = bb.cMax - bb.cMin + 1;
    // Fill with grey (128) as neutral background
    RGBBuf buf(rows * cols * 3, 128);
    for (int idx : indices) {
        int r = idx / img.cols - bb.rMin;
        int c = idx % img.cols - bb.cMin;
        int i = r * cols + c;
        buf[i*3+0] = img.pixels[idx].r;
        buf[i*3+1] = img.pixels[idx].g;
        buf[i*3+2] = img.pixels[idx].b;
    }
    return buf;
}

// Write corrected bbox patch back — only at region pixel positions
void writeBBoxRGB(Image& img, const RGBBuf& buf, const BBox& bb,
                   const std::vector<int>& indices) {
    int cols = bb.cMax - bb.cMin + 1;
    for (int idx : indices) {
        int r = idx / img.cols - bb.rMin;
        int c = idx % img.cols - bb.cMin;
        int i = r * cols + c;
        img.pixels[idx].r = buf[i*3+0];
        img.pixels[idx].g = buf[i*3+1];
        img.pixels[idx].b = buf[i*3+2];
    }
}


RegionDiagnosis diagnoseRegion(const RGBBuf& regionBuf, int rows, int cols,
                                bool runDCT) {
    // DFT diagnosis on luminance channel
    auto lumCh = extractChannel(regionBuf, 0); // approx with R; full lum below
    // Recompute proper lum
    int N = rows * cols;
    std::vector<double> lum(N);
    for (int i = 0; i < N; ++i)
        lum[i] = 0.299*regionBuf[i*3] + 0.587*regionBuf[i*3+1] + 0.114*regionBuf[i*3+2];

    ImageGray grayRegion = channelToGray(lum, rows, cols);
    DFTResult dftRes = processDFT(grayRegion, DFTMode::DIAGNOSE_ONLY);

    // Histogram diagnosis
    HistStats hist = computeHistStats(regionBuf);

    // DCT blockiness: average absolute difference across 8-px block boundaries
    double boundaryDiff = 0;
    int count = 0;
    if (runDCT && rows >= BLOCK && cols >= BLOCK) {
        for (int r = BLOCK-1; r+1 < rows; r += BLOCK) {
            for (int c = 0; c < cols; ++c) {
                double diff = std::abs(lum[r*cols+c] - lum[(r+1)*cols+c]);
                boundaryDiff += diff;
                ++count;
            }
        }
        for (int c = BLOCK-1; c+1 < cols; c += BLOCK) {
            for (int r = 0; r < rows; ++r) {
                double diff = std::abs(lum[r*cols+c] - lum[r*cols+(c+1)]);
                boundaryDiff += diff;
                ++count;
            }
        }
        if (count > 0) boundaryDiff /= count;
    }

    return {
        dftRes.isBlurry,
        dftRes.isNoisy,
        hist.underExposed,
        hist.overExposed,
        runDCT && boundaryDiff > 18.0,
        dftRes.scores.blurScore,
        dftRes.scores.noiseScore,
        hist.mean,
        hist.stddev,
        boundaryDiff
    };
}

void printDryRunReport(int k, int regionSize, const BBox& bb,
                        const RegionDiagnosis& d) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Region " << k
              << "  (" << regionSize << " px)"
              << "  bbox [" << bb.rMin << "," << bb.rMax
              << "]×["        << bb.cMin << "," << bb.cMax << "]\n";
    std::cout << "    blurScore   = " << d.blurScore
              << "  → " << (d.blurry       ? "BLURRY"   : "ok") << "\n";
    std::cout << "    noiseScore  = " << d.noiseScore
              << "  → " << (d.noisy        ? "NOISY"    : "ok") << "\n";
    std::cout << "    histMean    = " << d.histMean
              << "  → " << (d.underExposed ? "UNDEREXP" :
                             d.overExposed  ? "OVEREXP"  : "ok") << "\n";
    std::cout << "    boundaryDiff= " << d.boundaryDiff
              << "  → " << (d.blocky       ? "BLOCKY"   : "ok") << "\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./analyzer input.jpg [K=4] [--nodct] [--dryrun]\n";
        return 1;
    }

    std::string inputPath = argv[1];
    int  K       = 4;
    bool runDCT  = true;
    bool dryRun  = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--nodct")  runDCT = false;
        else if (a == "--dryrun") dryRun = true;
        else                      K = std::stoi(a);
    }

    std::cout << "══════════════════════════════════════════════\n";
    std::cout << "  Image Quality Analyzer\n";
    std::cout << "  Input : " << inputPath << "\n";
    std::cout << "  K=" << K << "  DCT=" << (runDCT?"yes":"no")
              << "  mode=" << (dryRun?"DRY-RUN (no pixels changed)":"ENHANCE") << "\n";
    std::cout << "══════════════════════════════════════════════\n\n";

    Image img = loadImage(inputPath);
    Image enhanced = img;  // will be modified only if !dryRun
    std::cout << "Image: " << img.rows << "×" << img.cols << " px\n\n";

    // Stage 2: K-Means
    std::cout << "[ Stage 2 ] K-Means segmentation (K=" << K << ")...\n";
    KMeansResult seg = kmeansSegment(img.pixels, img.rows, img.cols, K);
    if (!dryRun) {
        Image segVis = visualiseSegmentation(img, seg);
        saveImage(segVis, "output_segmentation.jpg");
        std::cout << "  Saved: output_segmentation.jpg\n";
    }
    std::cout << "\n";

    // Stage 1 + 3 + 4 per region
    ImageGray grayOrig = toGrayImage(img);
    std::vector<double> wholeOrig(grayOrig.data.begin(), grayOrig.data.end());

    if (dryRun)
        std::cout << "[ DRY RUN ] Diagnosis scores (no pixels modified):\n\n";

    for (int k = 0; k < K; ++k) {
        std::vector<int> indices = regionIndices(seg, k);
        if (indices.empty()) continue;

        // Bounding box
        BBox bb = boundingBox(indices, img.cols);
        int bbRows = bb.rMax - bb.rMin + 1;
        int bbCols = bb.cMax - bb.cMin + 1;

        // Extract bounding-box RGB patch
        RGBBuf regionBuf = extractBBoxRGB(img, bb, indices);

        // Stage 1: diagnose
        RegionDiagnosis diag = diagnoseRegion(regionBuf, bbRows, bbCols, runDCT);

        if (dryRun) {
            printDryRunReport(k, (int)indices.size(), bb, diag);
            continue;
        }

        std::cout << "[ Stage 1+3 ] Region " << k
                  << " (" << indices.size() << " px) —";
        if (diag.blurry)       std::cout << " BLURRY";
        if (diag.noisy)        std::cout << " NOISY";
        if (diag.underExposed) std::cout << " UNDEREXP";
        if (diag.overExposed)  std::cout << " OVEREXP";
        if (diag.blocky)       std::cout << " BLOCKY";
        if (!diag.blurry && !diag.noisy && !diag.underExposed &&
            !diag.overExposed && !diag.blocky)
            std::cout << " Clean";
        std::cout << "\n";

        // Stage 3: correct (only if something was detected)
        bool needsWork = diag.blurry || diag.noisy ||
                         diag.underExposed || diag.overExposed || diag.blocky;
        if (needsWork) {
            RGBBuf corrected = correctRegionRGB(regionBuf, bbRows, bbCols, diag);
            writeBBoxRGB(enhanced, corrected, bb, indices);
        }

        // Stage 4: evaluate
        ImageGray grayEnh = toGrayImage(enhanced);
        int regionSize = (int)indices.size();
        std::vector<double> rOrig(regionSize), rBefore(regionSize), rAfter(regionSize);
        for (int j = 0; j < regionSize; ++j) {
            rOrig[j]   = grayOrig.data[indices[j]];
            rBefore[j] = grayOrig.data[indices[j]];   // before = original (no separate degraded)
            rAfter[j]  = grayEnh.data[indices[j]];
        }

        std::string diagStr;
        if (diag.blurry)       diagStr += "BLURRY ";
        if (diag.noisy)        diagStr += "NOISY ";
        if (diag.underExposed) diagStr += "UNDEREXP ";
        if (diag.overExposed)  diagStr += "OVEREXP ";
        if (diag.blocky)       diagStr += "BLOCKY ";
        if (diagStr.empty())   diagStr = "Clean";

        RegionMetrics m = evaluateRegion(rOrig, rBefore, rAfter);
        std::cout << formatReport(k, diagStr, m);
    }

    if (dryRun) {
        std::cout << "Dry-run complete. No output written.\n";
        std::cout << "Run without --dryrun to apply corrections.\n";
        return 0;
    }

    saveImage(enhanced, "output_enhanced.jpg");
    std::cout << "\nSaved: output_enhanced.jpg\n\n";

    ImageGray grayEnh = toGrayImage(enhanced);
    std::vector<double> wholeAfter(grayEnh.data.begin(), grayEnh.data.end());
    std::cout << formatSummary(wholeOrig, wholeOrig, wholeAfter);

    showImage("Original", img);
    showImage("Enhanced", enhanced);
    cv::waitKey(0);
    return 0;
}
