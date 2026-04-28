#include "correction.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

constexpr int    BLOCK = 8;
constexpr double PI_C = 3.14159265358979323846;

// ── Channel helpers ───────────────────────────────────────────────────────────

std::vector<double> extractChannel(const RGBBuf& buf, int ch) {
    int N = (int)buf.size() / 3;
    std::vector<double> out(N);
    for (int i = 0; i < N; ++i) out[i] = buf[i*3 + ch];
    return out;
}

void writeChannel(RGBBuf& buf, const std::vector<double>& ch, int idx) {
    int N = (int)buf.size() / 3;
    for (int i = 0; i < N; ++i)
        buf[i*3 + idx] = (uint8_t)std::clamp((int)std::round(ch[i]), 0, 255);
}

std::vector<double> blendCh(const std::vector<double>& orig,
                             const std::vector<double>& corr, double strength) {
    std::vector<double> out(orig.size());
    for (size_t i = 0; i < orig.size(); ++i)
        out[i] = std::clamp(orig[i]*(1.0-strength) + corr[i]*strength, 0.0, 255.0);
    return out;
}

// ── Histogram stats ───────────────────────────────────────────────────────────

HistStats computeHistStats(const RGBBuf& pixels) {
    int N = (int)pixels.size() / 3;
    std::vector<double> lums(N);
    double sum = 0;
    for (int i = 0; i < N; ++i) {
        lums[i] = 0.299*pixels[i*3] + 0.587*pixels[i*3+1] + 0.114*pixels[i*3+2];
        sum += lums[i];
    }
    double mean = sum / N;
    double var  = 0;
    for (double l : lums) var += (l-mean)*(l-mean);
    double stddev = std::sqrt(var/N);

    auto sorted = lums;
    std::sort(sorted.begin(), sorted.end());
    double p2  = sorted[std::max(0, (int)(0.02*N))];
    double p98 = sorted[std::min(N-1, (int)(0.98*N))];

    return {mean, stddev, p2, p98,
            mean < 60.0  && p98 < 180.0,
            mean > 195.0 && p2  > 80.0};
}

// ── Histogram stretch in YCbCr ────────────────────────────────────────────────

RGBBuf histogramStretchLAB(const RGBBuf& pixels, double strength) {
    int N = (int)pixels.size() / 3;
    std::vector<double> Y(N), Cb(N), Cr(N);
    for (int i = 0; i < N; ++i) {
        double r = pixels[i*3], g = pixels[i*3+1], b = pixels[i*3+2];
        Y[i]  =  0.299*r + 0.587*g + 0.114*b;
        Cb[i] = -0.168736*r - 0.331264*g + 0.5*b + 128.0;
        Cr[i] =  0.5*r - 0.418688*g - 0.081312*b + 128.0;
    }

    auto sv = Y; std::sort(sv.begin(), sv.end());
    double lo = sv[std::max(0, (int)(0.01*N))];
    double hi = sv[std::min(N-1, (int)(0.99*N))];

    if (hi - lo < 30.0) {
        double mid = (hi + lo) / 2.0;
        lo = std::max(0.0,   mid - 60.0);
        hi = std::min(255.0, mid + 60.0);
    }

    std::vector<double> Ys(N);
    for (int i = 0; i < N; ++i)
        Ys[i] = std::clamp((Y[i] - lo) / (hi - lo) * 255.0, 0.0, 255.0);

    std::vector<double> Yb(N);
    for (int i = 0; i < N; ++i)
        Yb[i] = Y[i]*(1.0-strength) + Ys[i]*strength;

    RGBBuf out(pixels.size());
    for (int i = 0; i < N; ++i) {
        double y = Yb[i], cb = Cb[i]-128.0, cr = Cr[i]-128.0;
        out[i*3]   = (uint8_t)std::clamp(y + 1.402*cr,                   0.0, 255.0);
        out[i*3+1] = (uint8_t)std::clamp(y - 0.344136*cb - 0.714136*cr,  0.0, 255.0);
        out[i*3+2] = (uint8_t)std::clamp(y + 1.772*cb,                   0.0, 255.0);
    }
    return out;
}

// ── Sharpening (unsharp mask via FFT) ────────────────────────────────────────

static ImageGray channelToGray(const std::vector<double>& ch, int rows, int cols) {
    ImageGray g; g.rows = rows; g.cols = cols; g.data = ch; return g;
}

static std::vector<double> unsharpChannel(const std::vector<double>& ch,
                                           int rows, int cols,
                                           double blurCutoff, double amount) {
    ImageGray img    = channelToGray(ch, rows, cols);
    CMatrix padded   = toCMatrix(img);
    CMatrix dft      = dft2d(padded, false);
    CMatrix shifted  = fftShift(dft);
    CMatrix lp       = applyLowPassMask(shifted, blurCutoff);
    CMatrix back     = ifftShift(lp);
    CMatrix blurred  = dft2d(back, true);
    ImageGray blurImg = toImageGray(blurred, rows, cols);

    std::vector<double> out(ch.size());
    for (size_t i = 0; i < ch.size(); ++i)
        out[i] = std::clamp(ch[i] + amount*(ch[i] - blurImg.data[i]), 0.0, 255.0);
    return out;
}

RGBBuf sharpenRGB(const RGBBuf& pixels, int rows, int cols,
                   double blurCutoff, double amount, double strength) {
    RGBBuf out = pixels;
    for (int ch = 0; ch < 3; ++ch) {
        auto orig      = extractChannel(pixels, ch);
        auto sharpened = unsharpChannel(orig, rows, cols, blurCutoff, amount);
        writeChannel(out, blendCh(orig, sharpened, strength), ch);
    }
    return out;
}

// ── DFT denoising ─────────────────────────────────────────────────────────────

RGBBuf denoiseRGB(const RGBBuf& pixels, int rows, int cols,
                   double cutoffRatio, double strength) {
    RGBBuf out = pixels;
    for (int ch = 0; ch < 3; ++ch) {
        auto orig     = extractChannel(pixels, ch);
        ImageGray img = channelToGray(orig, rows, cols);
        CMatrix padded  = toCMatrix(img);
        CMatrix dft_m   = dft2d(padded, false);
        CMatrix shifted = fftShift(dft_m);
        CMatrix lp      = applyLowPassMask(shifted, cutoffRatio);
        CMatrix back    = ifftShift(lp);
        CMatrix recon   = dft2d(back, true);
        ImageGray denoised = toImageGray(recon, rows, cols);
        writeChannel(out, blendCh(orig, denoised.data, strength), ch);
    }
    return out;
}

// ── DCT blockiness removal ────────────────────────────────────────────────────

static void dct1d_8(const double in[BLOCK], double out[BLOCK]) {
    for (int u = 0; u < BLOCK; ++u) {
        double cu  = (u==0) ? 1.0/std::sqrt(2.0) : 1.0;
        double sum = 0;
        for (int x = 0; x < BLOCK; ++x)
            sum += in[x]*std::cos(PI_C*(2*x+1)*u/(2.0*BLOCK));
        out[u] = cu*sum*(2.0/BLOCK);
    }
}

static void idct1d_8(const double in[BLOCK], double out[BLOCK]) {
    for (int x = 0; x < BLOCK; ++x) {
        double sum = in[0]/std::sqrt(2.0);
        for (int u = 1; u < BLOCK; ++u)
            sum += in[u]*std::cos(PI_C*(2*x+1)*u/(2.0*BLOCK));
        out[x] = sum;
    }
}

static void dct2d_8x8(double block[BLOCK][BLOCK]) {
    double tmp[BLOCK][BLOCK];
    for (int r = 0; r < BLOCK; ++r) dct1d_8(block[r], tmp[r]);
    for (int c = 0; c < BLOCK; ++c) {
        double ci[BLOCK], co[BLOCK];
        for (int r = 0; r < BLOCK; ++r) ci[r] = tmp[r][c];
        dct1d_8(ci, co);
        for (int r = 0; r < BLOCK; ++r) block[r][c] = co[r];
    }
}

static void idct2d_8x8(double block[BLOCK][BLOCK]) {
    double tmp[BLOCK][BLOCK];
    for (int c = 0; c < BLOCK; ++c) {
        double ci[BLOCK], co[BLOCK];
        for (int r = 0; r < BLOCK; ++r) ci[r] = block[r][c];
        idct1d_8(ci, co);
        for (int r = 0; r < BLOCK; ++r) tmp[r][c] = co[r];
    }
    for (int r = 0; r < BLOCK; ++r) idct1d_8(tmp[r], block[r]);
}

static std::vector<double> dctSmoothChannel(const std::vector<double>& ch,
                                              int rows, int cols, double sf) {
    std::vector<double> out = ch;
    for (int r = 0; r+BLOCK <= rows; r += BLOCK) {
        for (int c = 0; c+BLOCK <= cols; c += BLOCK) {
            double blk[BLOCK][BLOCK];
            for (int br = 0; br < BLOCK; ++br)
                for (int bc = 0; bc < BLOCK; ++bc)
                    blk[br][bc] = ch[(r+br)*cols+(c+bc)] - 128.0;
            dct2d_8x8(blk);
            for (int u = 0; u < BLOCK; ++u)
                for (int v = 0; v < BLOCK; ++v)
                    if (u+v >= BLOCK) blk[u][v] *= sf;
            idct2d_8x8(blk);
            for (int br = 0; br < BLOCK; ++br)
                for (int bc = 0; bc < BLOCK; ++bc)
                    out[(r+br)*cols+(c+bc)] = std::clamp(blk[br][bc]+128.0, 0.0, 255.0);
        }
    }
    return out;
}

RGBBuf removeBlockinessRGB(const RGBBuf& pixels, int rows, int cols,
                             double sf, double strength) {
    RGBBuf out = pixels;
    for (int ch = 0; ch < 3; ++ch) {
        auto orig     = extractChannel(pixels, ch);
        auto smoothed = dctSmoothChannel(orig, rows, cols, sf);
        writeChannel(out, blendCh(orig, smoothed, strength), ch);
    }
    return out;
}

// ── Diagnosis ─────────────────────────────────────────────────────────────────

RegionDiagnosis diagnoseRegion(const RGBBuf& regionBuf, int rows, int cols, bool runDCT) {
    int N = rows * cols;
    std::vector<double> lum(N);
    for (int i = 0; i < N; ++i)
        lum[i] = 0.299*regionBuf[i*3] + 0.587*regionBuf[i*3+1] + 0.114*regionBuf[i*3+2];

    ImageGray grayRegion;
    grayRegion.rows = rows; grayRegion.cols = cols; grayRegion.data = lum;
    DFTResult dftRes = processDFT(grayRegion, DFTMode::DIAGNOSE_ONLY);
    HistStats hist   = computeHistStats(regionBuf);

    double boundaryDiff = 0;
    int count = 0;
    if (runDCT && rows >= BLOCK && cols >= BLOCK) {
        for (int r = BLOCK-1; r+1 < rows; r += BLOCK)
            for (int c = 0; c < cols; ++c) {
                boundaryDiff += std::abs(lum[r*cols+c] - lum[(r+1)*cols+c]);
                ++count;
            }
        for (int c = BLOCK-1; c+1 < cols; c += BLOCK)
            for (int r = 0; r < rows; ++r) {
                boundaryDiff += std::abs(lum[r*cols+c] - lum[r*cols+(c+1)]);
                ++count;
            }
        if (count > 0) boundaryDiff /= count;
    }

    return {
        dftRes.isBlurry, dftRes.isNoisy,
        hist.underExposed, hist.overExposed,
        runDCT && boundaryDiff > 18.0,
        dftRes.scores.blurScore, dftRes.scores.noiseScore,
        hist.mean, hist.stddev, boundaryDiff
    };
}

// ── Auto-correction ───────────────────────────────────────────────────────────

RGBBuf correctRegionRGB(const RGBBuf& pixels, int rows, int cols,
                         const RegionDiagnosis& diag) {
    RGBBuf result = pixels;
    if (diag.blocky)
        result = removeBlockinessRGB(result, rows, cols, 0.6, 0.6);
    if (diag.noisy)
        result = denoiseRGB(result, rows, cols, 0.35, 0.5);
    else if (diag.blurry)
        result = sharpenRGB(result, rows, cols, 0.15, 0.5, 0.5);
    if (diag.underExposed || diag.overExposed)
        result = histogramStretchLAB(result, 0.9);
    return result;
}
