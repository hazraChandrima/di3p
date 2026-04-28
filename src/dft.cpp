#include "dft.hpp"
#include <stdexcept>
#include <numeric>

int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

double toGray(uint8_t r, uint8_t g, uint8_t b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

void bitReversePermute(CRow& x) {
    int N = (int)x.size();
    int bits = 0;
    while ((1 << bits) < N) ++bits;
    for (int i = 0; i < N; ++i) {
        int rev = 0;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b)) rev |= (1 << (bits - 1 - b));
        if (i < rev) std::swap(x[i], x[rev]);
    }
}

void fft1d_inplace(CRow& x, bool forward) {
    int N = (int)x.size();
    bitReversePermute(x);
    for (int len = 2; len <= N; len <<= 1) {
        double angle = (forward ? -1.0 : +1.0) * 2.0 * PI / len;
        Complex W(std::cos(angle), std::sin(angle));
        for (int i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                Complex u = x[i + j];
                Complex t = w * x[i + j + len / 2];
                x[i + j]         = u + t;
                x[i + j + len/2] = u - t;
                w *= W;
            }
        }
    }
    if (!forward)
        for (auto& v : x) v /= double(N);
}

CRow dft1d(const CRow& x, bool inverse) {
    CRow y = x;
    int N = (int)y.size();
    int P = nextPow2(N);
    if (P != N) y.resize(P, {0.0, 0.0});
    fft1d_inplace(y, !inverse);
    return y;
}

CMatrix dft2d(const CMatrix& in, bool inverse) {
    int rows = (int)in.size();
    int cols = (int)in[0].size();
    CMatrix out(rows, CRow(cols));
    for (int r = 0; r < rows; ++r)
        out[r] = dft1d(in[r], inverse);
    for (int c = 0; c < cols; ++c) {
        CRow col(rows);
        for (int r = 0; r < rows; ++r) col[r] = out[r][c];
        CRow colT = dft1d(col, inverse);
        for (int r = 0; r < rows; ++r) out[r][c] = colT[r];
    }
    return out;
}

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

RealMatrix magnitudeSpectrum(const CMatrix& dft) {
    int rows = (int)dft.size(), cols = (int)dft[0].size();
    RealMatrix mag(rows, std::vector<double>(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mag[r][c] = std::log(1.0 + std::abs(dft[r][c]));
    return mag;
}

DFTScores computeScores(const CMatrix& shifted, double cutoffRatio) {
    int rows = (int)shifted.size(), cols = (int)shifted[0].size();
    int cr = rows / 2, cc = cols / 2;
    double maxR  = std::sqrt((double)(cr * cr + cc * cc));
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

CMatrix applyLowPassMask(const CMatrix& shifted, double cutoffRatio) {
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

CMatrix applyHighBoostMask(const CMatrix& shifted, double cutoffRatio, double boostFactor) {
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

DFTResult processDFT(const ImageGray& img, DFTMode mode,
                     double cutoffRatio, double boostFactor,
                     double blurThreshold, double noiseThreshold) {
    CMatrix padded       = toCMatrix(img);
    CMatrix spectrum     = dft2d(padded, false);
    CMatrix shifted      = fftShift(spectrum);
    DFTScores scores     = computeScores(shifted, cutoffRatio);

    CMatrix masked = shifted;
    if      (mode == DFTMode::DENOISE) masked = applyLowPassMask(shifted, cutoffRatio);
    else if (mode == DFTMode::SHARPEN) masked = applyHighBoostMask(shifted, cutoffRatio, boostFactor);

    CMatrix unshifted     = ifftShift(masked);
    CMatrix reconstructed = dft2d(unshifted, true);

    return {
        toImageGray(reconstructed, img.rows, img.cols),
        scores,
        scores.blurScore  < blurThreshold,
        scores.noiseScore < noiseThreshold
    };
}
