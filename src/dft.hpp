#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <algorithm>

using Complex    = std::complex<double>;
using CRow       = std::vector<Complex>;
using CMatrix    = std::vector<CRow>;
using RealMatrix = std::vector<std::vector<double>>;

constexpr double PI = 3.14159265358979323846;

struct ImageGray {
    int rows, cols;
    std::vector<double> data;
    double& at(int r, int c)       { return data[r * cols + c]; }
    double  at(int r, int c) const { return data[r * cols + c]; }
};

struct DFTScores {
    double blurScore;
    double noiseScore;
};

struct DFTResult {
    ImageGray processed;
    DFTScores scores;
    bool      isBlurry;
    bool      isNoisy;
};

enum class DFTMode { DENOISE, SHARPEN, DIAGNOSE_ONLY };

int     nextPow2(int n);
double  toGray(uint8_t r, uint8_t g, uint8_t b);

void    bitReversePermute(CRow& x);
void    fft1d_inplace(CRow& x, bool forward);
CRow    dft1d(const CRow& x, bool inverse = false);
CMatrix dft2d(const CMatrix& in, bool inverse = false);
CMatrix fftShift(const CMatrix& in);
CMatrix ifftShift(const CMatrix& in);

CMatrix   toCMatrix(const ImageGray& img);
ImageGray toImageGray(const CMatrix& m, int origRows, int origCols);

RealMatrix magnitudeSpectrum(const CMatrix& dft);
DFTScores  computeScores(const CMatrix& shifted, double cutoffRatio = 0.3);

CMatrix applyLowPassMask(const CMatrix& shifted, double cutoffRatio = 0.3);
CMatrix applyHighBoostMask(const CMatrix& shifted, double cutoffRatio = 0.3, double boostFactor = 1.8);

DFTResult processDFT(const ImageGray& img,
                     DFTMode mode         = DFTMode::DIAGNOSE_ONLY,
                     double cutoffRatio   = 0.3,
                     double boostFactor   = 1.8,
                     double blurThreshold = 0.05,
                     double noiseThreshold = 0.01);
