#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image.h"
#include "src/stb_image_write.h"

#include "src/dft.hpp"
#include "src/kmeans.hpp"
#include "src/correction.hpp"
#include "src/metrics.hpp"

namespace fs = std::filesystem;

// ── Image type ────────────────────────────────────────────────────────────────

struct Image {
	int rows = 0, cols = 0;
	std::vector<PixelRGB> pixels;
	PixelRGB& at(int r, int c)       { return pixels[r*cols+c]; }
	PixelRGB  at(int r, int c) const { return pixels[r*cols+c]; }
};

Image loadImage(const std::string& path) {
	int w, h, ch;
	uint8_t* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
	if (!data) throw std::runtime_error("cannot open: " + path);
	Image img; img.rows = h; img.cols = w;
	img.pixels.resize(h * w);
	for (int i = 0; i < h*w; ++i)
		img.pixels[i] = { data[i*3], data[i*3+1], data[i*3+2] };
	stbi_image_free(data);
	return img;
}

void saveImage(const Image& img, const std::string& path) {
	std::vector<uint8_t> buf(img.rows * img.cols * 3);
	for (int i = 0; i < img.rows*img.cols; ++i) {
		buf[i*3]   = img.pixels[i].r;
		buf[i*3+1] = img.pixels[i].g;
		buf[i*3+2] = img.pixels[i].b;
	}
	auto ext = fs::path(path).extension().string();
	for (auto& c : ext) c = (char)std::tolower(c);
	int ok = (ext == ".png")
		? stbi_write_png(path.c_str(), img.cols, img.rows, 3, buf.data(), img.cols*3)
		: stbi_write_jpg(path.c_str(), img.cols, img.rows, 3, buf.data(), 95);
	if (!ok) throw std::runtime_error("failed to save: " + path);
}

// Flat RGBBuf view of image pixels (no copy of struct, just raw bytes)
std::vector<uint8_t> imageToRGBBuf(const Image& img) {
	int N = img.rows * img.cols;
	std::vector<uint8_t> buf(N * 3);
	for (int i = 0; i < N; ++i) {
		buf[i*3]   = img.pixels[i].r;
		buf[i*3+1] = img.pixels[i].g;
		buf[i*3+2] = img.pixels[i].b;
	}
	return buf;
}

void rgbBufToImage(const std::vector<uint8_t>& buf, Image& img) {
	int N = img.rows * img.cols;
	for (int i = 0; i < N; ++i)
		img.pixels[i] = { buf[i*3], buf[i*3+1], buf[i*3+2] };
}

ImageGray toGrayImage(const Image& img) {
	ImageGray g; g.rows = img.rows; g.cols = img.cols;
	g.data.resize(img.rows * img.cols);
	for (int i = 0; i < img.rows*img.cols; ++i)
		g.data[i] = toGray(img.pixels[i].r, img.pixels[i].g, img.pixels[i].b);
	return g;
}

// ── Segmentation visualisations ───────────────────────────────────────────────

// Palette — distinct, reasonably saturated colours
static const PixelRGB SEG_PAL[] = {
	{220, 80,  80},   // red
	{ 80,160, 220},   // blue
	{ 80,200, 100},   // green
	{230,180,  50},   // yellow
	{180, 80, 220},   // purple
	{ 50,210, 200},   // teal
	{230,120,  50},   // orange
	{140,200,  80},   // lime
	{220,  80,160},   // pink
	{100,120, 220},   // indigo
	{200,200,  80},   // olive
	{ 80,200,170},    // mint
};
constexpr int PAL_SIZE = (int)(sizeof(SEG_PAL) / sizeof(SEG_PAL[0]));

// Each pixel replaced by its segment's solid palette colour
Image visualiseSegmentationFlat(const Image& img, const KMeansResult& seg) {
	Image vis; vis.rows = img.rows; vis.cols = img.cols;
	vis.pixels.resize(img.rows * img.cols);
	for (int i = 0; i < img.rows*img.cols; ++i)
		vis.pixels[i] = SEG_PAL[seg.labels[i] % PAL_SIZE];
	return vis;
}

// Each pixel is 50/50 blend of original and palette colour
Image visualiseSegmentationOverlay(const Image& img, const KMeansResult& seg) {
	Image vis = img;
	for (int i = 0; i < img.rows*img.cols; ++i) {
		const PixelRGB& c = SEG_PAL[seg.labels[i] % PAL_SIZE];
		vis.pixels[i].r = (uint8_t)((img.pixels[i].r + c.r) / 2);
		vis.pixels[i].g = (uint8_t)((img.pixels[i].g + c.g) / 2);
		vis.pixels[i].b = (uint8_t)((img.pixels[i].b + c.b) / 2);
	}
	return vis;
}

// ── Bounding-box helpers ──────────────────────────────────────────────────────

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

RGBBuf extractBBoxRGB(const Image& img, const BBox& bb,
		const std::vector<int>& indices) {
	int bbCols = bb.cMax - bb.cMin + 1;
	int bbRows = bb.rMax - bb.rMin + 1;
	RGBBuf buf(bbRows * bbCols * 3, 128);
	for (int idx : indices) {
		int r = idx / img.cols - bb.rMin;
		int c = idx % img.cols - bb.cMin;
		int i = r * bbCols + c;
		buf[i*3+0] = img.pixels[idx].r;
		buf[i*3+1] = img.pixels[idx].g;
		buf[i*3+2] = img.pixels[idx].b;
	}
	return buf;
}

void writeBBoxRGB(Image& img, const RGBBuf& buf, const BBox& bb,
		const std::vector<int>& indices) {
	int bbCols = bb.cMax - bb.cMin + 1;
	for (int idx : indices) {
		int r = idx / img.cols - bb.rMin;
		int c = idx % img.cols - bb.cMin;
		int i = r * bbCols + c;
		img.pixels[idx].r = buf[i*3+0];
		img.pixels[idx].g = buf[i*3+1];
		img.pixels[idx].b = buf[i*3+2];
	}
}

// ── Prompt helpers ────────────────────────────────────────────────────────────

static double askDouble(const std::string& prompt, double lo, double hi, double def) {
	while (true) {
		std::cout << prompt << " [" << lo << "-" << hi << ", def " << def << "]: ";
		std::string line;
		std::getline(std::cin, line);
		if (line.empty()) return def;
		try {
			double v = std::stod(line);
			if (v >= lo && v <= hi) return v;
		} catch (...) {}
		std::cout << "invalid, try again\n";
	}
}

static int askInt(const std::string& prompt, int lo, int hi, int def) {
	while (true) {
		std::cout << prompt << " [" << lo << "-" << hi << ", def " << def << "]: ";
		std::string line;
		std::getline(std::cin, line);
		if (line.empty()) return def;
		try {
			int v = std::stoi(line);
			if (v >= lo && v <= hi) return v;
		} catch (...) {}
		std::cout << "invalid, try again\n";
	}
}

static bool askYN(const std::string& prompt, bool def = false) {
	std::cout << prompt << " [" << (def ? "Y/n" : "y/N") << "]: ";
	std::string line;
	std::getline(std::cin, line);
	if (line.empty()) return def;
	return line[0] == 'y' || line[0] == 'Y';
}

static int askMenu(const std::string& title,
		const std::vector<std::string>& options,
		int def = 0) {
	std::cout << title << "\n";
	for (int i = 0; i < (int)options.size(); ++i)
		std::cout << "  " << (i+1) << ") " << options[i] << "\n";
	while (true) {
		std::cout << "> [def " << (def+1) << "]: ";
		std::string line;
		std::getline(std::cin, line);
		if (line.empty()) return def;
		try {
			int v = std::stoi(line);
			if (v >= 1 && v <= (int)options.size()) return v - 1;
		} catch (...) {}
		std::cout << "invalid, try again\n";
	}
}

// ── Effects ───────────────────────────────────────────────────────────────────

struct EffectParams {
	double sharpenAmount    = 0.5;
	double sharpenStrength  = 0.5;
	double denoiseStrength  = 0.5;
	double denoiseCutoff    = 0.35;
	double exposureStrength = 0.9;
	double blockSF          = 0.6;
	double blockStrength    = 0.6;
};

EffectParams askEffectParams(int choice) {
	EffectParams p;
	switch (choice) {
		case 0:
			p.sharpenAmount   = askDouble("  amount (edge emphasis)", 0.1, 2.0, 0.5);
			p.sharpenStrength = askDouble("  strength (blend)", 0.1, 1.0, 0.5);
			break;
		case 1:
			p.denoiseCutoff   = askDouble("  freq cutoff (lower = smoother)", 0.1, 0.8, 0.35);
			p.denoiseStrength = askDouble("  strength (blend)", 0.1, 1.0, 0.5);
			break;
		case 2:
			p.exposureStrength = askDouble("  strength (1.0 = full stretch)", 0.1, 1.0, 0.9);
			break;
		case 3:
			p.blockSF       = askDouble("  hi-freq factor (lower = smoother)", 0.1, 1.0, 0.6);
			p.blockStrength = askDouble("  strength (blend)", 0.1, 1.0, 0.6);
			break;
		default: break;
	}
	return p;
}

RGBBuf applyEffect(const RGBBuf& buf, int rows, int cols,
		int choice, const EffectParams& p) {
	switch (choice) {
		case 0: return sharpenRGB(buf, rows, cols, 0.15, p.sharpenAmount, p.sharpenStrength);
		case 1: return denoiseRGB(buf, rows, cols, p.denoiseCutoff, p.denoiseStrength);
		case 2: return histogramStretchLAB(buf, p.exposureStrength);
		case 3: return removeBlockinessRGB(buf, rows, cols, p.blockSF, p.blockStrength);
		default: return buf;
	}
}

// ── Segmentation output helper ────────────────────────────────────────────────

// Ask which seg visualisation(s) to save and do it.
// stem is the output filename without extension.
void saveSegVisuals(const Image& img, const KMeansResult& seg,
		const std::string& stem) {
	int choice = askMenu("segmentation output:",
			{"flat colours only",
			"overlay on original only",
			"both",
			"skip"},
			0);
	if (choice == 3) return;

	if (choice == 0 || choice == 2) {
		std::string p = stem + "_seg_flat.jpg";
		saveImage(visualiseSegmentationFlat(img, seg), p);
		std::cout << "saved: " << p << "\n";
	}
	if (choice == 1 || choice == 2) {
		std::string p = stem + "_seg_overlay.jpg";
		saveImage(visualiseSegmentationOverlay(img, seg), p);
		std::cout << "saved: " << p << "\n";
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
	std::cout << "image enhancer\n--------------\n";

	// Input: $1 or prompt
	std::string inputPath;
	if (argc >= 2) {
		inputPath = argv[1];
		if (!fs::exists(inputPath)) {
			std::cerr << "error: file not found: " << inputPath << "\n";
			return 1;
		}
	} else {
		while (true) {
			std::cout << "input image: ";
			std::getline(std::cin, inputPath);
			if (fs::exists(inputPath)) break;
			std::cout << "file not found, try again\n";
		}
	}

	Image img;
	try { img = loadImage(inputPath); }
	catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
	std::cout << "loaded " << img.cols << "x" << img.rows << " px\n";

	// Mode
	int mode = askMenu("mode:",
			{"manual  - pick effects yourself",
			"auto    - diagnose and fix automatically",
			"diagnose only - no changes",
			"segmentation view - just show regions"},
			0);

	// Output: $2 or prompt (not needed for diagnose-only or seg-view)
	std::string outPath;
	bool needsOutput = (mode == 0 || mode == 1);
	if (needsOutput) {
		if (argc >= 3) {
			outPath = argv[2];
		} else {
			std::cout << "output image [def enhanced.jpg]: ";
			std::getline(std::cin, outPath);
			if (outPath.empty()) outPath = "enhanced.jpg";
		}
	}

	// ── MANUAL ───────────────────────────────────────────────────────────────
	if (mode == 0) {
		RGBBuf buf = imageToRGBBuf(img);

		const std::vector<std::string> effectNames = {
			"sharpen",
			"denoise (DFT low-pass)",
			"exposure fix (histogram stretch)",
			"remove blockiness (DCT)",
			"done - save and exit"
		};

		while (true) {
			int choice = askMenu("effect:", effectNames, 4);
			if (choice == 4) break;
			EffectParams p = askEffectParams(choice);
			std::cout << "applying...";
			std::cout.flush();
			buf = applyEffect(buf, img.rows, img.cols, choice, p);
			std::cout << " done\n";
		}

		rgbBufToImage(buf, img);
		saveImage(img, outPath);
		std::cout << "saved: " << outPath << "\n";
		return 0;
	}

	// ── All other modes need segmentation ─────────────────────────────────────
	int K       = askInt("segments (K)", 2, 12, 4);
	bool runDCT = (mode != 3) && askYN("check for JPEG blockiness?", true);

	std::cout << "segmenting...";
	std::cout.flush();
	KMeansResult seg = kmeansSegment(img.pixels, img.rows, img.cols, K);
	std::cout << " done\n";

	// ── SEGMENTATION VIEW ─────────────────────────────────────────────────────
	if (mode == 3) {
		std::string stem = fs::path(inputPath).stem().string();
		saveSegVisuals(img, seg, stem);
		return 0;
	}

	// ── AUTO or DIAGNOSE — optionally save seg visuals ────────────────────────
	std::string outStem = fs::path(outPath).stem().string();
	if (askYN("save segmentation visuals?", false))
		saveSegVisuals(img, seg, outStem);

	// ── Per-region processing ─────────────────────────────────────────────────
	Image enhanced      = img;
	std::vector<uint8_t> origRGB = imageToRGBBuf(img);
	std::cout << "\n____________DIAGNOSIS____________\n";
	for (int k = 0; k < K; ++k) {
		std::vector<int> indices = regionIndices(seg, k);
		if (indices.empty()) continue;

		BBox bb    = boundingBox(indices, img.cols);
		int bbRows = bb.rMax - bb.rMin + 1;
		int bbCols = bb.cMax - bb.cMin + 1;

		RGBBuf regionBuf     = extractBBoxRGB(img, bb, indices);
		RegionDiagnosis diag = diagnoseRegion(regionBuf, bbRows, bbCols, runDCT);

		// Grayscale values for this region (for SSIM + entropy)
		std::vector<double> grayRegion(indices.size());
		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			grayRegion[i] = 0.299*origRGB[idx*3] + 0.587*origRGB[idx*3+1] + 0.114*origRGB[idx*3+2];
		}
		double regionSSIM    = computeSSIM(grayRegion, grayRegion); // self = 1.0, useful after correction
		double regionEntropy = computeEntropy(grayRegion);

		std::string diagStr;
		if (diag.blurry)       diagStr += "BLURRY ";
		if (diag.noisy)        diagStr += "NOISY ";
		if (diag.underExposed) diagStr += "UNDEREXP ";
		if (diag.overExposed)  diagStr += "OVEREXP ";
		if (diag.blocky)       diagStr += "BLOCKY ";
		if (diagStr.empty())   diagStr = "clean";

		std::cout << std::fixed << std::setprecision(4);
		std::cout << "\nRegion " << k << " (" << indices.size() << " px): " << diagStr;
		std::cout << "\nBLUR     : " << diag.blurScore
			<< "\nNOISE    : " << diag.noiseScore
			<< "\nMEAN     : " << diag.histMean
			<< "\nSSIM     : " << regionSSIM
			<< "\nENTROPY  : " << regionEntropy << " bits/px";
		if (runDCT) std::cout << "\nBOUNDARY : " << diag.boundaryDiff;
		std::cout << "\n";

		if (mode == 2) continue;

		bool needsWork = diag.blurry || diag.noisy ||
			diag.underExposed || diag.overExposed || diag.blocky;
		if (needsWork) {
			RGBBuf corrected = correctRegionRGB(regionBuf, bbRows, bbCols, diag);
			writeBBoxRGB(enhanced, corrected, bb, indices);
		}

		std::vector<uint8_t> afterRGB = imageToRGBBuf(enhanced);
		std::cout << formatReport(k, diagStr, evaluateRegion(origRGB, afterRGB, indices));
	}

	if (mode == 2) {
		std::cout << "\n\ndiagnosis complete, nothing written\n";
		return 0;
	}

	std::vector<uint8_t> afterRGB = imageToRGBBuf(enhanced);
	std::cout << formatSummary(origRGB, afterRGB);
	saveImage(enhanced, outPath);
	std::cout << "saved: " << outPath << "\n";
	return 0;
}
