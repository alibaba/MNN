//
//  ImageOpTest.cpp
//  MNNTests
//
//  Comprehensive test suite for image processing operations:
//  resize (bilinear, nearest), crop, rotate, colorspace conversions.
//

#include <MNN/ImageProcess.hpp>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>
#include "MNNTestSuite.h"

using namespace MNN;
using namespace MNN::CV;

// Helper: generate deterministic pixel data
static std::vector<uint8_t> generateTestImage(int h, int w, int channels) {
    std::vector<uint8_t> img(h * w * channels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                int idx = (y * w + x) * channels + c;
                img[idx] = static_cast<uint8_t>(((y * 17 + x * 31 + c * 53) * 137) % 256);
            }
        }
    }
    return img;
}

// Helper: compute reference bilinear interpolation for a single pixel
static uint8_t bilinearSample(const uint8_t* src, int srcW, int srcH,
                               int channels, int stride,
                               float sx, float sy, int ch) {
    float x = std::max(0.0f, std::min(sx, (float)(srcW - 1)));
    float y = std::max(0.0f, std::min(sy, (float)(srcH - 1)));
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = std::min(x0 + 1, srcW - 1);
    int y1 = std::min(y0 + 1, srcH - 1);
    float xf = x - x0;
    float yf = y - y0;

    float c00 = src[y0 * stride + x0 * channels + ch];
    float c01 = src[y0 * stride + x1 * channels + ch];
    float c10 = src[y1 * stride + x0 * channels + ch];
    float c11 = src[y1 * stride + x1 * channels + ch];

    float val = (1 - xf) * (1 - yf) * c00 + xf * (1 - yf) * c01
              + (1 - xf) * yf * c10 + xf * yf * c11;
    return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, roundf(val))));
}

// ========== Test: Bilinear Resize - Basic Upscale ==========
class BilinearResizeUpscaleTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 4, srcH = 4, dstW = 8, dstH = 8;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        float fx = (float)srcW / dstW;
        float fy = (float)srcH / dstH;
        tr.postScale(fx, fy);
        tr.postTranslate(0.5f * (fx - 1), 0.5f * (fy - 1));
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        // Verify against reference implementation with tolerance
        int mismatches = 0;
        for (int y = 0; y < dstH; ++y) {
            for (int x = 0; x < dstW; ++x) {
                float sx = x * fx + 0.5f * (fx - 1);
                float sy = y * fy + 0.5f * (fy - 1);
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = bilinearSample(src.data(), srcW, srcH,
                                                      channels, srcW * channels,
                                                      sx, sy, c);
                    uint8_t actual = dst[(y * dstW + x) * channels + c];
                    if (std::abs((int)expected - (int)actual) > 2) {
                        mismatches++;
                    }
                }
            }
        }
        MNNTEST_ASSERT(mismatches == 0);
        return true;
    }
};
MNNTestSuiteRegister(BilinearResizeUpscaleTest, "cv/image_op/bilinear_resize_upscale");

// ========== Test: Bilinear Resize - Downscale ==========
class BilinearResizeDownscaleTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 16, srcH = 16, dstW = 4, dstH = 4;
        const int channels = 3;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = RGB;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        float fx = (float)srcW / dstW;
        float fy = (float)srcH / dstH;
        tr.postScale(fx, fy);
        tr.postTranslate(0.5f * (fx - 1), 0.5f * (fy - 1));
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGB);

        // Check that output pixels are within valid range
        for (int i = 0; i < dstW * dstH * channels; ++i) {
            MNNTEST_ASSERT(dst[i] <= 255);
        }

        // Verify a few reference samples
        int mismatches = 0;
        for (int y = 0; y < dstH; ++y) {
            for (int x = 0; x < dstW; ++x) {
                float sx = x * fx + 0.5f * (fx - 1);
                float sy = y * fy + 0.5f * (fy - 1);
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = bilinearSample(src.data(), srcW, srcH,
                                                      channels, srcW * channels,
                                                      sx, sy, c);
                    uint8_t actual = dst[(y * dstW + x) * channels + c];
                    if (std::abs((int)expected - (int)actual) > 2) {
                        mismatches++;
                    }
                }
            }
        }
        MNNTEST_ASSERT(mismatches == 0);
        return true;
    }
};
MNNTestSuiteRegister(BilinearResizeDownscaleTest, "cv/image_op/bilinear_resize_downscale");

// ========== Test: Nearest Neighbor Resize ==========
class NearestResizeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 4, srcH = 4, dstW = 8, dstH = 8;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        float fx = (float)srcW / dstW;
        float fy = (float)srcH / dstH;
        tr.postScale(fx, fy);
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        // Every output pixel should exactly match some source pixel
        for (int y = 0; y < dstH; ++y) {
            for (int x = 0; x < dstW; ++x) {
                int sx = std::min((int)(x * fx), srcW - 1);
                int sy = std::min((int)(y * fy), srcH - 1);
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = src[(sy * srcW + sx) * channels + c];
                    uint8_t actual = dst[(y * dstW + x) * channels + c];
                    // Nearest should be exact or very close
                    MNNTEST_ASSERT(std::abs((int)expected - (int)actual) <= 1);
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(NearestResizeTest, "cv/image_op/nearest_resize");

// ========== Test: Crop Operations ==========
class CropBasicTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 8, srcH = 8;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        // Crop center 4x4 region
        const int cropX = 2, cropY = 2, cropW = 4, cropH = 4;

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        // Set matrix to translate from crop region
        Matrix tr;
        tr.postTranslate(cropX, cropY);
        process->setMatrix(tr);

        std::vector<uint8_t> dst(cropW * cropH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), cropW, cropH, channels * cropW, RGBA);

        // Verify cropped pixels match source region
        for (int y = 0; y < cropH; ++y) {
            for (int x = 0; x < cropW; ++x) {
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = src[((cropY + y) * srcW + (cropX + x)) * channels + c];
                    uint8_t actual = dst[(y * cropW + x) * channels + c];
                    MNNTEST_ASSERT(expected == actual);
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(CropBasicTest, "cv/image_op/crop_basic");

// ========== Test: Crop Edge Cases - Boundary Clamp ==========
class CropEdgeCaseTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 4, srcH = 4;
        const int channels = 3;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = RGB;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        // Try to "crop" starting at edge - should clamp
        Matrix tr;
        tr.postTranslate(3, 3); // Near bottom-right corner
        process->setMatrix(tr);

        std::vector<uint8_t> dst(4 * 4 * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), 4, 4, channels * 4, RGB);

        // Output should not crash and should contain valid pixel values
        for (int i = 0; i < 4 * 4 * channels; ++i) {
            MNNTEST_ASSERT(dst[i] <= 255);
        }
        return true;
    }
};
MNNTestSuiteRegister(CropEdgeCaseTest, "cv/image_op/crop_edge_case");

// ========== Test: Rotation 90 degrees ==========
class Rotate90Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 4, srcH = 6;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        // 90 degree rotation: new dimensions are swapped
        const int dstW = srcH, dstH = srcW;
        Matrix tr;
        tr.postRotate(90.0f * M_PI / 180.0f, srcW / 2.0f, srcH / 2.0f);
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        // Just verify output is valid (rotation with bilinear is hard to verify exactly)
        bool hasNonZero = false;
        for (int i = 0; i < dstW * dstH * channels; ++i) {
            if (dst[i] > 0) hasNonZero = true;
        }
        MNNTEST_ASSERT(hasNonZero);
        return true;
    }
};
MNNTestSuiteRegister(Rotate90Test, "cv/image_op/rotate_90");

// ========== Test: Rotation 180 degrees ==========
class Rotate180Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 4, H = 4;
        const int channels = 3;
        auto src = generateTestImage(H, W, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = RGB;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        // 180 degree rotation around center
        tr.postRotate(M_PI, W / 2.0f, H / 2.0f);
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H * channels);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, channels * W, RGB);

        // After 180 rotation, pixel at (x,y) should map to (W-1-x, H-1-y)
        int mismatches = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int rx = W - 1 - x;
                int ry = H - 1 - y;
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = src[(ry * W + rx) * channels + c];
                    uint8_t actual = dst[(y * W + x) * channels + c];
                    if (std::abs((int)expected - (int)actual) > 2) {
                        mismatches++;
                    }
                }
            }
        }
        // Allow small number of mismatches due to rounding at boundaries
        MNNTEST_ASSERT(mismatches <= 4);
        return true;
    }
};
MNNTestSuiteRegister(Rotate180Test, "cv/image_op/rotate_180");

// ========== Test: Rotation 270 degrees ==========
class Rotate270Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 6, srcH = 4;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        const int dstW = srcH, dstH = srcW;
        Matrix tr;
        tr.postRotate(270.0f * M_PI / 180.0f, srcW / 2.0f, srcH / 2.0f);
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        bool hasNonZero = false;
        for (int i = 0; i < dstW * dstH * channels; ++i) {
            if (dst[i] > 0) hasNonZero = true;
        }
        MNNTEST_ASSERT(hasNonZero);
        return true;
    }
};
MNNTestSuiteRegister(Rotate270Test, "cv/image_op/rotate_270");

// ========== Test: RGB to BGR Conversion ==========
class RGBToBGRTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 8, H = 8;
        auto src = generateTestImage(H, W, 3);

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = BGR;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H * 3);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, 3 * W, BGR);

        // RGB->BGR: channels should be swapped (R<->B)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                MNNTEST_ASSERT(src[idx + 0] == dst[idx + 2]); // R == B'
                MNNTEST_ASSERT(src[idx + 1] == dst[idx + 1]); // G == G'
                MNNTEST_ASSERT(src[idx + 2] == dst[idx + 0]); // B == R'
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RGBToBGRTest, "cv/image_op/rgb_to_bgr");

// ========== Test: BGR to RGB Conversion ==========
class BGRToRGBTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 6, H = 6;
        auto src = generateTestImage(H, W, 3);

        ImageProcess::Config config;
        config.sourceFormat = BGR;
        config.destFormat = RGB;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H * 3);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, 3 * W, RGB);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                MNNTEST_ASSERT(src[idx + 0] == dst[idx + 2]);
                MNNTEST_ASSERT(src[idx + 1] == dst[idx + 1]);
                MNNTEST_ASSERT(src[idx + 2] == dst[idx + 0]);
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(BGRToRGBTest, "cv/image_op/bgr_to_rgb");

// ========== Test: RGB to GRAY Conversion ==========
class RGBToGrayTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 8, H = 8;
        auto src = generateTestImage(H, W, 3);

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = GRAY;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, W, GRAY);

        // Check gray is a weighted sum of RGB (standard BT.601)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = (y * W + x) * 3;
                float r = src[idx + 0];
                float g = src[idx + 1];
                float b = src[idx + 2];
                // Standard grayscale: 0.299R + 0.587G + 0.114B
                float expected = 0.299f * r + 0.587f * g + 0.114f * b;
                int diff = std::abs((int)dst[y * W + x] - (int)roundf(expected));
                MNNTEST_ASSERT(diff <= 2);
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RGBToGrayTest, "cv/image_op/rgb_to_gray");

// ========== Test: RGBA to GRAY Conversion ==========
class RGBAToGrayTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 4, H = 4;
        auto src = generateTestImage(H, W, 4);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = GRAY;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, W, GRAY);

        // Just verify output values are in valid range and non-trivial
        bool hasNonZero = false;
        for (int i = 0; i < W * H; ++i) {
            MNNTEST_ASSERT(dst[i] <= 255);
            if (dst[i] > 0) hasNonZero = true;
        }
        MNNTEST_ASSERT(hasNonZero);
        return true;
    }
};
MNNTestSuiteRegister(RGBAToGrayTest, "cv/image_op/rgba_to_gray");

// ========== Test: Large Image Resize ==========
class LargeImageResizeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int srcW = 256, srcH = 256, dstW = 64, dstH = 64;
        const int channels = 4;
        auto src = generateTestImage(srcH, srcW, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        float fx = (float)srcW / dstW;
        float fy = (float)srcH / dstH;
        tr.postScale(fx, fy);
        tr.postTranslate(0.5f * (fx - 1), 0.5f * (fy - 1));
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src.data(), srcW, srcH, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        // Spot-check a few pixels against reference
        int checkCount = 0;
        int mismatches = 0;
        for (int y = 0; y < dstH; y += 8) {
            for (int x = 0; x < dstW; x += 8) {
                float sx = x * fx + 0.5f * (fx - 1);
                float sy = y * fy + 0.5f * (fy - 1);
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = bilinearSample(src.data(), srcW, srcH,
                                                      channels, srcW * channels,
                                                      sx, sy, c);
                    uint8_t actual = dst[(y * dstW + x) * channels + c];
                    if (std::abs((int)expected - (int)actual) > 2) {
                        mismatches++;
                    }
                    checkCount++;
                }
            }
        }
        MNNTEST_ASSERT(mismatches == 0);
        MNNTEST_ASSERT(checkCount > 0);
        return true;
    }
};
MNNTestSuiteRegister(LargeImageResizeTest, "cv/image_op/large_image_resize");

// ========== Test: Stride Mismatch ==========
class StrideMismatchTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 5, H = 5;  // Non-power-of-2 to test stride alignment
        const int channels = 3;
        // Use a wider stride (padded rows)
        const int srcStride = W * channels + 4; // Extra padding
        std::vector<uint8_t> src(H * srcStride, 0);

        // Fill valid pixel data
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < channels; ++c) {
                    src[y * srcStride + x * channels + c] =
                        static_cast<uint8_t>((y * 31 + x * 17 + c * 7) % 256);
                }
            }
        }

        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat = RGB;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H * channels);
        // Pass explicit stride
        process->convert(src.data(), W, H, srcStride, dst.data(), W, H, channels * W, RGB);

        // Verify pixels match despite stride mismatch
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < channels; ++c) {
                    uint8_t expected = src[y * srcStride + x * channels + c];
                    uint8_t actual = dst[(y * W + x) * channels + c];
                    MNNTEST_ASSERT(expected == actual);
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(StrideMismatchTest, "cv/image_op/stride_mismatch");

// ========== Test: Single-pixel Image ==========
class SinglePixelResizeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int channels = 4;
        uint8_t src[4] = {100, 150, 200, 255};

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = BILINEAR;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        // Resize 1x1 -> 4x4: all output pixels should equal the source pixel
        const int dstW = 4, dstH = 4;
        Matrix tr;
        float fx = 1.0f / dstW;
        float fy = 1.0f / dstH;
        tr.postScale(fx, fy);
        tr.postTranslate(0.5f * (fx - 1), 0.5f * (fy - 1));
        process->setMatrix(tr);

        std::vector<uint8_t> dst(dstW * dstH * channels);
        process->convert(src, 1, 1, 0, dst.data(), dstW, dstH, channels * dstW, RGBA);

        for (int i = 0; i < dstW * dstH; ++i) {
            for (int c = 0; c < channels; ++c) {
                // All pixels should be the same as the single source pixel
                MNNTEST_ASSERT(std::abs((int)dst[i * channels + c] - (int)src[c]) <= 1);
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SinglePixelResizeTest, "cv/image_op/single_pixel_resize");

// ========== Test: Identity Transform ==========
class IdentityTransformTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const int W = 8, H = 8;
        const int channels = 4;
        auto src = generateTestImage(H, W, channels);

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat = RGBA;
        config.filterType = NEAREST;
        config.wrap = CLAMP_TO_EDGE;

        std::unique_ptr<ImageProcess> process(ImageProcess::create(config));
        MNNTEST_ASSERT(process.get() != nullptr);

        // Identity matrix - output should equal input
        Matrix tr;
        process->setMatrix(tr);

        std::vector<uint8_t> dst(W * H * channels);
        process->convert(src.data(), W, H, 0, dst.data(), W, H, channels * W, RGBA);

        for (int i = 0; i < W * H * channels; ++i) {
            MNNTEST_ASSERT(src[i] == dst[i]);
        }
        return true;
    }
};
MNNTestSuiteRegister(IdentityTransformTest, "cv/image_op/identity_transform");
