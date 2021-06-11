//
//  ImageProcessTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include <cmath>
#include <memory>
#include <map>
#include "MNNTestSuite.h"

using namespace MNN;
using namespace MNN::CV;

static std::vector<uint8_t> genSourceData(int h, int w, int bpp) {
    std::vector<uint8_t> source(h * w * bpp);
    for (int y = 0; y < h; ++y) {
        auto pixelY = source.data() + w * y * bpp;
        int magicY  = ((h - y) * (h - y)) % 79;
        for (int x = 0; x < w; ++x) {
            auto pixelX = pixelY + x * bpp;
            int magicX  = (x * x) % 113;
            for (int p = 0; p < bpp; ++p) {
                int magic = (magicX + magicY + p * p * p) % 255;
                pixelX[p] = magic;
            }
        }
    }
    return source;
}

// format in {YUV_NV21, YUV_NV12, YUV_I420}
// dstFormat in {RGBA, BGRA, RGB, BGR, GRAY}
static int genYUVData(int h, int w, ImageFormat format, ImageFormat dstFormat,
                      std::vector<uint8_t>& source, std::vector<uint8_t>& dest) {
    // https://www.jianshu.com/p/e67f79f10c65
    if (format != YUV_NV21 && format != YUV_NV12 && /* YUV420sp(bi-planer): NV12, NV21 */
        format != YUV_I420                          /* YUV420p(planer): I420 or YV12 */) {
        return -1;
    }
    bool yuv420p = (format != YUV_NV12 && format != YUV_NV21);

    int bpp = 0;
    if (dstFormat == RGBA || dstFormat == BGRA) {
        bpp = 4;
    } else if (dstFormat == RGB || dstFormat == BGR) {
        bpp = 3;
    } else if (dstFormat == GRAY) {
        bpp = 1;
    }
    if (bpp == 0) {
        return -2;
    }

    // YUV420, Y: h*w, UV: (h/2)*(w/2)*2
    int ySize = h * w, uvSize = (h/2)*(w/2)*2;
    source.resize(ySize + uvSize);
    dest.resize(h * w * bpp);

    auto dstData = dest.data();
    for (int y = 0; y < h; ++y) {
        auto pixelY  = source.data() + w * y;
        auto pixelUV = source.data() + w * h + (y / 2) * (yuv420p ? w / 2 : w);
        int magicY   = ((h - y) * (h - y)) % 79;
        for (int x = 0; x < w; ++x) {
            int magicX  = ((x % 113) * (x % 113)) % 113, xx = x / 2;
            int yVal   = (magicX + magicY) % 255;

            int uVal, vVal;
            int uIndex = (yuv420p ? xx : 2 * xx);
            int vIndex = (yuv420p ? xx + (h/2)*(w/2) : 2 * xx + 1);
            if (format != YUV_NV12 && format != YUV_I420) {
                std::swap(uIndex, vIndex);
            }
            if (y % 2 == 0 && x % 2 == 0) {
                magicX      = ((((xx % 283) * (xx % 283)) % 283) * (((xx % 283) * (xx % 283)) % 283)) % 283;
                uVal  = (magicX + magicY) % 255;
                vVal  = (magicX + magicY * 179) % 255;
                pixelUV[uIndex] = uVal;
                pixelUV[vIndex] = vVal;
            } else {
                uVal = pixelUV[uIndex];
                vVal = pixelUV[vIndex];
            }
            pixelY[x]   = yVal;

            int Y = yVal, U = uVal - 128, V = vVal - 128;
            auto dstData = dest.data() + (y * w + x) * bpp;
            if (dstFormat == GRAY) {
                dstData[0] = Y;
                continue;
            }
            Y     = Y << 6;
#define CLAMP(x, minVal, maxVal) std::min(std::max((x), (minVal)), (maxVal))
            int r = CLAMP((Y + 73 * V) >> 6, 0, 255);
            int g = CLAMP((Y - 25 * U - 37 * V) >> 6, 0, 255);
            int b = CLAMP((Y + 130 * U) >> 6, 0, 255);

            dstData[0] = r;
            dstData[1] = g;
            dstData[2] = b;
            if (dstFormat == BGRA || dstFormat == BGR) {
                std::swap(dstData[0], dstData[2]);
            }
            if (bpp == 4) {
                dstData[3] = 255;
            }
        }
    }
    return 0;
}

class ImageProcessGrayToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessGrayToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 27, h = 1, size = w * h;
        auto integers = genSourceData(h, w, 1);
        std::vector<float> floats(size * 4);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<float>(std::vector<int>{1, 1, h, w}, floats.data(), Tensor::CAFFE_C4));

        ImageProcess::Config config;
        config.sourceFormat = GRAY;
        config.destFormat   = GRAY;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < floats.size() / 4; ++i) {
            int s = floats[4 * i + 0];
            if (s != integers[i]) {
                MNN_ERROR("Error for turn gray to float:%d, %d -> %f\n", i, integers[i], floats[4 * i]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessGrayToGrayTest, "cv/image_process/gray_to_gray");

class ImageProcessGrayToGrayBilinearTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessGrayToGrayBilinearTransformTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = GRAY;
        config.destFormat   = GRAY;
        config.filterType   = BILINEAR;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1280;
        int sh = 720;
        int dw = 360;
        int dh = 640;
        Matrix tr;
        tr.setScale(1.0 / sw, 1.0 / sh);
        tr.postRotate(30, 0.5f, 0.5f);
        tr.postScale(dw, dh);
        tr.invert(&tr);
        process->setMatrix(tr);

        auto integers = genSourceData(sh, sw, 1);
        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        for (int i = 0; i < 10; ++i) {
            process->convert(integers.data(), sw, sh, 0, tensor.get());
        }
        auto floats   = tensor->host<float>();
        int expects[] = {18, 36, 14, 36, 18, 44, 30, 60, 50, 24};
        for (int v = 0; v < 10; ++v) {
            if (fabsf(floats[4 * v] - (float)expects[v]) >= 2) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expects[v]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessGrayToGrayBilinearTransformTest, "cv/image_process/gray_to_gray_bilinear_transorm");

class ImageProcessGrayToGrayNearestTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessGrayToGrayNearestTransformTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = GRAY;
        config.destFormat   = GRAY;
        config.filterType   = NEAREST;
        config.wrap         = ZERO;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1280;
        int sh = 720;
        int dw = 360;
        int dh = 640;
        Matrix tr;
        tr.setScale(1.0 / sw, 1.0 / sh);
        tr.postRotate(90, 0.5f, 0.5f);
        tr.postScale(dw, dh);
        tr.invert(&tr);
        process->setMatrix(tr);

        auto integers = genSourceData(sh, sw, 1);
        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        for (int i = 0; i < 10; ++i) {
            process->convert(integers.data(), sw, sh, 0, tensor.get());
        }
        auto floats  = tensor->host<float>();
        int expect[] = {0, 4, 16, 36, 64, 21, 65, 38, 19, 8};
        for (int v = 0; v < 10; ++v) {
            if ((int)(floats[4 * v]) != expect[v]) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessGrayToGrayNearestTransformTest, "cv/image_process/gray_to_gray_nearest_transorm");

class ImageProcessGrayToRGBATest : public MNNTestCase {
public:
    virtual ~ImageProcessGrayToRGBATest() = default;
    virtual bool run(int precision) {
        int w = 15, h = 1, size = w * h;
        auto gray = genSourceData(h, w, 1);
        std::vector<uint8_t> rgba(size * 4);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 4}, rgba.data()));

        ImageProcess::Config config;
        config.sourceFormat = GRAY;
        config.destFormat   = RGBA;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(gray.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            int s = gray[i];
            int r = rgba[4 * i + 0];
            int g = rgba[4 * i + 1];
            int b = rgba[4 * i + 2];

            int y = s;
            int a = rgba[4 * i + 3];

            if (y != r || y != g || y != b || a != 255) {
                MNN_ERROR("Turn gray to RGBA:%d, %d -> %d,%d,%d,%d\n", i, s, r, g, b, a);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessGrayToRGBATest, "cv/image_process/gray_to_rgba");

class ImageProcessBGRToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessBGRToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 15, h = 1, size = w * h;
        auto bgr = genSourceData(h, w, 3);
        std::vector<uint8_t> gray(size);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 1}, gray.data()));

        ImageProcess::Config config;
        config.sourceFormat = BGR;
        config.destFormat   = GRAY;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(bgr.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            int s = gray[i];
            int r = bgr[3 * i + 2];
            int g = bgr[3 * i + 1];
            int b = bgr[3 * i + 0];
            int y = (19 * r + 38 * g + 7 * b) >> 6;
            if (abs(y - s) >= 2) {
                MNN_ERROR("Turn BGR to gray:%d, %d,%d,%d -> %d\n", i, r, g, b, s);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessBGRToGrayTest, "cv/image_process/bgr_to_gray");

class ImageProcessRGBToBGRTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int w = 27, h = 1, size = w * h;
        auto integers = genSourceData(h, w, 3);
        std::vector<uint8_t> resultData(size * 3);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 3}, resultData.data(), Tensor::TENSORFLOW));
        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            int r = resultData[3 * i + 2];
            int g = resultData[3 * i + 1];
            int b = resultData[3 * i + 0];
            if (r != integers[3 * i + 0] || g != integers[3 * i + 1] || b != integers[3 * i + 2]) {
                MNN_ERROR("Error for turn rgb to bgr:\n %d,%d,%d->%d, %d, %d\n", integers[3 * i + 0],
                          integers[3 * i + 1], integers[3 * i + 2], r, g, b);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBToBGRTest, "cv/image_process/rgb_to_bgr");

class ImageProcessRGBAToBGRATest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBAToBGRATest() = default;
    virtual bool run(int precision) {
        int w = 27, h = 1, size = w * h;
        auto integers = genSourceData(h, w, 4);
        std::vector<uint8_t> floats(size * 4);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 4}, floats.data(), Tensor::TENSORFLOW));
        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat   = BGRA;

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < floats.size() / 4; ++i) {
            int r = floats[4 * i + 2];
            int g = floats[4 * i + 1];
            int b = floats[4 * i + 0];
            if (r != integers[4 * i + 0] || g != integers[4 * i + 1] || b != integers[4 * i + 2]) {
                MNN_ERROR("Error for turn rgba to bgra:\n %d,%d,%d->%d, %d, %d, %d\n", integers[4 * i + 0],
                          integers[4 * i + 1], integers[4 * i + 2], floats[4 * i + 0], floats[4 * i + 1],
                          floats[4 * i + 2], floats[4 * i + 3]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBAToBGRATest, "cv/image_process/rgba_to_bgra");

class ImageProcessBGRToBGRTest : public MNNTestCase {
public:
    virtual ~ImageProcessBGRToBGRTest() = default;
    virtual bool run(int precision) {
        int w = 27, h = 1, size = w * h;
        auto integers = genSourceData(h, w, 3);
        std::vector<float> floats(size * 4);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<float>(std::vector<int>{1, 1, h, w}, floats.data(), Tensor::CAFFE_C4));
        ImageProcess::Config config;
        config.sourceFormat = BGR;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < floats.size() / 4; ++i) {
            int r = floats[4 * i + 0];
            int g = floats[4 * i + 1];
            int b = floats[4 * i + 2];
            if (r != integers[3 * i + 0] || g != integers[3 * i + 1] || b != integers[3 * i + 2]) {
                MNN_ERROR("Error for turn rgb to float:\n %d,%d,%d->%f, %f, %f, %f\n", integers[3 * i + 0],
                          integers[3 * i + 1], integers[3 * i + 2], floats[4 * i + 0], floats[4 * i + 1],
                          floats[4 * i + 2], floats[4 * i + 3]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessBGRToBGRTest, "cv/image_process/bgr_to_bgr");

class ImageProcessRGBToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 15, h = 1, size = w * h;
        auto rgb = genSourceData(h, w, 3);
        std::vector<uint8_t> gray(size);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 1}, gray.data()));
        ImageProcess::Config config;
        config.sourceFormat = RGB;
        config.destFormat   = GRAY;

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(rgb.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            int s = gray[i];
            int r = rgb[3 * i + 0];
            int g = rgb[3 * i + 1];
            int b = rgb[3 * i + 2];
            int y = (19 * r + 38 * g + 7 * b) >> 6;
            if (abs(y - s) >= 2) {
                MNN_ERROR("Error: Turn RGB to gray:%d, %d,%d,%d -> %d\n", i, r, g, b, s);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBToGrayTest, "cv/image_process/rgb_to_gray");

class ImageProcessRGBAToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBAToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 15, h = 1, size = w * h;
        auto rgba = genSourceData(h, w, 4);
        std::vector<uint8_t> gray(size);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 1}, gray.data()));

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat   = GRAY;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(rgba.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            int s = gray[i];
            int r = rgba[4 * i + 0];
            int g = rgba[4 * i + 1];
            int b = rgba[4 * i + 2];
            int y = (19 * r + 38 * g + 7 * b) >> 6;
            if (abs(y - s) >= 2) {
                MNN_ERROR("Turn RGBA to gray:%d, %d,%d,%d -> %d\n", i, r, g, b, s);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBAToGrayTest, "cv/image_process/rgba_to_gray");

class ImageProcessRGBAToGrayBilinearTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBAToGrayBilinearTransformTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat   = GRAY;
        config.filterType   = BILINEAR;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1280;
        int sh = 720;
        int dw = 360;
        int dh = 640;
        Matrix tr;
        tr.setScale(1.0 / sw, 1.0 / sh);
        tr.postRotate(30, 0.5f, 0.5f);
        tr.postScale(dw, dh);
        tr.invert(&tr);
        process->setMatrix(tr);

        auto integers = genSourceData(sh, sw, 4);
        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        process->convert(integers.data(), sw, sh, 0, tensor.get());
        auto floats  = tensor->host<float>();
        int expect[] = {19, 37, 15, 37, 19, 45, 31, 61, 51, 25};
        for (int v = 0; v < 10; ++v) {
            if (fabsf(floats[4 * v] - (float)expect[v]) >= 2) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBAToGrayBilinearTransformTest, "cv/image_process/rgba_to_gray_bilinear_transorm");

class ImageProcessRGBAToGrayNearestTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBAToGrayNearestTransformTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat   = GRAY;
        config.filterType   = NEAREST;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1280;
        int sh = 720;
        int dw = 360;
        int dh = 640;
        Matrix tr;
        tr.setScale(1.0 / sw, 1.0 / sh);
        tr.postRotate(60, 0.5f, 0.5f);
        tr.postScale(dw, dh);
        tr.invert(&tr);
        process->setMatrix(tr);

        auto integers = genSourceData(sh, sw, 4);
        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        for (int i = 0; i < 10; ++i) {
            process->convert(integers.data(), sw, sh, 0, tensor.get());
        }
        auto floats  = tensor->host<float>();
        int expect[] = {3, 50, 26, 17, 5, 1, 5, 10, 26, 50};
        for (int v = 0; v < 10; ++v) {
            if ((int)(floats[4 * v]) != expect[v]) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBAToGrayNearestTransformTest, "cv/image_process/rgba_to_gray_nearest_transorm");

class ImageProcessRGBAToBGRTest : public MNNTestCase {
public:
    virtual ~ImageProcessRGBAToBGRTest() = default;
    virtual bool run(int precision) {
        int w = 15, h = 1, size = w * h;
        auto rgba = genSourceData(h, w, 4);
        std::vector<uint8_t> bgr(size * 3);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, h, w, 3}, bgr.data()));

        ImageProcess::Config config;
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(rgba.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            if (rgba[4 * i + 0] != bgr[3 * i + 2] || rgba[4 * i + 1] != bgr[3 * i + 1] ||
                rgba[4 * i + 2] != bgr[3 * i + 0]) {
                MNN_ERROR("Error: Turn RGBA to BGR:%d, %d,%d,%d,%d -> %d,%d,%d\n", i, rgba[4 * i + 0], rgba[4 * i + 1],
                          rgba[4 * i + 2], rgba[4 * i + 3], bgr[3 * i + 0], bgr[3 * i + 1], bgr[3 * i + 2]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessRGBAToBGRTest, "cv/image_process/rgba_to_bgr");

// Test for _blitC3ToFloatC3
class ImageProcessBGRToBGRFloatBlitterTest : public MNNTestCase {
public:
    virtual ~ImageProcessBGRToBGRFloatBlitterTest() = default;
    virtual bool run(int precision) {
        int w = 27, h = 27, size = w * h;
        auto integers = genSourceData(h, w, 3);
        std::vector<float> floats(size * 3);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<float>(std::vector<int>{1, h, w, 3}, floats.data(), Tensor::TENSORFLOW));
        ImageProcess::Config config;
        config.sourceFormat = BGR;
        config.destFormat   = BGR;

        const float means[3]   = {127.5f, 127.5f, 127.5f};
        const float normals[3] = {2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f};
        memcpy(config.mean, means, sizeof(means));
        memcpy(config.normal, normals, sizeof(normals));

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < 3; ++j) {
                float result = floats[3 * i + j];
                float right  = (integers[3 * i + j] - means[j]) * normals[j];
                if (fabs(result - right) > 1e-6f) {
                    MNN_ERROR("Error for blitter bgr to bgr\n%d -> %f, right: %f\n", integers[3 * i + j], result,
                              right);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessBGRToBGRFloatBlitterTest, "cv/image_process/bgr_to_bgr_blitter");

// Test for _blitC1ToFloatC1
class ImageProcessGrayToGrayFloatBlitterTest : public MNNTestCase {
public:
    virtual ~ImageProcessGrayToGrayFloatBlitterTest() = default;
    virtual bool run(int precision) {
        int w = 27, h = 27, size = w * h;
        auto integers = genSourceData(h, w, 1);
        std::vector<float> floats(size);
        std::shared_ptr<MNN::Tensor> tensor(
            MNN::Tensor::create<float>(std::vector<int>{1, h, w, 1}, floats.data(), Tensor::TENSORFLOW));
        ImageProcess::Config config;
        config.sourceFormat = GRAY;
        config.destFormat   = GRAY;

        const float means[1]   = {127.5f};
        const float normals[1] = {2.0f / 255.0f};
        memcpy(config.mean, means, sizeof(means));
        memcpy(config.normal, normals, sizeof(normals));

        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
        process->convert(integers.data(), w, h, 0, tensor.get());
        for (int i = 0; i < size; ++i) {
            float result = floats[i];
            float right  = (integers[i] - means[0]) * normals[0];
            if (fabs(result - right) > 1e-6f) {
                MNN_PRINT("raw: %d, result: %f, right: %f\n", integers[i], result, right);
                MNN_ERROR("Error for blitter gray to gray\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessGrayToGrayFloatBlitterTest, "cv/image_process/gray_to_gray_blitter");

class ImageProcessYUVTestCommmon : public MNNTestCase {
protected:
    virtual ~ImageProcessYUVTestCommmon() = default;
    bool test(ImageFormat sourceFormat, ImageFormat destFormat, int bpp, int sw, int sh) {
        std::map<ImageFormat, std::string> formatMap = {
            {RGBA, "RGBA"}, {RGB, "RGB"}, {BGRA, "BGRA"}, {BGR, "BGR"}, {GRAY, "GRAY"},
            {YUV_NV21, "NV21"}, {YUV_NV12, "NV12"}, {YUV_I420, "I420"}
        };
        auto sourceStr = formatMap[sourceFormat].c_str(), destStr = formatMap[destFormat].c_str();
        //MNN_PRINT("%s_to_%s\n", sourceStr, destStr);

        ImageProcess::Config config;
        config.sourceFormat = sourceFormat;
        config.destFormat   = destFormat;
        //config.filterType   = NEAREST;
        //config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        //Matrix tr;
        //process->setMatrix(tr);
        std::vector<uint8_t> src, dst;
        genYUVData(sh, sw, sourceFormat, destFormat, src, dst);

        std::shared_ptr<Tensor> tensor(
            Tensor::create<uint8_t>(std::vector<int>{1, sh, sw, bpp}, nullptr, Tensor::TENSORFLOW));
        process->convert(src.data(), sw, sh, 0, tensor.get());
        for (int y = 0; y < sh; ++y) {
            auto srcY_Y  = src.data() + y * sw;
            auto srcY_UV = src.data() + (y / 2) * (sw / 2) * 2 + sw * sh;
            for (int x = 0; x < sw; ++x) {
                auto rightData = dst.data() + (y * sw + x) * bpp;
                auto testData = tensor->host<uint8_t>() + (y * sw + x) * bpp;

                bool wrong = false;
                for (int i = 0; i < bpp && !wrong; ++i) {
                    if (abs(rightData[i] - testData[i]) > 5) {
                        wrong = true;
                    }
                }
                if (wrong) {
                    int Y = srcY_Y[x], U = srcY_UV[(x / 2) * 2], V = srcY_UV[(x / 2) * 2 + 1];
                    MNN_ERROR("Error for %s to %s (%d, %d):  %d, %d, %d -> ", sourceStr, destStr, y, x, Y, U, V);
                    for (int i = 0; i < bpp; ++i) {
                        MNN_ERROR("%d, ", rightData[i]);
                    }
                    MNN_ERROR("wrong:");
                    for (int i = 0; i < bpp; ++i) {
                        MNN_ERROR(" %d%s", testData[i], (i < bpp ? ",": ""));
                    }
                    MNN_ERROR("\n");
                    return false;
                }
            }
        }
        return true;
    }
};

class ImageProcessYUVBlitterTest : public ImageProcessYUVTestCommmon {
public:
    virtual ~ImageProcessYUVBlitterTest() = default;
    virtual bool run(int precision) {
        std::vector<ImageFormat> srcFromats = {YUV_NV21, YUV_NV12, YUV_I420};
        std::vector<ImageFormat> dstFormats = {RGBA, RGB, BGRA, BGR, GRAY};
        std::vector<int> bpps = {4, 3, 4, 3, 1};
        bool succ = true;
        for (auto srcFormat : srcFromats) {
            for (int i = 0; i < dstFormats.size(); ++i) {
                succ = succ && test(srcFormat, dstFormats[i], bpps[i], 1920, 1080);
            }
        }
        return succ;
    }
};
// {YUV_NV21, YUV_NV12, YUV_I420} -> {RGBA, RGB, BGRA, BGR, GRAY} unit test
MNNTestSuiteRegister(ImageProcessYUVBlitterTest, "cv/image_process/yuv_blitter");
