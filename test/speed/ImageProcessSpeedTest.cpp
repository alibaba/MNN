//
//  ImageProcessSpeedTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/ImageProcess.hpp>
#include <cmath>
#include <memory>
#include "MNNTestSuite.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::CV;

class ImageProcessSpeedGrayToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedGrayToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 1080, h = 720, size = w * h;
        std::vector<uint8_t> integers(size);
        for (int i = 0; i < size; ++i) {
            int magic   = (i * 67 % 255);
            integers[i] = magic;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedGrayToGrayTest, "speed/cv/image_process/gray_to_gray");

class ImageProcessSpeedGrayToGrayBilinearTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedGrayToGrayBilinearTransformTest() = default;
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

        std::shared_ptr<unsigned char> integers(new unsigned char[sw * sh * 4]);
        auto pixels = integers.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY = pixels + sw * y;
            int magicY  = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + 4 * x;
                int magicX  = (x * x) % 113;
                for (int p = 0; p < 1; ++p) {
                    int magic = (magicX + magicY + p * p * p) % 255;
                    pixelX[p] = magic;
                }
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        process->convert(integers.get(), sw, sh, 0, tensor.get());
        auto floats   = tensor->host<float>();
        int expects[] = {18, 36, 14, 36, 18, 44, 30, 60, 50, 24};
        for (int v = 0; v < 10; ++v) {
            if (fabsf(floats[4 * v] - (float)expects[v]) >= 2) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expects[v]);
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedGrayToGrayBilinearTransformTest,
                     "speed/cv/image_process/gray_to_gray_bilinear_transorm");

class ImageProcessSpeedGrayToGrayNearestTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedGrayToGrayNearestTransformTest() = default;
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

        std::shared_ptr<unsigned char> integers(new unsigned char[sw * sh]);
        auto pixels = integers.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY = pixels + sw * y;
            int magicY  = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + x;
                int magicX  = (x * x) % 113;
                int magic   = (magicX + magicY) % 255;
                pixelX[0]   = magic;
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        process->convert(integers.get(), sw, sh, 0, tensor.get());
        auto floats  = tensor->host<float>();
        int expect[] = {0, 4, 16, 36, 64, 21, 65, 38, 19, 8};
        for (int v = 0; v < 10; ++v) {
            if ((int)(floats[4 * v]) != expect[v]) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedGrayToGrayNearestTransformTest,
                     "speed/cv/image_process/gray_to_gray_nearest_transorm");

class ImageProcessSpeedGrayToRGBATest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedGrayToRGBATest() = default;
    virtual bool run(int precision) {
        int w = 1080, h = 720, size = w * h;
        std::vector<uint8_t> gray(size);
        for (int i = 0; i < size; ++i) {
            int magic   = (i * 67 % 255);
            gray[i + 0] = (3 * magic + 0) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(gray.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedGrayToRGBATest, "speed/cv/image_process/gray_to_rgba");

class ImageProcessSpeedBGRToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedBGRToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 1080, h = 720, size = w * h;
        std::vector<uint8_t> bgr(size * 3);
        for (int i = 0; i < size; ++i) {
            int magic      = (i * 67 % 255);
            bgr[3 * i + 0] = (3 * magic + 0) % 255;
            bgr[3 * i + 1] = (3 * magic + 32) % 255;
            bgr[3 * i + 2] = (3 * magic + 64) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(bgr.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedBGRToGrayTest, "speed/cv/image_process/bgr_to_gray");
class ImageProcessSpeedRGBToBGRTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int w = 1081, h = 719, size = w * h;
        std::vector<uint8_t> integers(size * 3);
        for (int i = 0; i < size; ++i) {
            int magic           = (i * 67 % 255);
            integers[3 * i + 0] = (3 * magic + 0) % 255;
            integers[3 * i + 1] = (3 * magic + 1) % 255;
            integers[3 * i + 2] = (3 * magic + 2) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBToBGRTest, "speed/cv/image_process/rgb_to_bgr");

class ImageProcessSpeedRGBAToBGRATest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBAToBGRATest() = default;
    virtual bool run(int precision) {
        int w = 1081, h = 720, size = w * h;
        std::vector<uint8_t> integers(size * 4);
        for (int i = 0; i < size; ++i) {
            int magic           = (i * 67 % 255);
            integers[4 * i + 0] = (4 * magic + 0) % 255;
            integers[4 * i + 1] = (4 * magic + 1) % 255;
            integers[4 * i + 2] = (4 * magic + 2) % 255;
            integers[4 * i + 3] = (4 * magic + 3) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBAToBGRATest, "speed/cv/image_process/rgba_to_bgra");

class ImageProcessSpeedBGRToBGRTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedBGRToBGRTest() = default;
    virtual bool run(int precision) {
        int w = 1020, h = 960, size = w * h;
        std::vector<uint8_t> integers(size * 3);
        for (int i = 0; i < size; ++i) {
            int magic           = (i * 67 % 255);
            integers[3 * i + 0] = (3 * magic + 0) % 255;
            integers[3 * i + 1] = (3 * magic + 1) % 255;
            integers[3 * i + 2] = (3 * magic + 2) % 255;
        }
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
                MNN_ERROR("Error for turn rgb to float:\n, i:%d,  %d,%d,%d->%f, %f, %f, %f\n", i, integers[3 * i + 0],
                          integers[3 * i + 1], integers[3 * i + 2], floats[4 * i + 0], floats[4 * i + 1],
                          floats[4 * i + 2], floats[4 * i + 3]);
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedBGRToBGRTest, "speed/cv/image_process/bgr_to_bgr");

class ImageProcessSpeedRGBToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 1080, h = 720, size = w * h;
        std::vector<uint8_t> rgb(size * 3);
        for (int i = 0; i < size; ++i) {
            int magic      = (i * 67 % 255);
            rgb[3 * i + 0] = (3 * magic + 0) % 255;
            rgb[3 * i + 1] = (3 * magic + 32) % 255;
            rgb[3 * i + 2] = (3 * magic + 64) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(rgb.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBToGrayTest, "speed/cv/image_process/rgb_to_gray");

class ImageProcessSpeedRGBAToGrayTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBAToGrayTest() = default;
    virtual bool run(int precision) {
        int w = 1080, h = 720, size = w * h;
        std::vector<uint8_t> rgba(size * 4);
        for (int i = 0; i < size; ++i) {
            int magic       = (i * 67 % 255);
            rgba[4 * i + 0] = (4 * magic + 0) % 255;
            rgba[4 * i + 1] = (4 * magic + 32) % 255;
            rgba[4 * i + 2] = (4 * magic + 64) % 255;
            rgba[4 * i + 3] = 0;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(rgba.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBAToGrayTest, "speed/cv/image_process/rgba_to_gray");

class ImageProcessSpeedRGBAToGrayBilinearTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBAToGrayBilinearTransformTest() = default;
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

        std::shared_ptr<unsigned char> integers(new unsigned char[sw * sh * 4]);
        auto pixels = integers.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY = pixels + 4 * sw * y;
            int magicY  = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + 4 * x;
                int magicX  = (x * x) % 113;
                for (int p = 0; p < 4; ++p) {
                    int magic = (magicX + magicY + p * p * p) % 255;
                    pixelX[p] = magic;
                }
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        process->convert(integers.get(), sw, sh, 0, tensor.get());
        auto floats  = tensor->host<float>();
        int expect[] = {19, 37, 15, 37, 19, 45, 31, 61, 51, 25};
        for (int v = 0; v < 10; ++v) {
            if (fabsf(floats[4 * v] - (float)expect[v]) >= 2) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBAToGrayBilinearTransformTest,
                     "speed/cv/image_process/rgba_to_gray_bilinear_transorm");

class ImageProcessSpeedRGBAToGrayNearestTransformTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBAToGrayNearestTransformTest() = default;
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

        std::shared_ptr<unsigned char> integers(new unsigned char[sw * sh * 4]);
        auto pixels = integers.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY = pixels + 4 * sw * y;
            int magicY  = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + 4 * x;
                int magicX  = (x * x) % 113;
                for (int p = 0; p < 4; ++p) {
                    int magic = (magicX + magicY + p * p * p) % 255;
                    pixelX[p] = magic;
                }
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<float>(std::vector<int>{1, 1, dw, dh}, nullptr, Tensor::CAFFE_C4));
        process->convert(integers.get(), sw, sh, 0, tensor.get());
        auto floats  = tensor->host<float>();
        int expect[] = {3, 50, 26, 17, 5, 1, 5, 10, 26, 50};
        for (int v = 0; v < 10; ++v) {
            if ((int)(floats[4 * v]) != expect[v]) {
                MNN_ERROR("Error for %d, %.f, correct=%d\n", v, floats[4 * v], expect[v]);
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBAToGrayNearestTransformTest,
                     "speed/cv/image_process/rgba_to_gray_nearest_transorm");

class ImageProcessSpeedRGBAToBGRTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedRGBAToBGRTest() = default;
    virtual bool run(int precision) {
        int w = 1500, h = 1080, size = w * h;
        std::vector<uint8_t> rgba(size * 4);
        for (int i = 0; i < size; ++i) {
            int magic       = (i * 67 % 255);
            rgba[4 * i + 0] = (3 * magic + 32 * 0) % 255;
            rgba[4 * i + 1] = (3 * magic + 32 * 1) % 255;
            rgba[4 * i + 2] = (3 * magic + 32 * 2) % 255;
            rgba[4 * i + 3] = (3 * magic + 32 * 3) % 255;
        }
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
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(rgba.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedRGBAToBGRTest, "speed/cv/image_process/rgba_to_bgr");

class ImageProcessSpeedNV21ToRGBTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedNV21ToRGBTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = YUV_NV21;
        config.destFormat   = RGB;
        config.filterType   = NEAREST;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1920;
        int sh = 1080;
        Matrix tr;
        process->setMatrix(tr);
        std::shared_ptr<unsigned char> nv12(new unsigned char[sw * sh + (sw / 2) * (sh / 2) * 2]);
        auto pixels = nv12.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY  = pixels + sw * y;
            auto pixelUV = pixels + sw * sh + (y / 2) * sw;
            int magicY   = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + x;
                int magicX  = (x * x) % 113;
                int magic   = (magicX + magicY) % 255;
                pixelX[0]   = magic;
            }
            for (int x = 0; x < sw / 2; ++x) {
                auto pixelX = pixelUV + 2 * x;
                int magicX  = ((((x % 283) * (x % 283)) % 283) * (((x % 283) * (x % 283)) % 283)) % 283;
                int magic0  = (magicX + magicY) % 255;
                int magic1  = (magicX + magicY * 179) % 255;
                pixelX[0]   = magic0;
                pixelX[1]   = magic1;
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<uint8_t>(std::vector<int>{1, sh, sw, 3}, nullptr, Tensor::TENSORFLOW));
        process->convert(nv12.get(), sw, sh, 0, tensor.get());
        for (int y = 0; y < sh; ++y) {
            auto dstY    = tensor->host<uint8_t>() + 3 * y * sw;
            auto srcY_Y  = nv12.get() + y * sw;
            auto srcY_UV = nv12.get() + (y / 2) * (sw / 2) * 2 + sw * sh;
            for (int x = 0; x < sw; ++x) {
                auto dstX    = dstY + 3 * x;
                auto srcX_Y  = srcY_Y + x;
                auto srcX_UV = srcY_UV + (x / 2) * 2;
                int Y        = srcX_Y[0];
                int U        = (int)srcX_UV[1] - 128;
                int V        = (int)srcX_UV[0] - 128;

                Y     = Y << 6;
                int r = (Y + 73 * V) >> 6;
                int g = (Y - 25 * U - 37 * V) >> 6;
                int b = (Y + 130 * U) >> 6;

                r         = r < 0 ? 0 : r;
                r         = r > 255 ? 255 : r;
                g         = g < 0 ? 0 : g;
                g         = g > 255 ? 255 : g;
                b         = b < 0 ? 0 : b;
                b         = b > 255 ? 255 : b;
                auto diff = [](int a, int b) { return abs(a - b) > 5; };
                if (diff(dstX[0], r) || diff(dstX[1], g) || diff(dstX[2], b)) {
                    MNN_ERROR("%d, Error for NV12 to RGB: %d:  %d, %d, %d -> %d, %d, %d, wrong: %d, %d, %d\n", y, x, Y,
                              U, V, r, g, b, dstX[0], dstX[1], dstX[2]);
                    return false;
                }
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(nv12.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedNV21ToRGBTest, "speed/cv/image_process/nv21_to_rgb");

class ImageProcessSpeedNV12ToRGBTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedNV12ToRGBTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = YUV_NV12;
        config.destFormat   = RGB;
        config.filterType   = NEAREST;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1920;
        int sh = 1080;
        Matrix tr;
        process->setMatrix(tr);
        std::shared_ptr<unsigned char> nv12(new unsigned char[sw * sh + (sw / 2) * (sh / 2) * 2]);
        auto pixels = nv12.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY  = pixels + sw * y;
            auto pixelUV = pixels + sw * sh + (y / 2) * sw;
            int magicY   = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + x;
                int magicX  = (x * x) % 113;
                int magic   = (magicX + magicY) % 255;
                pixelX[0]   = magic;
            }
            for (int x = 0; x < sw / 2; ++x) {
                auto pixelX = pixelUV + 2 * x;
                int magicX  = ((((x % 283) * (x % 283)) % 283) * (((x % 283) * (x % 283)) % 283)) % 283;
                int magic0  = (magicX + magicY) % 255;
                int magic1  = (magicX + magicY * 179) % 255;
                pixelX[0]   = magic0;
                pixelX[1]   = magic1;
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<uint8_t>(std::vector<int>{1, sh, sw, 3}, nullptr, Tensor::TENSORFLOW));
        process->convert(nv12.get(), sw, sh, 0, tensor.get());
        for (int y = 0; y < sh; ++y) {
            auto dstY    = tensor->host<uint8_t>() + 3 * y * sw;
            auto srcY_Y  = nv12.get() + y * sw;
            auto srcY_UV = nv12.get() + (y / 2) * (sw / 2) * 2 + sw * sh;
            for (int x = 0; x < sw; ++x) {
                auto dstX    = dstY + 3 * x;
                auto srcX_Y  = srcY_Y + x;
                auto srcX_UV = srcY_UV + (x / 2) * 2;
                int Y        = srcX_Y[0];
                int U        = (int)srcX_UV[0] - 128;
                int V        = (int)srcX_UV[1] - 128;

                Y     = Y << 6;
                int r = (Y + 73 * V) >> 6;
                int g = (Y - 25 * U - 37 * V) >> 6;
                int b = (Y + 130 * U) >> 6;

                r         = r < 0 ? 0 : r;
                r         = r > 255 ? 255 : r;
                g         = g < 0 ? 0 : g;
                g         = g > 255 ? 255 : g;
                b         = b < 0 ? 0 : b;
                b         = b > 255 ? 255 : b;
                auto diff = [](int a, int b) { return abs(a - b) > 5; };
                if (diff(dstX[0], r) || diff(dstX[1], g) || diff(dstX[2], b)) {
                    MNN_ERROR("%d, Error for NV12 to RGB: %d:  %d, %d, %d -> %d, %d, %d, wrong: %d, %d, %d\n", y, x,
                              (int)srcX_Y[0], U, V, r, g, b, dstX[0], dstX[1], dstX[2]);
                    return false;
                }
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(nv12.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedNV12ToRGBTest, "speed/cv/image_process/nv12_to_rgb");

class ImageProcessSpeedNV12ToRGBATest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedNV12ToRGBATest() {
    }
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = YUV_NV21;
        config.destFormat   = RGBA;
        config.filterType   = NEAREST;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1920;
        int sh = 1080;
        Matrix tr;
        process->setMatrix(tr);
        std::shared_ptr<unsigned char> nv12(new unsigned char[sw * sh + (sw / 2) * (sh / 2) * 2]);
        auto pixels = nv12.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY  = pixels + sw * y;
            auto pixelUV = pixels + sw * sh + (y / 2) * sw;
            int magicY   = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + x;
                int magicX  = (x * x) % 113;
                int magic   = (magicX + magicY) % 255;
                pixelX[0]   = magic;
            }
            for (int x = 0; x < sw / 2; ++x) {
                auto pixelX = pixelUV + 2 * x;
                int magicX  = ((((x % 283) * (x % 283)) % 283) * (((x % 283) * (x % 283)) % 283)) % 283;
                int magic0  = (magicX + magicY) % 255;
                int magic1  = (magicX + magicY * 179) % 255;
                pixelX[0]   = magic0;
                pixelX[1]   = magic1;
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<uint8_t>(std::vector<int>{1, sh, sw, 4}, nullptr, Tensor::TENSORFLOW));
        process->convert(nv12.get(), sw, sh, 0, tensor.get());
        for (int y = 0; y < sh; ++y) {
            auto dstY    = tensor->host<uint8_t>() + 4 * y * sw;
            auto srcY_Y  = nv12.get() + y * sw;
            auto srcY_UV = nv12.get() + (y / 2) * (sw / 2) * 2 + sw * sh;
            for (int x = 0; x < sw; ++x) {
                auto dstX    = dstY + 4 * x;
                auto srcX_Y  = srcY_Y + x;
                auto srcX_UV = srcY_UV + (x / 2) * 2;
                int Y        = srcX_Y[0];
                int U        = (int)srcX_UV[1] - 128;
                int V        = (int)srcX_UV[0] - 128;

                Y     = Y << 6;
                int r = (Y + 73 * V) >> 6;
                int g = (Y - 25 * U - 37 * V) >> 6;
                int b = (Y + 130 * U) >> 6;

                r         = r < 0 ? 0 : r;
                r         = r > 255 ? 255 : r;
                g         = g < 0 ? 0 : g;
                g         = g > 255 ? 255 : g;
                b         = b < 0 ? 0 : b;
                b         = b > 255 ? 255 : b;
                auto diff = [](int a, int b) { return abs(a - b) > 5; };
                if (diff(dstX[0], r) || diff(dstX[1], g) || diff(dstX[2], b)) {
                    MNN_ERROR("%d, Error for NV12 to RGBA: %d:  %d, %d, %d -> %d, %d, %d, wrong: %d, %d, %d\n", y, x, Y,
                              U, V, r, g, b, dstX[0], dstX[1], dstX[2]);
                    return false;
                }
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(nv12.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedNV12ToRGBATest, "speed/cv/image_process/nv21_to_rgba");

// Test for _blitC3ToFloatC3
class ImageProcessSpeedBGRToBGRFloatBlitterTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedBGRToBGRFloatBlitterTest() = default;
    virtual bool run(int precision) {
        int w = 1020, h = 756, size = w * h;
        std::vector<uint8_t> integers(size * 3);
        for (int i = 0; i < size; ++i) {
            int magic           = (i * 67 % 255);
            integers[3 * i + 0] = (3 * magic + 0) % 255;
            integers[3 * i + 1] = (3 * magic + 1) % 255;
            integers[3 * i + 2] = (3 * magic + 2) % 255;
        }
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
                    MNN_ERROR("Error for blitter bgr to bgr\n%d ->i:%d, %f, right: %f\n", i, integers[3 * i + j],
                              result, right);
                    return false;
                }
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedBGRToBGRFloatBlitterTest, "speed/cv/image_process/bgr_to_bgr_blitter");

// Test for _blitC1ToFloatC1
class ImageProcessSpeedGrayToGrayFloatBlitterTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedGrayToGrayFloatBlitterTest() = default;
    virtual bool run(int precision) {
        int w = 1024, h = 1080, size = w * h;
        std::vector<uint8_t> integers(size);
        for (int i = 0; i < size; ++i) {
            int magic   = (i * 67 % 255);
            integers[i] = magic;
        }
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
                MNN_PRINT("i:%d, raw: %d, result: %f, right: %f\n", i, integers[i], result, right);
                MNN_ERROR("Error for blitter gray to gray\n");
                return false;
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(integers.data(), w, h, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedGrayToGrayFloatBlitterTest, "speed/cv/image_process/gray_to_gray_blitter");

class ImageProcessSpeedI420ToRGBTest : public MNNTestCase {
public:
    virtual ~ImageProcessSpeedI420ToRGBTest() = default;
    virtual bool run(int precision) {
        ImageProcess::Config config;
        config.sourceFormat = YUV_I420;
        config.destFormat   = RGB;
        config.filterType   = NEAREST;
        config.wrap         = CLAMP_TO_EDGE;
        std::shared_ptr<ImageProcess> process(ImageProcess::create(config));

        int sw = 1920;
        int sh = 1080;
        Matrix tr;
        process->setMatrix(tr);
        std::shared_ptr<unsigned char> nv12(new unsigned char[sw * sh + (sw / 2) * (sh / 2) * 2]);
        auto pixels = nv12.get();
        for (int y = 0; y < sh; ++y) {
            auto pixelY  = pixels + sw * y;
            auto pixelUV = pixels + sw * sh + (y / 2) * sw;
            int magicY   = ((sh - y) * (sh - y)) % 79;
            for (int x = 0; x < sw; ++x) {
                auto pixelX = pixelY + x;
                int magicX  = (x * x) % 113;
                int magic   = (magicX + magicY) % 255;
                pixelX[0]   = magic;
            }
            for (int x = 0; x < sw / 2; ++x) {
                auto pixelX = pixelUV + 2 * x;
                int magicX = ((((x % 283) * (x % 283)) % 283) * (((x % 283) * (x % 283)) % 283)) % 283;
                int magic0  = (magicX + magicY) % 255;
                int magic1  = (magicX + magicY * 179) % 255;
                pixelX[0]   = magic0;
                pixelX[1]   = magic1;
            }
        }

        std::shared_ptr<Tensor> tensor(
            Tensor::create<uint8_t>(std::vector<int>{1, sh, sw, 3}, nullptr, Tensor::TENSORFLOW));
        process->convert(nv12.get(), sw, sh, 0, tensor.get());
        for (int y = 0; y < sh; ++y) {
            auto dstY   = tensor->host<uint8_t>() + 3 * y * sw;
            auto srcY_Y = nv12.get() + y * sw;
            auto srcY_U = nv12.get() + (y / 2) * (sw / 2) + sw * sh;
            auto srcY_V = nv12.get() + (y / 2) * (sw / 2) + sw * sh + (sw / 2) * (sh / 2);
            for (int x = 0; x < sw; ++x) {
                auto dstX   = dstY + 3 * x;
                auto srcX_Y = srcY_Y + x;
                auto srcX_U = srcY_U + (x / 2);
                auto srcX_V = srcY_V + (x / 2);
                int Y       = srcX_Y[0];
                int U       = (int)srcX_U[0] - 128;
                int V       = (int)srcX_V[0] - 128;

                Y     = Y << 6;
                int r = (Y + 73 * V) >> 6;
                int g = (Y - 25 * U - 37 * V) >> 6;
                int b = (Y + 130 * U) >> 6;

                r         = r < 0 ? 0 : r;
                r         = r > 255 ? 255 : r;
                g         = g < 0 ? 0 : g;
                g         = g > 255 ? 255 : g;
                b         = b < 0 ? 0 : b;
                b         = b > 255 ? 255 : b;
                auto diff = [](int a, int b) { return abs(a - b) > 5; };
                if (diff(dstX[0], r) || diff(dstX[1], g) || diff(dstX[2], b)) {
                    MNN_ERROR("%d, Error for I420 to RGB: %d:  %d, %d, %d -> %d, %d, %d, wrong: %d, %d, %d\n", y, x, Y,
                              U, V, r, g, b, dstX[0], dstX[1], dstX[2]);
                    return false;
                }
            }
        }
        {
            AUTOTIME;
            for (int i = 0; i < 10; ++i) {
                process->convert(nv12.get(), sw, sh, 0, tensor.get());
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ImageProcessSpeedI420ToRGBTest, "speed/cv/image_process/I420_to_rgb");
