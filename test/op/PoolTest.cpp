//
//  PoolTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _MaxPoolWithExplicitPads(VARP input, int padX, int padY, const std::vector<int>& pads, bool isGlobal,
                                     DataType dataType = DataType_DT_FLOAT) {
    std::unique_ptr<PoolT> pool(new PoolT);
    pool->padX = padX;
    pool->padY = padY;
    pool->isGlobal = isGlobal;
    pool->kernelX = 5;
    pool->kernelY = 5;
    pool->strideX = 1;
    pool->strideY = 1;
    pool->type = PoolType_MAXPOOL;
    pool->padType = PoolPadType_CAFFE;
    pool->dataType = dataType;
    pool->ceilModel = false;
    pool->pads = pads;
    pool->countType = AvgPoolCountType_DEFAULT;

    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Pooling;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
    op->main.type = OpParameter_Pool;
    op->main.value = pool.release();

    return Variable::create(Expr::create(op.get(), {input}));
}

static std::vector<float> referenceMaxPool5x5(const std::vector<float>& input, int n, int c, int h, int w, int padY,
                                              int padX) {
    std::vector<float> output(input.size(), -16777216.0f);
    for (int b = 0; b < n; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oy = 0; oy < h; ++oy) {
                for (int ox = 0; ox < w; ++ox) {
                    float maxValue = -16777216.0f;
                    for (int fy = 0; fy < 5; ++fy) {
                        int iy = oy + fy - padY;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (int fx = 0; fx < 5; ++fx) {
                            int ix = ox + fx - padX;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            int offset = ((b * c + ch) * h + iy) * w + ix;
                            maxValue = std::max(maxValue, input[offset]);
                        }
                    }
                    int outputOffset = ((b * c + ch) * h + oy) * w + ox;
                    output[outputOffset] = maxValue;
                }
            }
        }
    }
    return output;
}

static std::vector<int8_t> referenceMaxPool5x5Int8(const std::vector<int8_t>& input, int n, int c, int h, int w,
                                                   int padY, int padX) {
    std::vector<int8_t> output(input.size(), std::numeric_limits<int8_t>::min());
    for (int b = 0; b < n; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oy = 0; oy < h; ++oy) {
                for (int ox = 0; ox < w; ++ox) {
                    int8_t maxValue = std::numeric_limits<int8_t>::min();
                    for (int fy = 0; fy < 5; ++fy) {
                        int iy = oy + fy - padY;
                        if (iy < 0 || iy >= h) {
                            continue;
                        }
                        for (int fx = 0; fx < 5; ++fx) {
                            int ix = ox + fx - padX;
                            if (ix < 0 || ix >= w) {
                                continue;
                            }
                            int offset = ((b * c + ch) * h + iy) * w + ix;
                            maxValue = std::max(maxValue, input[offset]);
                        }
                    }
                    int outputOffset = ((b * c + ch) * h + oy) * w + ox;
                    output[outputOffset] = maxValue;
                }
            }
        }
    }
    return output;
}

static std::vector<float> referenceGlobalMaxPool(const std::vector<float>& input, int n, int c, int h, int w) {
    std::vector<float> output(n * c, -16777216.0f);
    for (int b = 0; b < n; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            float maxValue = -16777216.0f;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = ((b * c + ch) * h + y) * w + x;
                    maxValue = std::max(maxValue, input[offset]);
                }
            }
            output[b * c + ch] = maxValue;
        }
    }
    return output;
}

class MaxPoolExplicitPadsTest : public MNNTestCase {
public:
    virtual ~MaxPoolExplicitPadsTest() = default;
    bool runCase(int precision, int padX, int padY, const std::vector<int>& pads) {
        const int n = 1;
        const int c = 16;
        const int h = 6;
        const int w = 7;
        std::vector<float> inputData(n * c * h * w);
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = (ch * h + y) * w + x;
                    inputData[offset] =
                        static_cast<float>((ch % 5) * 0.25f + y * 1.7f - x * 0.6f + (offset % 11) * 0.13f);
                }
            }
        }

        auto input = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
        auto output = _Convert(_MaxPoolWithExplicitPads(_Convert(input, NC4HW4), padX, padY, pads, false), NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();

        auto expected = referenceMaxPool5x5(inputData, n, c, h, w, pads.size() >= 2 ? pads[0] : padY,
                                            pads.size() >= 2 ? pads[1] : padX);
        const float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1.0f : 20.0f;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), expected.data(), expected.size(),
                                               0.001f * errorScale)) {
            MNN_ERROR("MaxPoolExplicitPads test failed\n");
            return false;
        }
        return true;
    }

    bool runGlobalCase(int precision) {
        const int n = 1;
        const int c = 16;
        const int h = 6;
        const int w = 7;
        std::vector<float> inputData(n * c * h * w);
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = (ch * h + y) * w + x;
                    inputData[offset] = static_cast<float>(ch * 0.5f + y * 2.0f + x * 0.25f);
                }
            }
        }

        auto input = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
        auto output = _Convert(_MaxPoolWithExplicitPads(_Convert(input, NC4HW4), 0, 0, {2, 2, 2, 2}, true), NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();

        auto expected = referenceGlobalMaxPool(inputData, n, c, h, w);
        const float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1.0f : 20.0f;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), expected.data(), expected.size(),
                                               0.001f * errorScale)) {
            MNN_ERROR("MaxPoolExplicitPads global test failed\n");
            return false;
        }
        return true;
    }

    bool runInt8Case() {
        const int n = 1;
        const int c = 16;
        const int h = 6;
        const int w = 7;
        std::vector<int8_t> inputData(n * c * h * w);
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = (ch * h + y) * w + x;
                    inputData[offset] = static_cast<int8_t>((ch * 11 + y * 17 - x * 13 + offset * 3) % 127 - 63);
                }
            }
        }

        auto input = _Input({n, c, h, w}, NCHW, halide_type_of<int8_t>());
        const std::vector<int> pads = {2, 2, 2, 2};
        auto output =
            _Convert(_MaxPoolWithExplicitPads(_Convert(input, NC4HW4), 0, 0, pads, false, DataType_DT_INT8), NCHW);
        ::memcpy(input->writeMap<int8_t>(), inputData.data(), inputData.size() * sizeof(int8_t));
        input->unMap();

        auto expected = referenceMaxPool5x5Int8(inputData, n, c, h, w, pads[0], pads[1]);
        if (!checkVector<int8_t>(output->readMap<int8_t>(), expected.data(), expected.size(), 0)) {
            MNN_ERROR("MaxPoolExplicitPads int8 test failed\n");
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        // Converted YOLOv10 MaxPool currently carries both legacy padX/padY and explicit pads metadata.
        if (!runCase(precision, 2, 2, {2, 2, 2, 2})) {
            return false;
        }
        // Distinguishes explicit pads() handling from an implementation that only reads padX/padY.
        if (!runCase(precision, 0, 0, {2, 2, 2, 2})) {
            return false;
        }
        auto backendType = getCurrentType();
        if ((backendType == MNN_FORWARD_CPU || backendType == MNN_FORWARD_CPU_EXTENSION) && !runInt8Case()) {
            return false;
        }
        return runGlobalCase(precision);
    }
};

MNNTestSuiteRegister(MaxPoolExplicitPadsTest, "op/MaxPoolExplicitPads");

class AvgPoolAccumulationTest : public MNNTestCase {
    bool runCase(int size, int kernel, int pad, AvgPoolCountType countType) {
        const int batch = 2, channels = 13;
        const float values[] = {1.0f, -0.5f, 2.0f, 8192.0f, 0.25f};
        auto input = _Input({batch, channels, size, size}, NCHW, halide_type_of<float>());
        auto inputData = input->writeMap<float>();
        for (int nc = 0; nc < batch * channels; ++nc) {
            std::fill(inputData + nc * size * size, inputData + (nc + 1) * size * size, values[nc % 5]);
        }
        input->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_Pooling;
        op->main.type = OpParameter_Pool;
        op->main.value = new PoolT;
        auto pool = op->main.AsPool();
        pool->type = PoolType_AVEPOOL;
        pool->kernelX = pool->kernelY = kernel;
        pool->strideX = pool->strideY = 1;
        pool->padX = pool->padY = pad;
        pool->padType = PoolPadType_CAFFE;
        pool->countType = countType;
        pool->ceilModel = false;
        auto output = _Convert(Variable::create(Expr::create(op.get(), {_Convert(input, NC4HW4)})), NCHW);
        const int outputSize = size + 2 * pad - kernel + 1;
        auto info = output->getInfo();
        auto actual = output->readMap<float>();
        if (info == nullptr || actual == nullptr ||
            info->dim != std::vector<int>({batch, channels, outputSize, outputSize})) {
            MNN_ERROR("AvgPoolAccumulation: invalid output shape\n");
            return false;
        }
        for (int nc = 0; nc < batch * channels; ++nc) {
            for (int y = 0; y < outputSize; ++y) {
                for (int x = 0; x < outputSize; ++x) {
                    const int validH = std::min(y - pad + kernel, size) - std::max(y - pad, 0);
                    const int validW = std::min(x - pad + kernel, size) - std::max(x - pad, 0);
                    const int count = countType == AvgPoolCountType_EXCLUDE_PADDING ? validH * validW : kernel * kernel;
                    const float expected = values[nc % 5] * validH * validW / count;
                    const int index = (nc * outputSize + y) * outputSize + x;
                    if (!std::isfinite(actual[index]) ||
                        std::fabs(actual[index] - expected) > 0.001f * std::max(1.0f, std::fabs(expected))) {
                        MNN_ERROR("AvgPoolAccumulation: size=%d kernel=%d pad=%d countType=%d at %d: %f != %f\n", size,
                                  kernel, pad, countType, index, actual[index], expected);
                        return false;
                    }
                }
            }
        }
        return true;
    }

public:
    bool run(int precision) override {
        MNNTEST_CPU_ONLY();
        bool valid = runCase(112, 112, 0, AvgPoolCountType_DEFAULT);
        valid = runCase(112, 112, 1, AvgPoolCountType_EXCLUDE_PADDING) && valid;
        valid = runCase(112, 112, 1, AvgPoolCountType_INCLUDE_PADDING) && valid;
        valid = runCase(7, 3, 1, AvgPoolCountType_EXCLUDE_PADDING) && valid;
        return valid;
    }
};

MNNTestSuiteRegister(AvgPoolAccumulationTest, "op/AvgPoolAccumulation");
