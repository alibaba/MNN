//
//  GridSampler3DTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <cfenv>
#include <cmath>
#include <random>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

static float getPosition(float x, int range, bool alignCorners, GridSamplePaddingMode paddingMode) {
    if (paddingMode == GRID_SAMPLE_PADDING_REFLECTION) {
        // if x is on the left side of -1.0, move it to the right side of 1.0
        if (x < -1.0f) {
            x = (x + ::ceil(1 - x) * 4);
        }
        // reflect
        if (x > 1.0f) {
            float l = (x - 1.0f);
            int reflectionNum = ::floor(l / 2.0);
            float offset = (l - reflectionNum * 2.0f);
            x = (reflectionNum % 2 == 0) ? (1 - offset) : (-1.0f + offset);
        }
    }

    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return (((1 + x) * (range - a) - b) / 2.0f);
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

static float sample(int d, int h, int w, const float *buffer, int depth, int height, int width, GridSamplePaddingMode paddingMode) {
    if (h < 0 || h >= height || w < 0 || w >= width || d < 0 || d >= depth) {
        if (paddingMode == GRID_SAMPLE_PADDING_ZEROS) {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height-1);
        w = CLAMP(w, 0, width-1);
        d = CLAMP(d, 0, depth-1);
    }

    return buffer[d * height * width + h * width + w];
}

static float interpolate(float d, float h, float w, const float *buffer, int depth, int height, int width, InterpolationMethod mode,
                         GridSamplePaddingMode paddingMode) {
    if (mode == NEAREST) {
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        int nd = ::floor(d+0.5f);
        return sample(nd, nh, nw, buffer, depth, height, width, paddingMode);
    }

    // mode == GridSampleMode_BILINEAR
    int d0 = ::floor(d);
    int d1 = ::ceil(d);
    int h0 = ::floor(h);
    int h1 = ::ceil(h);
    int w0 = ::floor(w);
    int w1 = ::ceil(w);
    float fx2 = w - w0;
    float fx1 = 1.0f - fx2;
    float fy2 = h - h0;
    float fy1 = 1.0f - fy2;
    float fz2 = d - d0;
    float fz1 = 1.0f - fz2;

    float i000 = sample(d0, h0, w0, buffer, depth, height, width, paddingMode);
    float i001 = sample(d0, h0, w1, buffer, depth, height, width, paddingMode);
    float i010 = sample(d0, h1, w0, buffer, depth, height, width, paddingMode);
    float i011 = sample(d0, h1, w1, buffer, depth, height, width, paddingMode);
    float i100 = sample(d1, h0, w0, buffer, depth, height, width, paddingMode);
    float i101 = sample(d1, h0, w1, buffer, depth, height, width, paddingMode);
    float i110 = sample(d1, h1, w0, buffer, depth, height, width, paddingMode);
    float i111 = sample(d1, h1, w1, buffer, depth, height, width, paddingMode);

    float i00 = ((i000) * fx1 + (i001) * fx2);
    float i01 = ((i010) * fx1 + (i011) * fx2);
    float i10 = ((i100) * fx1 + (i101) * fx2);
    float i11 = ((i110) * fx1 + (i111) * fx2);
    
    float i0 = i00 * fy1 + i01 * fy2;
    float i1 = i10 * fy1 + i11 * fy2;

    return ((i0 * fz1) + (i1 * fz2));
}

static void reference_grid_sample(const float *inputPtr, const float *gridPtr, std::vector<float> &output,
                                  int batch, int inDepth, int inHeight, int inWidth, int outDepth, int outHeight, int outWidth, int channel,
                                  InterpolationMethod mode, GridSamplePaddingMode paddingMode, bool alignCorners) {
    output.resize(batch * outHeight * outWidth * channel * outDepth);

    float *outputPtr = output.data();
    for (auto b = 0; b < batch; ++b) {
        const float *_inputPtr = inputPtr + b * inDepth * inHeight * inWidth * channel;
        const float *_gridPtr = gridPtr + b * outDepth * outHeight * outWidth * 3;
        float *_outputPtr = outputPtr + b * outDepth * outHeight * outWidth * channel;

        for (auto c = 0; c < channel; ++c) {
            auto __inputPtr = _inputPtr + c * inDepth * inHeight * inWidth;
            auto __outputPtr = _outputPtr + c * outDepth * outHeight * outWidth;
            for (int d = 0; d < outDepth; ++d) {
                for (auto h = 0; h < outHeight; ++h) {
                    auto __gridPtr = _gridPtr + (d * outWidth * outHeight + h * outWidth) * 3;
                    auto ___outputPtr = __outputPtr + d * outHeight * outWidth + h * outWidth;

                    for (auto w = 0; w < outWidth; ++w) {
                        auto x = getPosition(__gridPtr[3 * w + 0], inWidth, alignCorners, paddingMode);
                        auto y = getPosition(__gridPtr[3 * w + 1], inHeight, alignCorners, paddingMode);
                        auto z = getPosition(__gridPtr[3 * w + 2], inDepth, alignCorners, paddingMode);

                        ___outputPtr[w] = interpolate(z, y, x, __inputPtr, inDepth, inHeight, inWidth, mode, paddingMode);
                    }
                }
            }
        }
    }
}


class GridSample3DTest : public MNNTestCase {
public:
    virtual ~GridSample3DTest() = default;

    virtual bool run(int precision) {
        auto type = getCurrentType();

        const std::vector<std::vector<int>> configs({
            {1, 3, 5, 10, 5, 10, 3, 5},
            {1, 62, 6, 10, 12, 20, 1, 2},
            {2, 64, 12, 20, 6, 6, 5, 1},
            {1, 3, 384, 640, 384, 640, 2, 2},
        });

        for (auto config : configs) {
            const int batch = config[0];
            const int depth = config[1];
            const int inHeight = config[2];
            const int inWidth = config[3];
            const int outHeight = config[4];
            const int outWidth = config[5];
            const int inDepth = config[6];
            const int outDepth = config[7];

            std::vector<float> originInputData(batch * depth * inHeight * inWidth * inDepth);
            std::vector<float> originGridData(batch * outHeight * outWidth * outDepth * 3);

            auto inputPtr = originInputData.data();
            auto gridPtr = originGridData.data();

            std::random_device rd{};
            std::mt19937 gen{rd()};
            gen.seed(1024);
            std::normal_distribution<> inputDist{0.0f, 1.0};
            std::normal_distribution<> gridDist{0.0f, 3.0f / outWidth};

            for (int i = 0; i < batch * inHeight * inWidth * inDepth * depth; i++) {
                inputPtr[i] = inputDist(gen);
            }
            for (int b = 0; b < batch; b++) {
                for (int d=0; d<outDepth; ++d) {
                    for (int h = 0; h < outHeight; h++) {
                        for (int w = 0; w < outWidth; w++) {
                            float offsetH = gridDist(gen);
                            float offsetW = gridDist(gen);
                            float offsetD = gridDist(gen);
                            auto basic = b * outDepth * outHeight * outWidth + d * outWidth * outHeight +  h * outWidth + w;
                            gridPtr[3 * basic + 0] = (3.0f * w / (outWidth-1) - 1.0f + offsetW);
                            gridPtr[3 * basic + 1] = (3.0f * h / (outHeight-1) - 1.0f + offsetH);
                            gridPtr[3 * basic + 2] = (3.0f * d / (outDepth-0.999f) - 1.0f + offsetD);
                        }
                    }
                }
            }
            auto input = _Input({batch, depth, inDepth, inHeight, inWidth}, NCHW);
            auto grid = _Input({batch, outDepth, outHeight, outWidth, 3}, NCHW);
            ::memcpy(input->writeMap<float>(), inputPtr, originInputData.size() * sizeof(float));
            ::memcpy(grid->writeMap<float>(), gridPtr, originGridData.size() * sizeof(float));
            input = _Convert(input, NC4HW4);

            std::vector<InterpolationMethod> modes({BILINEAR});
            std::vector<GridSamplePaddingMode> paddingModes({GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER});
            std::vector<int> alignCornersVec = {1, 0};
            std::vector<float> expectedOutput;
            for (auto mode : modes) {
                for (auto paddingMode : paddingModes) {
                    for (auto alignCorners : alignCornersVec) {
                        reference_grid_sample(inputPtr, gridPtr, expectedOutput,
                                              batch, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth, depth,
                                              mode, paddingMode, alignCorners);
                        auto expectedOutPtr = expectedOutput.data();

                        grid->unMap();
                        input->unMap();

                        auto output = _GridSample(input, grid, mode, paddingMode, alignCorners);
                        output      = _Convert(output, NCHW);
                        auto outputPtr = output->readMap<float>();
//                        MNN_PRINT("GridSamplerTest, mode: %d, pad: %d, align: %d\n", mode, paddingMode, alignCorners);

                        if (mode == NEAREST) {
                            if (!checkVector<float>(outputPtr, expectedOutPtr, expectedOutput.size(), 0.01)) {
                                MNN_ERROR("GridSampleTest NEAREST test %d-%d-%d-%d-%d failed pad mode: %d, align: %d!\n", config[0], config[1], config[2], config[3], config[4], paddingMode, alignCorners);
                                return false;
                            }
                        } else {
                            if (!checkVector<float>(outputPtr, expectedOutPtr, expectedOutput.size(), 0.01)) {
                                MNN_ERROR("GridSampleTest BILINEAR test %d-%d-%d-%d-%d failed: pad mode: %d, align: %d!\n", config[0], config[1], config[2], config[3], config[4], paddingMode, alignCorners);
                                return false;
                            }
                        }
                    }
                }
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(GridSample3DTest, "op/GridSample3D");
