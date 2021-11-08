//
//  CropAndResizeTest.cpp
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
            x = x + ::ceil(1 - x) * 4;
        }
        // reflect
        if (x > 1.0f) {
            float l = x - 1.0f;
            int reflectionNum = ::floor(l / 2.0);
            float offset = l - reflectionNum * 2.0f;
            x = (reflectionNum % 2 == 0) ? (1 - offset) : (-1.0f + offset);
        }
    }

    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1 + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

static float sample(int h, int w, const float *buffer, int height, int width, GridSamplePaddingMode paddingMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if (paddingMode == GRID_SAMPLE_PADDING_ZEROS) {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height-1);
        w = CLAMP(w, 0, width-1);
    }

    return buffer[h * width + w];
}

static float interpolate(float h, float w, const float *buffer, int height, int width, InterpolationMethod mode,
                         GridSamplePaddingMode paddingMode) {
    if (mode == NEAREST) {
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        return sample(nh, nw, buffer, height, width, paddingMode);
    }

    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = w0_h + 1;
    int w1_w = w0_w + 1;

    float i00 = sample(w0_h, w0_w, buffer, height, width, paddingMode);
    float i01 = sample(w0_h, w1_w, buffer, height, width, paddingMode);
    float i10 = sample(w1_h, w0_w, buffer, height, width, paddingMode);
    float i11 = sample(w1_h, w1_w, buffer, height, width, paddingMode);

    float i0 = i00 * (w1_w - w) + i01 * (w - w0_w);
    float i1 = i10 * (w1_w - w) + i11 * (w - w0_w);

    return i0 * (w1_h - h) + i1 * (h - w0_h);
}

static void reference_grid_sample(const float *inputPtr, const float *gridPtr, std::vector<float> &output,
                                  int batch, int inHeight, int inWidth, int outHeight, int outWidth, int depth,
                                  InterpolationMethod mode, GridSamplePaddingMode paddingMode, bool alignCorners) {
    output.resize(batch * outHeight * outWidth * depth);

    float *outputPtr = output.data();
    for (auto b = 0; b < batch; ++b) {
        const float *_inputPtr = inputPtr + b * inHeight * inWidth * depth;
        const float *_gridPtr = gridPtr + b * outHeight * outWidth * 2;
        float *_outputPtr = outputPtr + b * outHeight * outWidth * depth;

        for (auto c = 0; c < depth; ++c) {
            auto __inputPtr = _inputPtr + c * inHeight * inWidth;
            auto __outputPtr = _outputPtr + c * outHeight * outWidth;

            for (auto h = 0; h < outHeight; ++h) {
                auto __gridPtr = _gridPtr + h * outWidth * 2;
                auto ___outputPtr = __outputPtr + h * outWidth;

                for (auto w = 0; w < outWidth; ++w) {
                    auto x = getPosition(__gridPtr[2 * w + 0], inWidth, alignCorners, paddingMode);
                    auto y = getPosition(__gridPtr[2 * w + 1], inHeight, alignCorners, paddingMode);

                    ___outputPtr[w] = interpolate(y, x, __inputPtr, inHeight, inWidth, mode, paddingMode);
                }
            }
        }
    }

}

/**
 @brief check the result with the ground truth
 @param result data
 @param rightData
 @param size
 @param threshold
 */
template <typename T>
bool checkVector(const T* result, const T* rightData, int size, T threshold, T ratio){
    MNN_ASSERT(result != nullptr);
    MNN_ASSERT(rightData != nullptr);
    MNN_ASSERT(size >= 0);
    int count = 0;
    for(int i = 0; i < size; ++i){
        if(fabs(result[i] - rightData[i]) > threshold){
            //std::cout << "right: " << rightData[i] << ", compute: " << result[i] << std::endl;
            count ++;
        }
    }

    float miss_match_ratio = 1.0f*count/size;
    if (miss_match_ratio > ratio) {
        std::cout << "ratio threshold: " << ratio << ", miss match ratio: " << miss_match_ratio << std::endl;
        return false;
    }

    return true;
}


class GridSampleTest : public MNNTestCase {
public:
    virtual ~GridSampleTest() = default;

    virtual bool run(int precision) {
        const std::vector<std::vector<int>> configs({
            {1, 3, 5, 10, 5, 10},
            {1, 62, 6, 10, 12, 20},
            {2, 64, 12, 20, 6, 6},
            {1, 3, 384, 640, 384, 640},
        });

        for (auto config : configs) {
            const int batch = config[0];
            const int depth = config[1];
            const int inHeight = config[2];
            const int inWidth = config[3];
            const int outHeight = config[4];
            const int outWidth = config[5];

            auto input = _Input({batch, depth, inHeight, inWidth}, NCHW);
            auto grid = _Input({batch, outHeight, outWidth, 2}, NCHW);

            auto inputPtr = input->writeMap<float>();
            auto gridPtr = grid->writeMap<float>();

            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> inputDist{0.0f, 1.0};
            std::normal_distribution<> gridDist{0.0f, 3.0f / outWidth};

            for (int i = 0; i < batch * inHeight * inWidth * depth; i++) {
                inputPtr[i] = inputDist(gen);
            }
            for (int b = 0; b < batch; b++) {
                for (int h = 0; h < outHeight; h++) {
                    for (int w = 0; w < outWidth; w++) {
                        float offsetH = gridDist(gen);
                        float offsetW = gridDist(gen);
                        gridPtr[b * outHeight * outWidth * 2 + h * outWidth * 2 + w * 2 + 0] =
                        2.0f * w / (outWidth-1) - 1.0f + offsetW;
                        gridPtr[b * outHeight * outWidth * 2 + h * outWidth * 2 + w * 2 + 1] =
                        2.0f * h / (outHeight-1) - 1.0f + offsetH;
                    }
                }
            }

            std::vector<InterpolationMethod> modes({BILINEAR});
            std::vector<GridSamplePaddingMode> paddingModes({GRID_SAMPLE_PADDING_ZEROS});
            std::vector<bool> alignCornersVec({false});

#define MNN_METAL_FULL_PRECISION 0
#if MNN_METAL_FULL_PRECISION
            bool usingMetalLowPrecision = false;
#else
            auto runtime = MNN::Express::Executor::getGlobalExecutor()->getRuntime();
            bool usingMetalLowPrecision = runtime.first.find(MNN_FORWARD_METAL) != runtime.first.end();
#endif

            std::vector<float> expectedOutput(batch * outHeight * outWidth * depth);
            for (auto mode : modes) {
                for (auto paddingMode : paddingModes) {
                    for (auto alignCorners : alignCornersVec) {
                        reference_grid_sample(inputPtr, gridPtr, expectedOutput,
                                              batch, inHeight, inWidth, outHeight, outWidth, depth,
                                              mode, paddingMode, alignCorners);
                        auto expectedOutPtr = expectedOutput.data();

                        grid->unMap();
                        input->unMap();
                        input = _Convert(input, NC4HW4);

                        auto output = _GridSample(input, grid, mode, paddingMode, alignCorners);
                        output      = _Convert(output, NCHW);
                        auto outputPtr = output->readMap<float>();

                        if (usingMetalLowPrecision) {
                            if (!checkVector<float>(outputPtr, expectedOutPtr, expectedOutput.size(), 0.2, 0.01)) {
                                MNN_ERROR("GridSampleTest test failed!\n");
                                return false;
                            }
                        } else {
                            if (mode == NEAREST) {
                                if (!checkVector<float>(outputPtr, expectedOutPtr, expectedOutput.size(), 0.01, 0.001)) {
                                    MNN_ERROR("GridSampleTest NEAREST test failed!\n");
                                    return false;
                                }
                            } else {
                                if (!checkVector<float>(outputPtr, expectedOutPtr, expectedOutput.size(), 0.01)) {
                                    MNN_ERROR("GridSampleTest BILINEAR test failed!\n");
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(GridSampleTest, "op/GridSample");
