//
//  NormalizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class NormalizeTest : public MNNTestCase {
public:
    static void _refNormalize(float* dst, const float* src, int batch, int channel, int area, float* scale, float eps) {
        // Normalize
        for (int b=0; b<batch; ++b) {
            for (int x=0; x<area; ++x) {
                auto dstX = dst + b * area * channel + x;
                auto srcX = src + b * area * channel + x;
                float sumSquare = 0.0f;
                for (int c=0; c<channel; ++c) {
                    sumSquare += (srcX[area * c] * srcX[area * c]);
                }
                float normalValue = 1.0f / sqrtf(sumSquare + eps);
                for (int c=0; c<channel; ++c) {
                    dstX[area*c] = srcX[area * c] * normalValue * scale[c];
                }
            }
        }
    }
    virtual ~NormalizeTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NCHW);
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input = _Convert(input, NC4HW4);
        std::vector<float> scaleData = {0.5f, 0.5f};
        float eps = 0.00f;
        auto output = _Normalize(input, 0, 0, eps, scaleData);
        output = _Convert(output, NCHW);
        std::vector<float> expectedOutput(4);
        _refNormalize(expectedOutput.data(), inpudata, 1, 2, 2, scaleData.data(), eps);
        auto gotOutput                        = output->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
        if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 4, 1e-5 * errorScale)) {
            MNN_ERROR("NormalizeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(NormalizeTest, "op/normalize");
