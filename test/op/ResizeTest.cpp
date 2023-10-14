//
//  ResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class ResizeTest : public MNNTestCase {
public:
    virtual ~ResizeTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
        auto output                             = _Resize(input, 2.0, 2.0);
        output                                  = _Convert(output, NHWC);
        const std::vector<float> expectedOutput = {-1.0, -1.5, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0,
                                                   3.0,  3.5,  4.0,  4.0,  3.0, 3.5, 4.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
            MNN_ERROR("ResizeTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {1, 4, 4, 1};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("ResizeTest test failed!\n");
            return false;
        }
        return true;
    }
};

class InterpTest : public MNNTestCase {
public:
    virtual ~InterpTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);

        float hScale = 2.0;
        float wScale = 2.0;
        float scales[] = {1.0, 1.0, hScale, wScale};
        auto scaleVar = _Const((void*)scales, {4}, NCHW);
        int outW = int(wScale * 2);
        int outH = int(hScale * 2);

        //Interp Type:1
        {
            auto output                             = _Interp({input, scaleVar}, wScale, hScale, outW, outH, 1, false);
            output                                  = _Convert(output, NHWC);
            const std::vector<float> expectedOutput = {-1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -2.0, -2.0,
                                                        3.0,  3.0,  4.0,  4.0,  3.0, 3.0, 4.0, 4.0};
            auto gotOutput                          = output->readMap<float>();

            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("InterpType:1 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:1 test failed!\n");
                return false;
            }
        }

        //Interp Type:2
        {
            auto output                             = _Interp({input, scaleVar}, wScale, hScale, outW, outH, 2, false);
            output                                  = _Convert(output, NHWC);
            const std::vector<float> expectedOutput = { -1.0000, -1.2500, -1.7500, -2.0000,  0.0000, -0.1250, -0.3750, -0.5000,
                                                        2.0000,  2.1250,  2.3750,  2.5000,  3.0000,  3.2500,  3.7500,  4.0000};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("InterpType:2 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:2 test failed!\n");
                return false;
            }
        }
        
        //Interp Type:3
        {
            auto output                             = _Interp({input, scaleVar}, wScale, hScale, outW, outH, 3, false);
            output                                  = _Convert(output, NHWC);
            const std::vector<float> expectedOutput = { 2.516724, 2.217651, 2.516724, 2.698303, -0.516724, -0.217651, -0.516724, -0.698303, 2.516724, 2.217651, 2.516724, 2.698303, 4.358459, 3.696228, 4.358459, 4.760529};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("InterpType:3 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:3 test failed!\n");
                return false;
            }
        }
        return true;
    }
};

class InterpInt8Test : public MNNTestCase {
public:
    virtual ~InterpInt8Test() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NHWC);
        input->setName("input_tensor");
        input->writeScaleMap(0.032, 1.f);
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);

        float hScale = 2.0;
        float wScale = 2.0;
        float scales[] = {1.0, 1.0, hScale, wScale};
        auto scaleVar = _Const((void*)scales, {4}, NCHW);
        int outW = int(wScale * 2);
        int outH = int(hScale * 2);
        
        //Interp Type:1
        {
            printf("InterpInt8 test: Type=1\n");
            auto output                             = _Interp({input, scaleVar}, wScale, hScale, outW, outH, 1, false);
            output                                  = _Convert(output, NHWC);
            output->writeScaleMap(0.031372549, 0.f);
            const std::vector<float> expectedOutput = {-1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -2.0, -2.0,
                                                        3.0,  3.0,  4.0,  4.0,  3.0, 3.0, 4.0, 4.0};
            auto gotOutput                          = output->readMap<float>();

            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.05)) {
                MNN_ERROR("InterpInt8 ResizeType=1 :test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpInt8 ResizeType=1: test failed!\n");
                return false;
            }
        }

        //Interp Type:2
        {
            printf("InterpInt8 test: Type=2\n");
            auto output                             = _Interp({input, scaleVar}, wScale, hScale, outW, outH, 2, false);
            output                                  = _Convert(output, NHWC);
            output->writeScaleMap(0.031372549, 2.);
            const std::vector<float> expectedOutput = { -1.0000, -1.2500, -1.7500, -2.0000,  0.0000, -0.1250, -0.3750, -0.5000,
                                                        2.0000,  2.1250,  2.3750,  2.5000,  3.0000,  3.2500,  3.7500,  4.0000};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.05)) {
                MNN_ERROR("InterpInt8 ResizeType=2 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpInt8 ResizeType=2 test failed!\n");
                return false;
            }
        }
        
        // Interp Type:3
        {
            printf("InterpInt8 test: Type=3\n");
            auto input0 = _Input({1, 2, 2, 1}, NHWC);
            input0->setName("input_tensor");
            input0->writeScaleMap(0.05, 1.f); // Fix quant scale.
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
            auto inputPtr          = input0->writeMap<float>();
            memcpy(inputPtr, inpudata, 4 * sizeof(float));
            input0->unMap();
            input0                                   = _Convert(input0, NC4HW4);
            auto output                             = _Interp({input0, scaleVar}, wScale, hScale, outW, outH, 3, false);
            output                                  = _Convert(output, NHWC);
            output->writeScaleMap(0.03967, 1.);
            const std::vector<float> expectedOutput = { 2.516724, 2.217651, 2.516724, 2.698303, -0.516724, -0.217651, -0.516724, -0.698303, 2.516724, 2.217651, 2.516724, 2.698303, 4.358459, 3.696228, 4.358459, 4.760529};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.02)) {
                MNN_ERROR("InterpType:3 test failed!\n");
                return false;
            }

            const std::vector<int> expectedDim = {1, 4, 4, 1};
            auto gotDim                        = output->getInfo()->dim;
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("InterpType:3 test failed!\n");
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(ResizeTest, "op/resize");
MNNTestSuiteRegister(InterpTest, "op/Interp");
MNNTestSuiteRegister(InterpInt8Test, "op/InterpInt8");
