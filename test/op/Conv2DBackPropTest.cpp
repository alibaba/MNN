//
//  Conv2DBackPropTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNForwardType.h>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN::Express;

class Conv2DBackPropTest : public MNNTestCase {
public:
    virtual ~Conv2DBackPropTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName) {
        const float inputGradData[] = {1., 1., 1., 1., 1., 1., 1., 1., 1}; // 1x1x3x3

        auto inputGrad = _Const(inputGradData, {1, 1, 3, 3}, NCHW);
        inputGrad      = _Convert(inputGrad, NC4HW4);

        const float weightData[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}; // 1x3x3x3
        auto weight              = _Const(weightData, {1, 3, 3, 3}, NCHW);
        auto bias                = _Const(0., {3}, NCHW);

        auto outputGrad    = _Deconv(weight, bias, inputGrad);
        outputGrad         = _Convert(outputGrad, NCHW);
        auto outputGradDim = outputGrad->getInfo()->dim;

        const int outSize = outputGrad->getInfo()->size;
        if (outputGrad->getInfo()->size != outSize) {
            return false;
        }

        const std::vector<int> expectedDim = {1, 3, 5, 5};
        if (!checkVector<int>(outputGradDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("Conv2DBackProp(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        const float expectedOutputGrad[] = {1., 2., 3., 2., 1., 2., 4., 6., 4., 2., 3., 6., 9., 6., 3., 2., 4., 6., 4.,
                                            2., 1., 2., 3., 2., 1., 1., 2., 3., 2., 1., 2., 4., 6., 4., 2., 3., 6., 9.,
                                            6., 3., 2., 4., 6., 4., 2., 1., 2., 3., 2., 1., 1., 2., 3., 2., 1., 2., 4.,
                                            6., 4., 2., 3., 6., 9., 6., 3., 2., 4., 6., 4., 2., 1., 2., 3., 2., 1.};
        auto outputGradData              = outputGrad->readMap<float>();

        if (!checkVector<float>(outputGradData, expectedOutputGrad, outSize, 0.005)) {
            MNN_ERROR("Conv2DBackProp(%s) test failed!\n", deviceName.c_str());
            return false;
        }

        return true;
    }
};

class Conv2DBackPropTestOnCPU : public Conv2DBackPropTest {
    virtual ~Conv2DBackPropTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU");
    }
};

class Conv2DBackPropTestOnOpencl : public Conv2DBackPropTest {
    virtual ~Conv2DBackPropTestOnOpencl() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL");
    }
};

MNNTestSuiteRegister(Conv2DBackPropTestOnCPU, "op/Conv2DBackPropTest");

class ConvBiasGradTest : public MNNTestCase {
public:
    virtual ~ConvBiasGradTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName, int precision) {
        const int height = 32, width = 32, channel = 32, batch = 16;
        std::vector<float> gradData(height * width * channel * batch, 0);
        for (unsigned int i = 0; i < gradData.size(); ++i) {
            gradData[i] = (float)rand() / RAND_MAX;
        }

        std::vector<float> outputData(channel, 0);
        for (unsigned int i = 0; i < gradData.size(); ++i) {
            outputData[(i / (height * width)) % channel] += gradData[i];
        }

        auto grad                     = _Input({batch, channel, height, width}, NCHW, halide_type_of<float>());
        auto output                   = _Convert(_ReduceSum(grad, {0, 2, 3}, false), NCHW);
        const std::vector<int> outDim = {channel};
        if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 1, 0)) {
            MNN_ERROR("ConvBiasGradTest(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(grad->writeMap<float>(), gradData.data(), gradData.size() * sizeof(float));
        // difference below 0.5% relative error is considered correct.
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(), 0.005 * errorScale)) {
            MNN_ERROR("ConvBiasGradTest(%s) test failed!\n", deviceName.c_str());
            return false;
        }
        return true;
    }
};

class ConvBiasGradTestOnCPU : public ConvBiasGradTest {
    virtual ~ConvBiasGradTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class ConvBiasGradTestOnOpencl : public ConvBiasGradTest {
    virtual ~ConvBiasGradTestOnOpencl() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL", precision);
    }
};

MNNTestSuiteRegister(ConvBiasGradTestOnCPU, "op/bias_grad");
