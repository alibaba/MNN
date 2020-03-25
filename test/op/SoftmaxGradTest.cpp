//
//  ReluGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <MNN/MNNForwardType.h>
#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>

using namespace MNN::Express;

static VARP _SoftmaxGrad(VARP originOutput, VARP outputGrad, int axis) {
    using namespace MNN;
    std::unique_ptr<OpT> softmax(new OpT);
    softmax->type       = OpType_SoftmaxGrad;
    softmax->main.type  = OpParameter_Axis;
    softmax->main.value = new AxisT;
    softmax->main.AsAxis()->axis = axis;
    return Variable::create(Expr::create(std::move(softmax), {originOutput, outputGrad}));
}

class SoftmaxGradTest : public MNNTestCase {
public:
    virtual ~SoftmaxGradTest() = default;
protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName) {
        auto creator = MNN::MNNGetExtraBackendCreator(type);
        if (creator == nullptr) {
            MNN_ERROR("backend %d not found!\n", type);
            return false;
        }

        const int batch = 4, channel = 4, size = batch * channel;
        float originOutputData[batch][channel] = {
            0.2, 0.23, 0.3, 0.27,
            0.18, 0.33, 0.16, 0.33,
            0.15, 0.18, 0.35, 0.32,
            0.29, 0.18, 0.22, 0.31
        };
        float outputGradData[batch][channel] = {
            1., 2., 3., 4.,
            2., 3., 4., 1.,
            3., 4., 1., 2.,
            4., 1., 2., 3.
        };
        float expectGrad[batch][channel];
        for (int b = 0; b < batch; ++b) {
            float sum = 0;
            for (int c = 0; c < channel; ++c) {
                sum += originOutputData[b][c] * outputGradData[b][c];
            }
            for (int c = 0; c < channel; ++c) {
                expectGrad[b][c] = originOutputData[b][c] * (outputGradData[b][c] - sum);
            }
        }

        auto output = _Input({batch, channel}, NCHW, halide_type_of<float>());
        auto outputGrad = _Input({batch, channel}, NCHW, halide_type_of<float>());
        auto outputConvert = _Convert(output, NC4HW4);
        auto outputGradConvert = _Convert(outputGrad, NC4HW4);
        auto softmaxGrad = _Convert(_SoftmaxGrad(outputConvert, outputGradConvert, 1), NCHW);

        if (type != MNN_FORWARD_CPU) {
            Optimizer::Config config;
            config.forwardType = type;
            auto optimizer = Optimizer::create(config);
            if (optimizer == nullptr) {
                MNN_ERROR("backend %s not support\n", deviceName.c_str());
                return false;
            }
            optimizer->onExecute({softmaxGrad});
        }

        const std::vector<int> outDim = {batch, channel};
        auto softmaxGradDim = softmaxGrad->getInfo()->dim;
        if (!checkVector<int>(softmaxGradDim.data(), outDim.data(), 2, 0)) {
            MNN_ERROR("SoftmaxGrad(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(output->writeMap<float>(), (const float *)originOutputData, size * sizeof(float));
        ::memcpy(outputGrad->writeMap<float>(), (const float *)outputGradData, size * sizeof(float));
        auto compute = softmaxGrad->readMap<float>();
        if(!checkVectorByRelativeError<float>(compute, (const float *)expectGrad, size, 0.005)){
            MNN_ERROR("SoftmaxGrad(%s) test failed!\n", deviceName.c_str());
            return false;
        }
        return true;
    }
};

class SoftmaxGradTestOnCPU : public SoftmaxGradTest {
public:
    virtual ~SoftmaxGradTestOnCPU() = default;
    virtual bool run() {
        return testOnBackend(MNN_FORWARD_CPU, "CPU");
    }
};

MNNTestSuiteRegister(SoftmaxGradTestOnCPU, "op/SoftmaxGrad");
