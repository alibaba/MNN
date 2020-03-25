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

static VARP _ReluGrad(VARP originInput, VARP inputGrad) {
    using namespace MNN;
    std::unique_ptr<OpT> relu(new OpT);
    relu->type       = OpType_ReluGrad;
    relu->main.type  = OpParameter_Relu;
    relu->main.value = new ReluT;
    relu->main.AsRelu()->slope = 0.0f;
    return Variable::create(Expr::create(std::move(relu), {originInput, inputGrad}));
}

static VARP _Relu6Grad(VARP originInput, VARP inputGrad) {
    using namespace MNN;
    std::unique_ptr<OpT> relu6(new OpT);
    relu6->type       = OpType_Relu6Grad;
    relu6->main.type  = OpParameter_Relu6;
    relu6->main.value = new Relu6T;
    relu6->main.AsRelu6()->slope = 0.0f;
    return Variable::create(Expr::create(std::move(relu6), {originInput, inputGrad}));
}

class ReluGradTest : public MNNTestCase {
public:
    virtual ~ReluGradTest() = default;
protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName) {
        auto creator = MNN::MNNGetExtraBackendCreator(type);
        if (creator == nullptr) {
            MNN_ERROR("backend %d not found!\n", type);
            return false;
        }

        const int h = 4, w = 4, size = h * w;
        const std::vector<float> originInputData = {
            6.2025,  -0.0156,  0.0765,  6.1872,
            0.0455,  6.3100,  0.0162, -0.1304,
            -0.0330,  0.0641,  6.2964,  0.0452,
            0.2203, -0.0665,  0.1727,  0.1119
        };
        const std::vector<float> inputGradData = {
            1., 2., 3., 4.,
            2., 3., 4., 1.,
            3., 4., 1., 2.,
            4., 1., 2., 3.
        };
        std::vector<float> reluExpectedGrad(size), relu6ExpectedGrad(size);
        for (int i = 0; i < size; ++i) {
            bool positive = (originInputData[i] > 0);
            bool under6 = (originInputData[i] < 6);
            reluExpectedGrad[i] = (positive ? inputGradData[i] : 0);
            relu6ExpectedGrad[i] = ((positive && under6) ? inputGradData[i] : 0);
        }

        auto input = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto inputGrad = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto inputConvert = _Convert(input, NC4HW4);
        auto inputGradConvert = _Convert(inputGrad, NC4HW4);
        auto reluGrad = _Convert(_ReluGrad(inputConvert, inputGradConvert), NCHW);
        auto relu6Grad = _Convert(_Relu6Grad(inputConvert, inputGradConvert), NCHW);

        const std::vector<int> outDim = {1, 1, h, w};
        auto reluGradDim = reluGrad->getInfo()->dim;
        auto relu6GradDim = relu6Grad->getInfo()->dim;
        if (!checkVector<int>(reluGradDim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("ReluGrad(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }
        if (!checkVector<int>(relu6GradDim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("Relu6Grad(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(input->writeMap<float>(), originInputData.data(), size * sizeof(float));
        ::memcpy(inputGrad->writeMap<float>(), inputGradData.data(), size * sizeof(float));
        if(!checkVector<float>(reluGrad->readMap<float>(), reluExpectedGrad.data(), size, 1e-6)){
            MNN_ERROR("ReluGrad(%s) test failed!\n", deviceName.c_str());
            return false;
        }
        if(!checkVector<float>(relu6Grad->readMap<float>(), relu6ExpectedGrad.data(), size, 1e-6)){
            MNN_ERROR("Relu6Grad(%s) test failed!\n", deviceName.c_str());
            return false;
        }
        return true;
    }
};

class ReluGradTestOnCPU : public ReluGradTest {
public:
    virtual ~ReluGradTestOnCPU() = default;
    virtual bool run() {
        return testOnBackend(MNN_FORWARD_CPU, "CPU");
    }
};

class ReluGradTestOnOpencl : public ReluGradTest {
public:
    virtual ~ReluGradTestOnOpencl() = default;
    virtual bool run() {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL");
    }
};

MNNTestSuiteRegister(ReluGradTestOnCPU, "op/ReluGrad");
