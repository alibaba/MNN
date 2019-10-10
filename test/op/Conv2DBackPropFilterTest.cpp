//
//  Conv2DBackPropFilterTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/9/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "Expr.hpp"
#include "ExprCreator.hpp"
#include "TestUtils.h"

using namespace MNN::Express;

class Conv2DBackPropFilterTest : public MNNTestCase{
public:
    virtual ~Conv2DBackPropFilterTest() = default;
    virtual bool run(){
        
        auto input = _Input({1,3,5,5}, NCHW);
        
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {
            0.9329, 0.8632, 0.4275, 0.6670, 0.1923,
            0.6141, 0.8261, 0.0899, 0.1442, 0.7056,
            0.5515, 0.0435, 0.5664, 0.3330, 0.8119,
            0.8131, 0.2928, 0.5145, 0.2485, 0.2596,
            0.3923, 0.8260, 0.7251, 0.7897, 0.9686, // the first channel
            0.5073, 0.2214, 0.2474, 0.3628, 0.0242,
            0.1869, 0.4747, 0.3383, 0.6147, 0.8212,
            0.0944, 0.4912, 0.2376, 0.2423, 0.6194,
            0.4229, 0.2750, 0.2160, 0.6690, 0.4680,
            0.6593, 0.6406, 0.7864, 0.0265, 0.3638, // the third channel
            0.6708, 0.3008, 0.4975, 0.8371, 0.4141,
            0.4837, 0.9709, 0.9418, 0.5752, 0.7287,
            0.4387, 0.4936, 0.5065, 0.1497, 0.3947,
            0.4060, 0.3319, 0.9262, 0.9229, 0.7986,
            0.8909, 0.5558, 0.7642, 0.5227, 0.9615}; // the last channel
        auto inputPtr = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 75 * sizeof(float));
        input->unMap();
        
        input = _Convert(input, NC4HW4);
        auto weight = _Const(1.0, {1, 3, 3, 3}, NCHW);
        auto bias = _Const(0.0, {1}, NCHW);
        
        auto convOut = _Conv(weight, bias, input);
        auto convOutDims = convOut->getInfo()->dim;
        
        auto grad = _Const(1.0, convOutDims, NCHW);
        grad = _Convert(grad, NC4HW4);
        auto weightGrad = _Conv2DBackPropFilter(weight, input, grad);
        weightGrad->setName("Conv2DBackPropFilter");
        weightGrad = _Convert(weightGrad, NCHW);
        weightGrad->setName("nc4hw4_to_nchw");
        
        
        auto weightGradDims = weightGrad->getInfo()->dim;
        const std::vector<int> expectedDims = {1,3,3,3};
        if(!checkVector<int>(weightGradDims.data(), expectedDims.data(), 4, 0)){
            MNN_ERROR("Conv2DBackPropFilter's output shape compute ERROR!\n");
            return false;
        }
        const std::vector<float> expectedWeightGrad = {
            4.9151, 3.9609, 3.9378,
            4.3119, 3.0589, 3.6735,
            4.7253, 4.3395, 5.2173,
            2.7992, 3.2303, 3.5078,
            2.7370, 3.5589, 4.2264,
            3.8235, 3.5845, 3.6288,
            5.3044, 5.2732, 5.0453,
            5.4994, 5.8188, 5.9443,
            5.3138, 5.1735, 5.9469};
        auto weightGradPtr = weightGrad->readMap<float>();
        if(!checkVector<float>(weightGradPtr, expectedWeightGrad.data(), 27, 0.01)){
            MNN_ERROR("Conv2DBackPropFilter test failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(Conv2DBackPropFilterTest, "op/Conv2DBackPropFilter");
