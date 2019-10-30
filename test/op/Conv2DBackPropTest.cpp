//
//  Conv2DBackPropTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/9/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "Expr.hpp"
#include "ExprCreator.hpp"
#include "TestUtils.h"

using namespace MNN::Express;

class Conv2DBackPropTest : public MNNTestCase{
    virtual ~Conv2DBackPropTest() = default;
    
    virtual bool run(){
        
        const float inputGradData[] = {
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1
        }; // 1x1x3x3
        
        auto inputGrad = _Const(inputGradData, {1, 1, 3, 3}, NCHW);
        inputGrad = _Convert(inputGrad, NC4HW4);
        
        const float weightData[] = {
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.,
            1., 1., 1.}; // 1x3x3x3
        auto weight = _Const(weightData, {1,3,3,3}, NCHW);
        auto bias = _Const(0., {1}, NCHW);
        
        auto outputGrad = _Deconv(weight, bias, inputGrad);
        outputGrad = _Convert(outputGrad, NCHW);
        auto outputGradDim = outputGrad->getInfo()->dim;
        
        const int outSize = outputGrad->getInfo()->size;
        if(outputGrad->getInfo()->size != outSize){
            return false;
        }
        
        const std::vector<int> expectedDim = {1,3,5,5};
        if(!checkVector<int>(outputGradDim.data(), expectedDim.data(), 4, 0)){
            MNN_ERROR("Conv2DBackProp shape test failed!\n");
            return false;
        }
        
        const float expectedOutputGrad[] = {
            1., 2., 3., 2., 1.,
            2., 4., 6., 4., 2.,
            3., 6., 9., 6., 3.,
            2., 4., 6., 4., 2.,
            1., 2., 3., 2., 1.,
            1., 2., 3., 2., 1.,
            2., 4., 6., 4., 2.,
            3., 6., 9., 6., 3.,
            2., 4., 6., 4., 2.,
            1., 2., 3., 2., 1.,
            1., 2., 3., 2., 1.,
            2., 4., 6., 4., 2.,
            3., 6., 9., 6., 3.,
            2., 4., 6., 4., 2.,
            1., 2., 3., 2., 1.};
        auto outputGradData = outputGrad->readMap<float>();
        
        if(!checkVector<float>(outputGradData, expectedOutputGrad, outSize, 0.01)){
            MNN_ERROR("Conv2DBackProp test failed!\n");
            return false;
        }
        
        return true;
    }
};

MNNTestSuiteRegister(Conv2DBackPropTest, "op/Conv2DBackPropTest");
