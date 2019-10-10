//
//  PoolGradTest.cpp
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

class PoolGradTest : public MNNTestCase{
public:
    virtual ~PoolGradTest() = default;
    virtual bool run(){
        
        const float originInputData[] = {
            0.2025,  0.0156,  0.0765,  0.1872,  0.2949, -0.0325,  0.0052, 0.4046,
            0.0455,  0.3100,  0.0162, -0.1304,  0.2245,  0.1622,  0.2437,0.1605,
            -0.0330,  0.0641,  0.2964,  0.0452, -0.1621,  0.2534,  0.3948,0.3268,
            0.2203, -0.0665,  0.1727,  0.1119, -0.1570,  0.1260,  0.3523,-0.0115,
            0.2305,  0.1664,  0.1277,  0.4092, -0.1601,  0.0929,  0.1138,0.0624,
            0.2331,  0.3501,  0.3382,  0.2309,  0.2175,  0.0826, -0.1567,0.1327,
            0.0320,  0.1205, -0.0566,  0.1267, -0.0004,  0.2930,  0.2353,-0.1668,
            0.1653,  0.3441, -0.0312,  0.2422,  0.1814,  0.1478,  0.2195,-0.0848};
        auto poolInput = _Const(originInputData, {1, 1, 8, 8}, NCHW);
        poolInput = _Convert(poolInput, NC4HW4);
        auto poolOut = _MaxPool(poolInput, {2, 2}, {2, 2});
        auto poolOutDim = poolOut->getInfo()->dim;
        const float poolInputGradData[] = {
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.};
        
        auto poolInputGrad = _Const(poolInputGradData, poolOutDim, NCHW);
        poolInputGrad = _Convert(poolInputGrad, NC4HW4);
        
        auto maxPoolOutputGrad = _PoolGrad(poolInput, poolOut, poolInputGrad, {2, 2}, {2, 2}, MAXPOOL);
        auto avePoolOutputGrad = _PoolGrad(poolInput, poolOut, poolInputGrad, {2, 2}, {2, 2}, AVEPOOL);
        
        maxPoolOutputGrad = _Convert(maxPoolOutputGrad, NCHW);
        avePoolOutputGrad = _Convert(avePoolOutputGrad, NCHW);
        const float maxExpectedGrad[] = {
            0., 0., 0., 1., 1., 0., 0., 1.,
            0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 1., 1., 0.,
            1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0.,
            0., 1., 0., 0., 1., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 1., 0.,
            0., 1., 0., 1., 0., 0., 0., 0.};
        
        const float aveExpectedGrad[] = {
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
            0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500
        };
        
        const std::vector<int> outDim = {1, 1, 8, 8};
        auto maxpoolOutputGradDim = maxPoolOutputGrad->getInfo()->dim;
        auto avepoolOutputGradDim = avePoolOutputGrad->getInfo()->dim;
        if(!checkVector<int>(maxpoolOutputGradDim.data(), outDim.data(), 4, 0)){
            MNN_ERROR("MaxpoolGrad shape test failed!\n");
            return false;
        }
        if(!checkVector<int>(avepoolOutputGradDim.data(), outDim.data(), 4, 0)){
            MNN_ERROR("AvepoolGrad shape test failed!\n");
            return false;
        }
        auto maxpoolOutputGradData = maxPoolOutputGrad->readMap<float>();
        auto avepoolOutputGradData = avePoolOutputGrad->readMap<float>();
        
        if(!checkVector<float>(maxpoolOutputGradData, maxExpectedGrad, 64, 0.0)){
            MNN_ERROR("MaxpoolGrad test failed!\n");
            return false;
        }
        if(!checkVector<float>(avepoolOutputGradData, aveExpectedGrad, 64, 0.0)){
            MNN_ERROR("AvepoolGrad test failed!\n");
            return false;
        }
        
        
        return true;
    }
};



MNNTestSuiteRegister(PoolGradTest, "op/PoolGrad");
