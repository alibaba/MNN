//
//  PoolGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class PoolGradTest : public MNNTestCase {
public:
    virtual ~PoolGradTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string &deviceName, int precision) {
        const int h = 7, w = 7, size = h * w;
        const float originInputData[]   = {0.3100,  0.0156,  0.0765, 0.1872, 0.2949,  0.2949,  0.0052, 0.0455,  0.3000,
                                         0.1872,  -0.1304, 0.2939, 0.2949, 0.2437,  -0.0330, 0.0641, 0.2934,  0.0452,
                                         -0.1621, 0.2534,  0.3948, 0.2203, -0.0665, 0.1727,  0.1119, -0.1570, 0.1260,
                                         0.3523,  0.2305,  0.1664, 0.1277, 0.4092,  -0.1601, 0.0929, 0.1138,  0.2331,
                                         0.3501,  0.3382,  0.2309, 0.2175, 0.0826,  -0.1567, 0.0320, 0.1205,  -0.0566,
                                         0.1267,  -0.0004, 0.2930, 0.2353};
        const float poolInputGradData[] = {1., 2., 3., 2., 3., 1., 3., 1., 2.};
        const float maxExpectedGrad[]   = {1., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.,
                                         0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0.,
                                         0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.};
        const float aveExpectedGrad[]   = {
            0.111111, 0.111111, 0.333333, 0.222222, 0.555556, 0.333333, 0.333333, 0.111111, 0.111111, 0.333333,
            0.222222, 0.555556, 0.333333, 0.333333, 0.333333, 0.333333, 0.888889, 0.555556, 1.000000, 0.444444,
            0.444444, 0.222222, 0.222222, 0.555556, 0.333333, 0.444444, 0.111111, 0.111111, 0.555556, 0.555556,
            1.000000, 0.444444, 0.777778, 0.333333, 0.333333, 0.333333, 0.333333, 0.444444, 0.111111, 0.333333,
            0.222222, 0.222222, 0.333333, 0.333333, 0.444444, 0.111111, 0.333333, 0.222222, 0.222222};

        auto poolInput        = _Input({1, 1, h, w}, NCHW, halide_type_of<float>());
        auto poolInputConvert = _Convert(poolInput, NC4HW4);
        auto maxPoolOut       = _MaxPool(poolInputConvert, {3, 3}, {2, 2});
        auto avePoolOut       = _AvePool(poolInputConvert, {3, 3}, {2, 2});
        auto poolOutDim       = maxPoolOut->getInfo()->dim;

        int poolSize = 1;
        for (auto length : poolOutDim) {
            poolSize *= length;
        }

        auto poolInputGrad        = _Input(poolOutDim, NCHW, halide_type_of<float>());
        auto poolInputGradConvert = _Convert(poolInputGrad, NC4HW4);

        auto maxPoolOutputGrad =
            _Convert(_PoolGrad(poolInputConvert, maxPoolOut, poolInputGradConvert, {3, 3}, {2, 2}, MAXPOOL), NCHW);
        auto avePoolOutputGrad =
            _Convert(_PoolGrad(poolInputConvert, avePoolOut, poolInputGradConvert, {3, 3}, {2, 2}, AVEPOOL), NCHW);

        const std::vector<int> outDim = {1, 1, h, w};
        auto maxpoolOutputGradDim     = maxPoolOutputGrad->getInfo()->dim;
        auto avepoolOutputGradDim     = avePoolOutputGrad->getInfo()->dim;
        if (!checkVector<int>(maxpoolOutputGradDim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("MaxpoolGrad(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }
        if (!checkVector<int>(avepoolOutputGradDim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("AvepoolGrad(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(poolInput->writeMap<float>(), (const float *)originInputData, size * sizeof(float));
        ::memcpy(poolInputGrad->writeMap<float>(), (const float *)poolInputGradData, poolSize * sizeof(float));
        auto compute = maxPoolOutputGrad->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 100;
        if (!checkVectorByRelativeError<float>(compute, maxExpectedGrad, size, 0.001 * errorScale)) {
            MNN_ERROR("MaxpoolGrad(%s) test failed!\n", deviceName.c_str());
            return false;
        }
        if (!checkVectorByRelativeError<float>(avePoolOutputGrad->readMap<float>(), aveExpectedGrad, size, 0.001 * errorScale)) {
            MNN_ERROR("AvepoolGrad(%s) test failed!\n", deviceName.c_str());
            return false;
        }

        return true;
    }
};

class PoolGradTestOnCPU : public PoolGradTest {
public:
    virtual ~PoolGradTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

MNNTestSuiteRegister(PoolGradTestOnCPU, "op/PoolGrad");
