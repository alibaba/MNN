//
//  Pool3DTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

// PoolType_MAXPOOL or PoolType_AVEPOOL
static VARP _Pool3D(VARP x, INTS kernels, INTS strides, PoolType type, PoolPadType padType, INTS pads) {
    std::unique_ptr<Pool3DT> pool3d(new Pool3DT);
    pool3d->strides = strides;
    pool3d->kernels = kernels;
    pool3d->pads    = pads;
    pool3d->type    = type;
    pool3d->padType = padType;

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Pooling3D;
    op->main.type  = OpParameter_Pool3D;
    op->main.value = pool3d.release();

    return (Variable::create(Expr::create(op.get(), {x})));
}

class Pool3DCommonTest : public MNNTestCase {
public:
    virtual ~Pool3DCommonTest() = default;

protected:
    static bool testOnBackend(MNNForwardType type, const std::string& deviceName, const std::string& test_op_name,
                              PoolType poolType, int precision) {
        // 1, 2, 3, 4, 4 -> 1, 2, 3, 3, 3
        const int h = 4, w = 4, depth = 3;
        const int poolSize = 2, poolDepth = 3;
        const int stride = 2, strideDepth = 1;
        const int pad = 1, padDepth = 1;

        const std::vector<float> inputData = {// channel = 0
                                              // depth = 0
                                              0.5488, 0.7152, 0.6028, 0.5449, 0.4237, 0.6459, 0.4376, 0.8918, 0.9637,
                                              0.3834, 0.7917, 0.5289, 0.568, 0.9256, 0.071, 0.0871,
                                              // depth = 1
                                              0.0202, 0.8326, 0.7782, 0.87, 0.9786, 0.7992, 0.4615, 0.7805, 0.1183,
                                              0.6399, 0.1434, 0.9447, 0.5218, 0.4147, 0.2646, 0.7742,
                                              // depth = 2
                                              0.4562, 0.5684, 0.0188, 0.6176, 0.6121, 0.6169, 0.9437, 0.6818, 0.3595,
                                              0.437, 0.6976, 0.0602, 0.6668, 0.6706, 0.2104, 0.1289,
                                              // channel = 1
                                              // depth = 0
                                              0.5488, 0.7152, 0.6028, 0.5449, 0.4237, 0.6459, 0.4376, 0.8918, 0.9637,
                                              0.3834, 0.7917, 0.5289, 0.568, 0.9256, 0.071, 0.0871,
                                              // depth = 1
                                              0.0202, 0.8326, 0.7782, 0.87, 0.9786, 0.7992, 0.4615, 0.7805, 0.1183,
                                              0.6399, 0.1434, 0.9447, 0.5218, 0.4147, 0.2646, 0.7742,
                                              // depth = 2
                                              0.4562, 0.5684, 0.0188, 0.6176, 0.6121, 0.6169, 0.9437, 0.6818, 0.3595,
                                              0.437, 0.6976, 0.0602, 0.6668, 0.6706, 0.2104, 0.1289
                                             };
        std::vector<float> outputData;

        if (poolType == PoolType_MAXPOOL) {
            outputData = std::vector<float>({// channel = 0
                                             // depth = 0
                                             0.5488, 0.8326, 0.87, 0.9786, 0.7992, 0.9447, 0.568, 0.9256, 0.7742,
                                             // depth = 1
                                             0.5488, 0.8326, 0.87, 0.9786, 0.9437, 0.9447, 0.6668, 0.9256, 0.7742,
                                             // depth = 2
                                             0.4562, 0.8326, 0.87, 0.9786, 0.9437, 0.9447, 0.6668, 0.6706, 0.7742,
                                             // channel = 1
                                             // depth = 0
                                             0.5488, 0.8326, 0.87, 0.9786, 0.7992, 0.9447, 0.568, 0.9256, 0.7742,
                                             // depth = 1
                                             0.5488, 0.8326, 0.87, 0.9786, 0.9437, 0.9447, 0.6668, 0.9256, 0.7742,
                                             // depth = 2
                                             0.4562, 0.8326, 0.87, 0.9786, 0.9437, 0.9447, 0.6668, 0.6706, 0.7742
                                            });
        } else {
            outputData = std::vector<float>(
                {// channel = 0
                 // depth = 0
                 0.071125, 0.366100, 0.176863, 0.310538, 0.537825, 0.393238, 0.136225, 0.209487, 0.107662,
                 // depth = 1
                 0.085433, 0.293000, 0.169375, 0.287992, 0.583150, 0.323992, 0.146383, 0.213075, 0.082517,
                 // depth = 2
                 0.059550, 0.274750, 0.185950, 0.258563, 0.592400, 0.308400, 0.148575, 0.195037, 0.112888,
                 // channel = 0
                 // depth = 0
                 0.071125, 0.366100, 0.176863, 0.310538, 0.537825, 0.393238, 0.136225, 0.209487, 0.107662,
                 // depth = 1
                 0.085433, 0.293000, 0.169375, 0.287992, 0.583150, 0.323992, 0.146383, 0.213075, 0.082517,
                 // depth = 2
                 0.059550, 0.274750, 0.185950, 0.258563, 0.592400, 0.308400, 0.148575, 0.195037, 0.112888,
                });
        }

        auto input  = _Input({1, 2, depth, h, w}, NCHW, halide_type_of<float>());
        auto output = _Pool3D(_Convert(input, NC4HW4), {poolDepth, poolSize, poolSize}, {strideDepth, stride, stride},
                              poolType, PoolPadType_CAFFE, {padDepth, pad, pad});
        output      = _Convert(output, NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(), 0.001 * errorScale)) {
            MNN_ERROR("%s(%s) test failed!: errorScale:%f\n", test_op_name.c_str(), deviceName.c_str(), errorScale);
            return false;
        }

        return true;
    }
};

class MaxPool3DTestOnCPU : public Pool3DCommonTest {
public:
    virtual ~MaxPool3DTestOnCPU() = default;
    virtual bool run(int precision) {
        return Pool3DCommonTest::testOnBackend(MNN_FORWARD_CPU, "CPU", "MaxPool3D", PoolType_MAXPOOL, precision);
    }
};

class AvePool3DTestOnCPU : public Pool3DCommonTest {
public:
    virtual ~AvePool3DTestOnCPU() = default;
    virtual bool run(int precision) {
        return Pool3DCommonTest::testOnBackend(MNN_FORWARD_CPU, "CPU", "AvePool3D", PoolType_AVEPOOL, precision);
    }
};

MNNTestSuiteRegister(MaxPool3DTestOnCPU, "op/MaxPool3d");
MNNTestSuiteRegister(AvePool3DTestOnCPU, "op/AvePool3d");
