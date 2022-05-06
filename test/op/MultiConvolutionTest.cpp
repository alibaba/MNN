//
//  MultiConvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/10/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN::Express;

class MultiConvolutionTest : public MNNTestCase {
public:
    virtual ~MultiConvolutionTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName, int precision) {
        // Multi input Conv
        {
            const int inputHeight = 5, inputWidth = 5, inputChannel = 2, outputChannel = 1;
            const int kernelSize = 3, stride = 2, pad = 1, batch = 1;
            const int height                   = (inputHeight + 2 * pad - kernelSize) / stride + 1; // height = 3
            const int width                    = (inputWidth + 2 * pad - kernelSize) / stride + 1;  // width = 3
            const std::vector<float> inputData = {
                // channel 0
                0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
                0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
                // channel 1
                0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
                0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803};
            const std::vector<float> filterData = {
                // outputChannel = 0, inputChannel = 0
                0.5567, 0.4559, 0.0203, 0.9659, 0.2679, 0.4117, 0.9696, 0.4567, 0.3787,
                // outputChannel = 0, inputChannel = 1
                0.3354, 0.2056, 0.0342, 0.023, 0.4683, 0.9966, 0.6097, 0.0873, 0.7917};
            const std::vector<float> biasData   = {1.0};
            const std::vector<float> outputData = {2.930293, 4.682340, 2.721255, 3.087505, 5.198602,
                                                   4.088373, 1.564287, 3.151330, 3.109602};

            auto input  = _Input({batch, inputChannel, inputHeight, inputWidth}, NCHW, halide_type_of<float>());
            auto filter = _Input({outputChannel, inputChannel, kernelSize, kernelSize}, NCHW, halide_type_of<float>());
            auto bias   = _Input({outputChannel}, NCHW, halide_type_of<float>());
            auto output = _Conv(filter, bias, _Convert(input, NC4HW4), CAFFE, {stride, stride}, {1, 1}, 1, {pad, pad});
            output      = _Convert(output, NCHW);

            const std::vector<int> outDim = {batch, outputChannel, height, width};
            if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
                MNN_ERROR("MultiConvolution(%s) shape test failed!\n", deviceName.c_str());
                return false;
            }

            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(filter->writeMap<float>(), filterData.data(), filterData.size() * sizeof(float));
            ::memcpy(bias->writeMap<float>(), biasData.data(), biasData.size() * sizeof(float));
            auto outputPtr = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
            if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.001 * errorScale)) {
                MNN_ERROR("MultiConvolution(%s) test failed!\n", deviceName.c_str());
                return false;
            }
        }

        // Multi input DepthwiseConv
        {
            const int inputHeight = 5, inputWidth = 5, inputChannel = 2, outputChannel = 2;
            const int kernelSize = 3, stride = 2, pad = 1, batch = 1;
            const int height                   = (inputHeight + 2 * pad - kernelSize) / stride + 1; // height = 3
            const int width                    = (inputWidth + 2 * pad - kernelSize) / stride + 1;  // width = 3
            const std::vector<float> inputData = {
                // channel 0
                0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
                0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
                // channel 1
                0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
                0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803};
            const std::vector<float> filterData = {
                // outputChannel = 0, inputChannel = 0
                0.5567, 0.4559, 0.0203, 0.9659, 0.2679, 0.4117, 0.9696, 0.4567, 0.3787,
                // outputChannel = 0, inputChannel = 1
                0.3354, 0.2056, 0.0342, 0.023, 0.4683, 0.9966, 0.6097, 0.0873, 0.7917};
            const std::vector<float> biasData   = {1.0f, 0.0f};
            const std::vector<float> outputData = {1.6428,  2.30901, 1.89379, 1.97912,  3.43648,  2.93648,
                                                   1.34808, 2.23674, 2.49887, 1.2875,   2.37333,  0.82747,
                                                   1.10839, 1.76213, 1.1519,  0.216204, 0.914589, 0.610736};

            auto input  = _Input({batch, inputChannel, inputHeight, inputWidth}, NCHW, halide_type_of<float>());
            auto filter = _Input({outputChannel, 1, kernelSize, kernelSize}, NCHW, halide_type_of<float>());
            auto bias   = _Input({outputChannel}, NCHW, halide_type_of<float>());
            auto output = _Conv(filter, bias, _Convert(input, NC4HW4), CAFFE, {stride, stride}, {1, 1}, 2, {pad, pad});
            output      = _Convert(output, NCHW);

            const std::vector<int> outDim = {batch, outputChannel, height, width};
            if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
                MNN_ERROR("MultiConvolution(%s) shape test failed!\n", deviceName.c_str());
                return false;
            }

            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(filter->writeMap<float>(), filterData.data(), filterData.size() * sizeof(float));
            ::memcpy(bias->writeMap<float>(), biasData.data(), biasData.size() * sizeof(float));
            auto outputPtr = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
            if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.001 * errorScale)) {
                MNN_ERROR("Depthwise MultiConvolution(%s) test failed!\n", deviceName.c_str());
                return false;
            }
        }
        return true;
    }
};

class MultiConvolutionTestOnCPU : public MultiConvolutionTest {
public:
    virtual ~MultiConvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class MultiConvolutionTestOnOpencl : public MultiConvolutionTest {
public:
    virtual ~MultiConvolutionTestOnOpencl() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL", precision);
    }
};

MNNTestSuiteRegister(MultiConvolutionTestOnCPU, "op/MultiConv");
