//
//  MultiDeconvolutionTest.cpp
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

class MultiDeconvolutionTest : public MNNTestCase {
public:
    virtual ~MultiDeconvolutionTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName, int precision) {
        // MultiInput Deconv
        {
            const int inputHeight = 3, inputWidth = 3, inputChannel = 3, outputChannel = 2;
            const int kernelSize = 3, stride = 2, pad = 1, batch = 2;
            const int height                   = (inputHeight - 1) * stride + kernelSize - pad * 2; // height = 5
            const int width                    = (inputWidth - 1) * stride + kernelSize - pad * 2;  // width = 5
            const std::vector<float> inputData = {
                // channel 0
                0.0500, 0.2283, 0.9916, 0.5502, 0.2731, 0.0964, 0.5169, 0.3492, 0.0057,
                // channel 1
                0.5207, 0.2388, 0.2215, 0.7307, 0.4999, 0.7638, 0.3025, 0.7966, 0.7117,
                // channel 2
                0.3264, 0.1317, 0.9161, 0.8626, 0.9634, 0.1032, 0.4114, 0.7719, 0.1408,
                // channel 0
                0.0500, 0.2283, 0.9916, 0.5502, 0.2731, 0.0964, 0.5169, 0.3492, 0.0057,
                // channel 1
                0.5207, 0.2388, 0.2215, 0.7307, 0.4999, 0.7638, 0.3025, 0.7966, 0.7117,
                // channel 2
                0.3264, 0.1317, 0.9161, 0.8626, 0.9634, 0.1032, 0.4114, 0.7719, 0.1408
            };
            const std::vector<float> filterData = {
                // outputChannel = 0, inputChannel = 0
                0.7648, 0.83, 0.3509, 0.8953, 0.7895, 0.4066, 0.5893, 0.9506, 0.4081,
                // outputChannel = 1, inputChannel = 0
                0.1982, 0.2179, 0.2756, 0.5602, 0.2062, 0.8441, 0.6934, 0.5666, 0.765,
                // outputChannel = 0, inputChannel = 1
                0.0375, 0.2276, 0.6908, 0.2677, 0.2822, 0.9121, 0.0821, 0.1406, 0.1126,
                // outputChannel = 1, inputChannel = 1
                0.3432, 0.4277, 0.6015, 0.0909, 0.957, 0.3732, 0.4586, 0.2034, 0.5555,
                // outputChannel = 0, inputChannel = 2
                0.8036, 0.8453, 0.226, 0.6534, 0.7527, 0.9455, 0.0295, 0.1798, 0.4561,
                // outputChannel = 1, inputChannel = 2
                0.3859, 0.1691, 0.7373, 0.246, 0.7928, 0.4552, 0.8937, 0.4109, 0.3926};
            const std::vector<float> biasData   = {1.0, 0.0};
            const std::vector<float> outputData = {
                // channel 0
                1.432098, 2.158248, 1.346763, 2.980813, 2.534924, 2.531556, 3.280517, 2.429089, 2.653877, 2.479560,
                2.289865, 3.713586, 2.081835, 2.836103, 1.369331, 2.626485, 3.331208, 2.626743, 2.721178, 1.503316,
                1.803119, 2.905308, 2.081503, 2.886019, 1.311322,
                // channel 1
                0.767390, 0.567106, 0.380019, 1.142767, 1.142727, 0.846633, 2.665777, 0.668269, 3.374221, 1.348453,
                1.496601, 1.565205, 1.298501, 1.004446, 0.832651, 1.126390, 3.713293, 1.199604, 2.818435, 0.581827,
                0.722235, 1.194398, 1.446314, 1.045943, 0.793899,
                // channel 0
                1.432098, 2.158248, 1.346763, 2.980813, 2.534924, 2.531556, 3.280517, 2.429089, 2.653877, 2.479560,
                2.289865, 3.713586, 2.081835, 2.836103, 1.369331, 2.626485, 3.331208, 2.626743, 2.721178, 1.503316,
                1.803119, 2.905308, 2.081503, 2.886019, 1.311322,
                // channel 1
                0.767390, 0.567106, 0.380019, 1.142767, 1.142727, 0.846633, 2.665777, 0.668269, 3.374221, 1.348453,
                1.496601, 1.565205, 1.298501, 1.004446, 0.832651, 1.126390, 3.713293, 1.199604, 2.818435, 0.581827,
                0.722235, 1.194398, 1.446314, 1.045943, 0.793899
            };

            auto input  = _Input({batch, inputChannel, inputHeight, inputWidth}, NCHW, halide_type_of<float>());
            auto filter = _Input({inputChannel, outputChannel, kernelSize, kernelSize}, NCHW, halide_type_of<float>());
            auto bias   = _Input({outputChannel}, NCHW, halide_type_of<float>());
            auto output =
                _Deconv(filter, bias, _Convert(input, NC4HW4), CAFFE, {stride, stride}, {1, 1}, 1, {pad, pad});
            output = _Convert(output, NCHW);

            const std::vector<int> outDim = {batch, outputChannel, height, width};
            if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
                MNN_ERROR("MultiDeconvolution(%s) shape test failed!\n", deviceName.c_str());
                return false;
            }

            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(filter->writeMap<float>(), filterData.data(), filterData.size() * sizeof(float));
            ::memcpy(bias->writeMap<float>(), biasData.data(), biasData.size() * sizeof(float));
            auto outputPtr = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
            if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.005 * errorScale)) {
                MNN_ERROR("MultiDeconvolution(%s) test failed!\n", deviceName.c_str());
                for (int v = 0; v < outputData.size(); ++v) {
                    MNN_ERROR("Correct:%f, Error:%f\n", outputData[v], outputPtr[v]);
                }
                return false;
            }
        }

        // MultiInput Depthwise Deconv
        {
            const int inputHeight = 3, inputWidth = 3, inputChannel = 2, outputChannel = 2;
            const int kernelSize = 3, stride = 2, pad = 1, batch = 1;
            const int height                   = (inputHeight - 1) * stride + kernelSize - pad * 2; // height = 5
            const int width                    = (inputWidth - 1) * stride + kernelSize - pad * 2;  // width = 5
            const std::vector<float> inputData = {
                // channel 0
                0.0500,
                0.2283,
                0.9916,
                0.5502,
                0.2731,
                0.0964,
                0.5169,
                0.3492,
                0.0057,
                // channel 1
                0.5207,
                0.2388,
                0.2215,
                0.7307,
                0.4999,
                0.7638,
                0.3025,
                0.7966,
                0.7117,
            };
            const std::vector<float> filterData = {
                // outputChannel = 0, inputChannel = 0
                0.7648,
                0.83,
                0.3509,
                0.8953,
                0.7895,
                0.4066,
                0.5893,
                0.9506,
                0.4081,
                // outputChannel = 1, inputChannel = 0
                0.1982,
                0.2179,
                0.2756,
                0.5602,
                0.2062,
                0.8441,
                0.6934,
                0.5666,
                0.765,
            };
            const std::vector<float> biasData   = {1.0, 0.0};
            const std::vector<float> outputData = {
                // channel 0
                1.03947, 1.22473, 1.18024, 1.98061, 1.78287, 1.5042, 1.55687, 1.44369, 1.84708, 2.02263, 1.43438,
                1.46822, 1.21561, 1.19735, 1.07611, 1.95205, 1.83392, 1.54944, 1.29515, 1.09637, 1.40809, 1.52281,
                1.27569, 1.14709, 1.0045,
                // channel 1
                0.107368, 0.573299, 0.0492406, 0.325655, 0.0456733, 0.454248, 0.864381, 0.244232, 0.625428, 0.291934,
                0.15067, 0.896828, 0.103079, 0.849846, 0.157496, 0.479929, 1.14687, 0.456822, 1.27264, 0.587849,
                0.0623755, 0.701596, 0.164259, 1.0711, 0.146753
            };

            auto input  = _Input({batch, inputChannel, inputHeight, inputWidth}, NCHW, halide_type_of<float>());
            auto filter = _Input({outputChannel, 1, kernelSize, kernelSize}, NCHW, halide_type_of<float>());
            auto bias   = _Input({outputChannel}, NCHW, halide_type_of<float>());
            auto output =
                _Deconv(filter, bias, _Convert(input, NC4HW4), CAFFE, {stride, stride}, {1, 1}, 2, {pad, pad});
            output = _Convert(output, NCHW);

            const std::vector<int> outDim = {batch, outputChannel, height, width};
            if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
                MNN_ERROR("MultiDeconvolution(%s) shape test failed!\n", deviceName.c_str());
                return false;
            }

            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(filter->writeMap<float>(), filterData.data(), filterData.size() * sizeof(float));
            ::memcpy(bias->writeMap<float>(), biasData.data(), biasData.size() * sizeof(float));
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
            if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                                   0.005 * errorScale)) {
                MNN_ERROR("Depthwise MultiDeconvolution(%s) test failed!\n", deviceName.c_str());
                return false;
            }
        }
        return true;
    }
};

class MultiDeconvolutionTestOnCPU : public MultiDeconvolutionTest {
public:
    virtual ~MultiDeconvolutionTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class MultiDeconvolutionTestOnOpencl : public MultiDeconvolutionTest {
public:
    virtual ~MultiDeconvolutionTestOnOpencl() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL", precision);
    }
};

MNNTestSuiteRegister(MultiDeconvolutionTestOnCPU, "op/MultiDeconv");
