//
//  Conv2DBackPropFilterTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>

#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN::Express;

class Conv2DBackPropFilterTest : public MNNTestCase {
public:
    virtual ~Conv2DBackPropFilterTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName, int precision) {
        const int inputHeight = 5, inputWidth = 5, inputChannel = 2, outputChannel = 3;
        const int kernelSize = 3, stride = 2, pad = 1, batch = 1;
        const int height                   = (inputHeight + 2 * pad - kernelSize) / stride + 1; // height = 3
        const int width                    = (inputWidth + 2 * pad - kernelSize) / stride + 1;  // width = 3
        const std::vector<float> inputData = {
            // channel 0
            0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638, 0.6367,
            0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
            // channel 1
            0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938, 0.2593,
            0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803};
        const std::vector<float> gradData = {// channel 0
                                             0.0229, 0.6325, 0.8646, 0.7819, 0.6056, 0.0749, 0.2162, 0.4768, 0.5742,
                                             // channel 1
                                             0.0241, 0.8462, 0.7895, 0.4366, 0.1978, 0.6466, 0.7126, 0.9574, 0.2927,
                                             // channel 2
                                             0.0020, 0.3654, 0.3904, 0.3954, 0.5271, 0.2788, 0.9785, 0.2899, 0.5009};
        const std::vector<float> filterData(outputChannel * inputChannel * kernelSize * kernelSize, 0.0);
        const std::vector<float> outputData = {
            // outputChannel = 0, inputChannel = 0
            1.067752, 1.259766, 1.313559, 1.076762, 1.769278, 1.249106, 1.514711, 0.683602, 1.379981,
            // outputChannel = 0, inputChannel = 1
            1.008152, 1.646069, 1.376681, 1.581137, 2.707695, 1.263700, 1.231126, 2.002633, 1.120040,
            // outputChannel = 1, inputChannel = 0
            1.474308, 0.766233, 1.428803, 1.223466, 1.743998, 1.367851, 1.556988, 1.172140, 1.069521,
            // outputChannel = 1, inputChannel = 1
            1.034659, 2.252174, 1.339982, 1.480274, 2.558655, 1.492689, 1.682971, 2.062799, 0.879627,
            // outputChannel = 2, inputChannel = 0
            0.990460, 1.033711, 1.519227, 0.987508, 1.567596, 1.128253, 1.048235, 0.580911, 0.835177,
            // outputChannel = 2, inputChannel = 1
            1.006851, 1.959918, 1.079935, 1.022828, 1.765439, 0.789565, 0.856232, 1.360733, 0.768066};

        auto input  = _Input({batch, inputChannel, inputHeight, inputWidth}, NCHW, halide_type_of<float>());
        auto grad   = _Input({batch, outputChannel, height, width}, NCHW, halide_type_of<float>());
        auto output = _Conv2DBackPropFilter(_Convert(input, NC4HW4), _Convert(grad, NC4HW4), {kernelSize, kernelSize},
                                            CAFFE, {stride, stride}, {1, 1}, 1, {pad, pad});
        output      = _Convert(output, NCHW);

        const std::vector<int> outDim = {outputChannel, inputChannel, kernelSize, kernelSize};
        if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("Conv2DBackPropFilter(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        ::memcpy(grad->writeMap<float>(), gradData.data(), gradData.size() * sizeof(float));
        auto size      = output->getInfo()->size;
        auto outputPtr = output->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.005 * errorScale)) {
            MNN_ERROR("Conv2DBackPropFilter(%s) test failed!\n", deviceName.c_str());
            for (int i = 0; i < size; ++i) {
                MNN_PRINT("%f - %f\n", outputPtr[i], outputData[i]);
            }
            return false;
        }
        return true;
    }
};

class Conv2DBackPropFilterTestOnCPU : public Conv2DBackPropFilterTest {
public:
    virtual ~Conv2DBackPropFilterTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

class Conv2DBackPropFilterTestOnOpencl : public Conv2DBackPropFilterTest {
public:
    virtual ~Conv2DBackPropFilterTestOnOpencl() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_OPENCL, "OPENCL", precision);
    }
};

MNNTestSuiteRegister(Conv2DBackPropFilterTestOnCPU, "op/Conv2DBackPropFilter");

class Conv2DDWBackPropFilterTest : public MNNTestCase {
public:
    virtual ~Conv2DDWBackPropFilterTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 3, 5, 5}, NCHW);

        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0.7023, 0.6672, 0.4087, 0.0406, 0.9464, 0.9814, 0.7135, 0.8183, 0.3256,
                                  0.9323, 0.1000, 0.6262, 0.7053, 0.6759, 0.6267, 0.3127, 0.2541, 0.5887,
                                  0.8536, 0.4462, 0.1815, 0.1685, 0.0113, 0.0132, 0.6045, // the first channel
                                  0.0315, 0.6133, 0.9989, 0.5813, 0.0218, 0.8548, 0.1491, 0.7521, 0.3627,
                                  0.0980, 0.2310, 0.1742, 0.2141, 0.1796, 0.2905, 0.9752, 0.8099, 0.2112,
                                  0.2591, 0.7598, 0.4165, 0.6857, 0.9767, 0.8897, 0.8165, // the second channel
                                  0.4202, 0.3214, 0.8497, 0.8358, 0.7235, 0.8389, 0.8026, 0.5240, 0.5476,
                                  0.1078, 0.5874, 0.3464, 0.8387, 0.3170, 0.6110, 0.8884, 0.7784, 0.8721,
                                  0.1358, 0.4529, 0.6801, 0.4875, 0.5604, 0.6948, 0.4249}; // the last channel
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 75 * sizeof(float));
        input->unMap();

        input            = _Convert(input, NC4HW4);
        auto weight      = _Const(1.0, {3, 1, 3, 3}, NCHW);
        auto bias        = _Const(0.0, {1}, NCHW);
        auto convOut     = _Conv(weight, bias, input, VALID, {1, 1}, {1, 1}, 3);
        auto convOutDims = convOut->getInfo()->dim;

        auto grad       = _Const(1.0, convOutDims, NCHW);
        grad            = _Convert(grad, NC4HW4);
        auto weightGrad = _Conv2DBackPropFilter(input, grad, {3, 3}, VALID, {1, 1}, {1, 1}, 3);
        weightGrad->setName("Conv2DDWBackPropFilter");
        weightGrad = _Convert(weightGrad, NCHW);
        weightGrad->setName("nc4hw4_to_nchw");

        auto weightGradDims                 = weightGrad->getInfo()->dim;
        const std::vector<int> expectedDims = {3, 1, 3, 3};
        if (!checkVector<int>(weightGradDims.data(), expectedDims.data(), 4, 0)) {
            MNN_ERROR("Conv2DBackPropFilter's output shape compute ERROR!\n");
            return false;
        }
        const std::vector<float> expectedWeightGrad = {5.7228, 4.9812, 5.4798, 5.1002, 5.5611, 5.9726, 2.9484,
                                                       3.8968, 4.5254, 4.0190, 4.0254, 3.4991, 4.3715, 3.1120,
                                                       3.1270, 4.6943, 4.4001, 4.5972, 5.5294, 5.3832, 5.3553,
                                                       6.4770, 5.1627, 4.4070, 6.0394, 5.0311, 4.9077};
        auto weightGradPtr                          = weightGrad->readMap<float>();
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        if (!checkVectorByRelativeError<float>(weightGradPtr, expectedWeightGrad.data(), 27, 0.01 * errorScale)) {
            MNN_ERROR("Conv2DBackPropFilter test failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(Conv2DDWBackPropFilterTest, "op/Conv2DBackPropFilterDW");
