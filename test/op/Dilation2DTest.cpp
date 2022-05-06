//
//  Dilation2DTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _Dilation2D(VARP input, const std::vector<float>& filterData, int depth, INTS kernels, INTS strides,
                        INTS dilations, PadMode mode) {
    std::unique_ptr<Convolution2DCommonT> common(new Convolution2DCommonT);
    common->dilateX     = dilations[0];
    common->dilateY     = dilations[1];
    common->strideX     = strides[0];
    common->strideY     = strides[1];
    common->kernelX     = kernels[0];
    common->kernelY     = kernels[1];
    common->outputCount = depth;
    common->padMode     = mode;

    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->weight = filterData;
    conv->common.reset(common.release());

    std::unique_ptr<OpT> dilation2d(new OpT);
    dilation2d->type       = OpType_Dilation2D;
    dilation2d->main.type  = OpParameter_Convolution2D;
    dilation2d->main.value = conv.release();

    return Variable::create(Expr::create(std::move(dilation2d), {input}));
}

class Dilation2DTest : public MNNTestCase {
public:
    virtual ~Dilation2DTest() = default;

protected:
    bool testOnBackend(MNNForwardType type, const std::string& deviceName, int precision) {
        const int batch = 2, hInput = 8, wInput = 8, depth = 2;
        const int kernel = 3, stride = 2, dilation = 2;
        const int hOutput = 4, wOutput = 4;
        PadMode mode = PadMode_SAME;

        const std::vector<float> inputData = {
            // batch 0, depth 0
            0.2442, 0.1112, 0.0019, 0.7956, 0.5193, 0.9469, 0.0734, 0.868, 0.938, 0.0044, 0.7733, 0.9659, 0.7684,
            0.4761, 0.0532, 0.1346, 0.5177, 0.8166, 0.3568, 0.0978, 0.031, 0.4323, 0.6097, 0.4109, 0.1825, 0.0286,
            0.7931, 0.0961, 0.3956, 0.5455, 0.9764, 0.0147, 0.7976, 0.6743, 0.4025, 0.0064, 0.9415, 0.0588, 0.3576,
            0.1793, 0.9301, 0.1929, 0.2146, 0.75, 0.5445, 0.1015, 0.9295, 0.8509, 0.7781, 0.4447, 0.7502, 0.938, 0.0157,
            0.2437, 0.0735, 0.1505, 0.2763, 0.5841, 0.6464, 0.0575, 0.2624, 0.3593, 0.5915, 0.5977,
            // batch 0, depth 1
            0.2905, 0.765, 0.6162, 0.7862, 0.6024, 0.4286, 0.6094, 0.1839, 0.2038, 0.9094, 0.1573, 0.8314, 0.8618,
            0.1735, 0.9426, 0.2599, 0.3691, 0.0707, 0.4993, 0.9102, 0.6286, 0.3101, 0.0336, 0.0018, 0.4176, 0.9939,
            0.5555, 0.8251, 0.6085, 0.0912, 0.002, 0.1107, 0.4421, 0.0648, 0.298, 0.3073, 0.1005, 0.0732, 0.6128,
            0.5606, 0.5251, 0.004, 0.0443, 0.9015, 0.641, 0.2778, 0.3342, 0.5899, 0.3267, 0.8305, 0.4335, 0.5785,
            0.7227, 0.9369, 0.1777, 0.8986, 0.5972, 0.3452, 0.7728, 0.331, 0.5725, 0.7188, 0.1314, 0.8734,
            // batch 1, depth 0
            1.2442, 0.1112, 0.0019, 0.7956, 0.5193, 0.9469, 0.0734, 1.868, 1.938, 0.0044, 0.7733, 0.9659, 0.7684,
            0.4761, 0.0532, 1.1346, 1.5177, 0.8166, 0.3568, 0.0978, 0.031, 0.4323, 0.6097, 1.4109, 1.1825, 0.0286,
            0.7931, 0.0961, 0.3956, 0.5455, 0.9764, 1.0147, 1.7976, 0.6743, 0.4025, 0.0064, 0.9415, 0.0588, 0.3576,
            1.1793, 1.9301, 0.1929, 0.2146, 0.75, 0.5445, 0.1015, 0.9295, 1.8509, 1.7781, 0.4447, 0.7502, 0.938, 0.0157,
            0.2437, 0.0735, 1.1505, 1.2763, 0.5841, 0.6464, 0.0575, 0.2624, 0.3593, 0.5915, 1.5977,
            // batch 1, depth 1
            1.2905, 0.765, 0.6162, 0.7862, 0.6024, 0.4286, 0.6094, 1.1839, 1.2038, 0.9094, 0.1573, 0.8314, 0.8618,
            0.1735, 0.9426, 1.2599, 1.3691, 0.0707, 0.4993, 0.9102, 0.6286, 0.3101, 0.0336, 1.0018, 1.4176, 0.9939,
            0.5555, 0.8251, 0.6085, 0.0912, 0.002, 1.1107, 1.4421, 0.0648, 0.298, 0.3073, 0.1005, 0.0732, 0.6128,
            1.5606, 1.5251, 0.004, 0.0443, 0.9015, 0.641, 0.2778, 0.3342, 1.5899, 1.3267, 0.8305, 0.4335, 0.5785,
            0.7227, 0.9369, 0.1777, 1.8986, 1.5972, 0.3452, 0.7728, 0.331, 0.5725, 0.7188, 0.1314, 1.8734};
        const std::vector<float> filterData = {// depth 0
                                               0.0707, 0.8473, 0.2599, 0.111, 0.0394, 0.8792, 0.3143, 0.5409, 0.527,
                                               // depth 1
                                               0.124, 0.4437, 0.5337, 0.057, 0.8509, 0.312, 0.2286, 0.419, 0.0331};
        const std::vector<float> outputData = {// batch 0, depth 0
                                               1.8451, 1.3553, 1.0864, 0.8598, 1.277, 1.8132, 1.3779, 1.3918, 1.6292,
                                               0.9807, 1.7301, 1.1386, 1.0402, 1.5973, 1.4769, 1.6982,
                                               // batch 0, depth 1
                                               1.7603, 1.6823, 1.0537, 1.1108, 1.8448, 1.676, 1.1301, 1.0089, 1.4376,
                                               1.7524, 1.1378, 1.4408, 1.4352, 1.3452, 1.5697, 1.7243,

                                               // batch 1, depth 0
                                               1.8451, 1.3553, 2.0138, 1.5556, 1.277, 1.8132, 2.3779, 2.3918, 1.6292,
                                               0.9807, 2.7301, 2.1386, 1.0402, 1.5973, 2.4769, 2.6982,
                                               // batch 1, depth 1
                                               1.7603, 1.6823, 1.5719, 2.1108, 1.8448, 1.676, 1.7936, 2.0089, 1.4376,
                                               1.7524, 1.9065, 2.4408, 1.4352, 1.3452, 2.1854, 2.7243};

        auto input  = _Input({batch, depth, hInput, wInput}, NCHW, halide_type_of<float>());
        auto output = _Dilation2D(_Convert(input, NC4HW4), filterData, depth, {kernel, kernel}, {stride, stride},
                                  {dilation, dilation}, mode);
        output      = _Convert(output, NCHW);

        if (type != MNN_FORWARD_CPU) {
            Optimizer::Config config;
            config.forwardType = type;
            auto optimizer     = Optimizer::create(config);
            if (optimizer == nullptr) {
                MNN_ERROR("backend %s not support\n", deviceName.c_str());
                return false;
            }
            optimizer->onExecute({output});
        }

        const std::vector<int> outDim = {batch, depth, hOutput, wOutput};
        if (!checkVector<int>(output->getInfo()->dim.data(), outDim.data(), 4, 0)) {
            MNN_ERROR("Dilation2D(%s) shape test failed!\n", deviceName.c_str());
            return false;
        }

        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(), 0.005 * errorScale)) {
            MNN_ERROR("Dilation2D(%s) test failed!\n", deviceName.c_str());
            return false;
        }

        return true;
    }
};

class Dilation2DTestOnCPU : public Dilation2DTest {
public:
    virtual ~Dilation2DTestOnCPU() = default;
    virtual bool run(int precision) {
        return testOnBackend(MNN_FORWARD_CPU, "CPU", precision);
    }
};

MNNTestSuiteRegister(Dilation2DTestOnCPU, "op/Dilation2D/cpu");
