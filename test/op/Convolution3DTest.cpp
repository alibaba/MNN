//
//  Convolution3DTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <numeric>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

#define TEST_RANDOM_SEED 100

using namespace MNN;
using namespace MNN::Express;
static void reference_conv3d(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output, int batch, int ic, int oc,
                             INTS inputShape, PadMode mode, INTS pads, INTS kernels, INTS strides, INTS dilations,
                             int group, ConvertFP32 functor) {
    INTS outputShape;
    if (mode == PadMode_SAME) {
        pads.clear();
        for (int i = 0; i < 3; ++i) {
            outputShape.push_back((inputShape[i] + strides[i] - 1) / strides[i]);
            pads.push_back(((outputShape[i] - 1) * strides[i] + (kernels[i] - 1) * dilations[i] + 1 - inputShape[i]) /
                           2);
        }
    } else {
        if (mode == PadMode_VALID) {
            pads = std::vector<int>(3, 0);
        }
        for (int i = 0; i < 3; ++i) {
            outputShape.push_back((inputShape[i] + 2 * pads[i] - (kernels[i] - 1) * dilations[i] - 1) / strides[i] + 1);
        }
    }

    MNN_ASSERT(oc % group == 0 && ic % group == 0);
    output.resize(batch * oc * outputShape[0] * outputShape[1] * outputShape[2]);
    int oc_step = oc / group, ic_step = ic / group;
    for (int b = 0; b < batch; ++b) {
        for (int o_c = 0; o_c < oc; ++o_c) {
            for (int o_d = 0; o_d < outputShape[0]; ++o_d) {
                for (int o_h = 0; o_h < outputShape[1]; ++o_h) {
                    for (int o_w = 0; o_w < outputShape[2]; ++o_w) {
                        float result_data = 0;
                        int g             = o_c / oc_step;
                        for (int i_c = g * ic_step; i_c < (g + 1) * ic_step; ++i_c) {
                            for (int k_d = 0; k_d < kernels[0]; ++k_d) {
                                for (int k_h = 0; k_h < kernels[1]; ++k_h) {
                                    for (int k_w = 0; k_w < kernels[2]; ++k_w) {
                                        int i_d = o_d * strides[0] - pads[0] + k_d * dilations[0];
                                        int i_h = o_h * strides[1] - pads[1] + k_h * dilations[1];
                                        int i_w = o_w * strides[2] - pads[2] + k_w * dilations[2];
                                        if (i_d < 0 || i_d >= inputShape[0] || i_h < 0 || i_h >= inputShape[1] ||
                                            i_w < 0 || i_w >= inputShape[2]) {
                                            continue;
                                        }
                                        float input_data =
                                            input[(((b * ic + i_c) * inputShape[0] + i_d) * inputShape[1] + i_h) *
                                                      inputShape[2] +
                                                  i_w];
                                        float weight_data =
                                            weight[((((g * oc_step + o_c % oc_step) * ic_step + i_c % ic_step) *
                                                         kernels[0] +
                                                     k_d) *
                                                        kernels[1] +
                                                    k_h) *
                                                       kernels[2] +
                                                   k_w];
                                        result_data += functor(input_data) * functor(weight_data);
                                    }
                                }
                            }
                        }
                        output[(((b * oc + o_c) * outputShape[0] + o_d) * outputShape[1] + o_h) * outputShape[2] +
                               o_w] = functor(result_data + functor(bias[o_c]));
                    }
                }
            }
        }
    }
}

static VARP _Conv3D(VARP input, const std::vector<float>& weight, const std::vector<float>& bias, INTS channel,
                    INTS kernelSize, PadMode mode, INTS pads, INTS strides, INTS dilates, int group) {
    MNN_ASSERT(group == 1);
    MNN_ASSERT(dilates.size() == 3 && strides.size() == 3 && kernelSize.size() == 3 && channel.size() == 2);
    MNN_ASSERT(mode != PadMode_CAFFE || pads.size());

    std::unique_ptr<Convolution3DT> conv3d(new Convolution3DT);
    conv3d->weight = weight;
    conv3d->bias   = bias;
    conv3d->common.reset(new Convolution3DCommonT);
    auto common     = conv3d->common.get();
    common->dilates = dilates;
    common->strides = strides;
    common->kernels = kernelSize;
    common->padMode = mode;
    common->pads    = std::vector<int>({0, 0, 0});
    if (mode == PadMode_CAFFE) {
        common->pads = pads;
    }
    common->inputCount  = channel[0];
    common->outputCount = channel[1];
    common->relu = common->relu6 = false;

    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type       = OpType_Convolution3D;
    convOp->main.type  = OpParameter_Convolution3D;
    convOp->main.value = conv3d.release();

    return (Variable::create(Expr::create(convOp.get(), {input})));
}

class Convolution3DCommonTest : public MNNTestCase {
public:
    virtual ~Convolution3DCommonTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name, int batch,
                     int ic, int oc, INTS inputShape, PadMode mode, INTS pads, INTS kernels, INTS strides,
                     INTS dilations, int group, int precision) {
        using namespace MNN::Express;
        std::vector<float> weightData, biasData;
        for (int i = 0; i < group * (oc / group) * (ic / group) * kernels[0] * kernels[1] * kernels[2]; i++) {
            weightData.push_back(rand() % 255 / 255.f / 1000.0f);
        }
        for (int i = 0; i < oc; i++) {
            biasData.push_back(rand() % 255 / 255.f);
        }
        std::vector<float> inputData, outputData;
        for (int i = 0; i < batch * ic * inputShape[0] * inputShape[1] * inputShape[2]; ++i) {
            inputData.push_back(rand() % 255 / 255.f);
        }
        reference_conv3d(inputData, weightData, biasData, outputData, batch, ic, oc, inputShape, mode, pads, kernels,
                         strides, dilations, group, FP32Converter[precision]);
        auto input  = _Input({batch, ic, inputShape[0], inputShape[1], inputShape[2]}, NCHW, halide_type_of<float>());
        auto output = _Conv3D(_Convert(input, NC4HW4), weightData, biasData, {ic, oc}, kernels, mode, pads, strides,
                              dilations, group);
        output      = _Convert(output, NCHW);

        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        // difference below 0.5% relative error is considered correct.
        auto outputPtr = output->readMap<float>();
        if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(), outputData.size(), 0.05)) {
            MNN_PRINT("%s expect:\t real:\n", test_op_name.c_str());
            for (int i = 0; i < outputData.size(); ++i) {
                MNN_PRINT("%f\t, %f\n", outputData[i], outputPtr[i]);
            }
            MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
#ifdef DEBUG
            auto subinput  = _Input({batch, ic, inputShape[0], inputShape[1], inputShape[2]}, NCHW, halide_type_of<float>());
            subinput->writeMap<float>();
            auto suboutput = _Conv3D(_Convert(subinput, NC4HW4), weightData, biasData, {ic, oc}, kernels, mode, pads, strides,
                                  dilations, group);
            suboutput      = _Convert(suboutput, NCHW);
            suboutput->readMap<float>();
#endif
            return false;
        }
        return true;
    }
};

class Convolution3DTest : public Convolution3DCommonTest {
public:
    virtual ~Convolution3DTest() = default;

protected:
    static bool test(MNNForwardType type, const std::string& device_name, int precision) {
        srand(TEST_RANDOM_SEED);
        for (int b = 1; b <= 2; b++) {
            for (int oc = 1; oc <= 8; oc *= 2) {
                for (int ic = 1; ic <= 8; ic *= 2) {
                    for (int is = 1; is <= 8; is *= 2) {
                        for (int id = 1; id <= 4; ++id) {
                            for (int kd = 1; kd <= 3 && kd <= id; ++kd) {
                                for (int kw = 1; kw <= 3 && kw <= is; ++kw) {
                                    for (int kh = 1; kh <= 3 && kh <= is; ++kh) {
                                        for (int p = 0; p <= 1; p++) {
                                            bool succ = Convolution3DCommonTest::test(
                                                type, device_name, "Conv3D", b, ic, oc, {id, is, is}, PadMode_CAFFE,
                                                {p, p, p}, {kd, kh, kw}, {1, 1, 1}, {1, 1, 1}, 1, precision);
                                            if (!succ) {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

class Convolution3DTestOnCPU : public Convolution3DTest {
public:
    virtual ~Convolution3DTestOnCPU() = default;
    virtual bool run(int precision) {
        return Convolution3DTest::test(MNN_FORWARD_CPU, "CPU", precision);
    }
};

MNNTestSuiteRegister(Convolution3DTestOnCPU, "op/convolution/conv3d");
