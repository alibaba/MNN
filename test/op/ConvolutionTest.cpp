//
//  ConvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include <MNN/Interpreter.hpp>
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "TestUtils.h"
#include "MNNTestSuite.h"
#include "MNN_generated.h"

#define TEST_RANDOM_SEED 100

using namespace MNN;

static void reference_conv2d(const std::vector<float>& input, const std::vector<float>& weight,
                             const std::vector<float>& bias, std::vector<float>& output,
                             int batch, int ic, int oc, int ih, int iw, PadMode mode, int pad_h, int pad_w,
                             int kh, int kw, int stride, int dilation, int group) {
    int oh, ow;
    if (mode == PadMode_SAME) {
        oh = (ih + stride - 1) / stride; // oh = ceil(ih / stride)
        ow = (iw + stride - 1) / stride; // ow = ceil(iw / stride)
        pad_h = ((oh - 1) * stride + (kh - 1) * dilation + 1 - ih) / 2;
        pad_w = ((ow - 1) * stride + (kw - 1) * dilation + 1 - iw) / 2;
    } else {
        if (mode == PadMode_VALID) {
            pad_h = pad_w = 0;
        }
        oh = (ih + 2 * pad_h - (kh - 1) * dilation - 1) / stride + 1;
        ow = (iw + 2 * pad_w - (kw - 1) * dilation - 1) / stride + 1;
    }

    MNN_ASSERT(oc % group == 0 && ic % group == 0);
    output.resize(batch * oh * ow * oc);
    int oc_step = oc / group, ic_step = ic / group;
    for (int b = 0; b < batch; ++b) {
        for (int o_c = 0; o_c < oc; ++o_c) {
            for (int o_h = 0; o_h < oh; ++o_h) {
                for (int o_w = 0; o_w < ow; ++o_w) {
                    float result_data = 0;
                    int g = o_c / oc_step;
                    for (int i_c = g * ic_step; i_c < (g + 1) * ic_step; ++i_c) {
                        for (int k_h = 0; k_h < kh; ++k_h) {
                            for (int k_w = 0; k_w < kw; ++k_w) {
                                int i_h = o_h * stride - pad_h + k_h * dilation;
                                int i_w = o_w * stride - pad_w + k_w * dilation;
                                if (i_h < 0 || i_h >= ih || i_w < 0 || i_w >= iw) {
                                    continue;
                                }
                                float input_data = input[((b * ic + i_c) * ih + i_h) * iw + i_w];
                                float weight_data = weight[(((g * oc_step + o_c % oc_step) * ic_step + i_c % ic_step) * kh + k_h) * kw + k_w];
                                result_data += input_data * weight_data;
                            }
                        }
                    }
                    output[((b * oc + o_c) * oh + o_h) * ow + o_w] = result_data + bias[o_c];
                }
            }
        }
    }
}

class ConvolutionCommonTest : public MNNTestCase {
public:
    virtual ~ConvolutionCommonTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name, const std::string& test_op_name,
                     int batch, int ic, int oc, int ih, int iw, PadMode mode,
                     int pad_h, int pad_w, int kh, int kw, int stride, int dilation, int group) {
        using namespace MNN::Express;
        auto creator = MNN::MNNGetExtraBackendCreator(type);
        if (creator == nullptr) {
            MNN_ERROR("backend %d not found!\n", type);
            return false;
        }
        std::map<PadMode, Express::PaddingMode> padMap = {
            {PadMode_CAFFE, CAFFE},
            {PadMode_VALID, VALID},
            {PadMode_SAME,  SAME}
        };
        std::vector<float> weightData, biasData;
        for (int i = 0; i < group * (oc / group) * (ic / group) * kw * kh; i++) {
            weightData.push_back(rand() % 255 / 255.f);
        }
        for (int i = 0; i < oc; i++) {
            biasData.push_back(rand() % 255 / 255.f);
        }
        std::vector<float> inputData, outputData;
        for (int i = 0; i < ih * iw * ic * batch; ++i) {
            inputData.push_back(rand() % 255 / 255.f);
        }
        reference_conv2d(inputData, weightData, biasData, outputData, batch, ic, oc, ih, iw,
                         mode, pad_h, pad_w, kh, kw, stride, dilation,  group);
        auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
        auto output = _Conv(std::move(weightData), std::move(biasData), _Convert(input, NC4HW4), {ic, oc}, {kw, kh},
                            padMap[mode], {stride, stride}, {dilation, dilation}, group, {pad_w, pad_h});
        output = _Convert(output, NCHW);

        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        // difference below 0.5% relative error is considered correct.
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(), 0.005)) {
            MNN_ERROR("%s(%s) test failed!\n", test_op_name.c_str(), device_name.c_str());
            return false;
        }
        return true;
    }
};

class ConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~ConvolutionTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name) {
        srand(TEST_RANDOM_SEED);
        for (int b = 1; b <= 2; b++) {
            for (int oc = 1; oc <= 8; oc *= 2) {
                for (int ic = 1; ic <= 8; ic *= 2) {
                    for (int is = 1; is <= 8; is *= 2) {
                        for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                            for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                for (int d = 1; d <= 2; d++) {
                                    if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                        continue;
                                    for (int s = 1; s <= 2; s++) {
                                        for (int p = 0; p <= 1; p++) {
                                            bool succ = ConvolutionCommonTest::test(type, device_name, "Conv2D",
                                                                                    b, ic, oc, is, is, PadMode_CAFFE,
                                                                                    p, p, kh, kw, s, d, 1);
                                            if (!succ) {
                                                MNN_ERROR("Error for conv b=%d, oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d,p=%d\n", b, oc, ic, is, kw, kh, d, s, p);
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

class ConvolutionTestOnCPU : public ConvolutionTest {
public:
    ~ConvolutionTestOnCPU() = default;
    virtual bool run() {
        return ConvolutionTest::test(MNN_FORWARD_CPU, "CPU");
    }
};

class ConvolutionTestOnOpencl : public ConvolutionTest {
public:
    ~ConvolutionTestOnOpencl() = default;
    virtual bool run() {
        return ConvolutionTest::test(MNN_FORWARD_OPENCL, "OPENCL");
    }
};

class DepthwiseConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~DepthwiseConvolutionTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name) {
        srand(TEST_RANDOM_SEED);
        for (int b = 1; b <= 2; b++) {
            for (int oc = 4; oc <= 8; oc *= 2) {
                for (int ic = oc; ic <= oc; ic++) {
                    for (int is = 1; is <= 8; is *= 2) {
                        for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                            for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                for (int d = 1; d <= 2; d++) {
                                    if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                        continue;
                                    for (int s = 1; s <= 2; s++) {
                                        for (int p = 0; p <= 1; p++) {
                                            // depthwise <==> group == outputChannel
                                            bool succ = ConvolutionCommonTest::test(type, device_name, "DepthwiseConv2D",
                                                                                    b, ic, oc, is, is, PadMode_CAFFE,
                                                                                    p, p, kh, kw, s, d, oc);
                                            if (!succ) {
                                                MNN_ERROR("Error for dw oc=%d, ic=%d, is=%d,kw=%d,kh=%d,d=%d,s=%d,p=%d\n", oc, ic, is, kw, kh, d, s, p);
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

class DepthwiseConvolutionTestOnCPU : public DepthwiseConvolutionTest {
public:
    ~DepthwiseConvolutionTestOnCPU() = default;
    virtual bool run() {
        return DepthwiseConvolutionTest::test(MNN_FORWARD_CPU, "CPU");
    }
};

class DepthwiseConvolutionTestOnOpencl : public DepthwiseConvolutionTest {
public:
    ~DepthwiseConvolutionTestOnOpencl() = default;
    virtual bool run() {
        return DepthwiseConvolutionTest::test(MNN_FORWARD_OPENCL, "OPENCL");
    }
};

class GroupConvolutionTest : public ConvolutionCommonTest {
public:
    virtual ~GroupConvolutionTest() = default;
protected:
    static bool test(MNNForwardType type, const std::string& device_name) {
        srand(TEST_RANDOM_SEED);
        for (int b = 1; b <= 2; b++) {
            for (int g = 2; g <= 4; g *= 2) {
                for (int oc = g * 4; oc <= 4 * g * 4; oc += g * 4) {
                    for (int ic = g * 4; ic <= 4 * g * 4; ic += g * 4) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                            continue;
                                        for (int s = 1; s <= 2; s++) {
                                            for (int p = 0; p <= 1; p++) {
                                                bool succ = ConvolutionCommonTest::test(type, device_name, "GroupConv2D",
                                                                                        b, ic, oc, is, is, PadMode_CAFFE,
                                                                                        p, p, kh, kw, s, d, g);
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
        }
        return true;
    }
};

class GroupConvolutionTestOnCPU : public GroupConvolutionTest {
public:
    virtual ~GroupConvolutionTestOnCPU() = default;
    virtual bool run() {
        return GroupConvolutionTest::test(MNN_FORWARD_CPU, "CPU");
    }
};

class GroupConvolutionTestOnOpencl : public GroupConvolutionTest {
public:
    virtual ~GroupConvolutionTestOnOpencl() = default;
    virtual bool run() {
        return GroupConvolutionTest::test(MNN_FORWARD_OPENCL, "OPENCL");
    }
};

MNNTestSuiteRegister(ConvolutionTestOnCPU, "op/convolution/conv");
MNNTestSuiteRegister(DepthwiseConvolutionTestOnCPU, "op/convolution/depthwise_conv");
MNNTestSuiteRegister(GroupConvolutionTestOnCPU, "op/convolution/conv_group");
