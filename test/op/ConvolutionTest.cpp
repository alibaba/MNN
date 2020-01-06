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
        if (type != MNN_FORWARD_CPU) {
            Optimizer::Config config;
            config.forwardType = type;
            auto optimizer = Optimizer::create(config);
            if (optimizer == nullptr) {
                MNN_ERROR("backend %s not support\n", device_name.c_str());
                return false;
            }
            optimizer->onExecute({output});
        }

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

MNNTestSuiteRegister(ConvolutionTestOnCPU, "op/convolution/conv/cpu");
MNNTestSuiteRegister(ConvolutionTestOnOpencl, "op/convolution/conv/opencl");
MNNTestSuiteRegister(DepthwiseConvolutionTestOnCPU, "op/convolution/depthwise_conv/cpu");
MNNTestSuiteRegister(DepthwiseConvolutionTestOnOpencl, "op/convolution/depthwise_conv/opencl");
MNNTestSuiteRegister(GroupConvolutionTestOnCPU, "op/convolution/conv_group/cpu");
MNNTestSuiteRegister(GroupConvolutionTestOnOpencl, "op/convolution/conv_group/opencl");

static Interpreter *create(int oc, // output channel
                           int w,  // input width
                           int h,  // input height
                           int c,  // input channel
                           int b,  // batch
                           int d,  // dilation
                           int kw, // kenrel width
                           int kh, // kenrel height
                           int s,  // stride
                           int p,  // padding
                           int g,  // group
                           std::vector<float> wt, std::vector<float> bias, std::vector<float> alpha, float scale,
                           int max, int min, bool depthwise) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input");
        auto iv    = fbb.CreateVector(std::vector<int>({0}));
        auto ov    = fbb.CreateVector(std::vector<int>({0}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Input);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Input);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
    }
    {
        int activation = rand() % 0b11;
        auto ccb       = Convolution2DCommonBuilder(fbb);
        ccb.add_dilateX(d);
        ccb.add_dilateY(d);
        ccb.add_strideX(s);
        ccb.add_strideY(s);
        ccb.add_kernelX(kw);
        ccb.add_kernelY(kh);
        ccb.add_padX(p);
        ccb.add_padY(p);
        ccb.add_padMode(PadMode_CAFFE);
        ccb.add_group(g);
        ccb.add_outputCount(oc);
        ccb.add_relu(activation & 0b01);
        ccb.add_relu6(activation & 0b10);
        auto common = ccb.Finish();

        int size = (int)wt.size();
        std::vector<int8_t> buffer;
        if (depthwise) {
            buffer.push_back(2);                    // dim
            buffer.push_back((oc & 0x00ff) >> 0);   // - short
            buffer.push_back((oc & 0xff00) >> 8);   // - extent
            buffer.push_back((size & 0x00ff) >> 0); // - short
            buffer.push_back((size & 0xff00) >> 8); // - extent
        } else if (size > UINT16_MAX) {
            buffer.push_back(2);                             // dim
            buffer.push_back((size & 0x000000ff) >> 0);      // - short
            buffer.push_back((size & 0x0000ff00) >> 8);      // - extent
            buffer.push_back((size / (size & 0xffff)) >> 0); // - short
            buffer.push_back((size / (size & 0xffff)) >> 8); // - extent
        } else {
            buffer.push_back(1);                    // dim
            buffer.push_back((size & 0x00ff) >> 0); // - short
            buffer.push_back((size & 0xff00) >> 8); // - extent
        }
        buffer.push_back(0); // sample count
        for (int i = -128; i < 128; i++)
            buffer.push_back(i); // samples
        for (float v : wt)
            buffer.push_back(v + 128); // value to index

        auto buffers = fbb.CreateVector(buffer);
        auto alphas  = fbb.CreateVector(alpha);
        auto iqb     = IDSTQuanBuilder(fbb);
        iqb.add_quantScale(scale);
        iqb.add_scaleIn(1.f);
        iqb.add_scaleOut(1.f);
        iqb.add_aMax(max);
        iqb.add_aMin(min);
        iqb.add_readType(0);
        iqb.add_has_scaleInt(true);
        iqb.add_buffer(buffers);
        iqb.add_alpha(alphas);
        iqb.add_type(1);
        iqb.add_useInt32(false);
        auto qnt = iqb.Finish();

        auto weights = fbb.CreateVector(wt);
        auto biases  = fbb.CreateVector(bias);
        auto cb      = Convolution2DBuilder(fbb);
        cb.add_common(common);
        cb.add_weight(weights);
        cb.add_bias(biases);
        cb.add_quanParameter(flatbuffers::Offset<IDSTQuan>(qnt.o));
        auto conv = cb.Finish();
        auto name = fbb.CreateString("qntconv");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(depthwise ? OpType_ConvolutionDepthwise : OpType_Convolution);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Convolution2D);
        builder.add_main(flatbuffers::Offset<void>(conv.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "output"});
    NetBuilder net(fbb);
    net.add_oplists(ops);
    net.add_tensorName(names);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class QuantizedConvolutionTest : public MNNTestCase {
public:
    virtual ~QuantizedConvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 1; b++) {     // CPU do not support batch now
            for (int g = 1; g <= 1; g++) { // 1
                for (int o = 1; o <= 16; o *= 4) {
                    for (int c = 1; c <= 16; c *= 4) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                            continue;

                                        for (int s = 1; s <= 2; s++) {
                                            // ARM do not support pad ...
                                            for (int p = 0; p <= 0 && p <= is && p <= std::min(kw, kh); p++) {
                                                dispatch([&](MNNForwardType backend) -> void {
                                                    if (backend == MNN_FORWARD_CPU)
                                                        return;
                                                    std::vector<float> wt, bias, alpha;
                                                    for (int i = 0; i < g * (o / g) * (c / g) * kw * kh; i++) {
                                                        wt.push_back(rand() % 16);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        bias.push_back(rand() % 255 / 255.f);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        alpha.push_back(rand() % 255 / 255.f);
                                                    }

                                                    // nets
                                                    auto net = create(o, is, is, c, b, d, kw, kh, s, p, g, wt, bias,
                                                                      alpha, 1.f / c / o, 127, -128, false);
                                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                                    auto GPU = createSession(net, backend);
                                                    if (!CPU || !GPU) {
                                                        delete net;
                                                        return;
                                                    }

                                                    // input/output
                                                    auto input = new Tensor(4);
                                                    {
                                                        input->buffer().dim[0].extent = b;
                                                        input->buffer().dim[1].extent = c;
                                                        input->buffer().dim[2].extent = is;
                                                        input->buffer().dim[3].extent = is;
                                                        TensorUtils::setLinearLayout(input);
                                                        input->buffer().host = (uint8_t *)malloc(input->size());
                                                        for (int i = 0; i < is * is * c * b; i++) {
                                                            input->host<float>()[i] = rand() % 16;
                                                        }

                                                        auto host   = net->getSessionInput(CPU, NULL);
                                                        auto device = net->getSessionInput(GPU, NULL);
                                                        net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                                        net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                                    }

                                                    // infer
                                                    assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU),
                                                                                       0.01));

                                                    // clean up
                                                    free(input->buffer().host);
                                                    delete input;
                                                    delete net;
                                                });
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

class QuantizedDepthwiseConvolutionTest : public MNNTestCase {
public:
    virtual ~QuantizedDepthwiseConvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int o = 1; o <= 8; o *= 2) {
                for (int g = o; g <= o; g++) {
                    for (int c = o; c <= o; c++) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    for (int d = 1; d <= 2; d++) {
                                        if (d > std::min(kw, kh) || d * (std::max(kw, kh) - 1) + 1 > is)
                                            continue;

                                        for (int s = 1; s <= 2; s++) {
                                            for (int p = 0; p <= 1; p++) {
                                                dispatch([&](MNNForwardType backend) -> void {
                                                    if (backend == MNN_FORWARD_CPU)
                                                        return;
                                                    std::vector<float> wt, bias, alpha;
                                                    for (int i = 0; i < g * (o / g) * (c / g) * kw * kh; i++) {
                                                        wt.push_back(rand() % 16);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        bias.push_back(rand() % 255 / 255.f);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        alpha.push_back(rand() % 255 / 255.f);
                                                    }

                                                    auto net = create(o, is, is, c, b, d, kw, kh, s, p, g, wt, bias,
                                                                      alpha, 1.f / c / o, 127, -128, true);
                                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                                    auto GPU = createSession(net, backend);
                                                    if (!CPU || !GPU) {
                                                        delete net;
                                                        return;
                                                    }

                                                    // input/output
                                                    auto input = new Tensor(4);
                                                    {
                                                        input->buffer().dim[0].extent = b;
                                                        input->buffer().dim[1].extent = c;
                                                        input->buffer().dim[2].extent = is;
                                                        input->buffer().dim[3].extent = is;
                                                        TensorUtils::setLinearLayout(input);
                                                        input->buffer().host = (uint8_t *)malloc(input->size());
                                                        for (int i = 0; i < is * is * c * b; i++) {
                                                            input->host<float>()[i] = rand() % 16;
                                                        }
                                                    }

                                                    auto host   = net->getSessionInput(CPU, NULL);
                                                    auto device = net->getSessionInput(GPU, NULL);
                                                    net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                                    net->getBackend(GPU, device)->onCopyBuffer(input, device);

                                                    // infer
                                                    assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU),
                                                                                       0.01));

                                                    // clean up
                                                    free(input->buffer().host);
                                                    delete input;
                                                    delete net;
                                                });
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

MNNTestSuiteRegister(QuantizedConvolutionTest, "op/convolution/qnt_conv");
MNNTestSuiteRegister(QuantizedDepthwiseConvolutionTest, "op/convolution/qnt_depthwise_conv");
