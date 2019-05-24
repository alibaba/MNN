//
//  ConvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Interpreter.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "MNN_generated.h"
#include "Session.hpp"
#include "TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

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
                           std::vector<float> wt, std::vector<float> bias, bool depthwise) {
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

        auto weights = fbb.CreateVector(wt);
        auto biases  = fbb.CreateVector(bias);
        auto cb      = Convolution2DBuilder(fbb);
        cb.add_common(common);
        cb.add_weight(weights);
        cb.add_bias(biases);
        auto conv = cb.Finish();
        auto name = fbb.CreateString("conv");
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

class ConvolutionTest : public MNNTestCase {
public:
    virtual ~ConvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int g = 1; g <= 1; g++) { // 1
                for (int o = 1; o <= 8; o *= 2) {
                    for (int c = 1; c <= 8; c *= 2) {
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
                                                    std::vector<float> wt, bias;
                                                    for (int i = 0; i < g * (o / g) * (c / g) * kw * kh; i++) {
                                                        wt.push_back(rand() % 255 / 255.f);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        bias.push_back(rand() % 255 / 255.f);
                                                    }

                                                    // nets
                                                    auto net =
                                                        create(o, is, is, c, b, d, kw, kh, s, p, g, wt, bias, false);
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
                                                            input->host<float>()[i] = rand() % 255 / 255.f;
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

class DepthwiseConvolutionTest : public MNNTestCase {
public:
    virtual ~DepthwiseConvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int o = 4; o <= 8; o *= 2) {
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
                                                    std::vector<float> wt, bias;
                                                    for (int i = 0; i < g * o / g * c / g * kw * kh; i++) {
                                                        wt.push_back(rand() % 255 / 255.f);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        bias.push_back(rand() % 255 / 255.f);
                                                    }

                                                    // nets
                                                    auto net =
                                                        create(o, is, is, c, b, d, kw, kh, s, p, g, wt, bias, true);
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
                                                            input->host<float>()[i] = rand() % 255 / 255.f;
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

class GroupConvolutionTest : public MNNTestCase {
public:
    virtual ~GroupConvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int g = 2; g <= 4; g *= 2) {
                for (int o = g * 4; o <= 4 * g * 4; o += g * 4) {
                    for (int c = g * 4; c <= 4 * g * 4; c += g * 4) {
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
                                                    std::vector<float> wt, bias;
                                                    for (int i = 0; i < g * o / g * c / g * kw * kh; i++) {
                                                        wt.push_back(rand() % 255 / 255.f);
                                                    }
                                                    for (int i = 0; i < o; i++) {
                                                        bias.push_back(rand() % 255 / 255.f);
                                                    }

                                                    // nets
                                                    auto net =
                                                        create(o, is, is, c, b, d, kw, kh, s, p, g, wt, bias, false);
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
                                                            input->host<float>()[i] = rand() % 255 / 255.f;
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

MNNTestSuiteRegister(ConvolutionTest, "op/convolution/conv");
MNNTestSuiteRegister(QuantizedConvolutionTest, "op/convolution/qnt_conv");
MNNTestSuiteRegister(DepthwiseConvolutionTest, "op/convolution/depthwise_conv");
MNNTestSuiteRegister(QuantizedDepthwiseConvolutionTest, "op/convolution/qnt_depthwise_conv");
MNNTestSuiteRegister(GroupConvolutionTest, "op/convolution/conv_group");
