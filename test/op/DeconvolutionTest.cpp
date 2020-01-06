//
//  DeconvolutionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Interpreter.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(int oc, // output channel
                           int is, // input size
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
        auto dims = fbb.CreateVector(std::vector<int>({b, c, is, is}));
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
        auto ccb = Convolution2DCommonBuilder(fbb);
        ccb.add_dilateX(d);
        ccb.add_dilateY(d);
        ccb.add_strideX(s);
        ccb.add_strideY(s);
        ccb.add_kernelX(kw);
        ccb.add_kernelY(kh);
        ccb.add_padX(p);
        ccb.add_padY(p);
        ccb.add_group(g);
        ccb.add_outputCount(oc);
        auto common  = ccb.Finish();
        auto weights = fbb.CreateVector(wt);
        auto biases  = fbb.CreateVector(bias);

        auto cb = Convolution2DBuilder(fbb);
        cb.add_common(common);
        cb.add_weight(weights);
        cb.add_bias(biases);
        auto conv = cb.Finish();
        auto name = fbb.CreateString("deconv");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(depthwise ? OpType_DeconvolutionDepthwise : OpType_Deconvolution);
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

class DeconvolutionTest : public MNNTestCase {
public:
    virtual ~DeconvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int g = 1; g <= 1; g++) {
                for (int o = 1; o <= 8; o *= 2) {
                    for (int c = 1; c <= 8; c *= 2) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 8 && kw <= is; kw *= 2) {
                                for (int kh = 2; kh <= 8 && kh <= is; kh *= 2) {
                                    for (int d = 1; d <= 2 && d <= std::min(kw, kh); d++) {
                                        for (int s = 1; s <= 4; s *= 2) {
                                            for (int p = 0; p <= 2 && p <= is && p <= std::min(kw, kh); p++) {
                                                if ((is - 1) * s + d * (kw - 1) + 1 - p * 2 <= 0)
                                                    continue;
                                                if ((is - 1) * s + d * (kh - 1) + 1 - p * 2 <= 0)
                                                    continue;

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
                                                    auto net = create(o, is, c, b, d, kw, kh, s, p, g, wt, bias, false);
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
                                                                                       0.015));

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

class DepthwiseDeconvolutionTest : public MNNTestCase {
public:
    virtual ~DepthwiseDeconvolutionTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int o = 4; o <= 8; o *= 2) {
                for (int g = o; g <= o; g++) {
                    for (int c = o; c <= o; c++) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 8 && kw <= is; kw *= 2) {
                                for (int kh = 1; kh <= 8 && kh <= is; kh *= 2) {
                                    for (int d = 1; d <= 2 && d <= std::min(kw, kh); d++) {
                                        for (int s = 1; s <= 4; s *= 2) {
                                            for (int p = 0; p <= 2 && p <= is && p <= std::min(kw, kh); p++) {
                                                if ((is - 1) * s + d * (kw - 1) + 1 - p * 2 <= 0)
                                                    continue;
                                                if ((is - 1) * s + d * (kh - 1) + 1 - p * 2 <= 0)
                                                    continue;

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
                                                    auto net = create(o, is, c, b, d, kw, kh, s, p, g, wt, bias, true);
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
                                                                                       0.015));

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

MNNTestSuiteRegister(DeconvolutionTest, "op/deconvolution/deconv");
MNNTestSuiteRegister(DepthwiseDeconvolutionTest, "op/deconvolution/depthwise_deconv");
// deconv do not support group now
