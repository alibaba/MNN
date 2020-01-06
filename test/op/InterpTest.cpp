//
//  InterpTest.cpp
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

static Interpreter *create(int type, float ws, float hs, bool a, int ow, int oh, int w, int h, int c, int b) {
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
        auto ib = InterpBuilder(fbb);
        ib.add_widthScale(ws);
        ib.add_heightScale(hs);
        ib.add_outputWidth(ow);
        ib.add_outputHeight(oh);
        ib.add_resizeType(type);
        ib.add_alignCorners(a);
        auto interp = ib.Finish();
        auto name   = fbb.CreateString("interp");
        auto iv     = fbb.CreateVector(std::vector<int>({0}));
        auto ov     = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Interp);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Interp);
        builder.add_main(flatbuffers::Offset<void>(interp.o));
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

class InterpBilinearTest : public MNNTestCase {
public:
    virtual ~InterpBilinearTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b *= 2) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int ow = 1; ow <= 8; ow *= 2) {
                            for (int oh = 1; oh <= 8; oh *= 2) {
                                for (int a = 0; a <= 1; a++) {
                                    if (a == 1 && (ow == 1 || oh == 1))
                                        continue;

                                    dispatch([&](MNNForwardType backend) -> void {
                                        if (backend == MNN_FORWARD_CPU)
                                            return;
                                        float ws = rand() % 255 / 255.f;
                                        float hs = rand() % 255 / 255.f;

                                        // nets
                                        auto net = create(2, ws, hs, a, ow, oh, w, h, c, b);
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
                                            input->buffer().dim[2].extent = h;
                                            input->buffer().dim[3].extent = w;
                                            TensorUtils::setLinearLayout(input);
                                            input->buffer().host = (uint8_t *)malloc(input->size());
                                            for (int i = 0; i < w * h * c * b; i++) {
                                                input->host<float>()[i] = rand() % 255 / 255.f;
                                            }
                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                        }

                                        // infer
                                        assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

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
        return true;
    }
};

class InterpCubicTest : public MNNTestCase {
public:
    virtual ~InterpCubicTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b *= 2) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int ow = 1; ow <= 8; ow *= 2) {
                            for (int oh = 1; oh <= 8; oh *= 2) {
                                for (int a = 0; a <= 1; a++) {
                                    dispatch([&](MNNForwardType backend) -> void {
                                        if (backend == MNN_FORWARD_CPU)
                                            return;
                                        float ws = rand() % 255 / 255.f;
                                        float hs = rand() % 255 / 255.f;

                                        // nets
                                        auto net = create(3, ws, hs, a, ow, oh, w, h, c, b);
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
                                            input->buffer().dim[2].extent = h;
                                            input->buffer().dim[3].extent = w;
                                            TensorUtils::setLinearLayout(input);
                                            input->buffer().host = (uint8_t *)malloc(input->size());
                                            for (int i = 0; i < w * h * c * b; i++) {
                                                input->host<float>()[i] = rand() % 255 / 255.f;
                                            }
                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                        }

                                        // infer
                                        assert(
                                            TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01, true));

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
        return true;
    }
};

MNNTestSuiteRegister(InterpBilinearTest, "op/interp/bilinear");
MNNTestSuiteRegister(InterpCubicTest, "op/interp/cubic");
