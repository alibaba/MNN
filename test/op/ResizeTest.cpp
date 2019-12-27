//
//  ResizeTest.cpp
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

static Interpreter *create(int type, float ws, float hs, int w, int h, int c, int b) {
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
        auto rb = ResizeBuilder(fbb);
        rb.add_xScale(ws);
        rb.add_yScale(hs);
        auto resize = rb.Finish();
        auto name   = fbb.CreateString("resize");
        auto iv     = fbb.CreateVector(std::vector<int>({0}));
        auto ov     = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Resize);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Resize);
        builder.add_main(flatbuffers::Offset<void>(resize.o));
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

class ResizeTest : public MNNTestCase {
public:
    virtual ~ResizeTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c++) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (float ws = 1; ws <= w; ws *= 2) {
                            for (float hs = 1; hs <= h; hs *= 2) {
                                dispatch([&](MNNForwardType backend) -> void {
                                    if (backend == MNN_FORWARD_CPU)
                                        return;
                                    // nets
                                    auto net = create(3, ws, hs, w, h, c, b);
                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU = createSession(net, backend);
                                    if (!CPU || !GPU) {
                                        delete net;
                                        return;
                                    }

                                    // input
                                    auto input = new Tensor(4, Tensor::TENSORFLOW);
                                    {
                                        input->buffer().dim[0].extent = b;
                                        input->buffer().dim[1].extent = h;
                                        input->buffer().dim[2].extent = w;
                                        input->buffer().dim[3].extent = c;
                                        TensorUtils::setLinearLayout(input);
                                        input->buffer().host = (uint8_t *)malloc(input->size());
                                        for (int i = 0; i < b * c * h * w; i++) {
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
        return true;
    }
};
MNNTestSuiteRegister(ResizeTest, "op/resize");
