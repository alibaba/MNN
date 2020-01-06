//
//  EltwiseTest.cpp
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

static Interpreter *create(EltwiseType type, int w, int h, int c, int b) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input0");
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
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input1");
        auto iv    = fbb.CreateVector(std::vector<int>({1}));
        auto ov    = fbb.CreateVector(std::vector<int>({1}));

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
        auto eb = EltwiseBuilder(fbb);
        eb.add_type(type);
        auto eltwise = eb.Finish();
        auto name    = fbb.CreateString("eltwise");
        auto iv      = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov      = fbb.CreateVector(std::vector<int>({2}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Eltwise);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Eltwise);
        builder.add_main(flatbuffers::Offset<void>(eltwise.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input0", "input1", "output"});
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

class EltwiseTest : public MNNTestCase {
public:
    virtual ~EltwiseTest() = default;
    virtual bool run() {
        EltwiseType types[] = {EltwiseType_PROD, EltwiseType_SUM, EltwiseType_MAXIMUM};

        for (int t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
            EltwiseType type = types[t];

            for (int b = 1; b <= 2; b++) {
                for (int c = 1; c <= 8; c *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int h = 1; h <= 8; h *= 2) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                auto net = create(type, w, h, c, b);
                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                auto GPU = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input
                                auto input0 = new Tensor(4);
                                {
                                    input0->buffer().dim[0].extent = b;
                                    input0->buffer().dim[1].extent = c;
                                    input0->buffer().dim[2].extent = h;
                                    input0->buffer().dim[3].extent = w;
                                    TensorUtils::setLinearLayout(input0);
                                    input0->buffer().host = (uint8_t *)malloc(input0->size());
                                    for (int i = 0; i < b * c * h * w; i++) {
                                        input0->host<float>()[i] = rand() % 100 / 100.f;
                                    }
                                    auto host   = net->getSessionInput(CPU, NULL);
                                    auto device = net->getSessionInput(GPU, NULL);
                                    net->getBackend(CPU, host)->onCopyBuffer(input0, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(input0, device);
                                }

                                auto input1 = new Tensor(4);
                                {
                                    input1->buffer().dim[0].extent = b;
                                    input1->buffer().dim[1].extent = c;
                                    input1->buffer().dim[2].extent = h;
                                    input1->buffer().dim[3].extent = w;
                                    TensorUtils::setLinearLayout(input1);
                                    input1->buffer().host = (uint8_t *)malloc(input1->size());
                                    for (int i = 0; i < b * c * h * w; i++) {
                                        input1->host<float>()[i] = rand() % 100 / 100.f;
                                    }
                                    auto host   = net->getSessionInput(CPU, "input1");
                                    auto device = net->getSessionInput(GPU, "input1");
                                    net->getBackend(CPU, host)->onCopyBuffer(input1, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(input1, device);
                                }

                                // infer
                                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                                // clean up
                                free(input0->buffer().host);
                                free(input1->buffer().host);
                                delete input0;
                                delete input1;
                                delete net;
                            });
                        }
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(EltwiseTest, "op/eltwise");
