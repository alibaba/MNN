//
//  BinaryOPTest.cpp
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

static Interpreter *create(int opType, int b0, int c0, int h0, int w0, int b1, int c1, int h1, int w1) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b0, c0, h0, w0}));
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
        auto dims = fbb.CreateVector(std::vector<int>({b1, c1, h1, w1}));
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
        BinaryOpBuilder bob(fbb);
        bob.add_opType(opType);
        auto binary = bob.Finish();
        auto name   = fbb.CreateString("binaryop");
        auto iv     = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov     = fbb.CreateVector(std::vector<int>({2}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_BinaryOp);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_BinaryOp);
        builder.add_main(flatbuffers::Offset<void>(binary.o));
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

class BinaryOPTest : public MNNTestCase {
public:
    virtual ~BinaryOPTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b *= 2) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            int optype = 0;
                            int b0, c0, h0, w0, b1, c1, h1, w1;
                            int b_1[] = {b, 1};
                            int c_1[] = {c, 1};
                            int h_1[] = {h, 1};
                            int w_1[] = {w, 1};
                            b0 = b_1[rand() % 2];
                            c0 = c_1[rand() % 2];
                            h0 = h_1[rand() % 2];
                            w0 = w_1[rand() % 2];
                            b1 = b_1[rand() % 2];
                            c1 = c_1[rand() % 2];
                            h1 = h_1[rand() % 2];
                            w1 = w_1[rand() % 2];

                            auto net   = create(optype, b0, c0, h0, w0, b1, c1, h1, w1);
                            auto CPU   = createSession(net, MNN_FORWARD_CPU);
                            auto GPU   = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input
                            auto input0 = new Tensor(4);
                            {
                                input0->buffer().dim[0].extent = b0;
                                input0->buffer().dim[1].extent = c0;
                                input0->buffer().dim[2].extent = h0;
                                input0->buffer().dim[3].extent = w0;
                                TensorUtils::setLinearLayout(input0);
                                input0->buffer().host = (uint8_t *)malloc(input0->size());
                                for (int i = 0; i < b0 * c0 * h0 * w0; i++) {
                                    input0->host<float>()[i] = rand() % 255 / 255.f;
                                }
                                auto host   = net->getSessionInput(CPU, NULL);
                                auto device = net->getSessionInput(GPU, NULL);
                                net->getBackend(CPU, host)->onCopyBuffer(input0, host);
                                net->getBackend(GPU, device)->onCopyBuffer(input0, device);
                            }

                            auto input1 = new Tensor(4);
                            {
                                input1->buffer().dim[0].extent = b1;
                                input1->buffer().dim[1].extent = c1;
                                input1->buffer().dim[2].extent = h1;
                                input1->buffer().dim[3].extent = w1;
                                TensorUtils::setLinearLayout(input1);
                                input1->buffer().host = (uint8_t *)malloc(input1->size());
                                for (int i = 0; i < b1 * c1 * h1 * w1; i++) {
                                    input1->host<float>()[i] = rand() % 255 / 255.f;
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
        return true;
    }
};
MNNTestSuiteRegister(BinaryOPTest, "op/binary");
