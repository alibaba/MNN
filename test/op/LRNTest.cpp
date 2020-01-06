//
//  LRNTest.cpp
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

static Interpreter *create(int type, int c, int size, int local, float alpha, float beta) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({1, c, size, size}));
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
        auto lb = LRNBuilder(fbb);
        lb.add_regionType(type);
        lb.add_localSize(local);
        lb.add_alpha(alpha);
        lb.add_beta(beta);
        auto lrn  = lb.Finish();
        auto name = fbb.CreateString("lrn");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_LRN);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_LRN);
        builder.add_main(flatbuffers::Offset<void>(lrn.o));
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

class LRNAcrossChannelTest : public MNNTestCase {
public:
    virtual ~LRNAcrossChannelTest() = default;
    virtual bool run() {
        for (int c = 3; c <= 8; c++) {
            for (int size = 1; size <= 8; size *= 2) {
                for (int local = 1; local <= c; local += 2) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        float alpha = rand() % 255 / 255.f;
                        float beta  = rand() % 255 / 255.f;

                        // nets
                        auto net = create(0, c, size, local, alpha, beta);
                        auto CPU = createSession(net, MNN_FORWARD_CPU);
                        auto GPU = createSession(net, backend);
                        if (!CPU || !GPU) {
                            delete net;
                            return;
                        }

                        // input
                        auto input = new Tensor(4);
                        {
                            input->buffer().dim[0].extent = 1;
                            input->buffer().dim[1].extent = c;
                            input->buffer().dim[2].extent = size;
                            input->buffer().dim[3].extent = size;
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int j = 0; j < 1 * c * size * size; j++) {
                                input->host<float>()[j] = j + 1;
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
        return true;
    }
};

class LRNWithinChannelTest : public MNNTestCase {
public:
    virtual ~LRNWithinChannelTest() = default;
    virtual bool run() {
        for (int c = 1; c <= 8; c *= 2) {
            for (int size = 3; size <= 8; size++) {
                for (int local = 1; local <= size; local += 2) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        float alpha = rand() % 255 / 255.f;
                        float beta  = rand() % 255 / 255.f;
                        // nets
                        auto net = create(1, c, size, local, alpha, beta);
                        auto CPU = createSession(net, MNN_FORWARD_CPU);
                        auto GPU = createSession(net, backend);
                        if (!CPU || !GPU) {
                            delete net;
                            return;
                        }

                        // input
                        auto input = new Tensor(4);
                        {
                            input->buffer().dim[0].extent = 1;
                            input->buffer().dim[1].extent = c;
                            input->buffer().dim[2].extent = size;
                            input->buffer().dim[3].extent = size;
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int j = 0; j < 1 * c * size * size; j++) {
                                input->host<float>()[j] = j + 1;
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
        return true;
    }
};

MNNTestSuiteRegister(LRNAcrossChannelTest, "op/lrn/across_channel");
MNNTestSuiteRegister(LRNWithinChannelTest, "op/lrn/within_channel");
