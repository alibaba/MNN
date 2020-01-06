//
//  CropTest.cpp
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

static Interpreter *create(int axis, std::vector<int> offsets, int b, int c, int h0, int w0, int h1, int w1) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h0, w0}));
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
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h1, w1}));
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
        auto data = fbb.CreateVector(offsets);
        auto cb   = CropBuilder(fbb);
        cb.add_axis(axis);
        cb.add_offset(data);
        auto crop = cb.Finish();
        auto name = fbb.CreateString("crop");
        auto iv   = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Crop);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Crop);
        builder.add_main(flatbuffers::Offset<void>(crop.o));
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

class CropWTest : public MNNTestCase {
public:
    virtual ~CropWTest() = default;
    virtual bool run() {
        int b = 3, c = 5, h0 = 7, w0 = 9;
        dispatch([&](MNNForwardType backend) -> void {
            if (backend == MNN_FORWARD_CPU)
                return;
            std::vector<int> offsets = {2};
            int h1 = h0, w1 = w0 - offsets[0];

            // nets
            auto net = create(3, offsets, b, c, h0, w0, h1, w1);
            auto CPU = createSession(net, MNN_FORWARD_CPU);
            auto GPU = createSession(net, backend);
            if (!CPU || !GPU) {
                delete net;
                return;
            }

            // input0
            auto input = new Tensor(4);
            {
                input->buffer().dim[0].extent = b;
                input->buffer().dim[1].extent = c;
                input->buffer().dim[2].extent = h0;
                input->buffer().dim[3].extent = w0;
                TensorUtils::setLinearLayout(input);
                input->buffer().host = (uint8_t *)malloc(input->size());
                for (int i = 0; i < b * c * h0 * w0; i++) {
                    input->host<float>()[i] = i + 1;
                }
                auto host   = net->getSessionInput(CPU, "input0");
                auto device = net->getSessionInput(GPU, "input0");
                net->getBackend(CPU, host)->onCopyBuffer(input, host);
                net->getBackend(GPU, device)->onCopyBuffer(input, device);
            }

            // infer
            assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.015));

            // clean up
            free(input->buffer().host);
            delete input;
            delete net;
        });
        return true;
    }
};

class CropHTest : public MNNTestCase {
public:
    virtual ~CropHTest() = default;
    virtual bool run() {
        int b = 3, c = 5, h0 = 7, w0 = 9;
        for (int i = 0; i < 2; i++) {
            dispatch([&](MNNForwardType backend) -> void {
                if (backend == MNN_FORWARD_CPU)
                    return;
                std::vector<int> offsets;
                int h1 = h0, w1 = w0;
                if (i == 0) {
                    offsets.push_back(1);
                    offsets.push_back(2);
                    h1 -= 1;
                    w1 -= 2;
                } else {
                    offsets.push_back(2);
                    h1 -= 2;
                    w1 -= 2;
                }

                // nets
                auto net = create(3, offsets, b, c, h0, w0, h1, w1);
                auto CPU = createSession(net, MNN_FORWARD_CPU);
                auto GPU = createSession(net, backend);
                if (!CPU || !GPU) {
                    delete net;
                    return;
                }

                // input
                auto input = new Tensor(4);
                {
                    input->buffer().dim[0].extent = b;
                    input->buffer().dim[1].extent = c;
                    input->buffer().dim[2].extent = h0;
                    input->buffer().dim[3].extent = w0;
                    TensorUtils::setLinearLayout(input);
                    input->buffer().host = (uint8_t *)malloc(input->size());
                    for (int j = 0; j < b * c * h0 * w0; j++) {
                        input->host<float>()[j] = j + 1;
                    }
                    auto host   = net->getSessionInput(CPU, "input0");
                    auto device = net->getSessionInput(GPU, "input0");
                    net->getBackend(CPU, host)->onCopyBuffer(input, host);
                    net->getBackend(GPU, device)->onCopyBuffer(input, device);
                }

                // infer
                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.015));

                // clean up
                free(input->buffer().host);
                delete input;
                delete net;
            });
        }
        return true;
    }
};
MNNTestSuiteRegister(CropWTest, "op/crop/w");
MNNTestSuiteRegister(CropHTest, "op/crop/h");
