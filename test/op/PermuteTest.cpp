//
//  PermuteTest.cpp
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

static Interpreter *create(std::vector<int> dims, int w, int h, int c, int b) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto idims = fbb.CreateVector(std::vector<int>({b, c, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(idims);
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
        auto pdims = fbb.CreateVector(dims);
        auto pb    = PermuteBuilder(fbb);
        pb.add_dims(pdims);
        auto permute = pb.Finish();
        auto name    = fbb.CreateString("permute");
        auto iv      = fbb.CreateVector(std::vector<int>({0}));
        auto ov      = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Permute);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Permute);
        builder.add_main(flatbuffers::Offset<void>(permute.o));
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

class PermuteTest : public MNNTestCase {
public:
    virtual ~PermuteTest() = default;
    virtual bool run() {
        std::vector<int> dims[] = {
            {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
        };

        for (int b = 1; b <= 2; b *= 2) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int d = 0; d < 6; d++) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                auto net = create(dims[d], w, h, c, b);
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
                                    input->buffer().dim[2].extent = h;
                                    input->buffer().dim[3].extent = w;
                                    TensorUtils::setLinearLayout(input);
                                    input->buffer().host = (uint8_t *)malloc(input->size());
                                    for (int j = 0; j < b * c * h * w; j++) {
                                        input->host<float>()[j] = rand() % 255 / 255.f;
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
                                delete net;
                                delete input;
                            });
                        }
                    }
                }
            }
        }
        
        {
            float inputData[] = {1.4, -0.9, 0.8, -1.9, 1.4, 0.5, 0.4, -1.9, -1.2, -0.9, 0.8, -0.1, -0.3, 0.5, -1.9, -1.2};
            float outputData[] = {1.4, 0.8, -0.9, -1.9, -1.2, 0.8, -0.9, -0.1, 1.4, 0.4, 0.5, -1.9, -0.3, -1.9, 0.5, -1.2};
            const int w = 2, h = 2, c = 2, b = 2;
            auto net = create({1, 0, 3, 2}, w, h, c, b);
            auto CPU = createSession(net, MNN_FORWARD_CPU);
            if (!CPU) {
                delete net;
                return false;
            }
            // input
            auto input = new Tensor(4);
            {
                input->buffer().dim[0].extent = b;
                input->buffer().dim[1].extent = c;
                input->buffer().dim[2].extent = h;
                input->buffer().dim[3].extent = w;
                TensorUtils::setLinearLayout(input);
                input->buffer().host = (uint8_t *)malloc(input->size());
                for (int j = 0; j < b * c * h * w; j++) {
                    input->host<float>()[j] = inputData[j];
                }
                auto host   = net->getSessionInput(CPU, NULL);
                net->getBackend(CPU, host)->onCopyBuffer(input, host);
            }
            // output
            auto correctOutput = new Tensor(4);
            {
                correctOutput->buffer().dim[0].extent = c;
                correctOutput->buffer().dim[1].extent = b;
                correctOutput->buffer().dim[2].extent = w;
                correctOutput->buffer().dim[3].extent = h;
                TensorUtils::setLinearLayout(correctOutput);
                correctOutput->buffer().host = (uint8_t *)malloc(correctOutput->size());
                for (int j = 0; j < b * c * h * w; j++) {
                    correctOutput->host<float>()[j] = outputData[j];
                }
            }
            assert(TensorUtils::compareTensors(infer(net, CPU), correctOutput, 0.001));
            delete net;
            free(input->buffer().host);
            delete input;
            free(correctOutput->buffer().host);
            delete correctOutput;
        }
        return true;
    }
};
MNNTestSuiteRegister(PermuteTest, "op/permute");
