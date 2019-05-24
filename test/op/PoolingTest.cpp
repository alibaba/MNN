//
//  PoolingTest.cpp
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

static Interpreter *create(PoolType type, int w, int h, int c, int b, int kernel, int stride, int pad, int g) {
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
        std::vector<float> scales;
        for (int i = 0; i < c * b; i++) {
            scales.push_back(i + 1);
        }
        auto pb = PoolBuilder(fbb);
        pb.add_kernelX(kernel);
        pb.add_kernelY(kernel);
        pb.add_strideX(stride);
        pb.add_strideY(stride);
        pb.add_padX(pad);
        pb.add_padY(pad);
        pb.add_isGlobal(g);
        pb.add_type(type);
        auto pool = pb.Finish();

        auto name = fbb.CreateString("pool");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Pooling);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Pool);
        builder.add_main(flatbuffers::Offset<void>(pool.o));
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

class PoolingMaxTest : public MNNTestCase {
public:
    virtual ~PoolingMaxTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int k = 1; k <= w && k <= h; k *= 2) {
                            for (int s = 1; s <= 4; s *= 2) {
                                for (int p = 0; p <= 2; p++) {
                                    for (int g = 0; g <= 1; g++) {
                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create(PoolType_MAXPOOL, w, h, c, b, k, s, p, g);
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
                                            }

                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);

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
        }
        return true;
    }
};

class PoolingAvgTest : public MNNTestCase {
public:
    virtual ~PoolingAvgTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int k = 1; k <= w && k <= h; k *= 2) {
                            for (int s = 1; s <= 4; s *= 2) {
                                for (int p = 0; p <= 2; p++) {
                                    for (int g = 0; g <= 1; g++) {
                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create(PoolType_AVEPOOL, w, h, c, b, k, s, p, g);
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
                                            }

                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);

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
        }
        return true;
    }
};
MNNTestSuiteRegister(PoolingMaxTest, "op/pool/max");
MNNTestSuiteRegister(PoolingAvgTest, "op/pool/avg");
