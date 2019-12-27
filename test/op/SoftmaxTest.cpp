//
//  SoftmaxTest.cpp
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

static Interpreter *create(int axis, std::vector<int> shape) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(shape);
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
        auto ab = AxisBuilder(fbb);
        ab.add_axis(axis);
        auto softmax = ab.Finish();
        auto name    = fbb.CreateString("softmax");
        auto iv      = fbb.CreateVector(std::vector<int>({0}));
        auto ov      = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Softmax);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Axis);
        builder.add_main(flatbuffers::Offset<void>(softmax.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "output"});
    NetBuilder builder(fbb);
    builder.add_oplists(ops);
    builder.add_tensorName(names);
    fbb.Finish(builder.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class SoftmaxDim4Test : public MNNTestCase {
public:
    virtual ~SoftmaxDim4Test() = default;
    virtual bool run() {
        for (int axis = 0; axis <= 3; axis++) {
            for (int b = 1; b <= 1; b *= 2) { // 1
                for (int c = 1; c <= 8; c *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int w = 1; w <= 8; w *= 2) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                auto net = create(axis, {b, c, h, w});
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
        return true;
    }
};

class SoftmaxDim3Test : public MNNTestCase {
public:
    virtual ~SoftmaxDim3Test() = default;
    virtual bool run() {
        for (int axis = 0; axis <= 2; axis++) {
            for (int c = 1; c <= 1; c *= 2) { // 1
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(axis, {c, h, w});
                            auto CPU = createSession(net, MNN_FORWARD_CPU);
                            auto GPU = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input/output
                            auto input = new Tensor(3);
                            {
                                input->buffer().dim[0].extent = c;
                                input->buffer().dim[1].extent = h;
                                input->buffer().dim[2].extent = w;
                                TensorUtils::setLinearLayout(input);
                                input->buffer().host = (uint8_t *)malloc(input->size());
                                for (int i = 0; i < c * h * w; i++) {
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
        return true;
    }
};

class SoftmaxDim2Test : public MNNTestCase {
public:
    virtual ~SoftmaxDim2Test() = default;
    virtual bool run() {
        for (int axis = 0; axis <= 1; axis++) {
            for (int h = 1; h <= 1; h *= 2) { // 1
                for (int w = 1; w <= 8; w *= 2) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        // nets
                        auto net = create(axis, {h, w});
                        auto CPU = createSession(net, MNN_FORWARD_CPU);
                        auto GPU = createSession(net, backend);
                        if (!CPU || !GPU) {
                            delete net;
                            return;
                        }

                        // input/output
                        auto input = new Tensor(2);
                        {
                            input->buffer().dim[0].extent = h;
                            input->buffer().dim[1].extent = w;
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int i = 0; i < h * w; i++) {
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
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxDim4Test, "op/softmax/dim4");
MNNTestSuiteRegister(SoftmaxDim3Test, "op/softmax/dim3");
MNNTestSuiteRegister(SoftmaxDim2Test, "op/softmax/dim2");
