//
//  SpatialProductTest.cpp
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

static Interpreter *create(int w, int h, int ic, int sc) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({1, ic, h, w}));
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
        auto dims = fbb.CreateVector(std::vector<int>({1, sc, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("weight");
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
        auto name = fbb.CreateString("spatial_product");
        auto iv   = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_SpatialProduct);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "weight", "output"});
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

class SpatialProductTest : public MNNTestCase {
public:
    virtual ~SpatialProductTest() = default;
    virtual bool run() {
        for (int ic = 2; ic <= 8; ic *= 2) {
            for (int sc = 1; sc <= 1; sc *= 2) { // support 1 only
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(w, h, ic, sc);
                            auto CPU = createSession(net, MNN_FORWARD_CPU);
                            auto GPU = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input/output
                            auto input = new Tensor(4);
                            {
                                input->buffer().dim[0].extent = 1;
                                input->buffer().dim[1].extent = ic;
                                input->buffer().dim[2].extent = h;
                                input->buffer().dim[3].extent = w;
                                TensorUtils::setLinearLayout(input);
                                input->buffer().host = (uint8_t *)malloc(input->size());
                                for (int i = 0; i < w * h * ic * 1; i++) {
                                    input->host<float>()[i] = rand() % 255 / 255.f;
                                }
                                auto host   = net->getSessionInput(CPU, "input");
                                auto device = net->getSessionInput(GPU, "input");
                                net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                net->getBackend(GPU, device)->onCopyBuffer(input, device);
                            }

                            auto weight = new Tensor(4);
                            {
                                weight->buffer().dim[0].extent = 1;
                                weight->buffer().dim[1].extent = sc;
                                weight->buffer().dim[2].extent = h;
                                weight->buffer().dim[3].extent = w;
                                TensorUtils::setLinearLayout(weight);
                                weight->buffer().host = (uint8_t *)malloc(weight->size());
                                for (int i = 0; i < w * h * sc * 1; i++) {
                                    weight->host<float>()[i] = rand() % 255 / 255.f;
                                }
                                auto host   = net->getSessionInput(CPU, "weight");
                                auto device = net->getSessionInput(GPU, "weight");
                                net->getBackend(CPU, host)->onCopyBuffer(weight, host);
                                net->getBackend(GPU, device)->onCopyBuffer(weight, device);
                            }

                            // infer
                            assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                            // clean up
                            free(input->buffer().host);
                            free(weight->buffer().host);
                            delete input;
                            delete weight;
                            delete net;
                        });
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SpatialProductTest, "op/spatial_product");
