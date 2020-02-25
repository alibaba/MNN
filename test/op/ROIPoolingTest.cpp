//
//  ROIPoolingTest.cpp
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

static Interpreter *create(float scale, int pw, int ph, int w, int h, int c, int b, int rb) {
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
        auto dims = fbb.CreateVector(std::vector<int>({rb, 5, 1, 1}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("roi");
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
        auto rp = RoiPoolingBuilder(fbb);
        rp.add_pooledWidth(pw);
        rp.add_pooledHeight(ph);
        rp.add_spatialScale(scale);
        auto roi  = rp.Finish();
        auto name = fbb.CreateString("ROIPooling");
        auto iv   = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_ROIPooling);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_RoiPooling);
        builder.add_main(flatbuffers::Offset<void>(roi.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "roi", "output"});
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

class ROIPoolingTest : public MNNTestCase {
public:
    virtual ~ROIPoolingTest() = default;
    virtual bool run() {
        for (int rb = 1; rb <= 8; rb *= 2) {
            for (int ib = 1; ib <= 8; ib *= 2) {
                for (int c = 1; c <= 8; c *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int w = 1; w <= 8; w *= 2) {
                            int pw = 5, ph = 7;
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                float scale = rand() % 255 / 255.f;
                                auto net    = create(scale, pw, ph, w, h, c, ib, rb);
                                auto CPU    = createSession(net, MNN_FORWARD_CPU);
                                auto GPU    = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input
                                auto input = new Tensor(4);
                                {
                                    input->buffer().dim[0].extent = ib;
                                    input->buffer().dim[1].extent = c;
                                    input->buffer().dim[2].extent = h;
                                    input->buffer().dim[3].extent = w;
                                    TensorUtils::setLinearLayout(input);
                                    input->buffer().host = (uint8_t *)malloc(input->size());
                                    for (int i = 0; i < ib * c * h * w; i++) {
                                        input->host<float>()[i] = rand() % 255 / 255.f;
                                    }
                                    auto host   = net->getSessionInput(CPU, "input");
                                    auto device = net->getSessionInput(GPU, "input");
                                    net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                }

                                // roi
                                auto roi = new Tensor(4);
                                {
                                    roi->buffer().dim[0].extent = rb;
                                    roi->buffer().dim[1].extent = 5;
                                    roi->buffer().dim[2].extent = 1;
                                    roi->buffer().dim[3].extent = 1;
                                    TensorUtils::setLinearLayout(roi);
                                    roi->buffer().host = (uint8_t *)malloc(roi->size());
                                    for (int i = 0; i < rb * 5; i++) {
                                        if ((i % 5) == 0) {
                                            roi->host<float>()[i] = std::min(i, ib - 1);
                                        } else {
                                            roi->host<float>()[i] = i;
                                        }
                                    }
                                    auto host   = net->getSessionInput(CPU, "roi");
                                    auto device = net->getSessionInput(GPU, "roi");
                                    net->getBackend(CPU, host)->onCopyBuffer(roi, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(roi, device);
                                }

                                // infer
                                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                                // clean up
                                free(input->buffer().host);
                                free(roi->buffer().host);
                                delete input;
                                delete roi;
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
MNNTestSuiteRegister(ROIPoolingTest, "op/roipooling");
