//
//  NormalizeTest.cpp
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

static Interpreter *create(float eps, int w, int h, int c, int b, int as, int cs) {
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
        for (int i = 0; i < c; i++) {
            scales.push_back(i + 1);
        }
        auto data = fbb.CreateVector(scales);
        auto nb   = NormalizeBuilder(fbb);
        nb.add_acrossSpatial(as);
        nb.add_channelShared(cs);
        nb.add_eps(eps);
        nb.add_scale(data);
        auto normalize = nb.Finish();
        auto name      = fbb.CreateString("normalize");
        auto iv        = fbb.CreateVector(std::vector<int>({0}));
        auto ov        = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Normalize);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Normalize);
        builder.add_main(flatbuffers::Offset<void>(normalize.o));
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

class NormalizeTest : public MNNTestCase {
public:
    virtual ~NormalizeTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 1; b *= 2) { // MNN_ASSERT(1 == inputTensor->batch()) in CPU
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int as = 0; as <= 0; as++) {     // MNN_ASSERT(!mAcrossSpatial); in CPU
                            for (int cs = 0; cs <= 0; cs++) { // MNN_ASSERT(!mChannelShared); in CPU
                                dispatch([&](MNNForwardType backend) -> void {
                                    if (backend == MNN_FORWARD_CPU)
                                        return;
                                    // nets
                                    float eps = rand() % 255 / 255.f;
                                    auto net  = create(eps, w, h, c, b, as, cs);
                                    auto CPU  = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU  = createSession(net, backend);
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
MNNTestSuiteRegister(NormalizeTest, "op/normalize");
