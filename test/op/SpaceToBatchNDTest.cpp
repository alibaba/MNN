//
//  SpaceToBatchNDTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Interpreter.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "Session.hpp"
#include "TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(std::vector<int> s, std::vector<int> pad, std::vector<int> dims) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto idims = fbb.CreateVector(std::vector<int>(dims));
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
        auto sdims  = fbb.CreateVector(std::vector<int>({2}));
        auto shapes = fbb.CreateVector(s);
        BlobBuilder bb(fbb);
        bb.add_dims(sdims);
        bb.add_dataType(DataType_DT_INT32);
        bb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        bb.add_int32s(shapes);
        auto shape = bb.Finish();

        auto pdims = fbb.CreateVector(std::vector<int>({4}));
        auto pads  = fbb.CreateVector(pad);
        BlobBuilder pb(fbb);
        pb.add_dims(pdims);
        pb.add_dataType(DataType_DT_INT32);
        pb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        pb.add_int32s(pads);
        auto padding = pb.Finish();

        auto sbb = SpaceBatchBuilder(fbb);
        sbb.add_blockShape(flatbuffers::Offset<Blob>(shape.o));
        sbb.add_padding(flatbuffers::Offset<Blob>(padding.o));
        auto sb = sbb.Finish();

        auto name = fbb.CreateString("space_to_batch");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_SpaceToBatchND);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_SpaceBatch);
        builder.add_main(flatbuffers::Offset<void>(sb.o));
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

class SpaceToBatchNDTest : public MNNTestCase {
public:
    virtual ~SpaceToBatchNDTest() = default;
    virtual void run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 4; c *= 2) {
                for (int h = 1; h <= 4; h *= 2) {
                    for (int w = 1; w <= 4; w *= 2) {
                        for (int pw = 0; pw <= 1; pw++) {
                            for (int ph = 0; ph <= 1; ph++) {
                                for (int sw = 1; sw <= 2; sw *= 2) {
                                    for (int sh = 1; sh <= 2; sh *= 2) {
                                        if (sh > h + 2 * ph || sw > w + 2 * pw)
                                            continue;

                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create({sh, sw}, {ph, ph, pw, pw}, {b, c, h, w});
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
            }
        }
    }
};
MNNTestSuiteRegister(SpaceToBatchNDTest, "op/space_to_batch_nd");
