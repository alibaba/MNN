//
//  SqueezeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Interpreter.hpp>
#include "MNNTestSuite.h"
#include "core/Session.hpp"
#include "TFQuantizeOp_generated.h"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(DataType type, std::vector<int> squeeze, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;
    std::vector<flatbuffers::Offset<flatbuffers::String>> names;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(type);
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
        auto dims = fbb.CreateVector(squeeze);
        auto spb  = SqueezeParamBuilder(fbb);
        spb.add_squeezeDims(dims);
        auto sp   = spb.Finish();
        auto name = fbb.CreateString("squeeze");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Squeeze);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_SqueezeParam);
        builder.add_main(flatbuffers::Offset<void>(sp.o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder builder(fbb);
    builder.add_dataType(type);
    builder.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto blob = builder.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(1);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto namesv = fbb.CreateVectorOfStrings({"input", "output"});
    auto extras = fbb.CreateVector(desc);
    NetBuilder net(fbb);
    net.add_oplists(ops);
    net.add_tensorName(namesv);
    net.add_extraTensorDescribe(extras);
    net.add_sourceType(NetSource_TENSORFLOW);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class SqueezeTest : public MNNTestCase {
public:
    virtual ~SqueezeTest() = default;
    virtual bool run() {
        DataType types[] = {
            DataType_DT_INT32, DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = types[t];
            int b = 3, c = 5, h = 7, w = 9;
            {
                for (int mask = 0b0001; mask <= 0b1110; mask++) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        std::vector<int> squeeze;
                        for (int j = 0; j < 4; j++) {
                            if (mask & (1 << j))
                                squeeze.push_back(j);
                        }

                        // nets
                        auto net = create(type, squeeze, b, c, h, w);
                        auto CPU = createSession(net, MNN_FORWARD_CPU);
                        auto GPU = createSession(net, backend);
                        if (!CPU || !GPU) {
                            delete net;
                            return;
                        }

                        // input/output
                        auto input = new Tensor(4, Tensor::TENSORFLOW);
                        {
                            input->setType(type);
                            input->buffer().dim[0].extent = b;
                            input->buffer().dim[1].extent = h;
                            input->buffer().dim[2].extent = w;
                            input->buffer().dim[3].extent = c;
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());

                            if (type == DataType_DT_FLOAT) {
                                for (int i = 0; i < b * c * h * w; i++) {
                                    input->host<float>()[i] = rand() % 255 / 255.f;
                                }
                            } else if (type == DataType_DT_INT32) {
                                for (int i = 0; i < b * c * h * w; i++) {
                                    input->host<int>()[i] = rand() % 255;
                                }
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
MNNTestSuiteRegister(SqueezeTest, "op/squeeze");
