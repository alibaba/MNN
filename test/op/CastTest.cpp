//
//  CastTest.cpp
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

static Interpreter *create(int b, int c, int h, int w, DataType src, DataType dst) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(src);
        ib.add_dformat(MNN_DATA_FORMAT_NHWC);
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
        CastParamBuilder cpb(fbb);
        cpb.add_srcT(src);
        cpb.add_dstT(dst);
        auto cast = cpb.Finish();
        auto name = fbb.CreateString("cast");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Cast);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_CastParam);
        builder.add_main(flatbuffers::Offset<void>(cast.o));
        vec.push_back(builder.Finish());
    }

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        BlobBuilder bb(fbb);
        bb.add_dataType(src);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();

        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        tdb.add_index(0);
        desc.push_back(tdb.Finish());
    }
    {
        BlobBuilder bb(fbb);
        bb.add_dataType(dst);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();

        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        tdb.add_index(1);
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVectorOfStrings({"input", "output"});
    auto extras = fbb.CreateVector(desc);
    NetBuilder net(fbb);
    net.add_sourceType(NetSource_TENSORFLOW);
    net.add_oplists(ops);
    net.add_tensorName(names);
    net.add_extraTensorDescribe(extras);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class CastTest : public MNNTestCase {
public:
    virtual ~CastTest() = default;
    virtual bool run() {
        DataType types[][2] = {
            {DataType_DT_FLOAT, DataType_DT_INT32},
            {DataType_DT_INT32, DataType_DT_FLOAT},
            {DataType_DT_UINT8, DataType_DT_FLOAT},
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType) / 2; t++) {
            DataType src = types[t][0], dst = types[t][1];
            for (int b = 1; b <= 2; b++) {
                for (int c = 1; c <= 8; c *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int w = 1; w <= 8; w *= 2) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                auto net = create(b, c, h, w, src, dst);
                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                auto GPU = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input/output
                                auto input = new Tensor(4, Tensor::TENSORFLOW);
                                {
                                    input->setType(src);
                                    input->buffer().dim[0].extent = b;
                                    input->buffer().dim[1].extent = h;
                                    input->buffer().dim[2].extent = w;
                                    input->buffer().dim[3].extent = c;
                                    TensorUtils::setLinearLayout(input);
                                    input->buffer().host = (uint8_t *)malloc(input->size());

                                    if (src == DataType_DT_FLOAT) {
                                        for (int i = 0; i < b * h * w * c; i++) {
                                            input->host<float>()[i] = rand() % 255;
                                        }
                                    } else if (src == DataType_DT_INT32) {
                                        for (int i = 0; i < b * h * w * c; i++) {
                                            input->host<int>()[i] = rand() % 255;
                                        }
                                    } else if (src == DataType_DT_UINT8) {
                                        for (int i = 0; i < b * h * w * c; i++) {
                                            input->host<uint8_t>()[i] = rand() % 255;
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
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(CastTest, "op/cast");
