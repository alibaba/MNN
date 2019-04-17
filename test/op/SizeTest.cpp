//
//  SizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Interpreter.hpp"
#include "MNNTestSuite.h"
#include "Session.hpp"
#include "TFQuantizeOp_generated.h"
#include "TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(DataType type, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    { // input
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        auto ib   = InputBuilder(fbb);
        ib.add_dims(dims);
        ib.add_dtype(type);
        auto input = ib.Finish();
        auto main  = flatbuffers::Offset<void>(input.o);
        auto name  = fbb.CreateString("input");
        auto iv    = fbb.CreateVector(std::vector<int>({0}));
        auto ov    = fbb.CreateVector(std::vector<int>({0}));

        auto builder = OpBuilder(fbb);
        builder.add_type(OpType_Input);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Input);
        builder.add_main(main);
        vec.push_back(builder.Finish());
    }
    {
        auto sb = SizeBuilder(fbb);
        sb.add_outputDataType(type);
        auto size = sb.Finish();

        auto main    = flatbuffers::Offset<void>(size.o);
        auto name    = fbb.CreateString("size");
        auto iv      = fbb.CreateVector(std::vector<int>({0}));
        auto ov      = fbb.CreateVector(std::vector<int>({1}));
        auto builder = OpBuilder(fbb);
        builder.add_type(OpType_Size);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Size);
        builder.add_main(main);
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "output"});
    std::vector<flatbuffers::Offset<TensorDescribe>> desc;

    {
        auto fb = BlobBuilder(fbb);
        fb.add_dataType(type);
        fb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto flt = fb.Finish();

        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }
    {
        auto qb = BlobBuilder(fbb);
        qb.add_dataType(DataType_DT_INT32);
        qb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto qnt = qb.Finish();

        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(1);
        tdb.add_blob(flatbuffers::Offset<Blob>(qnt.o));
        desc.push_back(tdb.Finish());
    }

    auto extras = fbb.CreateVector(desc);
    auto net    = NetBuilder(fbb);
    net.add_oplists(ops);
    net.add_tensorName(names);
    net.add_extraTensorDescribe(extras);
    net.add_sourceType(NetSource_TENSORFLOW);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class SizeTest : public MNNTestCase {
public:
    virtual ~SizeTest() = default;
    virtual void run() {
        DataType types[] = {
            DataType_DT_UINT8, DataType_DT_UINT16, DataType_DT_INT8,
            DataType_DT_INT16, DataType_DT_INT32,  DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = (DataType)types[t];
            for (int b = 1; b <= 2; b++) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int c = 1; c <= 8; c *= 2) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                // nets
                                auto net = create(type, b, c, h, w);
                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                auto GPU = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input
                                auto input = new Tensor(4, Tensor::TENSORFLOW);
                                {
                                    input->setType(type);
                                    input->buffer().dim[0].extent = b;
                                    input->buffer().dim[1].extent = h;
                                    input->buffer().dim[2].extent = w;
                                    input->buffer().dim[3].extent = c;
                                    TensorUtils::setLinearLayout(input);
                                    input->buffer().host = (uint8_t *)malloc(input->size());
                                    auto host            = net->getSessionInput(CPU, NULL);
                                    auto device          = net->getSessionInput(GPU, NULL);
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
};
MNNTestSuiteRegister(SizeTest, "op/size");
