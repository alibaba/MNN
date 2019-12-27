//
//  DequantizeTest.cpp
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

static Interpreter *create(DataType type, QuantizeMode mode, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    { // input
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(type);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input0");
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
        auto dims = fbb.CreateVector(std::vector<int>({1}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input1");
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
        auto dims = fbb.CreateVector(std::vector<int>({1}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input2");
        auto iv    = fbb.CreateVector(std::vector<int>({2}));
        auto ov    = fbb.CreateVector(std::vector<int>({2}));

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
        auto db = DequantizeBuilder(fbb);
        db.add_type(type);
        db.add_mode(mode);
        auto dequantize = db.Finish();
        auto name       = fbb.CreateString("dequantize");
        auto iv         = fbb.CreateVector(std::vector<int>({0, 1, 2}));
        auto ov         = fbb.CreateVector(std::vector<int>({3}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Dequantize);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Dequantize);
        builder.add_main(flatbuffers::Offset<void>(dequantize.o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder fb(fbb);
    fb.add_dataType(DataType_DT_FLOAT);
    fb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto flt = fb.Finish();
    BlobBuilder qb(fbb);
    qb.add_dataType(type);
    qb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto qnt = qb.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(qnt.o));
        desc.push_back(tdb.Finish());
    }
    for (int i = 1; i <= 3; i++) {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(i);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVectorOfStrings({"input0", "input1", "input2", "output"});
    auto extras = fbb.CreateVector(desc);
    NetBuilder net(fbb);
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

class DequantizeTest : public MNNTestCase {
public:
    virtual ~DequantizeTest() = default;
    virtual bool run() {
        DataType types[] = {DataType_DT_QUINT8, DataType_DT_QUINT16, DataType_DT_QINT8, DataType_DT_QINT16,
                            DataType_DT_QINT32};

        for (int m = QuantizeMode_MIN; m <= QuantizeMode_MAX; m++) {
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
                                    auto net = create(type, (QuantizeMode)m, b, c, h, w);
                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU = createSession(net, MNN_FORWARD_METAL);
                                    if (!CPU || !GPU) {
                                        delete net;
                                        return;
                                    }

                                    // input
                                    auto input0 = new Tensor(4, Tensor::TENSORFLOW);
                                    {
                                        input0->setType(type);
                                        input0->buffer().dim[0].extent = b;
                                        input0->buffer().dim[1].extent = h;
                                        input0->buffer().dim[2].extent = w;
                                        input0->buffer().dim[3].extent = c;
                                        TensorUtils::setLinearLayout(input0);
                                        input0->buffer().host = (uint8_t *)malloc(input0->size());

                                        if (type == DataType_DT_QUINT8) {
                                            for (int i                     = 0; i < b * c * h * w; i++)
                                                input0->host<uint8_t>()[i] = rand() % UINT8_MAX;
                                        } else if (type == DataType_DT_QUINT16) {
                                            for (int i                      = 0; i < b * c * h * w; i++)
                                                input0->host<uint16_t>()[i] = rand() % UINT16_MAX;
                                        } else if (type == DataType_DT_QINT8) {
                                            for (int i                    = 0; i < b * c * h * w; i++)
                                                input0->host<int8_t>()[i] = rand() % INT8_MAX;
                                        } else if (type == DataType_DT_QINT16) {
                                            for (int i                     = 0; i < b * c * h * w; i++)
                                                input0->host<int16_t>()[i] = rand() % INT16_MAX;
                                        } else if (type == DataType_DT_QINT32) {
                                            for (int i                     = 0; i < b * c * h * w; i++)
                                                input0->host<int32_t>()[i] = rand() % INT32_MAX;
                                        }

                                        auto host   = net->getSessionInput(CPU, "input0");
                                        auto device = net->getSessionInput(GPU, "input0");
                                        net->getBackend(CPU, host)->onCopyBuffer(input0, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(input0, device);
                                    }
                                    auto input1 = new Tensor(1);
                                    {
                                        input1->setType(DataType_DT_FLOAT);
                                        input1->buffer().dim[0].extent = 1;
                                        TensorUtils::setLinearLayout(input1);
                                        input1->buffer().host    = (uint8_t *)malloc(input1->size());
                                        input1->host<float>()[0] = 0;
                                        auto host                = net->getSessionInput(CPU, "input1");
                                        auto device              = net->getSessionInput(GPU, "input1");
                                        net->getBackend(CPU, host)->onCopyBuffer(input1, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(input1, device);
                                    }
                                    auto input2 = new Tensor(1);
                                    {
                                        input2->setType(DataType_DT_FLOAT);
                                        input2->buffer().dim[0].extent = 1;
                                        TensorUtils::setLinearLayout(input2);
                                        input2->buffer().host    = (uint8_t *)malloc(input2->size());
                                        input2->host<float>()[0] = 255;
                                        auto host                = net->getSessionInput(CPU, "input2");
                                        auto device              = net->getSessionInput(GPU, "input2");
                                        net->getBackend(CPU, host)->onCopyBuffer(input2, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(input2, device);
                                    }

                                    // infer
                                    assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                                    // clean up
                                    free(input0->buffer().host);
                                    free(input1->buffer().host);
                                    free(input2->buffer().host);
                                    delete input0;
                                    delete input1;
                                    delete input2;
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
MNNTestSuiteRegister(DequantizeTest, "op/dequantize");
