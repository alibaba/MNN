//
//  QuantizedAddTest.cpp
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

static Interpreter *create(FusedActivation act, int zp0, int zp1, int zpo, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_UINT8);
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
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_UINT8);
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
        auto iqpb1 = QuantizedParamBuilder(fbb);
        iqpb1.add_zeroPoint(zp0);
        iqpb1.add_scale(0.3);
        auto iqp1  = iqpb1.Finish();
        auto iqpb2 = QuantizedParamBuilder(fbb);
        iqpb2.add_zeroPoint(zp1);
        iqpb2.add_scale(0.5);
        auto iqp2 = iqpb2.Finish();
        auto oqpb = QuantizedParamBuilder(fbb);
        oqpb.add_zeroPoint(zpo);
        oqpb.add_scale(0.7);
        auto oqp = oqpb.Finish();

        auto qab = QuantizedAddBuilder(fbb);
        qab.add_activationType(act);
        qab.add_input1QuantizedParam(iqp1);
        qab.add_input2QuantizedParam(iqp2);
        qab.add_outputQuantizedParam(oqp);
        auto add = qab.Finish();

        auto name = fbb.CreateString("add");
        auto iv   = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_QuantizedAdd);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_QuantizedAdd);
        builder.add_main(flatbuffers::Offset<void>(add.o));
        vec.push_back(builder.Finish());
    }

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    BlobBuilder builder(fbb);
    builder.add_dataType(DataType_DT_UINT8);
    builder.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto blob = builder.Finish();
    for (int i = 0; i <= 2; i++) {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        tdb.add_index(i);
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVectorOfStrings({"input0", "input1", "output"});
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

class QuantizedAddTest : public MNNTestCase {
public:
    virtual ~QuantizedAddTest() = default;
    virtual bool run() {
        for (int act = FusedActivation_MIN; act <= FusedActivation_MAX; act++) {
            for (int b = 1; b <= 2; b++) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int c = 1; c <= 8; c *= 2) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                auto zp0 = rand() % UINT8_MAX;
                                auto zp1 = rand() % UINT8_MAX;
                                auto zpo = rand() % UINT8_MAX;

                                // nets
                                auto net = create((FusedActivation)act, zp0, zp1, zpo, b, c, h, w);
                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                auto GPU = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input
                                auto input0 = new Tensor(4, Tensor::TENSORFLOW);
                                {
                                    input0->setType(DataType_DT_UINT8);
                                    input0->buffer().dim[0].extent = b;
                                    input0->buffer().dim[1].extent = h;
                                    input0->buffer().dim[2].extent = w;
                                    input0->buffer().dim[3].extent = c;
                                    TensorUtils::setLinearLayout(input0);
                                    input0->buffer().host = (uint8_t *)malloc(input0->size());
                                    for (int i = 0; i < b * c * h * w; i++) {
                                        input0->host<uint8_t>()[i] = rand() % UINT8_MAX;
                                    }
                                    auto host   = net->getSessionInput(CPU, "input0");
                                    auto device = net->getSessionInput(GPU, "input0");
                                    net->getBackend(CPU, host)->onCopyBuffer(input0, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(input0, device);
                                }
                                auto input1 = new Tensor(4, Tensor::TENSORFLOW);
                                {
                                    input1->setType(DataType_DT_UINT8);
                                    input1->buffer().dim[0].extent = b;
                                    input1->buffer().dim[1].extent = h;
                                    input1->buffer().dim[2].extent = w;
                                    input1->buffer().dim[3].extent = c;
                                    TensorUtils::setLinearLayout(input1);
                                    input1->buffer().host = (uint8_t *)malloc(input1->size());
                                    for (int i = 0; i < b * c * h * w; i++) {
                                        input1->host<uint8_t>()[i] = rand() % UINT8_MAX;
                                    }
                                    auto host   = net->getSessionInput(CPU, "input1");
                                    auto device = net->getSessionInput(GPU, "input1");
                                    net->getBackend(CPU, host)->onCopyBuffer(input1, host);
                                    net->getBackend(GPU, device)->onCopyBuffer(input1, device);
                                }

                                // infer
                                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU)));

                                // clean up
                                free(input0->buffer().host);
                                free(input1->buffer().host);
                                delete input0;
                                delete input1;
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
MNNTestSuiteRegister(QuantizedAddTest, "op/quantized_add");
