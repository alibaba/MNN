//
//  MatMulTest.cpp
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

static Interpreter *create(int iw0, int ih0, int iw1, int ih1, int ow, int oh) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({iw0, ih0}));
        InputBuilder ib(fbb);
        auto input = ib.Finish();
        ib.add_dims(dims);

        OpBuilder builder(fbb);
        builder.add_type(OpType_Input);
        auto name = fbb.CreateString("input0");
        builder.add_name(name);
        auto iv = fbb.CreateVector(std::vector<int>({0}));
        builder.add_inputIndexes(iv);
        auto ov = fbb.CreateVector(std::vector<int>({0}));
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Input);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
    }
    {
        auto dims = fbb.CreateVector(std::vector<int>({iw1, ih1}));
        InputBuilder ib(fbb);
        auto input = ib.Finish();
        ib.add_dims(dims);

        OpBuilder builder(fbb);
        builder.add_type(OpType_Input);
        auto name = fbb.CreateString("input1");
        builder.add_name(name);
        auto iv = fbb.CreateVector(std::vector<int>({1}));
        builder.add_inputIndexes(iv);
        auto ov = fbb.CreateVector(std::vector<int>({1}));
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Input);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
    }
    {
        OpBuilder builder(fbb);
        builder.add_type(OpType_MatMul);
        auto name = fbb.CreateString("matMul");
        builder.add_name(name);
        auto iv = fbb.CreateVector(std::vector<int>({0, 1}));
        builder.add_inputIndexes(iv);
        auto ov = fbb.CreateVector(std::vector<int>({2}));
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_MatMul);
        builder.add_main(flatbuffers::Offset<void>(MatMulBuilder(fbb).Finish().o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder builder(fbb);
    builder.add_dataType(DataType_DT_FLOAT);
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
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(2);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        desc.push_back(tdb.Finish());
    }

    NetBuilder net(fbb);
    auto ops = fbb.CreateVector(vec);
    net.add_oplists(ops);
    auto names = fbb.CreateVectorOfStrings({"input0", "input1", "output"});
    net.add_tensorName(names);
    auto extras = fbb.CreateVector(desc);
    net.add_extraTensorDescribe(extras);
    net.add_sourceType(NetSource_TENSORFLOW);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class MatMulTest : public MNNTestCase {
public:
    virtual ~MatMulTest() = default;
    virtual void run() {
        for (int iw0 = 1; iw0 < 2; iw0++) {
            for (int ih0 = 10; ih0 < 20; ih0++) {
                int iw1 = ih0;
                for (int ih1 = 10; ih1 < 20; ih1++) {
                    int ow = iw0;
                    int oh = ih1;

                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        auto net = create(iw0, ih0, iw1, ih1, ow, oh);
                        auto CPU = createSession(net, MNN_FORWARD_CPU);
                        auto GPU = createSession(net, backend);
                        if (!CPU || !GPU) {
                            delete net;
                            return;
                        }

                        // input/output
                        auto input0 = new Tensor(2, Tensor::TENSORFLOW);
                        {
                            input0->buffer().dim[0].extent = iw0;
                            input0->buffer().dim[1].extent = ih0;
                            TensorUtils::setLinearLayout(input0);
                            input0->buffer().host = (uint8_t *)malloc(input0->size());
                            for (int i = 0; i < iw0 * ih0; i++) {
                                input0->host<float>()[i] = rand() % 255 / 255.f;
                            }
                            auto host   = net->getSessionInput(CPU, "input0");
                            auto device = net->getSessionInput(GPU, "input0");
                            net->getBackend(CPU, host)->onCopyBuffer(input0, host);
                            net->getBackend(GPU, device)->onCopyBuffer(input0, device);
                        }

                        auto input1 = new Tensor(2, Tensor::TENSORFLOW);
                        {
                            input1->buffer().dim[0].extent = iw1;
                            input1->buffer().dim[1].extent = ih1;
                            TensorUtils::setLinearLayout(input1);
                            input1->buffer().host = (uint8_t *)malloc(input1->size());
                            for (int i = 0; i < iw1 * ih1; i++) {
                                input1->host<float>()[i] = rand() % 255 / 255.f;
                            }
                            auto host   = net->getSessionInput(CPU, "input1");
                            auto device = net->getSessionInput(GPU, "input1");
                            net->getBackend(CPU, host)->onCopyBuffer(input1, host);
                            net->getBackend(GPU, device)->onCopyBuffer(input1, device);
                        }

                        // infer
                        assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

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
};
MNNTestSuiteRegister(MatMulTest, "op/matmul");
