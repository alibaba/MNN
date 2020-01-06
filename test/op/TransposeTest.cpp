//
//  TransposeTest.cpp
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

static Interpreter *create(std::vector<int> perms, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    { // input
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
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
        auto dims = fbb.CreateVector(std::vector<int>({(int)perms.size()}));
        auto data = fbb.CreateVector(perms);
        BlobBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dataType(DataType_DT_INT32);
        ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int32_t>>(data.o));
        auto input = ib.Finish();

        auto name = fbb.CreateString("perm");
        auto iv   = fbb.CreateVector(std::vector<int>({}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Const);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Blob);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
    }
    {
        auto tb = TransposeBuilder(fbb);
        tb.add_Tperm(DataType_DT_INT32);
        auto transpose = tb.Finish();
        auto name      = fbb.CreateString("transpose");
        auto iv        = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov        = fbb.CreateVector(std::vector<int>({2}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Transpose);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Transpose);
        builder.add_main(flatbuffers::Offset<void>(transpose.o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder fb(fbb);
    fb.add_dataType(DataType_DT_FLOAT);
    fb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto flt = fb.Finish();
    BlobBuilder qb(fbb);
    qb.add_dataType(DataType_DT_INT32);
    qb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto i32 = qb.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(1);
        tdb.add_blob(flatbuffers::Offset<Blob>(i32.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(2);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVectorOfStrings({"input", "perm", "output"});
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

class TransposeTest : public MNNTestCase {
public:
    virtual ~TransposeTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int h = 1; h <= 8; h *= 2) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int c = 1; c <= 8; c *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            std::vector<int> idx = {0, 1, 2, 3};
                            std::vector<int> perms;
                            int xxx    = rand() % 24;
                            auto index = 0;
                            index      = xxx / 6;
                            perms.push_back(idx[index]);
                            idx.erase(idx.begin() + index); // 0 ~ 3
                            index = (xxx % 6) / 2;
                            perms.push_back(idx[index]);
                            idx.erase(idx.begin() + index); // 0 ~ 2
                            index = xxx % 2;
                            perms.push_back(idx[index]);
                            idx.erase(idx.begin() + index); // 0 ~ 1
                            perms.push_back(idx[0]);

                            // nets
                            auto net = create(perms, b, c, h, w);
                            auto CPU = createSession(net, MNN_FORWARD_CPU);
                            auto GPU = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input
                            auto input = new Tensor(4, Tensor::TENSORFLOW);
                            {
                                input->setType(DataType_DT_FLOAT);
                                input->buffer().dim[0].extent = b;
                                input->buffer().dim[1].extent = h;
                                input->buffer().dim[2].extent = w;
                                input->buffer().dim[3].extent = c;
                                TensorUtils::setLinearLayout(input);
                                input->buffer().host = (uint8_t *)malloc(input->size());
                                for (int i = 0; i < b * c * h * w; i++) {
                                    input->host<float>()[i] = rand() % 255 / 255.f;
                                }
                                auto host   = net->getSessionInput(CPU, "input");
                                auto device = net->getSessionInput(GPU, "input");
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
        return true;
    }
};
MNNTestSuiteRegister(TransposeTest, "op/transpose");
