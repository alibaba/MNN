//
//  RangeTest.cpp
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

static Interpreter *create(DataType type) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    if (type == DataType_DT_INT32) {
        std::vector<int> start = {8};
        std::vector<int> limit = {128};
        std::vector<int> delta = {3};

        { // start
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(start);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int32_t>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input0");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({0}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_Const);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Blob);
            builder.add_main(flatbuffers::Offset<void>(input.o));
            vec.push_back(builder.Finish());
        }
        { // limit
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(limit);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int32_t>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input1");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({1}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_Const);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Blob);
            builder.add_main(flatbuffers::Offset<void>(input.o));
            vec.push_back(builder.Finish());
        }
        { // delta
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(delta);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int32_t>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input2");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({2}));

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
            auto rb = RangeBuilder(fbb);
            rb.add_Tidx(type);
            auto range = rb.Finish();
            auto name  = fbb.CreateString("range");
            auto iv    = fbb.CreateVector(std::vector<int>({0, 1, 2}));
            auto ov    = fbb.CreateVector(std::vector<int>({3}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_Range);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Range);
            builder.add_main(flatbuffers::Offset<void>(range.o));
            vec.push_back(builder.Finish());
        }
    } else if (type == DataType_DT_FLOAT) {
        std::vector<float> start = {8.f};
        std::vector<float> limit = {128.f};
        std::vector<float> delta = {3.f};

        { // start
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(start);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_float32s(flatbuffers::Offset<flatbuffers::Vector<float>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input0");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({0}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_Const);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Blob);
            builder.add_main(flatbuffers::Offset<void>(input.o));
            vec.push_back(builder.Finish());
        }
        { // limit
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(limit);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_float32s(flatbuffers::Offset<flatbuffers::Vector<float>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input1");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({1}));
            OpBuilder builder(fbb);
            builder.add_type(OpType_Const);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Blob);
            builder.add_main(flatbuffers::Offset<void>(input.o));
            vec.push_back(builder.Finish());
        }
        { // delta
            auto dims = fbb.CreateVector(std::vector<int>({1}));
            auto data = fbb.CreateVector(delta);
            BlobBuilder ib(fbb);
            ib.add_dims(dims);
            ib.add_dataType(type);
            ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
            ib.add_float32s(flatbuffers::Offset<flatbuffers::Vector<float>>(data.o));
            auto input = ib.Finish();
            auto name  = fbb.CreateString("input2");
            auto iv    = fbb.CreateVector(std::vector<int>({}));
            auto ov    = fbb.CreateVector(std::vector<int>({2}));
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
            auto rb = RangeBuilder(fbb);
            rb.add_Tidx(type);
            auto range = rb.Finish();
            auto name  = fbb.CreateString("range");
            auto iv    = fbb.CreateVector(std::vector<int>({0, 1, 2}));
            auto ov    = fbb.CreateVector(std::vector<int>({3}));
            OpBuilder builder(fbb);
            builder.add_type(OpType_Range);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_Range);
            builder.add_main(flatbuffers::Offset<void>(range.o));
            vec.push_back(builder.Finish());
        }
    }

    BlobBuilder builder(fbb);
    builder.add_dataType(type);
    builder.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto blob = builder.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    for (int i = 0; i <= 3; i++) {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        tdb.add_index(i);
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

class RangeTest : public MNNTestCase {
public:
    virtual ~RangeTest() = default;
    virtual bool run() {
        DataType types[] = {
            DataType_DT_INT32, DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = (DataType)types[t];

            dispatch([&](MNNForwardType backend) -> void {
                if (backend == MNN_FORWARD_CPU)
                    return;
                // nets
                auto net = create(type);
                auto CPU = createSession(net, MNN_FORWARD_CPU);
                auto GPU = createSession(net, backend);
                if (!CPU || !GPU) {
                    delete net;
                    return;
                }

                // infer
                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                // clean up
                delete net;
            });
        }
        return true;
    }
};
MNNTestSuiteRegister(RangeTest, "op/range");
