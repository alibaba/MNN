//
//  FillTest.cpp
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

#define kDim 4

static Interpreter *create(std::vector<int> dims, int scalar, bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto val   = fbb.CreateVector(dims);
        auto idims = fbb.CreateVector(std::vector<int>({(int)dims.size()}));
        BlobBuilder ib(fbb);
        ib.add_dims(idims);
        ib.add_dataType(DataType_DT_INT32);
        ib.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int>>(val.o));
        auto input = ib.Finish();
        auto name  = fbb.CreateString("input");
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
    {
        auto sdims = fbb.CreateVector(std::vector<int>({}));
        auto val   = fbb.CreateVector(std::vector<int>({scalar}));
        BlobBuilder bb(fbb);
        bb.add_dims(sdims);
        bb.add_dataType(DataType_DT_INT32);
        bb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        bb.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int>>(val.o));
        auto blob = bb.Finish();

        auto name = fbb.CreateString("scalar");
        auto iv   = fbb.CreateVector(std::vector<int>({}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Const);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Blob);
        builder.add_main(flatbuffers::Offset<void>(blob.o));
        vec.push_back(builder.Finish());
    }
    {
        auto name = fbb.CreateString("fill");
        auto iv   = fbb.CreateVector(std::vector<int>({0, 1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Fill);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Fill);
        builder.add_main(flatbuffers::Offset<void>(FillBuilder(fbb).Finish().o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "scalar", "output"});

    if (tensorflow) {
        BlobBuilder bb(fbb);
        bb.add_dataType(DataType_DT_INT32);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();

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
            tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
            tdb.add_index(2);
            desc.push_back(tdb.Finish());
        }
        auto extras = fbb.CreateVector(desc);
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        net.add_extraTensorDescribe(extras);
        net.add_sourceType(NetSource_TENSORFLOW);
        fbb.Finish(net.Finish());
    } else {
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        fbb.Finish(net.Finish());
    }
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class FillCaffeTest : public MNNTestCase {
public:
    virtual ~FillCaffeTest() = default;
    virtual bool run() {
        for (int i = 1; i <= 10; i++) {
            std::vector<int> dims;
            for (int j = 0; j < kDim; j++)
                dims.push_back(1 + rand() % 16);

            dispatch([&](MNNForwardType backend) -> void {
                if (backend == MNN_FORWARD_CPU)
                    return;
                // nets
                auto scalar = rand() % 255;
                auto net    = create(dims, scalar, false);
                auto CPU    = createSession(net, MNN_FORWARD_CPU);
                auto GPU    = createSession(net, backend);
                if (!CPU || !GPU) {
                    delete net;
                    return;
                }

                // infer
                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU)));

                // clean up
                delete net;
            });
        }
        return true;
    }
};

class FillTensorflowTest : public MNNTestCase {
public:
    virtual ~FillTensorflowTest() = default;
    virtual bool run() {
        for (int i = 1; i <= 10; i++) {
            std::vector<int> dims;
            for (int j = 0; j < kDim; j++)
                dims.push_back(1 + rand() % 16);

            dispatch([&](MNNForwardType backend) -> void {
                if (backend == MNN_FORWARD_CPU)
                    return;
                // nets
                auto scalar = rand() % 255;
                auto net    = create(dims, scalar, true);
                auto CPU    = createSession(net, MNN_FORWARD_CPU);
                auto GPU    = createSession(net, backend);
                if (!CPU || !GPU) {
                    delete net;
                    return;
                }

                // infer
                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU)));

                // clean up
                delete net;
            });
        }
        return true;
    }
};

MNNTestSuiteRegister(FillCaffeTest, "op/fill/caffe");
MNNTestSuiteRegister(FillTensorflowTest, "op/fill/tf");
