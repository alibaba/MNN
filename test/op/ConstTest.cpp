//
//  ConstTest.cpp
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

static Interpreter *create(int b, int c, int h, int w, std::vector<float> &data, bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims   = fbb.CreateVector(std::vector<int>({b, c, h, w}));
        auto floats = fbb.CreateVector(data);
        BlobBuilder bb(fbb);
        bb.add_dims(dims);
        bb.add_dataType(DataType_DT_FLOAT);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        bb.add_float32s(floats);
        auto cop = bb.Finish();

        auto name = fbb.CreateString("constop");
        auto iv   = fbb.CreateVector(std::vector<int>({}));
        auto ov   = fbb.CreateVector(std::vector<int>({0}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Const);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Blob);
        builder.add_main(flatbuffers::Offset<void>(cop.o));
        vec.push_back(builder.Finish());
    }

    auto names = fbb.CreateVectorOfStrings({"output"});
    auto ops   = fbb.CreateVector(vec);
    if (tensorflow) {
        BlobBuilder bb(fbb);
        bb.add_dataType(DataType_DT_FLOAT);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();

        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        auto extras = fbb.CreateVector(std::vector<flatbuffers::Offset<TensorDescribe>>({tdb.Finish()}));

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

class ConstCaffeTest : public MNNTestCase {
public:
    virtual ~ConstCaffeTest() = default;
    virtual void run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            std::vector<float> data;
                            for (int i = 0; i < b * c * h * w; i++)
                                data.push_back(rand() % 255 / 255.f);

                            // nets
                            auto net = create(b, c, h, w, data, true);
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
                }
            }
        }
    }
};

class ConstTensorflowTest : public MNNTestCase {
public:
    virtual ~ConstTensorflowTest() = default;
    virtual void run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            std::vector<float> data;
                            for (int i = 0; i < b * c * h * w; i++)
                                data.push_back(rand() % 255 / 255.f);

                            // nets
                            auto net = create(b, c, h, w, data, true);
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
                }
            }
        }
    }
};
MNNTestSuiteRegister(ConstCaffeTest, "op/const/caffe");
MNNTestSuiteRegister(ConstTensorflowTest, "op/const/tf");
