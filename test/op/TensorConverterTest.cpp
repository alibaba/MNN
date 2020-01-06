//
//  TensorConverterTest.cpp
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

static Interpreter *create(std::vector<int> inputs, std::vector<int> outputs, bool tensorflow, DataType type) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(inputs);
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

    if (!tensorflow) {
        auto tcib = TensorConvertInfoBuilder(fbb);
        tcib.add_source(MNN_DATA_FORMAT_NC4HW4);
        tcib.add_dest(MNN_DATA_FORMAT_NHWC);
        auto cvt = tcib.Finish();

        {
            auto name = fbb.CreateString("cvt");
            auto iv   = fbb.CreateVector(std::vector<int>({0}));
            auto ov   = fbb.CreateVector(std::vector<int>({1}));
            OpBuilder builder(fbb);
            builder.add_type(OpType_ConvertTensor);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_TensorConvertInfo);
            builder.add_main(flatbuffers::Offset<void>(cvt.o));
            vec.push_back(builder.Finish());
        }

        auto ops   = fbb.CreateVector(vec);
        auto names = fbb.CreateVectorOfStrings({"input", "output"});
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        fbb.Finish(net.Finish());
    } else if (type == DataType_DT_FLOAT) {
        auto tcib = TensorConvertInfoBuilder(fbb);
        tcib.add_source(MNN_DATA_FORMAT_NHWC);
        tcib.add_dest(MNN_DATA_FORMAT_NC4HW4);
        auto cvt = tcib.Finish();

        {
            auto name = fbb.CreateString("cvt");
            auto iv   = fbb.CreateVector(std::vector<int>({0}));
            auto ov   = fbb.CreateVector(std::vector<int>({1}));
            OpBuilder builder(fbb);
            builder.add_type(OpType_ConvertTensor);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_TensorConvertInfo);
            builder.add_main(flatbuffers::Offset<void>(cvt.o));
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
        auto names  = fbb.CreateVectorOfStrings({"input", "output"});
        auto extras = fbb.CreateVector(desc);
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        net.add_extraTensorDescribe(extras);
        net.add_sourceType(NetSource_TENSORFLOW);
        fbb.Finish(net.Finish());
    } else {
        {
            auto tcib = TensorConvertInfoBuilder(fbb);
            tcib.add_source(MNN_DATA_FORMAT_NHWC);
            tcib.add_dest(MNN_DATA_FORMAT_NC4HW4);
            auto cvt  = tcib.Finish();
            auto name = fbb.CreateString("cvt");
            auto iv   = fbb.CreateVector(std::vector<int>({0}));
            auto ov   = fbb.CreateVector(std::vector<int>({1}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_ConvertTensor);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_TensorConvertInfo);
            builder.add_main(flatbuffers::Offset<void>(cvt.o));
            vec.push_back(builder.Finish());
        }
        {
            auto cpb = CastParamBuilder(fbb);
            cpb.add_srcT(type);
            cpb.add_dstT(DataType_DT_FLOAT);
            auto cast = cpb.Finish();
            auto name = fbb.CreateString("cast");
            auto iv   = fbb.CreateVector(std::vector<int>({1}));
            auto ov   = fbb.CreateVector(std::vector<int>({2}));

            OpBuilder builder(fbb);
            builder.add_type(OpType_Cast);
            builder.add_name(name);
            builder.add_inputIndexes(iv);
            builder.add_outputIndexes(ov);
            builder.add_main_type(OpParameter_CastParam);
            builder.add_main(flatbuffers::Offset<void>(cast.o));
            vec.push_back(builder.Finish());
        }

        BlobBuilder bb(fbb);
        bb.add_dataType(type);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();
        BlobBuilder fb(fbb);
        fb.add_dataType(DataType_DT_FLOAT);
        fb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto flt = fb.Finish();

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
            tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
            desc.push_back(tdb.Finish());
        }

        auto ops    = fbb.CreateVector(vec);
        auto extras = fbb.CreateVector(desc);
        auto names  = fbb.CreateVectorOfStrings({"input", "output", "cast"});
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(names);
        net.add_extraTensorDescribe(extras);
        net.add_sourceType(NetSource_TENSORFLOW);
        fbb.Finish(net.Finish());
    }
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class TensorConverterCaffeTest : public MNNTestCase {
public:
    virtual ~TensorConverterCaffeTest() = default;
    virtual bool run() {
        for (int i = 0; i < 24; i++) {
            dispatch([&](MNNForwardType backend) -> void {
                if (backend == MNN_FORWARD_CPU)
                    return;
                int b = 3, c = 5, h = 7, w = 9;
                std::vector<int> inputs = {b, c, h, w};
                std::vector<int> rest   = inputs;
                std::vector<int> outputs;

                auto index = 0;
                index      = i / 6;
                outputs.push_back(rest[index]);
                rest.erase(rest.begin() + index); // 0 ~ 3
                index = (i % 6) / 2;
                outputs.push_back(rest[index]);
                rest.erase(rest.begin() + index); // 0 ~ 2
                index = i % 2;
                outputs.push_back(rest[index]);
                rest.erase(rest.begin() + index); // 0 ~ 1
                outputs.push_back(rest[0]);

                // nets
                auto net = create(inputs, outputs, false, DataType_DT_FLOAT);
                auto CPU = createSession(net, MNN_FORWARD_CPU);
                auto GPU = createSession(net, backend);
                if (!CPU || !GPU) {
                    delete net;
                    return;
                }

                // input/output
                auto input = new Tensor(4);
                {
                    input->buffer().dim[0].extent = b;
                    input->buffer().dim[1].extent = c;
                    input->buffer().dim[2].extent = h;
                    input->buffer().dim[3].extent = w;
                    TensorUtils::setLinearLayout(input);
                    input->buffer().host = (uint8_t *)malloc(input->size());
                    for (int j = 0; j < b * c * h * w; j++) {
                        input->host<float>()[j] = j + 1;
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
        return true;
    }
};

class TensorConverterTensorflowTest : public MNNTestCase {
public:
    virtual ~TensorConverterTensorflowTest() = default;
    virtual bool run() {
        DataType types[] = {DataType_DT_FLOAT, DataType_DT_UINT8};
        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = types[t];
            for (int i = 0; i < 24; i++) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    int b = 1, c = 16, h = 5, w = 4;
                    std::vector<int> inputs = {b, h, w, c};
                    std::vector<int> rest   = inputs;
                    std::vector<int> outputs;

                    auto index = 0;
                    index      = i / 6;
                    outputs.push_back(rest[index]);
                    rest.erase(rest.begin() + index); // 0 ~ 3
                    index = (i % 6) / 2;
                    outputs.push_back(rest[index]);
                    rest.erase(rest.begin() + index); // 0 ~ 2
                    index = i % 2;
                    outputs.push_back(rest[index]);
                    rest.erase(rest.begin() + index); // 0 ~ 1
                    outputs.push_back(rest[0]);

                    // nets
                    auto net = create(inputs, outputs, true, type);
                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                    auto GPU = createSession(net, backend);
                    if (!CPU || !GPU) {
                        delete net;
                        return;
                    }

                    // input/output
                    auto input = new Tensor(4, Tensor::TENSORFLOW);
                    {
                        input->buffer().dim[0].extent = b;
                        input->buffer().dim[1].extent = h;
                        input->buffer().dim[2].extent = w;
                        input->buffer().dim[3].extent = c;
                        input->setType(type);
                        TensorUtils::setLinearLayout(input);
                        input->buffer().host = (uint8_t *)malloc(input->size());
                        if (type == DataType_DT_FLOAT) {
                            for (int j = 0; j < b * c * h * w; j++) {
                                input->host<float>()[j] = j + 1;
                            }
                        } else {
                            for (int j = 0; j < b * c * h * w; j++) {
                                input->host<uint8_t>()[j] = j + 1;
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
        return true;
    }
};
MNNTestSuiteRegister(TensorConverterCaffeTest, "op/tensor_converter/caffe");
MNNTestSuiteRegister(TensorConverterTensorflowTest, "op/tensor_converter/tensorflow");
