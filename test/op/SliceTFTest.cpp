//
//  SliceTFTest.cpp
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

static Interpreter *create(DataType type, std::vector<int> begins, std::vector<int> sizes, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;
    std::vector<flatbuffers::Offset<flatbuffers::String>> names;

    {
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
        auto dims = fbb.CreateVector(std::vector<int>({4}));
        auto data = fbb.CreateVector(begins);
        BlobBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dataType(DataType_DT_INT32);
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
    {
        auto dims = fbb.CreateVector(std::vector<int>({4}));
        auto data = fbb.CreateVector(sizes);
        BlobBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dataType(DataType_DT_INT32);
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
        auto sb = SliceTfBuilder(fbb);
        sb.add_T(type);
        auto slicetf = sb.Finish();
        auto name    = fbb.CreateString("slice_tf");
        auto iv      = fbb.CreateVector(std::vector<int>({0, 1, 2}));
        auto ov      = fbb.CreateVector(std::vector<int>({3}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_SliceTf);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_SliceTf);
        builder.add_main(flatbuffers::Offset<void>(slicetf.o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder fb(fbb);
    fb.add_dataType(type);
    fb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto flt = fb.Finish();
    BlobBuilder qb(fbb);
    qb.add_dataType(DataType_DT_INT32);
    qb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto qnt = qb.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }
    for (int i = 1; i <= 2; i++) {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(qnt.o));
        tdb.add_index(i);
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(3);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto namesv = fbb.CreateVectorOfStrings({"input0", "input1", "input2", "output"});
    auto extras = fbb.CreateVector(desc);
    NetBuilder net(fbb);
    net.add_oplists(ops);
    net.add_tensorName(namesv);
    net.add_extraTensorDescribe(extras);
    net.add_sourceType(NetSource_TENSORFLOW);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class SliceTFTest : public MNNTestCase {
public:
    virtual ~SliceTFTest() = default;
    virtual bool run() {
        DataType types[] = {
            DataType_DT_INT32, DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = types[t];
            for (int b = 1; b <= 2; b++) {
                for (int c = 2; c <= 8; c *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int w = 2; w <= 8; w++) {
                            dispatch([&](MNNForwardType backend) -> void {
                                if (backend == MNN_FORWARD_CPU)
                                    return;
                                std::vector<int> dims = {b, h, w, c};
                                std::vector<int> begins, sizes;
                                for (int i = 0; i < 4; i++) {
                                    begins.push_back(dims[i] - 1 > 0 ? rand() % (dims[i] - 1) : 0);
                                    sizes.push_back(1 + rand() % (dims[i] - begins[i]));
                                }

                                // nets
                                auto net = create(type, begins, sizes, b, c, h, w);
                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                auto GPU = createSession(net, backend);
                                if (!CPU || !GPU) {
                                    delete net;
                                    return;
                                }

                                // input/output
                                auto input = new Tensor(4, Tensor::TENSORFLOW);
                                {
                                    input->setType(type);
                                    input->buffer().dim[0].extent = b;
                                    input->buffer().dim[1].extent = h;
                                    input->buffer().dim[2].extent = w;
                                    input->buffer().dim[3].extent = c;
                                    TensorUtils::setLinearLayout(input);
                                    input->buffer().host = (uint8_t *)malloc(input->size());

                                    if (type == DataType_DT_FLOAT) {
                                        for (int i = 0; i < b * c * h * w; i++) {
                                            input->host<float>()[i] = rand() % 255 / 255.f;
                                        }
                                    } else if (type == DataType_DT_INT32) {
                                        for (int i = 0; i < b * c * h * w; i++) {
                                            input->host<int>()[i] = rand() % 255;
                                        }
                                    }
                                    auto host   = net->getSessionInput(CPU, "input0");
                                    auto device = net->getSessionInput(GPU, "input0");
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
MNNTestSuiteRegister(SliceTFTest, "op/slicetf");
