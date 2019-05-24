//
//  QuantizedSoftmaxTest.cpp
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

static Interpreter *create(std::vector<int> inputs, float beta, float scale) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(inputs);
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_UINT8);
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
        auto qsb = QuantizedSoftmaxBuilder(fbb);
        qsb.add_beta(beta);
        qsb.add_inputScale(scale);
        auto softmax = qsb.Finish();
        auto name    = fbb.CreateString("softmax");
        auto iv      = fbb.CreateVector(std::vector<int>({0}));
        auto ov      = fbb.CreateVector(std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_QuantizedSoftmax);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_QuantizedSoftmax);
        builder.add_main(flatbuffers::Offset<void>(softmax.o));
        vec.push_back(builder.Finish());
    }

    BlobBuilder builder(fbb);
    builder.add_dataType(DataType_DT_UINT8);
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
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class QuantizedSoftmax2DimsTest : public MNNTestCase {
public:
    virtual ~QuantizedSoftmax2DimsTest() = default;
    virtual bool run() {
        for (int c = 1; c <= 8; c *= 2) {
            for (int w = 1; w <= 8; w *= 2) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    std::vector<int> inputs = {w, c};
                    float beta              = rand() % 255 / 255.f;
                    float scale             = (rand() + 1) % 255 / 255.f;

                    // nets
                    auto net = create(inputs, beta, scale);
                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                    auto GPU = createSession(net, backend);
                    if (!CPU || !GPU) {
                        delete net;
                        return;
                    }

                    // input/output
                    auto input = new Tensor(2, Tensor::TENSORFLOW);
                    {
                        input->setType(DataType_DT_UINT8);
                        input->buffer().dim[0].extent = w;
                        input->buffer().dim[1].extent = c;
                        TensorUtils::setLinearLayout(input);
                        input->buffer().host = (uint8_t *)malloc(input->size());
                        for (int i = 0; i < c * w; i++) {
                            input->host<uint8_t>()[i] = i + 1;
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

class QuantizedSoftmax4DimsTest : public MNNTestCase {
public:
    virtual ~QuantizedSoftmax4DimsTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            std::vector<int> inputs = {b, h, w, c};
                            float beta              = (rand() % 254 + 1) / 255.f;
                            float scale             = (rand() % 254 + 1) / 255.f;

                            // nets
                            auto net = create(inputs, beta, scale);
                            auto CPU = createSession(net, MNN_FORWARD_CPU);
                            auto GPU = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input/output
                            auto input = new Tensor(4, Tensor::TENSORFLOW);
                            {
                                input->setType(DataType_DT_UINT8);
                                input->buffer().dim[0].extent = b;
                                input->buffer().dim[1].extent = h;
                                input->buffer().dim[2].extent = w;
                                input->buffer().dim[3].extent = c;
                                TensorUtils::setLinearLayout(input);
                                input->buffer().host = (uint8_t *)malloc(input->size());
                                for (int i = 0; i < b * c * h * w; i++) {
                                    input->host<uint8_t>()[i] = i + 1;
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
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(QuantizedSoftmax2DimsTest, "op/quantized_softmax/2dims");
MNNTestSuiteRegister(QuantizedSoftmax4DimsTest, "op/quantized_softmax/4dims");
