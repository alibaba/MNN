//
//  ConcatTest.cpp
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

static Interpreter *create(int axis, int n, int b, int c, int h, int w, bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    std::vector<flatbuffers::Offset<flatbuffers::String>> ns;
    for (int i = 0; i < n; i++) {
        auto dims = fbb.CreateVector(tensorflow ? std::vector<int>({b, h, w, c}) : std::vector<int>({b, c, h, w}));
        auto name = fbb.CreateString(std::to_string(i));
        auto iv   = fbb.CreateVector(std::vector<int>({i}));
        auto ov   = fbb.CreateVector(std::vector<int>({i}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();

        OpBuilder builder(fbb);
        builder.add_type(OpType_Input);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Input);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
        ns.push_back(name);
    }
    {
        auto name = fbb.CreateString("concat");
        std::vector<int> ips;
        for (int i = 0; i < n; i++) {
            ips.push_back(i);
        }
        auto iv = fbb.CreateVector(ips);
        auto ov = fbb.CreateVector(std::vector<int>({n}));
        AxisBuilder ab(fbb);
        ab.add_axis(axis);
        auto concat = ab.Finish();

        OpBuilder builder(fbb);
        builder.add_type(OpType_Concat);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Axis);
        builder.add_main(flatbuffers::Offset<void>(concat.o));
        vec.push_back(builder.Finish());
        ns.push_back(name);
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVector(ns);

    if (tensorflow) {
        BlobBuilder bb(fbb);
        bb.add_dataType(DataType_DT_FLOAT);
        bb.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = bb.Finish();

        std::vector<flatbuffers::Offset<TensorDescribe>> desc;
        for (int i = 0; i < n; i++) {
            TensorDescribeBuilder tdb(fbb);
            tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
            tdb.add_index(i);
            desc.push_back(tdb.Finish());
        }
        TensorDescribeBuilder tdb(fbb);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        tdb.add_index(n);
        desc.push_back(tdb.Finish());

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

class ConcatCaffeTest : public MNNTestCase {
public:
    virtual ~ConcatCaffeTest() = default;
    virtual bool run() {
        for (int axis = 1; axis <= 3; axis++) {
            for (int n = 2; n <= 4; n++) {
                for (int b = 1; b <= 2; b++) {
                    for (int c = 1; c <= 8; c *= 2) {
                        for (int h = 1; h <= 8; h *= 2) {
                            for (int w = 1; w <= 8; w *= 2) {
                                dispatch([&](MNNForwardType backend) -> void {
                                    if (backend == MNN_FORWARD_CPU)
                                        return;
                                    // nets
                                    auto net = create(axis, n, b, c, h, w, false);
                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU = createSession(net, backend);
                                    if (!CPU || !GPU) {
                                        delete net;
                                        return;
                                    }

                                    // input
                                    for (int i = 0; i < n; i++) {
                                        auto input                    = new Tensor(4);
                                        input->buffer().dim[0].extent = b;
                                        input->buffer().dim[1].extent = c;
                                        input->buffer().dim[2].extent = h;
                                        input->buffer().dim[3].extent = w;
                                        TensorUtils::setLinearLayout(input);
                                        input->buffer().host = (uint8_t *)malloc(input->size());
                                        for (int j = 0; j < b * c * h * w; j++) {
                                            input->host<float>()[j] = rand() % 255 / 255.f;
                                        }

                                        auto host   = net->getSessionInput(CPU, std::to_string(i).c_str());
                                        auto device = net->getSessionInput(GPU, std::to_string(i).c_str());
                                        net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(input, device);

                                        // clean up
                                        free(input->buffer().host);
                                        delete input;
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
        }
        return true;
    }
};

class ConcatTensorflowTest : public MNNTestCase {
public:
    virtual ~ConcatTensorflowTest() = default;
    virtual bool run() {
        for (int axis = 1; axis <= 3; axis++) {
            for (int n = 2; n <= 4; n++) {
                for (int b = 1; b <= 2; b++) {
                    for (int c = 1; c <= 8; c *= 2) {
                        for (int h = 1; h <= 8; h *= 2) {
                            for (int w = 1; w <= 8; w *= 2) {
                                dispatch([&](MNNForwardType backend) -> void {
                                    if (backend == MNN_FORWARD_CPU)
                                        return;
                                    // nets
                                    auto net = create(axis, n, b, c, h, w, true);
                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU = createSession(net, backend);
                                    if (!CPU || !GPU) {
                                        delete net;
                                        return;
                                    }

                                    // input
                                    for (int i = 0; i < n; i++) {
                                        auto input = new Tensor(4, Tensor::TENSORFLOW);
                                        {
                                            input->buffer().dim[0].extent = b;
                                            input->buffer().dim[1].extent = h;
                                            input->buffer().dim[2].extent = w;
                                            input->buffer().dim[3].extent = c;
                                            TensorUtils::setLinearLayout(input);
                                            input->buffer().host = (uint8_t *)malloc(input->size());
                                            for (int j = 0; j < b * c * h * w; j++) {
                                                input->host<float>()[j] = rand() % 255 / 255.f;
                                            }
                                        }

                                        auto host   = net->getSessionInput(CPU, std::to_string(i).c_str());
                                        auto device = net->getSessionInput(GPU, std::to_string(i).c_str());
                                        net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(input, device);

                                        // clean up
                                        free(input->buffer().host);
                                        delete input;
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
        }
        return true;
    }
};

MNNTestSuiteRegister(ConcatCaffeTest, "op/concat/caffe");
MNNTestSuiteRegister(ConcatTensorflowTest, "op/concat/tf");
