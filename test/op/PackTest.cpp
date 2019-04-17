//
//  PackTest.cpp
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

static Interpreter *create(DataType type, int axis, int n, std::vector<int> shape) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    std::vector<flatbuffers::Offset<flatbuffers::String>> ns;
    for (int i = 0; i < n; i++) {
        auto dims = fbb.CreateVector(shape);
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(type);
        auto input = ib.Finish();
        auto name  = fbb.CreateString(std::to_string(i));
        auto iv    = fbb.CreateVector(std::vector<int>({i}));
        auto ov    = fbb.CreateVector(std::vector<int>({i}));

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
        auto ppb = PackParamBuilder(fbb);
        ppb.add_dataType(type);
        ppb.add_axis(axis);
        auto pack = ppb.Finish();
        auto name = fbb.CreateString("pack");
        std::vector<int> ips;
        for (int i = 0; i < n; i++) {
            ips.push_back(i);
        }
        auto iv = fbb.CreateVector(ips);
        auto ov = fbb.CreateVector(std::vector<int>({n}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Pack);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_PackParam);
        builder.add_main(flatbuffers::Offset<void>(pack.o));
        vec.push_back(builder.Finish());
        ns.push_back(name);
    }

    BlobBuilder builder(fbb);
    builder.add_dataType(type);
    builder.add_dataFormat(MNN_DATA_FORMAT_NHWC);
    auto blob = builder.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    for (int i = 0; i < n; i++) {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(i);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(n);
        tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVector(ns);
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

class PackTensorTest : public MNNTestCase {
public:
    virtual ~PackTensorTest() = default;
    virtual void run() {
        DataType types[] = {
            DataType_DT_INT32, DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = types[t];
            for (int axis = 0; axis <= 3; axis++) {
                for (int n = 2; n <= 4; n++) {
                    for (int b = 1; b <= 2; b++) {
                        for (int c = 1; c <= 8; c *= 2) {
                            for (int h = 1; h <= 8; h *= 2) {
                                for (int w = 1; w <= 8; w *= 2) {
                                    dispatch([&](MNNForwardType backend) -> void {
                                        if (backend == MNN_FORWARD_CPU)
                                            return;
                                        // nets
                                        auto net = create(type, axis, n, {b, h, w, c});
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
                                                input->setType(type);
                                                input->buffer().dim[0].extent = b;
                                                input->buffer().dim[1].extent = h;
                                                input->buffer().dim[2].extent = w;
                                                input->buffer().dim[3].extent = c;
                                                TensorUtils::setLinearLayout(input);
                                                input->buffer().host = (uint8_t *)malloc(input->size());
                                                if (type == DataType_DT_INT32) {
                                                    for (int j = 0; j < b * c * h * w; j++) {
                                                        input->host<int>()[j] = rand() % 255;
                                                    }
                                                } else if (type == DataType_DT_FLOAT) {
                                                    for (int j = 0; j < b * c * h * w; j++) {
                                                        input->host<float>()[j] = rand() % 255 / 255.f;
                                                    }
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
        }
    }
};

class PackScalarTest : public MNNTestCase {
public:
    virtual ~PackScalarTest() = default;
    virtual void run() {
        DataType types[] = {
            DataType_DT_INT32, DataType_DT_FLOAT,
        };

        for (int t = 0; t < sizeof(types) / sizeof(DataType); t++) {
            DataType type = types[t];
            for (int n = 2; n <= 4; n++) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    // nets
                    auto net = create(type, 0, n, {});
                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                    auto GPU = createSession(net, backend);
                    if (!CPU || !GPU) {
                        delete net;
                        return;
                    }

                    // input
                    for (int i = 0; i < n; i++) {
                        auto input = new Tensor(1, Tensor::TENSORFLOW);
                        {
                            input->setType(type);
                            input->buffer().dim[0].extent = 1;
                            TensorUtils::setLinearLayout(input);
                            input->buffer().dimensions = 0;
                            input->buffer().host       = (uint8_t *)malloc(input->size());
                            if (type == DataType_DT_INT32) {
                                input->host<int>()[0] = rand() % 255;
                            } else if (type == DataType_DT_FLOAT) {
                                input->host<float>()[0] = rand() % 255 / 255.f;
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
};
MNNTestSuiteRegister(PackTensorTest, "op/pack/tensor");
MNNTestSuiteRegister(PackScalarTest, "op/pack/scalar");
