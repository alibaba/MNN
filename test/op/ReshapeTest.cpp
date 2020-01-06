//
//  ReshapeTest.cpp
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

static Interpreter *create(MNN_DATA_FORMAT fmt, std::vector<int> inputs, std::vector<int> outputs, bool dynamic,
                           bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(inputs);
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
        ib.add_dformat(tensorflow ? MNN_DATA_FORMAT_NHWC : MNN_DATA_FORMAT_NC4HW4);
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
    if (dynamic) {
        auto dims = fbb.CreateVector(std::vector<int>({(int)outputs.size()}));
        auto data = fbb.CreateVector(outputs);
        BlobBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dataType(DataType_DT_INT32);
        ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        ib.add_int32s(flatbuffers::Offset<flatbuffers::Vector<int32_t>>(data.o));
        auto input = ib.Finish();
        auto name  = fbb.CreateString("shape");
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
        auto dims = fbb.CreateVector(outputs);
        auto rb   = ReshapeBuilder(fbb);
        rb.add_dims(dims);
        rb.add_dimType(fmt);
        auto reshape = rb.Finish();
        auto name    = fbb.CreateString("reshape");
        auto iv      = fbb.CreateVector(dynamic ? std::vector<int>({0, 1}) : std::vector<int>({0}));
        auto ov      = fbb.CreateVector(dynamic ? std::vector<int>({2}) : std::vector<int>({1}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Reshape);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Reshape);
        builder.add_main(flatbuffers::Offset<void>(reshape.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = dynamic ? fbb.CreateVectorOfStrings({"input", "shape", "output"})
                         : fbb.CreateVectorOfStrings({"input", "output"});
    if (tensorflow) {
        BlobBuilder fb(fbb);
        fb.add_dataType(DataType_DT_FLOAT);
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
        if (dynamic) {
            TensorDescribeBuilder tdb1(fbb);
            tdb1.add_index(1);
            tdb1.add_blob(flatbuffers::Offset<Blob>(qnt.o));
            desc.push_back(tdb1.Finish());

            TensorDescribeBuilder tdb2(fbb);
            tdb2.add_index(2);
            tdb2.add_blob(flatbuffers::Offset<Blob>(flt.o));
            desc.push_back(tdb2.Finish());
        } else {
            TensorDescribeBuilder tdb(fbb);
            tdb.add_index(1);
            tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
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

class ReshapeCaffe4Test : public MNNTestCase {
public:
    virtual ~ReshapeCaffe4Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
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
                        auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, false);
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
                                input->host<float>()[j] = rand() % 255 / 255.f;
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
        return true;
    }
};

class ReshapeTensorflow4Test : public MNNTestCase {
public:
    virtual ~ReshapeTensorflow4Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
                for (int i = 0; i < 24; i++) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        int b = 3, c = 5, h = 7, w = 9;
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
                        auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, true);
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
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int j = 0; j < b * c * h * w; j++) {
                                input->host<float>()[j] = rand() % 255 / 255.f;
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
        return true;
    }
};

class ReshapeCaffe3Test : public MNNTestCase {
public:
    virtual ~ReshapeCaffe3Test() = default;
    virtual bool run() {
        int f = MNN_DATA_FORMAT_NCHW;
        for (int d = 0; d <= 1; d++) {
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
                    outputs.push_back(rest[1] * rest[0]);

                    // nets
                    auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, false);
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
                            input->host<float>()[j] = rand() % 255 / 255.f;
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

class ReshapeTensorflow3Test : public MNNTestCase {
public:
    virtual ~ReshapeTensorflow3Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
                for (int i = 0; i < 24; i++) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        int b = 3, c = 5, h = 7, w = 9;
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
                        outputs.push_back(rest[1] * rest[0]);

                        // nets
                        auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, true);
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
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int j = 0; j < b * c * h * w; j++) {
                                input->host<float>()[j] = rand() % 255 / 255.f;
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
        return true;
    }
};

class ReshapeCaffe2Test : public MNNTestCase {
public:
    virtual ~ReshapeCaffe2Test() = default;
    virtual bool run() {
        int f = MNN_DATA_FORMAT_NCHW;
        for (int d = 0; d <= 1; d++) {
            for (int i = 0; i < 24; i++) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    int b = 3, c = 5, h = 7, w = 9;
                    std::vector<int> inputs = {b, c, h, w};
                    std::vector<int> rest   = inputs;
                    std::vector<int> outputs;

                    auto index0 = i / 6, index1 = (i % 6) / 2;
                    int tmp = rest[index0];
                    rest.erase(rest.begin() + index0);
                    outputs.push_back(tmp * rest[index1]);
                    rest.erase(rest.begin() + index1);
                    outputs.push_back(rest[1] * rest[0]);

                    // nets
                    auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, false);
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
                            input->host<float>()[j] = rand() % 255 / 255.f;
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

class ReshapeTensorflow2Test : public MNNTestCase {
public:
    virtual ~ReshapeTensorflow2Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
                for (int i = 0; i < 24; i++) {
                    dispatch([&](MNNForwardType backend) -> void {
                        if (backend == MNN_FORWARD_CPU)
                            return;
                        int b = 3, c = 5, h = 7, w = 9;
                        std::vector<int> inputs = {b, h, w, c};
                        std::vector<int> rest   = inputs;
                        std::vector<int> outputs;

                        auto index0 = i / 6, index1 = (i % 6) / 2;
                        int tmp = rest[index0];
                        rest.erase(rest.begin() + index0);
                        outputs.push_back(tmp * rest[index1]);
                        rest.erase(rest.begin() + index1);
                        outputs.push_back(rest[1] * rest[0]);

                        // nets
                        auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, true);
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
                            TensorUtils::setLinearLayout(input);
                            input->buffer().host = (uint8_t *)malloc(input->size());
                            for (int j = 0; j < b * c * h * w; j++) {
                                input->host<float>()[j] = rand() % 255 / 255.f;
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
        return true;
    }
};

class ReshapeCaffe1Test : public MNNTestCase {
public:
    virtual ~ReshapeCaffe1Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    int b = 3, c = 5, h = 7, w = 9;
                    std::vector<int> inputs  = {b, c, h, w};
                    std::vector<int> outputs = {b * c * h * w};

                    // nets
                    auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, false);
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
                        for (int i = 0; i < b * c * h * w; i++) {
                            input->host<float>()[i] = rand() % 255 / 255.f;
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

class ReshapeTensorflow1Test : public MNNTestCase {
public:
    virtual ~ReshapeTensorflow1Test() = default;
    virtual bool run() {
        for (int f = MNN_DATA_FORMAT_NCHW; f <= MNN_DATA_FORMAT_NHWC; f++) {
            for (int d = 0; d <= 1; d++) {
                dispatch([&](MNNForwardType backend) -> void {
                    if (backend == MNN_FORWARD_CPU)
                        return;
                    int b = 3, c = 5, h = 7, w = 9;
                    std::vector<int> inputs  = {b, h, w, c};
                    std::vector<int> outputs = {b * c * h * w};

                    // nets
                    auto net = create((MNN_DATA_FORMAT)f, inputs, outputs, d, true);
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
                        TensorUtils::setLinearLayout(input);
                        input->buffer().host = (uint8_t *)malloc(input->size());
                        for (int i = 0; i < b * c * h * w; i++) {
                            input->host<float>()[i] = rand() % 255 / 255.f;
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
MNNTestSuiteRegister(ReshapeCaffe4Test, "op/reshape/caffe_4");
MNNTestSuiteRegister(ReshapeTensorflow4Test, "op/reshape/tensorflow_4");
MNNTestSuiteRegister(ReshapeCaffe3Test, "op/reshape/caffe_3");
MNNTestSuiteRegister(ReshapeTensorflow3Test, "op/reshape/tensorflow_3");
MNNTestSuiteRegister(ReshapeCaffe2Test, "op/reshape/caffe_2");
MNNTestSuiteRegister(ReshapeTensorflow2Test, "op/reshape/tensorflow_2");
MNNTestSuiteRegister(ReshapeCaffe1Test, "op/reshape/caffe_1");
MNNTestSuiteRegister(ReshapeTensorflow1Test, "op/reshape/tensorflow_1");
