//
//  SliceTest.cpp
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

static Interpreter *create(int axis, int n, int extent, int b, int c, int h, int w, bool tensorflow) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;
    std::vector<flatbuffers::Offset<flatbuffers::String>> names;

    {
        auto dims = fbb.CreateVector(tensorflow ? std::vector<int>({b, h, w, c}) : std::vector<int>({b, c, h, w}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
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
        names.push_back(name);
    }
    {
        auto step = extent / n;
        std::vector<int> steps, outputs;
        for (int i = 1; i <= n; i++) {
            if (tensorflow) {
                steps.push_back(i < n ? step : extent - (n - 1) * step);
            } else if (i < n) {
                steps.push_back(i * step);
            }
            outputs.push_back(i);
            names.push_back(fbb.CreateString(std::to_string(i).c_str()));
        }

        auto points = fbb.CreateVector(steps);
        auto sb     = SliceBuilder(fbb);
        sb.add_axis(axis);
        sb.add_slicePoints(points);
        sb.add_sourceType(tensorflow ? NetSource_TENSORFLOW : NetSource_CAFFE);
        auto slice = sb.Finish();
        auto name  = fbb.CreateString("slice");
        auto iv    = fbb.CreateVector(std::vector<int>({0}));
        auto ov    = fbb.CreateVector(outputs);

        OpBuilder builder(fbb);
        builder.add_type(OpType_Slice);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Slice);
        builder.add_main(flatbuffers::Offset<void>(slice.o));
        vec.push_back(builder.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto namesv = fbb.CreateVector(names);
    if (tensorflow) {
        BlobBuilder builder(fbb);
        builder.add_dataType(DataType_DT_FLOAT);
        builder.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        auto blob = builder.Finish();

        std::vector<flatbuffers::Offset<TensorDescribe>> desc;
        {
            TensorDescribeBuilder tdb(fbb);
            tdb.add_index(0);
            tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
            desc.push_back(tdb.Finish());
        }
        for (int i = 1; i <= n; i++) {
            TensorDescribeBuilder tdb(fbb);
            tdb.add_blob(flatbuffers::Offset<Blob>(blob.o));
            tdb.add_index(i);
            desc.push_back(tdb.Finish());
        }

        auto extras = fbb.CreateVector(desc);
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(namesv);
        net.add_extraTensorDescribe(extras);
        net.add_sourceType(NetSource_TENSORFLOW);
        fbb.Finish(net.Finish());
    } else {
        NetBuilder net(fbb);
        net.add_oplists(ops);
        net.add_tensorName(namesv);
        fbb.Finish(net.Finish());
    }
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static std::vector<Tensor *> infer(const Interpreter *net, Session *session, int n) {
    net->runSession(session);
    std::vector<Tensor *> outputs;
    auto all = net->getSessionOutputAll(session);
    for (int i = 1; i <= n; i++) {
        outputs.push_back(all[std::to_string(i)]);
    }
    return outputs;
}

class SliceCaffeChannelTest : public MNNTestCase {
public:
    virtual ~SliceCaffeChannelTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 2; c <= 8; c++) {
            for (int h = 1; h <= 8; h *= 2) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int n = 2; n <= c; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(1, n, c, b, c, h, w, false);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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

class SliceTensorflowChannelTest : public MNNTestCase {
public:
    virtual ~SliceTensorflowChannelTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 2; c <= 8; c++) {
            for (int h = 1; h <= 8; h *= 2) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int n = 2; n <= c; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(3, n, c, b, c, h, w, true);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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

class SliceCaffeHeightTest : public MNNTestCase {
public:
    virtual ~SliceCaffeHeightTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 1; c <= 8; c *= 2) {
            for (int h = 2; h <= 8; h++) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int n = 2; n <= h; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(2, n, h, b, c, h, w, false);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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

class SliceTensorflowHeightTest : public MNNTestCase {
public:
    virtual ~SliceTensorflowHeightTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 1; c <= 8; c *= 2) {
            for (int h = 2; h <= 8; h++) {
                for (int w = 1; w <= 8; w *= 2) {
                    for (int n = 2; n <= h; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(1, n, h, b, c, h, w, true);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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

class SliceCaffeWidthTest : public MNNTestCase {
public:
    virtual ~SliceCaffeWidthTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 1; c <= 8; c *= 2) {
            for (int h = 1; h <= 8; h *= 2) {
                for (int w = 2; w <= 8; w++) {
                    for (int n = 2; n <= w; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(3, n, w, b, c, h, w, false);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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

class SliceTensorflowWidthTest : public MNNTestCase {
public:
    virtual ~SliceTensorflowWidthTest() = default;
    virtual bool run() {
        int b = 1;
        for (int c = 2; c <= 8; c *= 2) {
            for (int h = 1; h <= 8; h *= 2) {
                for (int w = 2; w <= 8; w++) {
                    for (int n = 2; n <= w; n++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            // nets
                            auto net = create(2, n, w, b, c, h, w, true);
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
                            auto outputs  = infer(net, CPU, n);
                            auto compares = infer(net, GPU, n);
                            for (int i = 0; i < outputs.size(); i++) {
                                assert(TensorUtils::compareTensors(compares[i], outputs[i], 0.01));
                            }

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
MNNTestSuiteRegister(SliceCaffeChannelTest, "op/slice/caffe/channel");
MNNTestSuiteRegister(SliceCaffeHeightTest, "op/slice/caffe/height");
MNNTestSuiteRegister(SliceCaffeWidthTest, "op/slice/caffe/width");
MNNTestSuiteRegister(SliceTensorflowChannelTest, "op/slice/tensorflow/channel");
MNNTestSuiteRegister(SliceTensorflowHeightTest, "op/slice/tensorflow/height");
MNNTestSuiteRegister(SliceTensorflowWidthTest, "op/slice/tensorflow/width");
