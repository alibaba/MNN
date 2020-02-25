//
//  CropAndResizeTest.cpp
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

static Interpreter *create(CropAndResizeMethod method, float e, int n, int b, int c, int h, int w, int ch, int cw) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    std::vector<flatbuffers::Offset<flatbuffers::String>> ns;
    {
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("image");
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
        ns.push_back(name);
    }
    {
        auto dims = fbb.CreateVector(std::vector<int>({n, 4}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_FLOAT);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("boxes");
        auto iv    = fbb.CreateVector(std::vector<int>({1}));
        auto ov    = fbb.CreateVector(std::vector<int>({1}));

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
        auto dims = fbb.CreateVector(std::vector<int>({n}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_INT32);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("indexes");
        auto iv    = fbb.CreateVector(std::vector<int>({2}));
        auto ov    = fbb.CreateVector(std::vector<int>({2}));

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
        auto dims = fbb.CreateVector(std::vector<int>({2}));
        auto data = fbb.CreateVector(std::vector<int>({ch, cw}));
        BlobBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dataType(DataType_DT_INT32);
        ib.add_dataFormat(MNN_DATA_FORMAT_NHWC);
        ib.add_int32s(data);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("size");
        auto iv    = fbb.CreateVector(std::vector<int>({}));
        auto ov    = fbb.CreateVector(std::vector<int>({3}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_Const);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Blob);
        builder.add_main(flatbuffers::Offset<void>(input.o));
        vec.push_back(builder.Finish());
        ns.push_back(name);
    }
    {
        auto name = fbb.CreateString("car");
        auto carb = CropAndResizeBuilder(fbb);
        carb.add_extrapolationValue(e);
        carb.add_method(method);
        auto car = carb.Finish();
        auto iv  = fbb.CreateVector(std::vector<int>({0, 1, 2, 3}));
        auto ov  = fbb.CreateVector(std::vector<int>({4}));

        OpBuilder builder(fbb);
        builder.add_type(OpType_CropAndResize);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_CropAndResize);
        builder.add_main(flatbuffers::Offset<void>(car.o));
        vec.push_back(builder.Finish());
        ns.push_back(name);
    }

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
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(1);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(2);
        tdb.add_blob(flatbuffers::Offset<Blob>(qnt.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(3);
        tdb.add_blob(flatbuffers::Offset<Blob>(qnt.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(4);
        tdb.add_blob(flatbuffers::Offset<Blob>(flt.o));
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

class CropAndResizeTest : public MNNTestCase {
public:
    virtual ~CropAndResizeTest() = default;
    virtual bool run() {
        int methods[] = {CropAndResizeMethod_BILINEAR, CropAndResizeMethod_NEAREST};

        for (int m = 0; m < sizeof(methods) / sizeof(int); m++) {
            CropAndResizeMethod method = (CropAndResizeMethod)m;
            for (int n = 1; n <= 4; n *= 2) {
                for (int b = 1; b <= 2; b++) {
                    for (int c = 1; c <= 8; c *= 2) {
                        for (int h = 1; h <= 8; h *= 2) {
                            for (int w = 1; w <= 8; w *= 2) {
                                dispatch([&](MNNForwardType backend) -> void {
                                    if (backend == MNN_FORWARD_CPU)
                                        return;
                                    float e = rand() % 255 / 255.f;
                                    int ch  = rand() % 4 + 1;
                                    int cw  = rand() % 4 + 1;

                                    // nets
                                    auto net = create(method, e, n, b, c, h, w, ch, cw);
                                    auto CPU = createSession(net, MNN_FORWARD_CPU);
                                    auto GPU = createSession(net, backend);
                                    if (!CPU || !GPU) {
                                        delete net;
                                        return;
                                    }

                                    // image
                                    auto image = new Tensor(4, Tensor::TENSORFLOW);
                                    {
                                        image->setType(DataType_DT_FLOAT);
                                        image->buffer().dim[0].extent = b;
                                        image->buffer().dim[1].extent = h;
                                        image->buffer().dim[2].extent = w;
                                        image->buffer().dim[3].extent = c;
                                        TensorUtils::setLinearLayout(image);
                                        image->buffer().host = (uint8_t *)malloc(image->size());
                                        for (int i = 0; i < b * c * h * w; i++) {
                                            image->host<float>()[i] = i + 1; // rand() % 255 / 255.f;
                                        }
                                        auto host   = net->getSessionInput(CPU, "image");
                                        auto device = net->getSessionInput(GPU, "image");
                                        net->getBackend(CPU, host)->onCopyBuffer(image, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(image, device);
                                    }
                                    // boxes
                                    auto boxes = new Tensor(2, Tensor::TENSORFLOW);
                                    {
                                        boxes->setType(DataType_DT_FLOAT);
                                        boxes->buffer().dim[0].extent = n;
                                        boxes->buffer().dim[1].extent = 4;
                                        TensorUtils::setLinearLayout(boxes);
                                        boxes->buffer().host = (uint8_t *)malloc(boxes->size());
                                        for (int i = 0; i < n; i++) {
                                            auto y1                         = rand() % 255 / 255.f * (h - ch);
                                            auto y2                         = rand() % 255 / 255.f * (h - ch);
                                            auto x1                         = rand() % 255 / 255.f * (w - cw);
                                            auto x2                         = rand() % 255 / 255.f * (w - cw);
                                            boxes->host<float>()[i * 4 + 0] = std::min(y1, y2);
                                            boxes->host<float>()[i * 4 + 1] = std::min(x1, x2);
                                            boxes->host<float>()[i * 4 + 2] = std::max(y1, y2);
                                            boxes->host<float>()[i * 4 + 3] = std::max(x1, x2);
                                        }
                                        auto host   = net->getSessionInput(CPU, "boxes");
                                        auto device = net->getSessionInput(GPU, "boxes");
                                        net->getBackend(CPU, host)->onCopyBuffer(boxes, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(boxes, device);
                                    }
                                    // indexes
                                    auto indexes = new Tensor(1, Tensor::TENSORFLOW);
                                    {
                                        indexes->setType(DataType_DT_INT32);
                                        indexes->buffer().dim[0].extent = n;
                                        TensorUtils::setLinearLayout(indexes);
                                        indexes->buffer().host = (uint8_t *)malloc(indexes->size());
                                        for (int i = 0; i < n; i++) {
                                            indexes->host<int>()[i] = rand() % n;
                                        }
                                        auto host   = net->getSessionInput(CPU, "indexes");
                                        auto device = net->getSessionInput(GPU, "indexes");
                                        net->getBackend(CPU, host)->onCopyBuffer(indexes, host);
                                        net->getBackend(GPU, device)->onCopyBuffer(indexes, device);
                                    }

                                    // infer
                                    assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                                    // clean up
                                    free(image->buffer().host);
                                    free(boxes->buffer().host);
                                    free(indexes->buffer().host);
                                    delete image;
                                    delete boxes;
                                    delete indexes;
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
MNNTestSuiteRegister(CropAndResizeTest, "op/crop_and_resize");
