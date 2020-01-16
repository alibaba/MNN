//
//  QuantizedMaxPoolTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Interpreter.hpp>
#include "MNNTestSuite.h"
#include "core/Session.hpp"
#include "TFQuantizeOp_generated.h"
#include "core/TensorUtils.hpp"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(PoolPadType type, int kernel, int stride, int pad, int min, int max, int w, int h, int c,
                           int b) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_UINT8);
        ib.add_dformat(MNN_DATA_FORMAT_NHWC);
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
        auto pb = QuantizedMaxPoolBuilder(fbb);
        pb.add_type(DataType_DT_UINT8);
        pb.add_kernelX(kernel);
        pb.add_kernelY(kernel);
        pb.add_strideX(stride);
        pb.add_strideY(stride);
        pb.add_padX(pad);
        pb.add_padY(pad);
        pb.add_padType(type);
        pb.add_modelFormat(ModeFormat_TFLITE);
        pb.add_outputActivationMin(min);
        pb.add_outputActivationMax(max);
        auto pool = pb.Finish();

        auto name = fbb.CreateString("pool");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_QuantizedMaxPool);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_QuantizedMaxPool);
        builder.add_main(flatbuffers::Offset<void>(pool.o));
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

class QuantizedMaxPoolTest : public MNNTestCase {
public:
    virtual ~QuantizedMaxPoolTest() = default;
    virtual bool run() {
        int types[] = {PoolPadType_VALID, PoolPadType_SAME};

        for (int t = 0; t < sizeof(types) / sizeof(int); t++) {
            PoolPadType type = (PoolPadType)types[t];
            for (int b = 1; b <= 2; b++) {
                for (int c = 1; c <= 8; c *= 2) {
                    for (int h = 1; h <= 8; h *= 2) {
                        for (int w = 1; w <= 8; w *= 2) {
                            for (int k = 1; k <= w && k <= h; k *= 2) {
                                for (int s = 1; s <= 4; s *= 2) {
                                    for (int p = 0; p <= 2; p++) {
                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create(type, k, s, p, 0, 255, w, h, c, b);
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
                                                for (int i = 0; i < w * h * c * b; i++) {
                                                    input->host<uint8_t>()[i] = rand() % 255;
                                                }
                                                auto host   = net->getSessionInput(CPU, NULL);
                                                auto device = net->getSessionInput(GPU, NULL);
                                                net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                                net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                            }

                                            // infer
                                            auto output  = infer(net, CPU);
                                            auto compare = infer(net, GPU);
                                            assert(TensorUtils::compareTensors(compare, output));

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
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(QuantizedMaxPoolTest, "op/quantized_max_pool");
