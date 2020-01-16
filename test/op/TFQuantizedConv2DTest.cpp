//
//  TFQuantizedConv2DTest.cpp
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

static Interpreter *create(int oc, // output channel
                           int is, // input size
                           int c,  // input channel
                           int b,  // batch
                           int d,  // dilation
                           int kw, // kenrel width
                           int kh, // kenrel height
                           int s,  // stride
                           int g,  // group
                           std::vector<uint8_t> weights, std::vector<int32_t> bias, bool depthwise) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, c, is, is}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        ib.add_dtype(DataType_DT_UINT8);
        ib.add_dformat(MNN_DATA_FORMAT_NC4HW4);
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
        FusedActivation act = (FusedActivation)(rand() % FusedActivation_MAX);

        auto ccb = Convolution2DCommonBuilder(fbb);
        ccb.add_dilateX(d);
        ccb.add_dilateY(d);
        ccb.add_strideX(s);
        ccb.add_strideY(s);
        ccb.add_kernelX(kw);
        ccb.add_kernelY(kh);
        ccb.add_padX(0);
        ccb.add_padY(0);
        ccb.add_padMode(PadMode_SAME);
        ccb.add_group(g);
        ccb.add_outputCount(oc);
        ccb.add_relu(false);
        ccb.add_relu6(false);
        auto common = ccb.Finish();

        auto fqpb = QuantizedParamBuilder(fbb);
        fqpb.add_zeroPoint(0);
        fqpb.add_scale(0.3f);
        auto filter = fqpb.Finish();
        auto iqpb   = QuantizedParamBuilder(fbb);
        iqpb.add_zeroPoint(0);
        iqpb.add_scale(0.5f);
        auto input = iqpb.Finish();
        auto oqpb  = QuantizedParamBuilder(fbb);
        oqpb.add_zeroPoint(0);
        oqpb.add_scale(0.7f);
        auto output = oqpb.Finish();

        auto wts = fbb.CreateVector(weights);
        auto bs  = fbb.CreateVector(bias);
        auto cb  = TfQuantizedConv2DBuilder(fbb);
        cb.add_common(common);
        cb.add_weight(wts);
        cb.add_biasflag(true);
        cb.add_bias(bs);
        cb.add_activationType(act);
        cb.add_modelFormat(ModeFormat_TENSORFLOW);
        cb.add_filterQuantizedParam(flatbuffers::Offset<QuantizedParam>(filter.o));
        cb.add_inputQuantizedParam(flatbuffers::Offset<QuantizedParam>(input.o));
        cb.add_outputQuantizedParam(flatbuffers::Offset<QuantizedParam>(output.o));
        auto conv = cb.Finish();

        auto name = fbb.CreateString("conv");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(depthwise ? OpType_QuantizedDepthwiseConv2D : OpType_TfQuantizedConv2D);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_TfQuantizedConv2D);
        builder.add_main(flatbuffers::Offset<void>(conv.o));
        vec.push_back(builder.Finish());
    }
    {
        auto cast = CreateCastParam(fbb, DataType_DT_UINT8, DataType_DT_FLOAT);
        auto main = flatbuffers::Offset<void>(cast.o);
        auto name = fbb.CreateString("cast");
        auto iv   = fbb.CreateVector(std::vector<int>({1}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Cast);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_CastParam);
        builder.add_main(main);
        vec.push_back(builder.Finish());
    }

    BlobBuilder fb(fbb);
    fb.add_dataType(DataType_DT_FLOAT);
    fb.add_dataFormat(MNN_DATA_FORMAT_NC4HW4);
    auto f32 = fb.Finish();
    BlobBuilder qb(fbb);
    qb.add_dataType(DataType_DT_UINT8);
    qb.add_dataFormat(MNN_DATA_FORMAT_NC4HW4);
    auto u8 = qb.Finish();

    std::vector<flatbuffers::Offset<TensorDescribe>> desc;
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(0);
        tdb.add_blob(flatbuffers::Offset<Blob>(u8.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(1);
        tdb.add_blob(flatbuffers::Offset<Blob>(u8.o));
        desc.push_back(tdb.Finish());
    }
    {
        TensorDescribeBuilder tdb(fbb);
        tdb.add_index(2);
        tdb.add_blob(flatbuffers::Offset<Blob>(f32.o));
        desc.push_back(tdb.Finish());
    }

    auto ops    = fbb.CreateVector(vec);
    auto names  = fbb.CreateVectorOfStrings({"input", "conv", "output"});
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

class TFQuantizedConv2DTest : public MNNTestCase {
public:
    virtual ~TFQuantizedConv2DTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 1; b++) { // CPU do not support batch now
            int g = 1;
            {
                for (int o = 4; o <= 256; o *= 4) { // 4 to avoid dirty NC4HW4 blank
                    for (int c = 1; c <= 16; c *= 2) {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    int d = 1;
                                    { // CPU support 1 only
                                        for (int s = 1; s <= 2; s++) {
                                            dispatch([&](MNNForwardType backend) -> void {
                                                if (backend == MNN_FORWARD_CPU)
                                                    return;
                                                std::vector<uint8_t> weights;
                                                std::vector<int32_t> bias;
                                                for (int i = 0; i < g * (o / g) * (c / g) * kw * kh; i++) {
                                                    weights.push_back(rand() % 16);
                                                }
                                                for (int i = 0; i < o; i++) {
                                                    bias.push_back(rand() % UINT8_MAX);
                                                }

                                                // nets
                                                auto net = create(o, is, c, b, d, kw, kh, s, g, weights, bias, false);
                                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                                auto GPU = createSession(net, backend);
                                                if (!CPU || !GPU) {
                                                    delete net;
                                                    return;
                                                }

                                                // input/output
                                                auto input = new Tensor(4, Tensor::CAFFE_C4);
                                                {
                                                    input->setType(DataType_DT_UINT8);
                                                    input->buffer().dim[0].extent = b;
                                                    input->buffer().dim[1].extent = c;
                                                    input->buffer().dim[2].extent = is;
                                                    input->buffer().dim[3].extent = is;
                                                    TensorUtils::setLinearLayout(input);
                                                    input->buffer().host = (uint8_t *)malloc(input->size());
                                                    for (int i = 0; i < is * is * c * b; i++) {
                                                        input->host<uint8_t>()[i] = rand() % 16;
                                                    }
                                                    auto host   = net->getSessionInput(CPU, NULL);
                                                    auto device = net->getSessionInput(GPU, NULL);
                                                    net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                                    net->getBackend(GPU, device)->onCopyBuffer(input, device);
                                                }

                                                // infer
                                                assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU),
                                                                                   0.01 * UINT8_MAX));

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
        }
        return true;
    }
};

class TFQuantizedDepthwiseConv2DTest : public MNNTestCase {
public:
    virtual ~TFQuantizedDepthwiseConv2DTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int o = 4; o <= 16; o *= 2) { // 4 to avoid dirty NC4HW4 blank
                int g = o;
                {
                    int c = o;
                    {
                        for (int is = 1; is <= 8; is *= 2) {
                            for (int kw = 1; kw <= 3 && kw <= is; kw++) {
                                for (int kh = 1; kh <= 3 && kh <= is; kh++) {
                                    for (int d = 1;
                                         d <= 2 && d <= std::min(kw, kh) && d * (std::max(kw, kh) - 1) + 1 <= is; d++) {
                                        for (int s = 1; s <= 2; s++) {
                                            dispatch([&](MNNForwardType backend) -> void {
                                                if (backend == MNN_FORWARD_CPU)
                                                    return;
                                                std::vector<uint8_t> weights;
                                                std::vector<int32_t> bias;
                                                for (int i = 0; i < g * kw * kh; i++) {
                                                    weights.push_back(rand() % 16);
                                                }
                                                for (int i = 0; i < o; i++) {
                                                    bias.push_back(rand() % UINT8_MAX);
                                                }

                                                // nets
                                                auto net = create(o, is, c, b, d, kw, kh, s, g, weights, bias, true);
                                                auto CPU = createSession(net, MNN_FORWARD_CPU);
                                                auto GPU = createSession(net, backend);
                                                if (!CPU || !GPU) {
                                                    delete net;
                                                    return;
                                                }

                                                // input/output
                                                auto input = new Tensor(4, Tensor::CAFFE_C4);
                                                {
                                                    input->setType(DataType_DT_UINT8);
                                                    input->buffer().dim[0].extent = b;
                                                    input->buffer().dim[1].extent = c;
                                                    input->buffer().dim[2].extent = is;
                                                    input->buffer().dim[3].extent = is;
                                                    TensorUtils::setLinearLayout(input);
                                                    input->buffer().host = (uint8_t *)malloc(input->size());
                                                    for (int i = 0; i < is * is * c * b; i++) {
                                                        input->host<uint8_t>()[i] = rand() % 16;
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
        }
        return true;
    }
};
MNNTestSuiteRegister(TFQuantizedConv2DTest, "op/quantized_conv2d/conv");
MNNTestSuiteRegister(TFQuantizedDepthwiseConv2DTest, "op/quantized_conv2d/depthwise_conv");
