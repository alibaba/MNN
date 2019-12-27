//
//  PoolingTest.cpp
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

static Interpreter *create(PoolType type, PoolPadType padType, int w, int h, int c, int b, int kernel, int stride, int pad, int g) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({b, c, h, w}));
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
    }
    {
        std::vector<float> scales;
        for (int i = 0; i < c * b; i++) {
            scales.push_back(i + 1);
        }
        auto pb = PoolBuilder(fbb);
        pb.add_kernelX(kernel);
        pb.add_kernelY(kernel);
        pb.add_strideX(stride);
        pb.add_strideY(stride);
        pb.add_padX(pad);
        pb.add_padY(pad);
        pb.add_isGlobal(g);
        pb.add_type(type);
        pb.add_padType(padType);
        auto pool = pb.Finish();

        auto name = fbb.CreateString("pool");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Pooling);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_Pool);
        builder.add_main(flatbuffers::Offset<void>(pool.o));
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "output"});
    NetBuilder net(fbb);
    net.add_oplists(ops);
    net.add_tensorName(names);
    fbb.Finish(net.Finish());
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

static void testRoutine(PoolType poolType, PoolPadType poolPadType, const float* inputData, const float* correctData, int iw, int ih, int ow, int oh, int c, int b, int kernel, int stride, int pad, int global) {
    auto net = create(poolType, poolPadType, iw, ih, c, b, kernel, stride, pad, global);
    auto sess = createSession(net, MNN_FORWARD_CPU);
    auto input = new Tensor(4); // input, caffe format
    {
        input->buffer().dim[0].extent = b;
        input->buffer().dim[1].extent = c;
        input->buffer().dim[2].extent = ih;
        input->buffer().dim[3].extent = iw;
        TensorUtils::setLinearLayout(input);
        input->buffer().host = (uint8_t *)malloc(input->size());
        for (int i = 0; i < iw * ih * c * b; i++) {
            input->host<float>()[i] = inputData[i];
        }
    }
    auto correct = new Tensor(4); // correct output, caffe format
    {
        correct->buffer().dim[0].extent = b;
        correct->buffer().dim[1].extent = c;
        correct->buffer().dim[2].extent = oh;
        correct->buffer().dim[3].extent = ow;
        TensorUtils::setLinearLayout(correct);
        correct->buffer().host = (uint8_t *)malloc(correct->size());
        for (int i = 0; i < ow * oh * c * b; i++) {
            correct->host<float>()[i] = correctData[i];
        }
    }
    auto host   = net->getSessionInput(sess, NULL);
    net->getBackend(sess, host)->onCopyBuffer(input, host);
    assert(TensorUtils::compareTensors(infer(net, sess), correct, 0.01));
    delete input;
    delete correct;
    delete net;
}

class PoolingMaxTest : public MNNTestCase {
public:
    virtual ~PoolingMaxTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int k = 1; k <= w && k <= h; k *= 2) {
                            for (int s = 1; s <= 4; s *= 2) {
                                for (int p = 0; p <= 2; p++) {
                                    for (int g = 0; g <= 1; g++) {
                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create(PoolType_MAXPOOL, PoolPadType_VALID, w, h, c, b, k, s, p, g);
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
                                                for (int i = 0; i < w * h * c * b; i++) {
                                                    input->host<float>()[i] = rand() % 255 / 255.f;
                                                }
                                            }

                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);

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
                }
            }
        }
        return true;
    }
};

class PoolingAvgTest : public MNNTestCase {
public:
    virtual ~PoolingAvgTest() = default;
    virtual bool run() {
        for (int b = 1; b <= 2; b++) {
            for (int c = 1; c <= 8; c *= 2) {
                for (int h = 1; h <= 8; h *= 2) {
                    for (int w = 1; w <= 8; w *= 2) {
                        for (int k = 1; k <= w && k <= h; k *= 2) {
                            for (int s = 1; s <= 4; s *= 2) {
                                for (int p = 0; p <= 2; p++) {
                                    for (int g = 0; g <= 1; g++) {
                                        dispatch([&](MNNForwardType backend) -> void {
                                            if (backend == MNN_FORWARD_CPU)
                                                return;
                                            // nets
                                            auto net = create(PoolType_AVEPOOL, PoolPadType_VALID, w, h, c, b, k, s, p, g);
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
                                                for (int i = 0; i < w * h * c * b; i++) {
                                                    input->host<float>()[i] = rand() % 255 / 255.f;
                                                }
                                            }

                                            auto host   = net->getSessionInput(CPU, NULL);
                                            auto device = net->getSessionInput(GPU, NULL);
                                            net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                            net->getBackend(GPU, device)->onCopyBuffer(input, device);

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
                }
            }
        }
        const int iw = 7, ih = 7, c = 1, b = 1, global = 0;
        const float inputData[] = {
            0.45, 0.15, 0.27, 0.25, 0.02, 0.87, 0.63,
            0.57, 0.99, 0.87, 0.04, 0.89, 0.45, 0.39,
            0.65, 0.64, 0.88, 0.9 , 0.31, 0.22, 0.45,
            0.59, 0.93, 0.19, 0.84, 0.26, 0.98, 0.26,
            0.52, 0.8 , 0.96, 0.2 , 0.05, 0.12, 0.49,
            0.66, 0.2 , 0.3 , 0.33, 0.8 , 0.65, 0.03,
            0.02, 0.93, 0.63, 0.21, 0.92, 0.96, 0.37
        };
        { // caffe pool test when stride = 1 and pad = 1
            const int stride = 1, pad = 1, kernel = 3, ow = 7, oh = 7;
            const float correctData[] = {
                0.24  , 0.3667, 0.2856, 0.26  , 0.28  , 0.3611, 0.26  ,
                0.3833, 0.6078, 0.5544, 0.4922, 0.4389, 0.47  , 0.3344,
                0.4856, 0.7011, 0.6978, 0.5756, 0.5433, 0.4678, 0.3056,
                0.4589, 0.6844, 0.7044, 0.51  , 0.4311, 0.3489, 0.28  ,
                0.4111, 0.5722, 0.5278, 0.4367, 0.47  , 0.4044, 0.2811,
                0.3478, 0.5578, 0.5067, 0.4889, 0.4711, 0.4878, 0.2911,
                0.2011, 0.3044, 0.2889, 0.3544, 0.43  , 0.4144, 0.2233
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_CAFFE, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, pad, global);
        }
        { // tensorflow pool test when stride = 1 and use valid pad type
            const int stride = 1, kernel = 3, ow = 5, oh = 5;
            const float correctData[] = {
                0.6078, 0.5544, 0.4922, 0.4389, 0.47  ,
                0.7011, 0.6978, 0.5756, 0.5433, 0.4678,
                0.6844, 0.7044, 0.51  , 0.4311, 0.3489,
                0.5722, 0.5278, 0.4367, 0.47  , 0.4044,
                0.5578, 0.5067, 0.4889, 0.4711, 0.4878
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_VALID, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, 0, global);
        }
        { // tensorflow pool test when stride = 1 and use same pad type
            const int stride = 1, kernel = 3, ow = 7, oh = 7;
            const float correctData[] = {
                0.54  , 0.55  , 0.4283, 0.39  , 0.42  , 0.5417, 0.585 ,
                0.575 , 0.6078, 0.5544, 0.4922, 0.4389, 0.47  , 0.5017,
                0.7283, 0.7011, 0.6978, 0.5756, 0.5433, 0.4678, 0.4583,
                0.6883, 0.6844, 0.7044, 0.51  , 0.4311, 0.3489, 0.42  ,
                0.6167, 0.5722, 0.5278, 0.4367, 0.47  , 0.4044, 0.4217,
                0.5217, 0.5578, 0.5067, 0.4889, 0.4711, 0.4878, 0.4367,
                0.4525, 0.4567, 0.4333, 0.5317, 0.645 , 0.6217, 0.5025
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_SAME, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, 0, global);
        }
        { // caffe pool test when stride = 2 and pad = 1
            const int stride = 2, pad = 1, kernel = 3, ow = 4, oh = 4;
            const float correctData[] = {
                0.24  , 0.2856, 0.28  , 0.26  ,
                0.4856, 0.6978, 0.5433, 0.3056,
                0.4111, 0.5278, 0.47  , 0.2811,
                0.2011, 0.2889, 0.43  , 0.2233,
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_CAFFE, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, pad, global);
        }
        { // tensorflow pool test when stride = 2 and use valid pad type
            const int stride = 2, kernel = 2, ow = 3, oh = 3;
            const float correctData[] = {
                0.54  , 0.3575, 0.5575,
                0.7025, 0.7025, 0.4425,
                0.545 , 0.4475, 0.405
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_VALID, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, 0, global);
        }
        { // tensorflow pool test when stride = 2 and use same pad type
            const int stride = 2, kernel = 2, ow = 4, oh = 4;
            const float correctData[] = {
                0.54  , 0.3575, 0.5575, 0.51 ,
                0.7025, 0.7025, 0.4425, 0.355,
                0.545 , 0.4475, 0.405 , 0.26 ,
                0.475 , 0.42  , 0.94  , 0.37
            };
            testRoutine(PoolType_AVEPOOL, PoolPadType_SAME, inputData, correctData, iw, ih, ow, oh, c, b, kernel, stride, 0, global);
        }
        return true;
    }
};
MNNTestSuiteRegister(PoolingMaxTest, "op/pool/max");
MNNTestSuiteRegister(PoolingAvgTest, "op/pool/avg");
