//
//  LSTMTest.cpp
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
#include "Tensor_generated.h"
#include "TestUtils.h"

using namespace MNN;

static Interpreter *create(int cont, int w, int c, int ow, std::vector<float> hws, std::vector<float> iws,
                           std::vector<float> bias) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto dims = fbb.CreateVector(std::vector<int>({1, c, 1, w}));
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
    if (cont) {
        auto dims = fbb.CreateVector(std::vector<int>({1, c, 1, 1}));
        InputBuilder ib(fbb);
        ib.add_dims(dims);
        auto input = ib.Finish();
        auto name  = fbb.CreateString("cont");
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
    }
    {
        auto HDims    = fbb.CreateVector(std::vector<int>({1, 1, 4 * ow, ow}));
        auto HWeights = fbb.CreateVector(hws);
        BlobBuilder hbb(fbb);
        hbb.add_dims(HDims);
        hbb.add_dataType(DataType_DT_FLOAT);
        hbb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        hbb.add_float32s(HWeights);
        auto H = hbb.Finish();

        auto IDims    = fbb.CreateVector(std::vector<int>({1, 1, 4 * ow, w}));
        auto IWeights = fbb.CreateVector(iws);
        BlobBuilder ibb(fbb);
        ibb.add_dims(IDims);
        ibb.add_dataType(DataType_DT_FLOAT);
        ibb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        ibb.add_float32s(IWeights);
        auto I = ibb.Finish();

        auto BDims  = fbb.CreateVector(std::vector<int>({1, 1, 4 * ow, 1}));
        auto biases = fbb.CreateVector(bias);
        BlobBuilder bbb(fbb);
        bbb.add_dims(BDims);
        bbb.add_dataType(DataType_DT_FLOAT);
        bbb.add_dataFormat(MNN_DATA_FORMAT_NCHW);
        bbb.add_float32s(biases);
        auto B = bbb.Finish();

        auto lb = LSTMBuilder(fbb);
        lb.add_outputCount(ow);
        lb.add_weightI(I);
        lb.add_weightH(H);
        lb.add_bias(B);
        auto lstm = lb.Finish();

        auto name = fbb.CreateString("lstm");
        auto iv   = cont ? fbb.CreateVector(std::vector<int>({0, 1})) : fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({2}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_LSTM);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_LSTM);
        builder.add_main(lstm.o);
        vec.push_back(builder.Finish());
    }

    auto ops   = fbb.CreateVector(vec);
    auto names = fbb.CreateVectorOfStrings({"input", "cont", "output"});
    NetBuilder builder(fbb);
    builder.add_oplists(ops);
    builder.add_tensorName(names);
    auto net = builder.Finish();
    fbb.Finish(net);
    return Interpreter::createFromBuffer((const char *)fbb.GetBufferPointer(), fbb.GetSize());
}

static Tensor *infer(const Interpreter *net, Session *session) {
    net->runSession(session);
    return net->getSessionOutputAll(session).begin()->second;
}

class LSTMTest : public MNNTestCase {
public:
    virtual ~LSTMTest() = default;
    virtual bool run() {
        for (int c = 1; c < 16; c++) {
            for (int ow = 1; ow < 16; ow *= 2) {
                for (int iw = 1; iw < 16; iw *= 2) {
                    for (int t = 0; t <= 1; t++) {
                        dispatch([&](MNNForwardType backend) -> void {
                            if (backend == MNN_FORWARD_CPU)
                                return;
                            std::vector<float> H, I, B;
                            for (int i = 0; i < ow * 4 * ow; i++)
                                H.push_back(rand() % 255 / 255.f);
                            for (int i = 0; i < iw * 4 * ow; i++)
                                I.push_back(rand() % 255 / 255.f);
                            for (int i = 0; i < 1 * 4 * ow; i++)
                                B.push_back(rand() % 255 / 255.f);

                            auto net = create(t, iw, c, ow, H, I, B);
                            auto CPU = createSession(net, MNN_FORWARD_CPU);
                            auto GPU = createSession(net, backend);
                            if (!CPU || !GPU) {
                                delete net;
                                return;
                            }

                            // input
                            auto input = new Tensor(4);
                            {
                                input->buffer().dim[0].extent = 1;
                                input->buffer().dim[1].extent = c;
                                input->buffer().dim[2].extent = 1;
                                input->buffer().dim[3].extent = iw;
                                TensorUtils::setLinearLayout(input);
                                input->buffer().host = (uint8_t *)malloc(input->size());
                                for (int j = 0; j < 1 * c * 1 * iw; j++) {
                                    input->host<float>()[j] = rand() % 255 / 255.f;
                                }
                                auto host   = net->getSessionInput(CPU, "input");
                                auto device = net->getSessionInput(GPU, "input");
                                net->getBackend(CPU, host)->onCopyBuffer(input, host);
                                net->getBackend(GPU, device)->onCopyBuffer(input, device);
                            }
                            auto cont = new Tensor(4);
                            if (t) {
                                cont->buffer().dim[0].extent = 1;
                                cont->buffer().dim[1].extent = c;
                                cont->buffer().dim[2].extent = 1;
                                cont->buffer().dim[3].extent = 1;
                                TensorUtils::setLinearLayout(cont);
                                cont->buffer().host = (uint8_t *)malloc(cont->size());
                                for (int j = 0; j < 1 * c * 1 * 1; j++) {
                                    cont->host<float>()[j] = rand() % 2;
                                }
                                auto host   = net->getSessionInput(CPU, "cont");
                                auto device = net->getSessionInput(GPU, "cont");
                                net->getBackend(CPU, host)->onCopyBuffer(cont, host);
                                net->getBackend(GPU, device)->onCopyBuffer(cont, device);
                            }

                            // infer
                            assert(TensorUtils::compareTensors(infer(net, GPU), infer(net, CPU), 0.01));

                            // clean up
                            free(input->buffer().host);
                            free(cont->buffer().host);
                            delete input;
                            delete cont;
                            delete net;
                        });
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(LSTMTest, "op/lstm");
