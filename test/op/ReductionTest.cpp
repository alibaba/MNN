//
//  ReductionTest.cpp
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

static Interpreter *create(ReductionType op, std::vector<int> dims, bool kd, int b, int c, int h, int w) {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<Op>> vec;

    {
        auto idims = fbb.CreateVector(std::vector<int>({b, h, w, c}));
        InputBuilder ib(fbb);
        ib.add_dims(idims);
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
        auto rdims = fbb.CreateVector(dims);
        auto rpb   = ReductionParamBuilder(fbb);
        rpb.add_operation(op);
        rpb.add_dType(DataType_DT_FLOAT);
        rpb.add_dim(rdims);
        rpb.add_coeff(0.f);
        rpb.add_keepDims(kd);
        auto reduction = rpb.Finish();

        auto name = fbb.CreateString("reduction");
        auto iv   = fbb.CreateVector(std::vector<int>({0}));
        auto ov   = fbb.CreateVector(std::vector<int>({1}));
        OpBuilder builder(fbb);
        builder.add_type(OpType_Reduction);
        builder.add_name(name);
        builder.add_inputIndexes(iv);
        builder.add_outputIndexes(ov);
        builder.add_main_type(OpParameter_ReductionParam);
        builder.add_main(flatbuffers::Offset<void>(reduction.o));
        vec.push_back(builder.Finish());
    }

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

class ReductionTest : public MNNTestCase {
public:
    virtual ~ReductionTest() = default;
    virtual bool run() {
        ReductionType ops[] = {ReductionType_SUM, ReductionType_MEAN, ReductionType_MAXIMUM, ReductionType_MINIMUM,
                               ReductionType_PROD};

        for (int i = 0; i < sizeof(ops) / sizeof(ReductionType); i++) {
            ReductionType op = ops[i];

            for (int d = 1; d <= 0b1111; d++) {
                for (int kd = 0; kd <= 1; kd++) {
                    for (int b = 3; b <= 3; b++) {
                        for (int c = 5; c <= 5; c++) {
                            for (int h = 7; h <= 7; h++) {
                                for (int w = 9; w <= 9; w++) {
                                    dispatch([&](MNNForwardType backend) -> void {
                                        if (backend == MNN_FORWARD_CPU)
                                            return;
                                        // nets
                                        std::vector<int> dims;
                                        if (d & 0b0001)
                                            dims.push_back(0);
                                        if (d & 0b0010)
                                            dims.push_back(1);
                                        if (d & 0b0100)
                                            dims.push_back(2);
                                        if (d & 0b1000)
                                            dims.push_back(3);

                                        auto net = create(op, dims, kd, b, c, h, w);
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
        return true;
    }
};
MNNTestSuiteRegister(ReductionTest, "op/reduction");
