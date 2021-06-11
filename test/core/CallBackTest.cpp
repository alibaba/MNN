//
//  CallBackTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/9/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/Interpreter.hpp>
using namespace MNN;

class CallBackTest : public MNNTestCase {
public:
    virtual ~CallBackTest() = default;
    virtual bool run(int precision) {
        // build net
        std::unique_ptr<NetT> net(new NetT);
        std::unique_ptr<OpT> input(new OpT);
        input->type = OpType_Input;
        auto param(new InputT);
        param->dims.push_back(1);
        param->dims.push_back(1);
        param->dims.push_back(1);
        param->dims.push_back(64);
        input->main.type = OpParameter_Input;
        input->main.value = param;
        input->outputIndexes.push_back(0);
        net->oplists.emplace_back(std::move(input));
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_TanH;
        op->inputIndexes.push_back(0);
        op->outputIndexes.push_back(1);
        net->oplists.emplace_back(std::move(op));
        net->tensorName.push_back("tensor_0");
        net->tensorName.push_back("tensor_1");
        net->tensorNumber = 2;
        net->usage = Usage_INFERENCE;
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = MNN::Net::Pack(builder, net.get());
        builder.Finish(offset);
        int size      = builder.GetSize();
        auto buffer = builder.GetBufferPointer();
        std::shared_ptr<Interpreter> interpreter(Interpreter::createFromBuffer(buffer, size));
        ScheduleConfig config;
        Session* session = interpreter->createSession(config);
        // run callback
        bool opType = false, opInput = false, opOutput = false;
        TensorCallBackWithInfo before = [&](const std::vector<Tensor*>& nTensors, const OperatorInfo* info) {
            opType = info->type() == "UnaryOp";
            opInput = nTensors.size() == 1 && nTensors[0]->shape()[3] == 64;
            return false;
        };
        TensorCallBackWithInfo after = [&](const std::vector<Tensor*>& nTensors, const OperatorInfo* info) {
            opType &= info->type() == "UnaryOp";
            opOutput = nTensors.size() == 1 &&  nTensors[0]->shape()[3] == 64;
            return true;
        };
        interpreter->runSessionWithCallBackInfo(session, before, after);
        return  opType && opInput && opOutput;
    }
};
MNNTestSuiteRegister(CallBackTest, "core/callback");
