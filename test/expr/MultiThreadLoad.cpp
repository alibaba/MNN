//
//  MultiThreadLoadTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"
#include <thread>
using namespace MNN::Express;
using namespace MNN;

class MultiThreadLoadTest : public MNNTestCase {
public:
    virtual bool run() {
        auto x1 = _Input({4}, NHWC, halide_type_of<float>());
        auto x0 = _Input({4}, NCHW, halide_type_of<float>());
        auto y = _Add(x1, x0);
        std::unique_ptr<MNN::NetT> net(new NetT);
        Variable::save({y}, net.get());
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, net.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();

        std::vector<std::thread> threads;
        for (int i=0; i<100; ++i) {
            threads.emplace_back([&](){
                std::shared_ptr<Interpreter> interp(Interpreter::createFromBuffer(bufferOutput, sizeOutput));
                ScheduleConfig config;
                auto session = interp->createSession(config);
                interp->runSession(session);
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        return true;
    }
};
MNNTestSuiteRegister(MultiThreadLoadTest, "expr/MultiThreadLoad");
