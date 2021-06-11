//
//  ModuleTest.cpp
//  MNNTests
//
//  Created by MNN on b'2020/12/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <thread>
#include "MNNTestSuite.h"
#include "core/Backend.hpp"
#include <MNN/expr/Executor.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "MNN_generated.h"
using namespace MNN::Express;
using namespace MNN;

// When we use MNNConverter to convert other mobilenet model to MNN model,
// {Conv3x3Depthwise + BN + Relu + Conv1x1 + BN + Relu} will be converted
// and optimized to {Conv3x3Depthwise + Conv1x1}
static VARP convBlock(VARP x, INTS channels, int stride) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int group = inputChannel;
    x         = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
    x         = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}
static VARP convBlocTemp(VARP x, INTS channels, int stride) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int group = inputChannel;
    x         = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1});
    x         = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}
static VARP _mobileNetV1Expr() {
    int inputSize = 224, poolSize; // MobileNet_224, MobileNet_192, MobileNet_160, MobileNet_128
    {
        inputSize = 224;
        poolSize  = inputSize / 32;
    }

    int channels[6]; // MobileNet_100, MobileNet_075, MobileNet_050, MobileNet_025
    { channels[0] = 32; }

    for (int i = 1; i < 6; ++i) {
        channels[i] = channels[0] * (1 << i);
    }

    auto x = _Input({1, 3, inputSize, inputSize}, NC4HW4);
    x->setName("Input");
    x      = _Conv(0.0f, 0.0f, x, {3, channels[0]}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
    x      = convBlock(x, {channels[0], channels[1]}, 1);
    x      = convBlock(x, {channels[1], channels[2]}, 2);
    x      = convBlock(x, {channels[2], channels[2]}, 1);
    x      = convBlock(x, {channels[2], channels[3]}, 2);
    x      = convBlock(x, {channels[3], channels[3]}, 1);
    x      = convBlock(x, {channels[3], channels[4]}, 2);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlocTemp(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[5]}, 2);
    x      = convBlock(x, {channels[5], channels[5]}, 1);
    x      = _AvePool(x, {poolSize, poolSize}, {1, 1}, VALID);
    x      = _Conv(0.0f, 0.0f, x, {channels[5], 1001}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    x      = _Softmax(x, -1);
    x      = _Convert(x, NCHW);
    x->setName("Prob");
    return x;
}
class ModuleTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto y = _mobileNetV1Expr();
        std::unique_ptr<MNN::NetT> net(new NetT);
        Variable::save({y}, net.get());
        y = nullptr;
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, net.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        // Force use CPU Runtime
        BackendConfig bnConfig;
        auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
        ExecutorScope scope(exe);
        auto rtInfo = Express::ExecutorScope::Current()->getRuntime();
        auto rt = rtInfo.first.begin()->second;
        auto mem0 = rt->onGetMemoryInMB();
        Module::Config config;
        config.shapeMutable = false;
        config.rearrange = true;
        std::shared_ptr<Module> interp0(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, &config));
        auto mem1 = rt->onGetMemoryInMB();
        MNN_PRINT("Increase: %f in rt\n", mem1 - mem0);
        std::shared_ptr<Module> interp1(Module::clone(interp0.get(), true));
        auto mem2 = rt->onGetMemoryInMB();
        MNN_PRINT("Increase: %f in rt\n", mem2 - mem1);
        if (mem2 - mem1 > mem1 - mem0) {
            return false;
        }
        config.rearrange = false;
        std::shared_ptr<Module> interp2(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, &config));
        std::shared_ptr<Module> interp3(Module::clone(interp2.get()));
        auto x = _Input({1, 3, 224, 224}, NC4HW4, halide_type_of<float>());
        auto xPtr = x->writeMap<float>();
        ::memset(xPtr, 0, 1*3*224*224*sizeof(float));
        x->unMap();
        auto y0 = interp0->onForward({x});
        auto y1 = interp1->onForward({x});
        if (y0.size() != 1) {
            return false;
        }
        {
            auto info = y0[0]->getInfo();
            if (info->size != 1001) {
                return false;
            }
            if (y0[0]->readMap<float>() == nullptr) {
                return false;
            }
        }
        if (y1.size() != 1) {
            return false;
        }
        {
            auto info = y1[0]->getInfo();
            if (info->size != 1001) {
                return false;
            }
            if (y1[0]->readMap<float>() == nullptr) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ModuleTest, "expr/ModuleTest");


class SessionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            auto y = _mobileNetV1Expr();
            std::unique_ptr<MNN::NetT> net(new NetT);
            Variable::save({y}, net.get());
            y = nullptr;
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput));
        ScheduleConfig config;
        config.numThread = 1;
        auto s1 = net->createSession(config);
        int runTime = 10;
        {
            AUTOTIME;
            for (int t = 0; t < runTime; ++t) {
                net->runSession(s1);
            }
        }
        net->releaseSession(s1);
        std::vector<Session*> sessions;
        for (int i = 0; i < 4; ++i) {
            auto s = net->createSession(config);
            sessions.emplace_back(s);
        }
        std::vector<std::thread> allThreads;
        for (int i = 0; i < 4; ++i) {
            auto s = sessions[i];
            allThreads.emplace_back(std::thread([s, net, config, runTime] {
                {
                    AUTOTIME;
                    for (int t = 0; t < runTime; ++t) {
                        net->runSession(s);
                    }
                }
            }));
        }
        for (auto& t : allThreads) {
            t.join();
        }
        return true;
    }
};
MNNTestSuiteRegister(SessionTest, "expr/SessionTest");
