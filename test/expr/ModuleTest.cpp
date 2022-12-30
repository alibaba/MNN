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
    x         = _Conv(0.01f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
    x         = _Conv(0.03f, -1.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}
static VARP convBlocTemp(VARP x, INTS channels, int stride) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int group = inputChannel;
    x         = _Conv(0.002f, 1.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1});
    x         = _Conv(0.05f, -2.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}
static VARP _mobileNetV1Expr(VARP x = nullptr) {
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
    if (nullptr == x) {
        x = _Input({1, 3, inputSize, inputSize}, NC4HW4);
        x->setName("Input");
    }
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
        std::shared_ptr<Module> interp0(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, &config), Module::destroy);
        auto mem1 = rt->onGetMemoryInMB();
        MNN_PRINT("Increase: %f in rt\n", mem1 - mem0);
        std::shared_ptr<Module> interp1(Module::clone(interp0.get(), true), Module::destroy);
        auto mem2 = rt->onGetMemoryInMB();
        MNN_PRINT("Increase: %f in rt\n", mem2 - mem1);
        if (mem2 - mem1 > mem1 - mem0) {
            return false;
        }
        config.rearrange = false;
        std::shared_ptr<Module> interp2(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, &config), Module::destroy);
        std::shared_ptr<Module> interp3(Module::clone(interp2.get()), Module::destroy);
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
        // Test Release order, should be test in debug mode
        interp0.reset();
        interp1.reset();
        MNN::ScheduleConfig sconfig;
        std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
        std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs), Executor::RuntimeManager::destroy);
        interp0.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        auto z0 = interp0->onForward({x});
        // Release runtime and module, then trigger var's release
        interp0.reset();
        rtMgr.reset();
        z0.clear();
        return true;
    }
};
MNNTestSuiteRegister(ModuleTest, "expr/ModuleTest");

class RefTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<int8_t> buffer;
        // construct
        {
            auto x = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
            x->setName("data");
            auto x1 = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
            x1->setName("data1");
            auto x1Ptr = x1->writeMap<int>();
            for (int i=0; i<x1->getInfo()->size; ++i) {
                x1Ptr[i] = 1;
            }
            x1.fix(VARP::CONSTANT);
            auto y = x + x1;
            y->setName("o0");
            auto y1 = x - x1;
            y1->setName("o1");
            buffer = Variable::save({y, y1});
        }
        // Execute
        std::shared_ptr<Module> refModule(Module::load({"data"}, {"o0", "o1", "data1"}, (const uint8_t*)buffer.data(), buffer.size()), Module::destroy);
        auto x = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
        auto size = x->getInfo()->size;
        std::vector<int> inputPtr(size);
        for (int i=0; i<size; ++i) {
            inputPtr[i] = i;
        }
        ::memcpy(x->writeMap<int>(), inputPtr.data(), size * sizeof(int));
        auto outputVars = refModule->onForward({x});
        refModule.reset();
        auto p0 = outputVars[0]->readMap<int>();
        for (int i=0; i<size; ++i) {
            if (p0[i] != inputPtr[i] + 1) {
                FUNC_PRINT(1);
                return false;
            }
        }
        auto p1 = outputVars[1]->readMap<int>();
        for (int i=0; i<size; ++i) {
            if (p1[i] != inputPtr[i] - 1) {
                FUNC_PRINT(1);
                return false;
            }
        }
        auto p2 = outputVars[2]->readMap<int>();
        for (int i=0; i<size; ++i) {
            if (p2[i] != 1) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RefTest, "expr/RefTest");

class LoopTest : public MNNTestCase {
public:
    static void _computeLoop(int size, int* o0, int* o1, const int* i0, const int* i1, int loop) {
        for (int v=0; v<size; ++v) {
            auto x = i0[v];
            auto x1 = i1[v];
            auto y = i0[v];
            auto y1 = i1[v];
            for (int i=0; i<loop; ++i) {
                y = x + x1;
                y1 = x - x1;
                x = y;
                x1 = y1;
            }
            o0[v] = y;
            o1[v] = y1;
        }
    }
    virtual bool run(int precision) {
        std::vector<int8_t> buffer;
        // construct
        {
            auto x = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
            x->setName("data");
            auto x1 = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
            x1->setName("data1");
            auto y = x + x1;
            y->setName("o0");
            auto y1 = x - x1;
            y1->setName("o1");
            auto limit = _Input({}, NCHW, halide_type_of<int>());
            limit->setName("limit");
            auto cond = _Input({}, NCHW, halide_type_of<int>());
            cond->setName("cond");
            auto resCond = _Scalar<int>(1);
            resCond->setName("condresult");
            ExecutorScope::Current()->registerSubGraph("body", {resCond, y, y1}, {limit, cond, x, x1});
            auto u = _Loop({limit, resCond, x, x1}, "body");
            u[0]->setName("o0");
            u[1]->setName("o1");
            buffer = Variable::save(u);
        }
        // Execute
        std::shared_ptr<Module> loopModule(Module::load({"limit", "data", "data1"}, {"o0", "o1"}, (const uint8_t*)buffer.data(), buffer.size()), Module::destroy);
        auto limit = _Input({}, NCHW, halide_type_of<int>());
        auto x = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
        auto x1 = _Input({1, 3, 5, 7}, NCHW, halide_type_of<int>());
        auto size = x->getInfo()->size;
        std::vector<int> inputPtr(size);
        std::vector<int> inputPtr2(size);
        for (int i=0; i<size; ++i) {
            inputPtr[i] = i;
            inputPtr2[i] = i / 2;
        }
        std::vector<int> outputPtr(size);
        std::vector<int> outputPtr2(size);
        {
            auto xPtr = x->writeMap<int>();
            ::memcpy(xPtr, inputPtr.data(), inputPtr.size() * sizeof(int));
            auto x1Ptr = x1->writeMap<int>();
            ::memcpy(x1Ptr, inputPtr2.data(), inputPtr2.size() * sizeof(int));
        }
        auto testFunc = [&](int limitIndex) {
            limit->writeMap<int>()[0] = limitIndex;
            auto y = loopModule->onForward({limit, x, x1});
            auto yPtr = y[0]->readMap<int>();
            auto yPtr1 = y[1]->readMap<int>();
            _computeLoop(size, outputPtr.data(), outputPtr2.data(), inputPtr.data(), inputPtr2.data(), limitIndex);
            for (int i=0; i<size; ++i) {
                if (yPtr[i] != outputPtr[i]) {
                    MNN_PRINT("Error for loop index:%d, %d - %d,%d\n", i, yPtr[i], limitIndex, outputPtr[i]);
                    return false;
                }
                if (yPtr1[i] != outputPtr2[i]) {
                    MNN_PRINT("Error for loop index:%d, %d - %d,%d\n", i, yPtr1[i], limitIndex, outputPtr2[i]);
                    return false;
                }
            }
            return true;
        };
        bool res0 = testFunc(1);
        bool res1 = testFunc(4);
        bool res2 = testFunc(7);
        bool res3 = testFunc(0);
        limit->writeMap<int>()[0] = 2;
        auto y = loopModule->onForward({limit, x, x1});
        loopModule.reset();
        auto yPtr = y[0]->readMap<int>();
        return res0 && res1 && res2 && res3;
    }
};
MNNTestSuiteRegister(LoopTest, "expr/LoopTest");

class ModuleCloneTest : public MNNTestCase {
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
        Module::Config config;
        config.shapeMutable = false;
        config.rearrange = true;
        std::shared_ptr<Module> moduleBasic;
        {
            MNN::ScheduleConfig sconfig;
            sconfig.numThread = 1;
            std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
            std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
            moduleBasic.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        }
        auto makeInput = []() {
            auto varp = _Input({1, 3, 224, 224}, NC4HW4, halide_type_of<float>());
            auto ptr = varp->writeMap<float>();
            int size = varp->getInfo()->size;
            for (int i=0; i < size; ++i) {
                ptr[i] = (float) i / 1000.0f;
            }
            return varp;
        };
        auto basicResult = moduleBasic->onForward({makeInput()});
        float targetAvage = _ReduceMean(basicResult[0])->readMap<float>()[0];

        /* Clone Module Begin */
        int cloneNumber = 4;
        std::vector<std::shared_ptr<Executor>> cloneExecutors(cloneNumber);
        std::vector<std::shared_ptr<Module>> cloneModules(cloneNumber);
        for (int i=0; i<cloneNumber; ++i) {
            cloneExecutors[i] = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
            ExecutorScope current(cloneExecutors[i]);
            cloneModules[i].reset(Module::clone(moduleBasic.get()));
        }
        /* Clone Module End */
        
        /* Execute Module with Multi-Thread Begin*/
        std::vector<bool> result(cloneNumber);
        {
            std::vector<std::thread> threads;
            for (int i=0; i<cloneNumber; ++i) {
                auto curExe = cloneExecutors[i];
                auto curMod = cloneModules[i];
                threads.emplace_back(([curExe, curMod, i, &result, &makeInput, targetAvage] {
                    result[i] = true;
                    ExecutorScope current(curExe);
                    auto varp = makeInput();
                    auto res = curMod->onForward({varp})[0];
                    res = _ReduceMean(res);
                    auto currentAvage = res->readMap<float>()[0];
                    result[i] = targetAvage == currentAvage;
                }));
            }
            for (auto& t : threads) {
                t.join();
            }
        }
        /* Execute Module with Multi-Thread End*/

        /* Release Module Begin*/
        bool res = true;
        for (int i=0; i<cloneNumber; ++i) {
            ExecutorScope current(cloneExecutors[i]);
            cloneModules[i].reset();
            if (!result[i]) {
                res = false;
            }
        }
        /* Release Module End*/
        return res;
    };
};
MNNTestSuiteRegister(ModuleCloneTest, "expr/ModuleClone");


class ModuleTestSpeed : public MNNTestCase {
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
        Module::Config config;
        config.shapeMutable = false;
        config.rearrange = true;
        std::shared_ptr<Module> interp0;
        {
            MNN::ScheduleConfig sconfig;
            sconfig.numThread = 1;
            std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
            std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
            interp0.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        }
        std::shared_ptr<Module> interp1;
        {
            MNN::ScheduleConfig sconfig;
            sconfig.numThread = 4;
            std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
            std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
            interp1.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        }
        auto x = _Input({1, 3, 224, 224}, NC4HW4, halide_type_of<float>());
        auto xPtr = x->writeMap<float>();
        ::memset(xPtr, 0, 1*3*224*224*sizeof(float));
        x->unMap();
        int runTime = 10;
        {
            Timer _l;
            for (int i=0; i<runTime; ++i) {
                auto y0 = interp0->onForward({x});
            }
            MNN_PRINT("Thread 1 avg cost: %f ms\n", (float)_l.durationInUs() / 1000.0f / runTime);
        }
        {
            Timer _l;
            for (int i=0; i<runTime; ++i) {
                auto y0 = interp1->onForward({x});
            }
            MNN_PRINT("Thread 4 avg cost: %f ms\n", (float)_l.durationInUs() / 1000.0f / runTime);
        }
        return true;
    }
};
MNNTestSuiteRegister(ModuleTestSpeed, "expr/ModuleTestSpeed");

class SessionTest : public MNNTestCase {
public:
    bool _run(int precision, bool lazy) {
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
        std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput), Interpreter::destroy);
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
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        auto res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }
};
MNNTestSuiteRegister(SessionTest, "expr/SessionTest");
