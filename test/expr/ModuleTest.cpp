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
#include "RuntimeAttr.hpp"
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
static VARP _mobileNetV1Expr(VARP x = nullptr, bool softmax = true) {
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
    x      = _Conv(0.01f, 0.0f, x, {3, channels[0]}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
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
    x      = _Conv(0.01f, 0.0f, x, {channels[5], 1001}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    if (softmax) {
        x      = _Softmax(x, -1);
    }
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
        rtMgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 1); // eager
        float defer_mem0, defer_mem1;
        rtMgr->getInfo(MNN::Interpreter::MEMORY, &defer_mem0);
        interp0.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        auto z0 = interp0->onForward({x});
        rtMgr->getInfo(MNN::Interpreter::MEMORY, &defer_mem1);
        float eager_increase = defer_mem1 - defer_mem0;
        MNN_PRINT("EagerAllocator Increase: %f\n", eager_increase);
        rtMgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0); // defer
        rtMgr->getInfo(MNN::Interpreter::MEMORY, &defer_mem0);
        interp1.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        auto z1 = interp1->onForward({x});
        rtMgr->getInfo(MNN::Interpreter::MEMORY, &defer_mem1);
        float defer_increase = defer_mem1 - defer_mem0;
        MNN_PRINT("DeferAllocator Increase: %f\n", defer_increase);
        MNNTEST_ASSERT(defer_increase <= eager_increase);
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

class ModuleReleaseTest : public MNNTestCase {
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
        auto rtInfo = exe->getRuntime();
        float memory;
        auto countMemory = [&rtInfo, &memory]() {
            memory = 0.0f;
            for (auto& iter : rtInfo.first) {
                memory += iter.second->onGetMemoryInMB();
            }
            memory += rtInfo.second->onGetMemoryInMB();
        };
        countMemory();
        FUNC_PRINT_ALL(memory, f);
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
        countMemory();
        FUNC_PRINT_ALL(memory, f);
        interp0.reset();
        countMemory();
        FUNC_PRINT_ALL(memory, f);
        if (memory > 1.0f) {
            return false;
        }
        return true;
    };
};
MNNTestSuiteRegister(ModuleReleaseTest, "expr/ModuleReleaseTest");


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
        auto x = _Input({1, 3, 224, 224}, NC4HW4, halide_type_of<float>());
        auto xPtr = x->writeMap<float>();
        ::memset(xPtr, 0, 1*3*224*224*sizeof(float));
        x->unMap();
        int runTime = 10;
        std::shared_ptr<Module> interp0;
        {
            MNN::ScheduleConfig sconfig;
            sconfig.numThread = 1;
            std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
            std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
            interp0.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        }
        {
            Timer _l;
            for (int i=0; i<runTime; ++i) {
                auto y0 = interp0->onForward({x});
            }
            MNN_PRINT("Thread 1 avg cost: %f ms\n", (float)_l.durationInUs() / 1000.0f / runTime);
        }
        std::shared_ptr<Module> interp1;
        {
            MNN::ScheduleConfig sconfig;
            sconfig.numThread = 4;
            std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
            std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
            rtMgr->setHint(Interpreter::STRICT_CHECK_MODEL, 0);
            interp1.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
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

class SpecialSessionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        {
            int expect = 5;
            auto x = _Input({10}, NHWC, halide_type_of<int>());
            auto y = _Scalar<int>(expect);
            auto z = x * x + y;
            z->setName("test");
            auto res = z + y;
            auto buffer = Variable::save({res});
            std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)buffer.data(), buffer.size()), Interpreter::destroy);
            ScheduleConfig config;
            config.numThread = 1;
            net->setSessionMode(Interpreter::Session_Debug);
            auto session = net->createSession(config);

            int directValue = -1;
            int copyValue = -1;
            MNN::TensorCallBack beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
                auto origin = ntensors[1];
                if (opName == "test") {
                    directValue = origin->host<int>()[0];
                    std::shared_ptr<MNN::Tensor> copyTensor(new MNN::Tensor(origin, MNN::Tensor::TENSORFLOW));
                    origin->copyToHostTensor(copyTensor.get());
                    copyValue = copyTensor->host<int>()[0];
                }
                return true;
            };
            MNN::TensorCallBack afterCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
                if (opName == "test") {
                    return false;
                }
                return true;
            };
            net->runSessionWithCallBack(session, beforeCallBack, afterCallBack);
            if (expect != directValue) {
                FUNC_PRINT(1);
                return false;
            }
            if (expect != copyValue) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }

};
MNNTestSuiteRegister(SpecialSessionTest, "expr/SpecialSessionTest");

class SessionCircleTest : public MNNTestCase {
public:
    bool _run(int precision, bool loop) {
        int channel = 10;
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            auto x = _Input({2, channel, 1, 1}, NC4HW4);
            x->setName("x");
            auto ox = x * x;
            ox->setName("ox");
            auto y = _Const(1.0f, {1, channel, 1, 1}, NC4HW4);
            y->setName("y");
            y.fix(VARP::TRAINABLE);
            auto z = x * y;
            z->setName("xy");
            z = _ReduceMean(z);
            z->setName("l");
            z = y + z;
            z = _Convert(z, NCHW);
            z = _Unsqueeze(z, {0});
            z = _Squeeze(z, {0});
            z = _Convert(z, NC4HW4);
            z->setName("z");
            std::unique_ptr<MNN::NetT> net(new NetT);
            Variable::save({z, ox}, net.get());
            z = nullptr;
            if (loop) {
                // Make Loop
                // Find x index
                int yIndex = -1;
                int zIndex = -1;
                for (int i=0; i<net->tensorName.size(); ++i) {
                    if (net->tensorName[i] == "y") {
                        yIndex = i;
                    } else if (net->tensorName[i] == "z") {
                        zIndex = i;
                    }
                }
                if (yIndex == -1 || zIndex == -1) {
                    FUNC_PRINT(1);
                    return false;
                }
                for (auto& op : net->oplists) {
                    for (int i=0; i<op->outputIndexes.size(); ++i) {
                        if (op->outputIndexes[i] == zIndex) {
                            op->outputIndexes[i] = yIndex;
                        }
                    }
                }
            }
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput), Interpreter::destroy);
        auto rt = MNN::Express::Executor::getGlobalExecutor()->getRuntime().first;
        auto type = MNN_FORWARD_CPU;
        for (auto& iter : rt) {
            if (iter.first != MNN_FORWARD_CPU) {
                type = iter.first;
                break;
            }
        }
        net->setSessionMode(Interpreter::Session_Output_User);
        ScheduleConfig config;
        config.type = type;
        config.numThread = 4;
        config.saveTensors = {"l", "ox", "xy"};
        BackendConfig bnConfig;
        bnConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
        config.backendConfig = &bnConfig;
        auto session = net->createSession(config);
        auto x = net->getSessionInput(session, "x");
        auto l = net->getSessionOutput(session, "l");
        auto z2 = net->getSessionOutput(session, "xy");
        if (nullptr == x || nullptr == l || nullptr == z2) {
            return false;
        }
        std::vector<float> values(10);
        std::vector<float> z2values(10);
        float basicValue = 0.5f;
        for (int range=0; range<10; ++range) {
            int curSize = range+1;
            net->resizeTensor(x, {curSize, channel, 1, 1});
            net->resizeSession(session);
            std::shared_ptr<MNN::Tensor> xh(new Tensor(x));
            for (int i=0; i<curSize*channel; ++i) {
                xh->host<float>()[i] = basicValue;
            }
            x->copyFromHostTensor(xh.get());
            net->runSession(session);
            std::shared_ptr<MNN::Tensor> lh(new Tensor(l));
            l->copyToHostTensor(lh.get());
            values[range] = lh->host<float>()[0];
            std::shared_ptr<MNN::Tensor> z2h(new Tensor(z2));
            z2->copyToHostTensor(z2h.get());
            auto z2hSize = z2h->elementSize();
            float summer = 0.0f;
            for (int i=0; i<z2hSize; ++i) {
                summer += z2h->host<float>()[i];
            }
            z2values[range] = summer;
        }
        MNN_PRINT("loop: %d, %f -> %f, %f -> %f\n", loop, values[0], values[9], z2values[0], z2values[9]);
        if (fabsf(values[0] - basicValue) > 0.001f) {
            return false;
        }
        if (loop && values[9] <= values[0] + basicValue) {
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        auto res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        return _run(precision, false);
    }
};
MNNTestSuiteRegister(SessionCircleTest, "expr/SessionCircleTest");

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
        int runTime = 5;
        auto s0 = net->createSession(config);
        {
            AUTOTIME;
            for (int t = 0; t < runTime; ++t) {
                net->runSession(s0);
            }
        }
        net->releaseSession(s0);
        config.numThread = 4;
        auto s1 = net->createSession(config);
        {
            AUTOTIME;
            for (int t = 0; t < runTime; ++t) {
                net->runSession(s1);
            }
        }
        net->releaseSession(s1);
        std::vector<std::thread> allThreads;
        for (int i = 0; i < 4; ++i) {
            allThreads.emplace_back(std::thread([runTime, i, bufferOutput, sizeOutput] {
                {
                    std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput), Interpreter::destroy);
                    ScheduleConfig config;
                    config.numThread = 4 - i;
                    BackendConfig bnConfig;
                    bnConfig.power = MNN::BackendConfig::Power_Normal;
                    config.backendConfig = &bnConfig;
                    auto s = net->createSession(config);
                    AUTOTIME;
                    for (int t = 0; t < runTime; ++t) {
                        net->runSession(s);
                    }
                    net->releaseSession(s);
                }
            }));
        }
        for (auto& t : allThreads) {
            t.join();
        }
        for (int i=0; i<3; ++i) {
            auto rt = Interpreter::createRuntime({config});
            auto s0 = net->createSession(config, rt);
            auto s1 = net->createSession(config, rt);
            int numberThread = 0;
            net->getSessionInfo(s0, MNN::Interpreter::THREAD_NUMBER, &numberThread);
            if (numberThread != 4) {
                FUNC_PRINT(i);
                return false;
            }
            net->getSessionInfo(s1, MNN::Interpreter::THREAD_NUMBER, &numberThread);
            if (numberThread != 4) {
                FUNC_PRINT(i);
                return false;
            }
            {
                AUTOTIME;
                for (int t = 0; t < runTime; ++t) {
                    net->runSession(s0);
                }
            }
            net->releaseSession(s0);
            net->releaseSession(s1);
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

class MultiThreadOneSessionTest : public MNNTestCase {
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
        config.numThread = 4;
        auto s1 = net->createSession(config);
        std::vector<std::thread> allThreads;
        for (int i = 0; i < 4; ++i) {
            allThreads.emplace_back(std::thread([net, s1] {
                net->runSession(s1);
            }));
        }
        for (auto& t : allThreads) {
            t.join();
        }
        return true;
    }
    virtual bool run(int precision) {
        auto res = _run(precision, true);
        return res;
    }
};
MNNTestSuiteRegister(MultiThreadOneSessionTest, "expr/MultiThreadOneSessionTest");

class MemeoryUsageTest : public MNNTestCase {
public:
    bool _run(int precision, bool lazy) {
        auto func = [precision](VARP y, float limit) {
            flatbuffers::FlatBufferBuilder builderOutput(1024);
            {
                std::unique_ptr<MNN::NetT> net(new NetT);
                Variable::save({y}, net.get());
                auto len = MNN::Net::Pack(builderOutput, net.get());
                builderOutput.Finish(len);
            }
            int sizeOutput    = builderOutput.GetSize();
            auto bufferOutput = builderOutput.GetBufferPointer();
            std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput), Interpreter::destroy);
            ScheduleConfig config;
            BackendConfig bnConfig;
            bnConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
            config.numThread = 1;
            config.type = ExecutorScope::Current()->getAttr()->firstType;
            config.backendConfig = &bnConfig;
            auto s1 = net->createSession(config);
            float memory = 0.0f;
            net->getSessionInfo(s1, MNN::Interpreter::MEMORY, &memory);
            if (memory < 0.01f) {
                FUNC_PRINT(precision);
                return false;
            }
            if (memory > limit) {
                MNN_ERROR("memory %f larger than limit: %f, precision=%d\n", memory, limit, precision);
                return false;
            }
            FUNC_PRINT_ALL(memory, f);
            return true;
        };
        auto y = _mobileNetV1Expr();
        bool res = func(y, 62.0f);
        if (!res) {
            return false;
        }
        auto x = _Input({1, 3, 1024, 1024}, NCHW);
        y = _Sigmoid(x);
        res = func(y, 35.0f);
        if (!res) {
            return false;
        }
        auto weightVar = MNN::Express::_Const(0.0f, {100, 10000}, NCHW);
        x = MNN::Express::_Input({1, 100}, NCHW);
        auto x2 = MNN::Express::_Input({1, 10000}, NCHW);
        y = MNN::Express::_MatMul(x, weightVar);
        auto weightVar2 = MNN::Express::_Const(0.0f, {10000, 100}, NCHW);
        y = MNN::Express::_MatMul(y, weightVar2);
        res = func(y, 8.0f);
        if (!res) {
            return false;
        }
        weightVar = MNN::Express::_Const(0.0f, {100, 10000, 1, 1}, NC4HW4);
        x = MNN::Express::_Input({100, 10000, 1, 1}, NC4HW4);
        y = MNN::Express::_Add(x, weightVar);
        res = func(y, 12.0f);
        if (!res) {
            return false;
        }
        auto w2 = weightVar * weightVar;
        y = MNN::Express::_Add(x, w2);
        // TODO: Optimize the memory to 10.0f
        res = func(y, 20.0f);
        if (!res) {
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        auto res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        return res;
    }
};
MNNTestSuiteRegister(MemeoryUsageTest, "expr/MemeoryUsageTest");

// This test shoule use gpu to test
class ConstMemoryReplaceTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x = _Input({1, 4, 1, 1}, NC4HW4);
        auto y = _Const(0.3f, {1, 1, 4, 1}, NC4HW4);
        auto z = x * y;
        auto w0 = _Round(_ReduceSum(_Convert(y, NHWC)));
        z = z + _Unsqueeze(w0, {0});
        auto w1 = _Scalar<int>(1);
        auto shape = _Stack({w1, _Cast<int>(w0), w1, w1}, -1);
        auto ones = _Fill(shape, _Scalar<float>(0.3f));
        auto res = z + ones;
        x->writeMap<float>();
        auto ptr = res->readMap<float>();
        if (nullptr == ptr) {
            FUNC_PRINT(1);
            return false;
        }
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            std::shared_ptr<MNN::NetT> net(new NetT);
            Variable::save({res}, net.get());
            y = nullptr;
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)bufferOutput, sizeOutput), Interpreter::destroy);
        ScheduleConfig config;
        config.numThread = 4;
        config.type = ExecutorScope::Current()->getAttr()->firstType;
        auto s1 = net->createSession(config);
        int resizeCode;
        net->getSessionInfo(s1, Interpreter::RESIZE_STATUS, &resizeCode);
        if (resizeCode != 0) {
            FUNC_PRINT(1);
            return false;
        }
        net->runSession(s1);
        net->resizeTensor(net->getSessionInput(s1, nullptr), {1, 1, 1, 1});
        net->resizeSession(s1);
        return resizeCode == 0;
    }
};
MNNTestSuiteRegister(ConstMemoryReplaceTest, "expr/ConstMemoryReplaceTest");

class MutlThreadConstReplaceTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto func = [precision](VARP y, int thread) {
            flatbuffers::FlatBufferBuilder builderOutput(1024);
            {
                std::unique_ptr<MNN::NetT> net(new NetT);
                Variable::save({y}, net.get());
                auto len = MNN::Net::Pack(builderOutput, net.get());
                builderOutput.Finish(len);
            }
            int sizeOutput    = builderOutput.GetSize();
            auto bufferOutput = builderOutput.GetBufferPointer();
            MNN::Express::Module::Config modConfig;
            modConfig.rearrange = true;
            std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{}, bufferOutput, sizeOutput, &modConfig), MNN::Express::Module::destroy);

            ScheduleConfig config;
            BackendConfig bnConfig;
            bnConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
            config.numThread = 1;
            config.type = ExecutorScope::Current()->getAttr()->firstType;
            config.backendConfig = &bnConfig;

            std::vector<std::thread> threads;
            std::vector<float> summer(thread);
            std::mutex moduleMutex;

            for (int t = 0; t<thread; ++t) {
                threads.emplace_back([&, t]() {
                    auto newExe = Executor::newExecutor(config.type, bnConfig, 1);
                    ExecutorScope scope(newExe);
                    std::shared_ptr<Module> tempModule;
                    {
                        std::unique_lock<std::mutex> _l(moduleMutex);
                        tempModule.reset(Module::clone(net.get()), Module::destroy);
                    }
                    // Create Input
                    auto x = MNN::Express::_Input({1, 100}, NCHW);
                    auto xPtr = x->writeMap<float>();
                    for (int j=0; j<100; ++j) {
                        xPtr[j] = j / 100.0f;
                    }
                    x->unMap();
                    auto y = tempModule->onForward({x});
                    auto yPtr = y[0]->readMap<float>();
                    auto ySize = y[0]->getInfo()->size;
                    float sum = 0.0f;
                    for (int j=0; j<ySize; ++j) {
                        sum += yPtr[j];
                    }
                    y[0]->unMap();
                    {
                        std::unique_lock<std::mutex> _l(moduleMutex);
                        summer[t] = sum;
                    }
                });
            }
            for (auto& t : threads) {
                t.join();
            }
            MNN_PRINT("Summer: ");
            for (auto t : summer) {
                MNN_PRINT("%f, ", t);
            }
            MNN_PRINT("\n");
            return true;
        };
        auto weightVar = MNN::Express::_Const(0.001f, {100, 10000}, NCHW);
        auto x = MNN::Express::_Input({1, 100}, NCHW);
        x->setName("x");
        auto y = MNN::Express::_MatMul(x, weightVar);
        auto weightVar2 = MNN::Express::_Const(0.0002f, {10000, 100}, NCHW);
        y = MNN::Express::_MatMul(y, weightVar2);
        y->setName("y");
        func(y, 4);

        return true;
    };
};
MNNTestSuiteRegister(MutlThreadConstReplaceTest, "expr/MutlThreadConstReplaceTest");

class ResizeOptimizationTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<int8_t> buffer;
        {
            // Make Buffer
            auto x0 = _Input({1, 3, 32, 32}, NCHW, halide_type_of<float>());
            x0->setName("x0");
            {
                auto x1s = _Shape(x0);
                auto ss = _Unstack(x1s);
                auto w = ss[2];
                auto h = ss[3];
                int batchNumber = 1;
                int channelNumber = 3;
                auto batch = _Const(&batchNumber, {}, NCHW, halide_type_of<int32_t>());
                auto channel = _Const(&channelNumber, {}, NCHW, halide_type_of<int32_t>());
                x0 = _Reshape(x0, _Stack({batch * channel, w * h}));
                x0 = _Reshape(x0, x1s);
            }
            auto y0 = _mobileNetV1Expr(_Convert(x0, NC4HW4), false);
            y0->setName("y0");
            auto x1 = _Input({1, 3, 64, 64}, NCHW, halide_type_of<float>());
            x1->setName("x1");
            auto y1 = _mobileNetV1Expr(_Convert(x1, NC4HW4), false);
            y1->setName("y1");
            auto z = y0 + y1;
            z->setName("z");
            buffer = Variable::save({z});
        }
        std::vector<std::pair<std::vector<int>, std::vector<int>>> inputShapes {
            {{1, 3, 32, 32}, {1, 3, 24, 24}},
            {{1, 3, 16, 16}, {1, 3, 24, 24}},
            {{1, 3, 48, 48}, {1, 3, 24, 24}},
        };
        {
            // Test For Interpreter API
            std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer((void*)buffer.data(), buffer.size()), Interpreter::destroy);
            ScheduleConfig config;
            config.numThread = 1;
            net->setSessionMode(Interpreter::Session_Debug);
            auto session = net->createSession(config);
            auto getResult = [session, net, &inputShapes] {
                std::vector<float> resultSummer(inputShapes.size());
                auto x0 = net->getSessionInput(session, "x0");
                auto x1 = net->getSessionInput(session, "x1");
                auto z = net->getSessionOutput(session, "z");
                auto fillInput = [](MNN::Tensor* t, float v) {
                    std::shared_ptr<MNN::Tensor> tensor(new MNN::Tensor(t, t->getDimensionType()));
                    auto size = tensor->elementSize();
                    auto ptr = tensor->host<float>();
                    float cv = v;
                    for (int i=0; i<size; ++i) {
                        ptr[i] = cv;
                        cv = cv * -1.0f;
                    }
                    t->copyFromHostTensor(tensor.get());
                };
                for (int u=0; u<inputShapes.size(); ++u) {
                    net->resizeTensor(x0, inputShapes[u].first);
                    net->resizeTensor(x1, inputShapes[u].second);
                    net->resizeSession(session);
                    float u0 = (float)x0->elementSize();
                    float u1 = (float)x1->elementSize();
                    fillInput(x0, 0.0001f * (float)u);
                    fillInput(x1, 0.0001f * (float)u);
                    net->runSession(session);
                    std::shared_ptr<MNN::Tensor> tensor(new MNN::Tensor(z, z->getDimensionType()));
                    z->copyToHostTensor(tensor.get());
                    auto size = tensor->elementSize();
                    auto resPtr = tensor->host<float>();
                    float summer = 0.0f;
                    float decrate = 1.0f / u0 / u1;
                    for (int i=0; i<size; ++i) {
                        summer += (resPtr[i] * resPtr[i]) * decrate;
                    }
                    resultSummer[u] = summer;
                    FUNC_PRINT_ALL(summer, f);
                }
                return resultSummer;
            };
            auto originRes = getResult();
            net->setSessionMode(Interpreter::Session_Resize_Check);
            auto checkRes = getResult();
            net->setSessionMode(Interpreter::Session_Resize_Fix);
            auto fixRes = getResult();
            for (int u=0; u<inputShapes.size(); ++u) {
                auto v1error = fabsf(originRes[u]-checkRes[u]);
                auto v2error = fabsf(originRes[u]-fixRes[u]);
                if (v1error > 0.05f || v2error > 0.05f) {
                    FUNC_PRINT(u);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ResizeOptimizationTest, "expr/ResizeOptimizationTest");

class WinogradMemoryTest : public MNNTestCase {
public:
    float memoryUsed(int level) {
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
        std::shared_ptr<Module> interp0;
        
        MNN::ScheduleConfig sconfig;
        sconfig.numThread = 1;
        std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
        auto rtInfo = Express::ExecutorScope::Current()->getRuntime();
        auto rt = rtInfo.first.begin()->second;
        std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
        rtMgr->setMode(Interpreter::Session_Memory_Collect);
        rtMgr->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, level);
        config.rearrange = false; // When set WINOGRAD_MEMORY_LEVEL=0 to test memory, must set rearrange=false.
        interp0.reset(Module::load({"Input"}, {"Prob"}, bufferOutput, sizeOutput, rtMgr, &config), Module::destroy);
        float memoryInMB = 0.0f;
        rtMgr->getInfo(Interpreter::MEMORY, &memoryInMB);
        return memoryInMB;
    }
    virtual bool run(int precision) {
        float mem0 = memoryUsed(0);
        float mem3 = memoryUsed(3);
        printf("level=0,3: %fMb, %fMb\n", mem0,mem3);
        if (mem3 < mem0) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(WinogradMemoryTest, "expr/WinogradMemoryTest");
