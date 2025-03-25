//
//  MemoryIncrease.cpp
//  MNNTests
//
//  Created by MNN on b'2020/08/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include <thread>
#include "MNNTestSuite.h"
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

VARP _mobileNetV1Expr() {
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
    x      = _Conv(0.0f, 0.0f, x, {3, channels[0]}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
    x      = convBlock(x, {channels[0], channels[1]}, 1);
    x      = convBlock(x, {channels[1], channels[2]}, 2);
    x      = convBlock(x, {channels[2], channels[2]}, 1);
    x      = convBlock(x, {channels[2], channels[3]}, 2);
    x      = convBlock(x, {channels[3], channels[3]}, 1);
    x      = convBlock(x, {channels[3], channels[4]}, 2);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[4]}, 1);
    x      = convBlock(x, {channels[4], channels[5]}, 2);
    x      = convBlock(x, {channels[5], channels[5]}, 1);
    x      = _AvePool(x, {poolSize, poolSize}, {1, 1}, VALID);
    x      = _Conv(0.0f, 0.0f, x, {channels[5], 1001}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
    x      = _Softmax(x, -1);
    return x;
}
class MemoryIncreaseMobileNetV1Test : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto y = _mobileNetV1Expr();
        std::unique_ptr<MNN::NetT> net(new NetT);
        Variable::save({y}, net.get());
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, net.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> interp(Interpreter::createFromBuffer(bufferOutput, sizeOutput));
        ScheduleConfig config;
        config.type      = MNN_FORWARD_CPU;
        auto session     = interp->createSession(config);
        auto input       = interp->getSessionInput(session, nullptr);
        float initMemory = 0.0f;
        interp->getSessionInfo(session, MNN::Interpreter::MEMORY, &initMemory);
        for (int i = 0; i < 100; ++i) {
            if (i % 2 == 0) {
                interp->resizeTensor(input, {1, 3, 112, 112});
            } else {
                interp->resizeTensor(input, {1, 3, 224, 224});
            }
            interp->resizeSession(session);
        }
        float lastMemory = 0.0f;
        interp->getSessionInfo(session, MNN::Interpreter::MEMORY, &lastMemory);
        MNN_PRINT("From init %f mb to %f mb\n", initMemory, lastMemory);
        if (lastMemory > initMemory) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MemoryIncreaseMobileNetV1Test, "expr/MemoryIncrease/mobilenetv1");

class MemoryIncreaseInterpTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x = _Input({1, 3, 224, 224}, NCHW, halide_type_of<float>());
        auto y = _Interp({x}, 0.25, 0.25, 56, 56, 2, true);
        y = _Convert(y, NCHW);
        auto size = y->getInfo()->size;
        int e = 14;
        y = _Reshape(y, {e, -1});
        int l = size / e;
        VARP res;
        {
            std::unique_ptr<OpT> mat(new OpT);
            mat->type = OpType_MatMul;
            mat->main.type = OpParameter_MatMul;
            mat->main.value = new MatMulT;
            mat->main.AsMatMul()->transposeA = false;
            mat->main.AsMatMul()->transposeB = false;

            std::vector<float> bias(e, 0.0f);
            auto biasVar = _Const(bias.data(), {e}, NCHW, halide_type_of<float>());
            auto weightVar = _Input({l, 50}, NCHW, halide_type_of<float>());
            res = Variable::create(Expr::create(mat.get(), {y, weightVar, biasVar}));
        }
        std::unique_ptr<MNN::NetT> net(new NetT);
        Variable::save({res}, net.get());
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, net.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> interp(Interpreter::createFromBuffer(bufferOutput, sizeOutput));
        ScheduleConfig config;
        config.type      = MNN_FORWARD_CPU;
        auto session     = interp->createSession(config);
        auto input       = interp->getSessionInput(session, nullptr);

        {
            interp->resizeTensor(input, {1, 3, 112, 112});
            interp->resizeSession(session);
            interp->resizeTensor(input, {1, 3, 224, 224});
            interp->resizeSession(session);
        }
        float initMemory = 0.0f;
        interp->getSessionInfo(session, MNN::Interpreter::MEMORY, &initMemory);

        for (int i = 0; i < 100; ++i) {
            if (i % 2 == 0) {
                interp->resizeTensor(input, {1, 3, 112, 112});
            } else {
                interp->resizeTensor(input, {1, 3, 224, 224});
            }
            interp->resizeSession(session);
        }
        float lastMemory = 0.0f;
        interp->getSessionInfo(session, MNN::Interpreter::MEMORY, &lastMemory);
        MNN_PRINT("From init %f mb to %f mb\n", initMemory, lastMemory);
        if (lastMemory > initMemory) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MemoryIncreaseInterpTest, "expr/MemoryIncrease/interp");

class MidOutputTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x = _Input({100}, NCHW, halide_type_of<float>());
        auto y = x * x;
        std::string midName = "midTensor";
        y->setName(midName);
        auto z = _Exp(y);
        z = _Sqrt(z);
        z = _Abs(z);
        std::unique_ptr<MNN::NetT> net(new NetT);
        Variable::save({z}, net.get());
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, net.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> interp(Interpreter::createFromBuffer(bufferOutput, sizeOutput));
        ScheduleConfig config;
        config.type      = MNN_FORWARD_CPU;
        config.saveTensors = {midName};
        auto session     = interp->createSession(config);
        auto input       = interp->getSessionInput(session, nullptr);
        auto output = interp->getSessionOutput(session, midName.c_str());
        std::vector<float> inputValues(100);
        for (int i=0; i<100; ++i) {
            inputValues[i] = (float)i;
        }
        ::memcpy(input->host<void>(), inputValues.data(), 100 * sizeof(float));
        interp->runSession(session);
        auto outputPtr = output->host<float>();
        for (int i=0; i<100; ++i) {
            auto diff = outputPtr[i] - inputValues[i] * inputValues[i];
            if (diff < 0) {
                diff = -diff;
            }
            if (diff > 0.1f) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(MidOutputTest, "expr/MidOutputTest");

class ConstFoldMemoryTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BackendConfig bnConfig;
        auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
        ExecutorScope scope(exe);
        Module::Config config;
        config.shapeMutable = true;
        config.rearrange = true;
        std::vector<int8_t> buffer;
        {
            // Make Buffer
            auto x0 = _Input({1}, NCHW, halide_type_of<float>());
            x0->setName("x0");
            auto x1 = _Const(1.0f, {256, 1024}, NCHW);
            x1 = x1 * x1 * _Cos(x1) * _Sin(x1);
            auto y0 = x0 * x1;
            y0->setName("y0");
            buffer = Variable::save({y0});
        }
        auto rtInfo = Express::ExecutorScope::Current()->getRuntime();
        auto rt = rtInfo.first.begin()->second;
        MNN::ScheduleConfig sconfig;
        std::vector<MNN::ScheduleConfig> sconfigs = {sconfig};
        std::shared_ptr<Executor::RuntimeManager> rtMgr(Executor::RuntimeManager::createRuntimeManager(sconfigs));
        rtMgr->setMode(Interpreter::Session_Memory_Collect);
        std::shared_ptr<MNN::Express::Module> m0(Module::load({"x0"}, {"y0"}, (const unsigned char*)buffer.data(), buffer.size(), rtMgr, &config), Module::destroy);
        std::shared_ptr<MNN::Express::Module> m1(Module::load({"x0"}, {"y0"}, (const unsigned char*)buffer.data(), buffer.size(), rtMgr, &config), Module::destroy);
        float memoryInit = 0.0f;
        rtMgr->getInfo(Interpreter::MEMORY, &memoryInit);
        FUNC_PRINT_ALL(memoryInit, f);
        auto x = _Input({1}, NCHW, halide_type_of<float>());
        x->writeMap<float>();
        x->unMap();
        float memoryCurrent = 0.0f;
        auto compute = [&](){
            m0->onForward({x});
            rtMgr->getInfo(Interpreter::MEMORY, &memoryCurrent);
            auto static0 = memoryCurrent - memoryInit;
            FUNC_PRINT_ALL(static0, f);
            if (static0 > 2.1f) {
                MNN_ERROR("Constant folder Memory too large\n");
                return false;
            }
            memoryInit = memoryCurrent;
            m1->traceOrOptimize(Interpreter::Session_Resize_Check);
            m1->onForward({x});
            rtMgr->getInfo(Interpreter::MEMORY, &memoryCurrent);
            auto static1 = memoryCurrent - memoryInit;
            FUNC_PRINT_ALL(static1, f);
            if (static1 <= static0) {
                MNN_ERROR("Check mod the memory should be larger than init mode\n");
                return false;
            }
            m1->traceOrOptimize(Interpreter::Session_Resize_Fix);
            m1->onForward({x});
            rtMgr->getInfo(Interpreter::MEMORY, &memoryCurrent);
            auto static2 = memoryCurrent - memoryInit;
            FUNC_PRINT_ALL(static2, f);
            if (static2 >= static1) {
                MNN_ERROR("TODO: Fix mod the memory should be less than check mode\n");
            }
            return true;
        };
        bool res = compute();
        if (!res) {
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ConstFoldMemoryTest, "expr/ConstFoldMemoryTest");
