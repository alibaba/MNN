//
//  CheckNAN.cpp
//  MNNTests
//
//  Created by MNN on b'2020/08/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef _MSC_VER
#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include <thread>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <math.h>

using namespace MNN::Express;
using namespace MNN;

class CheckNANTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        // Make Model Buffer
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        {
            auto x = _Input({1, 3, 56, 56}, NCHW, halide_type_of<float>());
            x->setName("x");
            auto size = x->getInfo()->size;
            int e = 14;
            auto y = _Reshape(x, {e, -1});
            auto l = size / e;
            VARP res;
            {
                std::unique_ptr<OpT> mat(new OpT);
                mat->type = OpType_MatMul;
                mat->main.type = OpParameter_MatMul;
                mat->main.value = new MatMulT;
                mat->main.AsMatMul()->transposeA = false;
                mat->main.AsMatMul()->transposeB = false;

                std::vector<float> bias(50, 0.0f);
                auto biasVar = _Const(bias.data(), {50}, NCHW, halide_type_of<float>());
                std::vector<float> weight(l * 50, 0.0f);
                auto weightVar = _Const(weight.data(), {l, 50}, NCHW, halide_type_of<float>());
                res = Variable::create(Expr::create(mat.get(), {y, weightVar, biasVar}));
            }
            res->setName("y");
            std::unique_ptr<MNN::NetT> net(new NetT);
            Variable::save({res}, net.get());
            auto len = MNN::Net::Pack(builderOutput, net.get());
            builderOutput.Finish(len);
        }
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::shared_ptr<Interpreter> interp(Interpreter::createFromBuffer(bufferOutput, sizeOutput));
        ScheduleConfig config;
        config.type      = MNN_FORWARD_CPU;
        auto session     = interp->createSession(config);
        auto input       = interp->getSessionInput(session, nullptr);
        auto code = interp->runSession(session);
        if (NO_ERROR != code) {
            return false;
        }
        BackendConfig bnConfig;
        // Set 1 to check NAN
        bnConfig.flags = 1;
        ScheduleConfig config2;
        config2.type      = MNN_FORWARD_CPU;
        config2.backendConfig = &bnConfig;

        auto session2 = interp->createSession(config2);
        auto inputTensor = interp->getSessionInput(session2, nullptr);
        auto tensorSize = inputTensor->elementSize();
        for (int i=0; i<tensorSize; ++i) {
            inputTensor->host<float>()[i] = 1.0f / 0.0f;
        }
        code = interp->runSession(session2);
        if (NO_ERROR == code) {
            MNN_ERROR("CheckNAN for Basic(Session) not valid\n");
            return false;
        }
        std::shared_ptr<MNN::Express::Executor> executor(Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1));
        MNN::Express::ExecutorScope scope(executor);
        std::shared_ptr<MNN::Express::Module> module(MNN::Express::Module::load({"x"}, {"y"}, bufferOutput, sizeOutput));
        {
            auto x = _Input({1, 3, 56, 56}, NCHW, halide_type_of<float>());
            ::memset(x->writeMap<float>(), 0, x->getInfo()->size * sizeof(float));
            auto y = module->onForward({x});
            if (y.empty()) {
                MNN_ERROR("Normal input error in checknan\n");
                return false;
            }
            x = _Input({1, 3, 56, 56}, NCHW, halide_type_of<float>());
            for (int i=0; i<x->getInfo()->size; ++i) {
                x->writeMap<float>()[i] = 1.0f / 0.0f;
            }
            y = module->onForward({x});
            if (!y.empty()) {
                MNN_ERROR("CheckNAN for Module not valid\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(CheckNANTest, "expr/CheckNAN");
#endif
