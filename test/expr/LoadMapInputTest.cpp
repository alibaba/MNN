//
//  LoadMapInputTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"
#include "MNNTestSuite.h"

using namespace MNN::Express;

// Regression test for #4731: Variable::loadMap input tensor lost its host buffer,
// making writeMap() return NULL and downstream format conversion crash with a
// null source pointer (e.g. expressDemo SIGSEGV on NCHW-input models).
class LoadMapInputWriteMapTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Build a tiny NCHW-input model (Conv3x3 + ReLU) and save it.
        auto x = _Input({1, 3, 8, 8}, NCHW);
        std::vector<float> weight(4 * 3 * 3 * 3, 0.1f);
        std::vector<float> bias(4, 0.01f);
        auto w = _Const(weight.data(), {4, 3, 3, 3}, NCHW);
        auto b = _Const(bias.data(), {4}, NCHW);
        auto y = _Relu(_Conv(w, b, x));
        Variable::save({y}, "regression_4731.mnn");

        // Load the model and check the input VARP is writable.
        auto varMap = Variable::loadMap("regression_4731.mnn");
        auto io = Variable::getInputAndOutput(varMap);
        if (io.first.empty() || io.second.empty()) {
            MNN_PRINT("LoadMapInputTest: no input/output found\n");
            return false;
        }
        auto input = io.first.begin()->second;
        auto ptr = input->writeMap<float>();
        if (nullptr == ptr) {
            // Before the fix this was NULL (input tensor host was dropped by
            // Tensor::clone in Variable::load), causing SIGSEGV downstream.
            MNN_PRINT("LoadMapInputTest: writeMap returned NULL (bug #4731)\n");
            return false;
        }
        auto inInfo = input->getInfo();
        int size = 1;
        for (auto d : inInfo->dim) {
            size *= d;
        }
        for (int i = 0; i < size; ++i) {
            ptr[i] = 0.5f;
        }
        // Forward must not crash (compute succeeds; output reading is a separate
        // follow-up concern, see issue #4731).
        auto output = io.second.begin()->second;
        (void)output->readMap<float>();
        return true;
    }
};
MNNTestSuiteRegister(LoadMapInputWriteMapTest, "expr/LoadMapInputWriteMap");

// Regression test for #4750: reading the loadMap output via readMap() returned
// NULL because Variable::load replaced the output tensor with a Tensor::clone
// whose shared describe carried memoryType=MEMORY_BACKEND, so mapOutput's copy
// branch could not allocate a host buffer.
class LoadMapOutputReadMapTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Build a tiny NCHW-input model (Conv3x3 + ReLU) and save it.
        auto x = _Input({1, 3, 8, 8}, NCHW);
        std::vector<float> weight(4 * 3 * 3 * 3, 0.1f);
        std::vector<float> bias(4, 0.01f);
        auto w = _Const(weight.data(), {4, 3, 3, 3}, NCHW);
        auto b = _Const(bias.data(), {4}, NCHW);
        auto y = _Relu(_Conv(w, b, x));
        MNNCreateDir("tmp");
        Variable::save({y}, "tmp/regression_4750.mnn");

        auto varMap = Variable::loadMap("tmp/regression_4750.mnn");
        auto io = Variable::getInputAndOutput(varMap);
        if (io.first.size() != 1 || io.second.size() != 1) {
            MNN_PRINT("LoadMapOutputTest: expected single input/output, got %zu/%zu\n", io.first.size(),
                      io.second.size());
            return false;
        }
        auto input = io.first.begin()->second;
        auto output = io.second.begin()->second;

        auto ptr = input->writeMap<float>();
        if (nullptr == ptr) {
            MNN_PRINT("LoadMapOutputTest: writeMap returned NULL\n");
            return false;
        }
        auto inInfo = input->getInfo();
        int size = 1;
        for (auto d : inInfo->dim) {
            size *= d;
        }
        for (int i = 0; i < size; ++i) {
            ptr[i] = 0.5f;
        }

        auto out = output->readMap<float>();
        if (nullptr == out) {
            // Before the fix this was NULL (output tensor memoryType was
            // overwritten to MEMORY_BACKEND by Tensor::clone in Variable::load).
            MNN_PRINT("LoadMapOutputTest: readMap returned NULL (bug #4750)\n");
            return false;
        }
        // Expected first value: 27 * 0.1 * 0.5 + 0.01 = 1.36. FP16 stores
        // 0.1 approximately and accumulates the rounding error across the convolution.
        constexpr float expected = 1.36f;
        const float tolerance = precision == BackendConfig::Precision_Low ? 0.005f : 0.001f;
        auto val = out[0];
        if (val < expected - tolerance || val > expected + tolerance) {
            MNN_PRINT("LoadMapOutputTest: unexpected output value %f (expected %f +/- %f)\n", val, expected, tolerance);
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(LoadMapOutputReadMapTest, "expr/LoadMapOutputReadMap");
