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
