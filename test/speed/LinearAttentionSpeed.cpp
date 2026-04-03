//
//  LinearAttentionSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2026/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <cmath>
#include <cstring>
#include <vector>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;

static void fillRandom(float* data, int size, float scale = 0.1f) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((i % 17) - 8) * scale;
    }
}

static void fillGate(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = -0.1f * ((i % 5) + 1);
    }
}

static void fillBeta(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0.1f * ((i % 9) + 1);
    }
}

static std::shared_ptr<Module> _makeModule(
    int numKHeads, int numVHeads, int headKDim, int headVDim, bool useL2Norm, int numThread = 1)
{
    auto qkv   = _Input();
    auto gate   = _Input();
    auto beta   = _Input();
    auto convW  = _Input();

    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    auto* param = op->main.AsLinearAttentionParam();
    param->attn_type     = "gated_delta_rule";
    param->num_k_heads   = numKHeads;
    param->num_v_heads   = numVHeads;
    param->head_k_dim    = headKDim;
    param->head_v_dim    = headVDim;
    param->use_qk_l2norm = useL2Norm;

    auto o = Variable::create(Expr::create(op.get(), {qkv, gate, beta, convW}));
    auto buffer = Variable::save({o});

    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bnConfig;
    bnConfig.memory    = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power     = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    config.numThread = numThread;

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

class LinearAttentionSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) {
        struct TestCase {
            int numKHeads, numVHeads, headKDim, headVDim, seqLen, K_conv;
            const char* name;
        };

        std::vector<TestCase> cases = {
            // Decode scenarios (L=1) - most common in LLM inference
            {4, 4, 64, 64, 1, 4, "decode_H4_d64"},
            {16, 16, 64, 64, 1, 4, "decode_H16_d64"},
            {4, 4, 128, 128, 1, 4, "decode_H4_d128"},
            {16, 16, 128, 128, 1, 4, "decode_H16_d128"},
            // Prefill scenarios
            {4, 4, 64, 64, 16, 4, "prefill16_H4_d64"},
            {16, 16, 64, 64, 16, 4, "prefill16_H16_d64"},
            {4, 4, 64, 64, 64, 4, "prefill64_H4_d64"},
            {16, 16, 64, 64, 64, 4, "prefill64_H16_d64"},
            {4, 4, 64, 64, 128, 4, "prefill128_H4_d64"},
            {16, 16, 64, 64, 128, 4, "prefill128_H16_d64"},
            {4, 4, 64, 64, 2048, 4, "prefill2048_H4_d64"},
            {16, 16, 64, 64, 2048, 4, "prefill2048_H16_d64"},
        };

        const int B = 1;
        const bool useL2Norm = true;
        const int numThread = 4;
        const int warmup = 5;
        const int repeat = 20;

        for (auto& tc : cases) {
            int key_dim = tc.numKHeads * tc.headKDim;
            int val_dim = tc.numVHeads * tc.headVDim;
            int D = 2 * key_dim + val_dim;

            auto module = _makeModule(tc.numKHeads, tc.numVHeads, tc.headKDim, tc.headVDim, useL2Norm, numThread);
            if (!module) {
                MNN_PRINT("Error: failed to create module for %s\n", tc.name);
                return false;
            }

            auto qkvVar  = _Input({B, D, tc.seqLen}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, tc.seqLen, tc.numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, tc.seqLen, tc.numVHeads}, NCHW, halide_type_of<float>());
            auto convWVar = _Input({D, 1, tc.K_conv}, NCHW, halide_type_of<float>());

            fillRandom(qkvVar->writeMap<float>(), B * D * tc.seqLen);
            fillGate(gateVar->writeMap<float>(), B * tc.seqLen * tc.numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * tc.seqLen * tc.numVHeads);
            fillRandom(convWVar->writeMap<float>(), D * 1 * tc.K_conv, 0.05f);

            // Warmup
            for (int t = 0; t < warmup; ++t) {
                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: empty output for %s\n", tc.name);
                    return false;
                }
            }

            // Benchmark
            MNN::Timer _t;
            for (int t = 0; t < repeat; ++t) {
                module->onForward({qkvVar, gateVar, betaVar, convWVar});
            }
            float avgMs = _t.durationInUs() / 1000.0f / (float)repeat;

            MNN_PRINT("[%s] B=%d H=%d dk=%d dv=%d L=%d, Avg: %.3f ms\n",
                      tc.name, B, tc.numVHeads, tc.headKDim, tc.headVDim, tc.seqLen, avgMs);
        }

        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionSpeed, "speed/LinearAttentionSpeed");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
