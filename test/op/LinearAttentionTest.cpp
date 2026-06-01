//
//  LinearAttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sys/stat.h>

using namespace MNN::Express;

// ─── Naive reference implementation of the Gated Delta Rule ───
// This function replicates the full pipeline: Conv1D + SiLU → Split QKV → GQA → L2Norm → Gated Delta Rule
// All shapes follow the convention used in CPULinearAttention.cpp.

struct NaiveLinearAttention {
    // Conv1D state: [B, D, convStateSize]
    std::vector<float> convState;
    // Recurrent state S: [B, H, d_k, d_v]
    std::vector<float> rnnState;
    bool initialized = false;

    int B, D, convStateSize, H, dk, dv;

    void init(int batch, int convDim, int convKernel, int numVHeads, int headKDim, int headVDim) {
        B = batch;
        D = convDim;
        convStateSize = convKernel - 1;
        H = numVHeads;
        dk = headKDim;
        dv = headVDim;
        convState.assign(B * D * convStateSize, 0.0f);
        rnnState.assign(B * H * dk * dv, 0.0f);
        initialized = true;
    }

    // qkv: [B, D, L], gate: [B, L, H], beta: [B, L, H], convW: [D, 1, K]
    // output: [B, L, H_v, d_v]
    std::vector<float> forward(
        const float* qkvPtr, const float* gatePtr, const float* betaPtr, const float* convWPtr,
        int batch, int L, int convDim, int K_conv,
        int numKHeads, int numVHeads, int headKDim, int headVDim,
        bool useL2Norm)
    {
        const int key_dim = numKHeads * headKDim;
        const int val_dim = numVHeads * headVDim;
        const int gqa_factor = (numVHeads > numKHeads) ? (numVHeads / numKHeads) : 1;
        const int HH = numVHeads;
        const int ddk = headKDim;
        const int ddv = headVDim;

        // Step 1: Build conv_input = cat(convState, qkv) along dim L
        const int totalLen = convStateSize + L;
        std::vector<float> convInput(B * D * totalLen, 0.0f);

        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                float* dst = convInput.data() + b * D * totalLen + d * totalLen;
                const float* stateChannel = convState.data() + b * D * convStateSize + d * convStateSize;
                ::memcpy(dst, stateChannel, convStateSize * sizeof(float));
                const float* inputChannel = qkvPtr + b * D * L + d * L;
                ::memcpy(dst + convStateSize, inputChannel, L * sizeof(float));
            }
        }

        // Depthwise Conv1D padding=0, output length = L
        std::vector<float> convOut(B * D * L, 0.0f);
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                const float* src = convInput.data() + b * D * totalLen + d * totalLen;
                const float* weight = convWPtr + d * K_conv;
                float* out = convOut.data() + b * D * L + d * L;
                for (int l = 0; l < L; ++l) {
                    float sum = 0.0f;
                    for (int k = 0; k < K_conv; ++k) {
                        sum += src[l + k] * weight[k];
                    }
                    float sigmoid_val = 1.0f / (1.0f + expf(-sum));
                    out[l] = sum * sigmoid_val;
                }
            }
        }

        // Update convState
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                const float* src = convInput.data() + b * D * totalLen + d * totalLen + (totalLen - convStateSize);
                float* dst = convState.data() + b * D * convStateSize + d * convStateSize;
                ::memcpy(dst, src, convStateSize * sizeof(float));
            }
        }

        // Step 2: Split Q, K, V with GQA expansion
        std::vector<float> Q(B * L * HH * ddk, 0.0f);
        std::vector<float> K(B * L * HH * ddk, 0.0f);
        std::vector<float> V(B * L * HH * ddv, 0.0f);

        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                for (int h = 0; h < numKHeads; ++h) {
                    for (int di = 0; di < ddk; ++di) {
                        int srcChannel = h * ddk + di;
                        float val = convOut[b * D * L + srcChannel * L + l];
                        for (int r = 0; r < gqa_factor; ++r) {
                            int dstHead = h * gqa_factor + r;
                            Q[(b * L + l) * HH * ddk + dstHead * ddk + di] = val;
                        }
                    }
                }
                for (int h = 0; h < numKHeads; ++h) {
                    for (int di = 0; di < ddk; ++di) {
                        int srcChannel = key_dim + h * ddk + di;
                        float val = convOut[b * D * L + srcChannel * L + l];
                        for (int r = 0; r < gqa_factor; ++r) {
                            int dstHead = h * gqa_factor + r;
                            K[(b * L + l) * HH * ddk + dstHead * ddk + di] = val;
                        }
                    }
                }
                for (int h = 0; h < numVHeads; ++h) {
                    for (int di = 0; di < ddv; ++di) {
                        int srcChannel = 2 * key_dim + h * ddv + di;
                        float val = convOut[b * D * L + srcChannel * L + l];
                        V[(b * L + l) * HH * ddv + h * ddv + di] = val;
                    }
                }
            }
        }

        // Step 3: L2 Norm
        if (useL2Norm) {
            const float eps = 1e-6f;
            for (int i = 0; i < B * L * HH; ++i) {
                float* qHead = Q.data() + i * ddk;
                float sumSq = 0.0f;
                for (int di = 0; di < ddk; ++di) sumSq += qHead[di] * qHead[di];
                float invNorm = 1.0f / sqrtf(sumSq + eps);
                for (int di = 0; di < ddk; ++di) qHead[di] *= invNorm;

                float* kHead = K.data() + i * ddk;
                sumSq = 0.0f;
                for (int di = 0; di < ddk; ++di) sumSq += kHead[di] * kHead[di];
                invNorm = 1.0f / sqrtf(sumSq + eps);
                for (int di = 0; di < ddk; ++di) kHead[di] *= invNorm;
            }
        }

        // Step 4: Scale Q
        const float qScale = 1.0f / sqrtf((float)ddk);
        for (int i = 0; i < B * L * HH * ddk; ++i) {
            Q[i] *= qScale;
        }

        // Step 5: Gated Delta Rule with persistent state
        std::vector<float> output(B * L * HH * ddv, 0.0f);

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < L; ++t) {
                for (int h = 0; h < HH; ++h) {
                    float* state = rnnState.data() + (b * HH + h) * ddk * ddv;

                    const float* q_t = Q.data() + (b * L + t) * HH * ddk + h * ddk;
                    const float* k_t = K.data() + (b * L + t) * HH * ddk + h * ddk;
                    const float* v_t = V.data() + (b * L + t) * HH * ddv + h * ddv;
                    float g_t = gatePtr[b * L * HH + t * HH + h];
                    float beta_t = betaPtr[b * L * HH + t * HH + h];

                    // Decay
                    float decay = expf(g_t);
                    for (int i = 0; i < ddk * ddv; ++i) state[i] *= decay;

                    // Read: v_pred = S^T @ k_t
                    std::vector<float> v_pred(ddv, 0.0f);
                    for (int di = 0; di < ddk; ++di) {
                        for (int dj = 0; dj < ddv; ++dj) {
                            v_pred[dj] += state[di * ddv + dj] * k_t[di];
                        }
                    }

                    // Delta
                    std::vector<float> delta(ddv);
                    for (int dj = 0; dj < ddv; ++dj) {
                        delta[dj] = beta_t * (v_t[dj] - v_pred[dj]);
                    }

                    // Write: S += k_t @ delta^T
                    for (int di = 0; di < ddk; ++di) {
                        for (int dj = 0; dj < ddv; ++dj) {
                            state[di * ddv + dj] += k_t[di] * delta[dj];
                        }
                    }

                    // Query: o_t = S^T @ q_t
                    float* o_t = output.data() + (b * L + t) * HH * ddv + h * ddv;
                    for (int dj = 0; dj < ddv; ++dj) {
                        float sum = 0.0f;
                        for (int di = 0; di < ddk; ++di) {
                            sum += state[di * ddv + dj] * q_t[di];
                        }
                        o_t[dj] = sum;
                    }
                }
            }
        }

        return output;
    }
};

// ─── Helper: create a LinearAttention Module via FlatBuffers ───
static std::shared_ptr<Module> _makeLinearAttentionModule(
    int numKHeads, int numVHeads, int headKDim, int headVDim, bool useL2Norm,
    const std::string& attnType = "gated_delta_rule")
{
    auto qkv      = _Input();
    auto gate     = _Input();
    auto beta     = _Input();
    auto convW    = _Input();

    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    auto* param = op->main.AsLinearAttentionParam();
    param->attn_type    = attnType;
    param->num_k_heads  = numKHeads;
    param->num_v_heads  = numVHeads;
    param->head_k_dim   = headKDim;
    param->head_v_dim   = headVDim;
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
    config.numThread = 1;

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

// ─── Helper: generate deterministic float data ───
static void fillDeterministic(float* data, int size, float scale = 0.1f, float offset = 0.0f) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((i % 17) - 8) * scale + offset;
    }
}

// ─── Helper: generate conv weight (small values so conv output stays reasonable) ───
static void fillConvWeight(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((i % 7) - 3) * 0.05f;
    }
}

// ─── Helper: generate gate values (negative, for exp decay < 1) ───
static void fillGate(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = -0.1f * ((i % 5) + 1);  // range [-0.1, -0.5]
    }
}

// ─── Helper: generate beta values (learning rate in [0, 1]) ───
static void fillBeta(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0.1f * ((i % 9) + 1);  // range [0.1, 0.9]
    }
}

// ─── Test class ───
class LinearAttentionTest : public MNNTestCase {
public:
    LinearAttentionTest() = default;
    virtual ~LinearAttentionTest() = default;

    virtual bool run(int precision) {
        // Test parameters
        const int B = 1;
        const int numKHeads = 2;
        const int numVHeads = 2;
        const int headKDim  = 4;
        const int headVDim  = 4;
        const int K_conv    = 4;  // conv kernel size
        const int key_dim   = numKHeads * headKDim;
        const int val_dim   = numVHeads * headVDim;
        const int D         = 2 * key_dim + val_dim;  // conv_dim
        const bool useL2Norm = true;
        const float tolerance = 0.001f;

        // ─── Test 1: Prefill (seq_len > 1) ───
        {
            const int L = 4;

            // Create Module
            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, useL2Norm);
            if (!module) {
                MNN_PRINT("Error: Failed to create LinearAttention module\n");
                return false;
            }

            // Prepare inputs
            auto qkvVar  = _Input({B, D, L}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());

            fillDeterministic(qkvVar->writeMap<float>(),  B * D * L, 0.1f);
            fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            // Naive reference
            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);
            auto expected = naive.forward(
                qkvVar->readMap<float>(), gateVar->readMap<float>(),
                betaVar->readMap<float>(), convWVar->readMap<float>(),
                B, L, D, K_conv, numKHeads, numVHeads, headKDim, headVDim, useL2Norm);

            // Run MNN op
            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("Error: LinearAttention module returned empty output\n");
                return false;
            }
            auto output = outputs[0];
            const float* resultPtr = output->readMap<float>();
            const int outSize = B * L * numVHeads * headVDim;

            // Compare
            for (int i = 0; i < outSize; ++i) {
                float diff = fabs(resultPtr[i] - expected[i]);
                if (diff > tolerance) {
                    MNN_PRINT("Prefill Test FAILED at index %d: expected %.6f, got %.6f (diff=%.6f)\n",
                              i, expected[i], resultPtr[i], diff);
                    return false;
                }
            }
            MNN_PRINT("LinearAttention Prefill Test (L=%d) PASSED\n", L);
        }

        // ─── Test 2: Multi-step decode (seq_len = 1, tests state persistence) ───
        {
            const int decodeSteps = 4;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, useL2Norm);
            if (!module) {
                MNN_PRINT("Error: Failed to create LinearAttention module for decode test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;

                auto qkvVar  = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                // Slightly different input per step
                fillDeterministic(qkvVar->writeMap<float>(),  B * D * L, 0.1f, 0.01f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                // Naive reference
                auto expected = naive.forward(
                    qkvVar->readMap<float>(), gateVar->readMap<float>(),
                    betaVar->readMap<float>(), convWVar->readMap<float>(),
                    B, L, D, K_conv, numKHeads, numVHeads, headKDim, headVDim, useL2Norm);

                // Run MNN op
                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode step %d returned empty output\n", step);
                    return false;
                }
                auto output = outputs[0];
                const float* resultPtr = output->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Decode Test FAILED at step %d, index %d: expected %.6f, got %.6f (diff=%.6f)\n",
                                  step, i, expected[i], resultPtr[i], diff);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Multi-step Decode Test (%d steps) PASSED\n", decodeSteps);
        }

        // ─── Test 3: Without L2 Normalization ───
        {
            const int L = 3;
            const bool noL2Norm = false;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, noL2Norm);
            if (!module) {
                MNN_PRINT("Error: Failed to create LinearAttention module (no L2)\n");
                return false;
            }

            auto qkvVar   = _Input({B, D, L}, NCHW, halide_type_of<float>());
            auto gateVar  = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar  = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());

            fillDeterministic(qkvVar->writeMap<float>(),  B * D * L, 0.05f);
            fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);
            auto expected = naive.forward(
                qkvVar->readMap<float>(), gateVar->readMap<float>(),
                betaVar->readMap<float>(), convWVar->readMap<float>(),
                B, L, D, K_conv, numKHeads, numVHeads, headKDim, headVDim, noL2Norm);

            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("Error: LinearAttention module (no L2) returned empty output\n");
                return false;
            }
            auto output = outputs[0];
            const float* resultPtr = output->readMap<float>();
            const int outSize = B * L * numVHeads * headVDim;

            for (int i = 0; i < outSize; ++i) {
                float diff = fabs(resultPtr[i] - expected[i]);
                if (diff > tolerance) {
                    MNN_PRINT("No-L2Norm Test FAILED at index %d: expected %.6f, got %.6f (diff=%.6f)\n",
                              i, expected[i], resultPtr[i], diff);
                    return false;
                }
            }
            MNN_PRINT("LinearAttention No-L2Norm Test (L=%d) PASSED\n", L);
        }

        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionTest, "op/linear_attention");

// ─── Decode fast path test: focuses on L=1 correctness and state consistency ───
class LinearAttentionDecodeTest : public MNNTestCase {
public:
    LinearAttentionDecodeTest() = default;
    virtual ~LinearAttentionDecodeTest() = default;

    virtual bool run(int precision) {
        const float tolerance = 0.001f;

        // ─── Test 1: Single decode step (L=1) basic correctness ───
        {
            const int B = 1, numKHeads = 2, numVHeads = 2;
            const int headKDim = 8, headVDim = 8, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int L = 1;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for decode single step test\n");
                return false;
            }

            auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());

            fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.1f);
            fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);
            auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                          betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                          numKHeads, numVHeads, headKDim, headVDim, true);

            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("Error: Decode single step returned empty output\n");
                return false;
            }
            const float* resultPtr = outputs[0]->readMap<float>();
            const int outSize = B * L * numVHeads * headVDim;

            for (int i = 0; i < outSize; ++i) {
                float diff = fabs(resultPtr[i] - expected[i]);
                if (diff > tolerance) {
                    MNN_PRINT("Decode single step FAILED at index %d: expected %.6f, got %.6f (diff=%.6f)\n", i,
                              expected[i], resultPtr[i], diff);
                    return false;
                }
            }
            MNN_PRINT("LinearAttention Decode single step (dk=%d, dv=%d) PASSED\n", headKDim, headVDim);
        }

        // ─── Test 2: Prefill then multi-step decode (state continuity) ───
        {
            const int B = 1, numKHeads = 2, numVHeads = 2;
            const int headKDim = 4, headVDim = 4, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int prefillLen = 3;
            const int decodeSteps = 6;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for prefill+decode test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            // Prefill phase
            {
                auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.08f, 0.02f);
                fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, prefillLen, D,
                                              K_conv, numKHeads, numVHeads, headKDim, headVDim, true);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Prefill phase returned empty output\n");
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * prefillLen * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Prefill+Decode: Prefill FAILED at index %d: expected %.6f, got %.6f\n", i,
                                  expected[i], resultPtr[i]);
                        return false;
                    }
                }
            }

            // Decode phase (L=1 per step, state should carry over from prefill)
            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;
                auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.1f, 0.03f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, true);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Prefill+Decode: Decode step %d FAILED at index %d: expected %.6f, got %.6f\n", step,
                                  i, expected[i], resultPtr[i]);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Prefill(%d)+Decode(%d steps) state continuity PASSED\n", prefillLen,
                      decodeSteps);
        }

        // ─── Test 3: Decode without L2 Norm ───
        {
            const int B = 1, numKHeads = 2, numVHeads = 2;
            const int headKDim = 4, headVDim = 4, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int decodeSteps = 4;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, false);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for decode no-L2 test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;
                auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.05f, 0.02f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, false);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode no-L2 step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Decode no-L2 step %d FAILED at index %d: expected %.6f, got %.6f\n", step, i,
                                  expected[i], resultPtr[i]);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Decode no-L2Norm (%d steps) PASSED\n", decodeSteps);
        }

        // ─── Test 4: Decode with GQA (numVHeads > numKHeads) ───
        {
            const int B = 1, numKHeads = 2, numVHeads = 4;
            const int headKDim = 4, headVDim = 4, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int decodeSteps = 3;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for decode GQA test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;
                auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.1f, 0.05f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, true);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode GQA step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Decode GQA step %d FAILED at index %d: expected %.6f, got %.6f\n", step, i,
                                  expected[i], resultPtr[i]);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Decode GQA (H_k=%d, H_v=%d, %d steps) PASSED\n", numKHeads, numVHeads,
                      decodeSteps);
        }

        // ─── Test 5: Decode with larger head dimensions ───
        {
            const int B = 1, numKHeads = 1, numVHeads = 1;
            const int headKDim = 16, headVDim = 16, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int decodeSteps = 3;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for decode large-dim test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;
                auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.08f, 0.01f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, true);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode large-dim step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT("Decode large-dim step %d FAILED at index %d: expected %.6f, got %.6f\n", step, i,
                                  expected[i], resultPtr[i]);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Decode large-dim (dk=%d, dv=%d, %d steps) PASSED\n", headKDim, headVDim,
                      decodeSteps);
        }

        // ─── Test 6: Decode with batch size > 1 ───
        {
            const int B = 3, numKHeads = 2, numVHeads = 2;
            const int headKDim = 4, headVDim = 4, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int decodeSteps = 4;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
            if (!module) {
                MNN_PRINT("Error: Failed to create module for decode batch test\n");
                return false;
            }

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

            for (int step = 0; step < decodeSteps; ++step) {
                const int L = 1;
                auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                auto gateVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
                auto betaVar = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());

                // Use different offsets per step so each batch element gets distinct data
                fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.1f, 0.02f * step);
                fillGate(gateVar->writeMap<float>(), B * L * numVHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numVHeads);

                auto expected = naive.forward(qkvVar->readMap<float>(), gateVar->readMap<float>(),
                                              betaVar->readMap<float>(), convWVar->readMap<float>(), B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, true);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: Decode batch step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numVHeads * headVDim;

                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > tolerance) {
                        MNN_PRINT(
                            "Decode batch(B=%d) step %d FAILED at index %d: expected %.6f, got %.6f (diff=%.6f)\n", B,
                            step, i, expected[i], resultPtr[i], diff);
                        return false;
                    }
                }
            }
            MNN_PRINT("LinearAttention Decode batch (B=%d, %d steps) PASSED\n", B, decodeSteps);
        }

        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionDecodeTest, "op/linear_attention_decode");

// ─── Local mirror of MNN::Transformer::KVMeta (must match layout) ───
// Used to drive the rollback signal mMeta->remove without depending on the
// internal Llm header. See test/op/AttentionTest.cpp for the same pattern.
struct LATestKVMeta {
    enum { NoChange, PendingWrite, PendingRead } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;
};

// Variant of _makeLinearAttentionModule that wires a caller-owned KVMeta
// pointer via the runtime KVCACHE_INFO hint. The CPULinearAttention op reads
// this through backend->getMetaPtr() in its constructor.
static std::shared_ptr<Module> _makeLinearAttentionModuleWithMeta(int numKHeads, int numVHeads, int headKDim,
                                                                  int headVDim, bool useL2Norm, LATestKVMeta* meta) {
    auto qkv = _Input();
    auto gate = _Input();
    auto beta = _Input();
    auto convW = _Input();

    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    auto* param = op->main.AsLinearAttentionParam();
    param->attn_type = "gated_delta_rule";
    param->num_k_heads = numKHeads;
    param->num_v_heads = numVHeads;
    param->head_k_dim = headKDim;
    param->head_v_dim = headVDim;
    param->use_qk_l2norm = useL2Norm;

    auto o = Variable::create(Expr::create(op.get(), {qkv, gate, beta, convW}));
    auto buffer = Variable::save({o});

    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bnConfig;
    bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    config.numThread = 1;

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, meta);
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

// ─── Rollback test: verify the explicit-rollback path in CPULinearAttention::onResize ───
//
// Background: LinearAttention's recurrent state has no token-level structure,
// so Llm::eraseHistory() cannot truncate it. The fix introduces:
//   (a) a snapshot of the post-prefix state taken inside the prefix-cache
//       disk file_flag branches (PendingRead/PendingWrite), and
//   (b) an explicit-rollback branch in onResize that, when mMeta->remove > 0,
//       restores from the snapshot if mSnapshotValid is true, otherwise zeros
//       the state.
//
// This test covers branch (b) without snapshot (mSnapshotValid=false), the
// most common rollback path when prefix cache is not in use:
//
//   Module A: prefill(X)  -> internal state advances  (no snapshot taken,
//                            since file_flag stays NoChange throughout)
//             [simulate Llm: meta.previous = X.len, meta.remove = X.len]
//             prefill(Y)  -> isExplicitRollback fires, mSnapshotValid=false,
//                            so state is zeroed before Y is applied.
//
//   Module B: prefill(Y) on a brand-new module starting from zero state.
//
// Module A's second-prefill output and Module B's only-prefill output must
// match byte-for-byte (within float tolerance), proving:
//   - the rollback branch is hit when remove>0 in prefill,
//   - it correctly clears state to zero when no snapshot is available, and
//   - subsequent computation is equivalent to a fresh-init module.
//
// The "snapshot exists" path (mSnapshotValid=true) requires real prefix-cache
// disk files (.k/.v + setExternalPath(EXTERNAL_PATH_PREFIXCACHE_DIR, ...)) and
// is exercised by rollback_demo Stage 3 against actual model bundles.
class LinearAttentionRollbackTest : public MNNTestCase {
public:
    LinearAttentionRollbackTest() = default;
    virtual ~LinearAttentionRollbackTest() = default;

    virtual bool run(int precision) {
        const int B = 1, numKHeads = 2, numVHeads = 2;
        const int headKDim = 4, headVDim = 4, K_conv = 4;
        const int key_dim = numKHeads * headKDim;
        const int val_dim = numVHeads * headVDim;
        const int D = 2 * key_dim + val_dim;
        const int prefillLen = 4;
        const float tolerance = 0.001f;
        const int outSize = B * prefillLen * numVHeads * headVDim;

        // Shared conv weight across both modules so the only variable is state.
        auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

        // ─── Module A: prefill(X) -> simulate eraseHistory -> prefill(Y) ───
        LATestKVMeta metaA;
        auto moduleA = _makeLinearAttentionModuleWithMeta(numKHeads, numVHeads, headKDim, headVDim, true, &metaA);
        if (!moduleA) {
            MNN_PRINT("RollbackTest: failed to create moduleA\n");
            return false;
        }

        // Step 1: prefill X. metaA.previous=0 here makes onResize treat this as
        // a fresh prefill (zeros state, drops any snapshot — none yet anyway).
        {
            auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.07f, 0.0f);
            fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);
            auto outputs = moduleA->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("RollbackTest: moduleA prefill X returned empty output\n");
                return false;
            }
            // Force evaluation so internal state is fully updated before next call.
            (void)outputs[0]->readMap<float>();
        }

        // Step 2: simulate the meta updates Llm performs around eraseHistory.
        //   - updateContext after prefill X: meta.previous += prefillLen
        //   - eraseHistory(0, previous): meta.remove = previous
        metaA.previous = prefillLen;
        metaA.remove = prefillLen;

        // Step 3: prefill Y. onResize sees remove>0 -> isExplicitRollback;
        // mSnapshotValid=false (no PendingRead/PendingWrite ever fired), so
        // the rollback branch zeros mConvState/mRecurrentState before forward.
        std::vector<float> outputA(outSize, 0.0f);
        {
            auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            // Use distinct input from X so we don't accidentally pass the test
            // when the rollback is silently skipped (state would carry X's effect).
            fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.05f, 0.1f);
            fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);
            auto outputs = moduleA->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("RollbackTest: moduleA prefill Y after rollback returned empty output\n");
                return false;
            }
            const float* p = outputs[0]->readMap<float>();
            ::memcpy(outputA.data(), p, outSize * sizeof(float));
        }

        // ─── Module B: fresh prefill(Y) baseline ───
        LATestKVMeta metaB;
        auto moduleB = _makeLinearAttentionModuleWithMeta(numKHeads, numVHeads, headKDim, headVDim, true, &metaB);
        if (!moduleB) {
            MNN_PRINT("RollbackTest: failed to create moduleB\n");
            return false;
        }
        std::vector<float> outputB(outSize, 0.0f);
        {
            auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            // Identical Y inputs as moduleA's step 3.
            fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.05f, 0.1f);
            fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);
            auto outputs = moduleB->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("RollbackTest: moduleB fresh prefill Y returned empty output\n");
                return false;
            }
            const float* p = outputs[0]->readMap<float>();
            ::memcpy(outputB.data(), p, outSize * sizeof(float));
        }

        // ─── Compare: A's post-rollback prefill must equal B's fresh prefill ───
        for (int i = 0; i < outSize; ++i) {
            float diff = fabs(outputA[i] - outputB[i]);
            if (diff > tolerance) {
                MNN_PRINT(
                    "Rollback (mSnapshotValid=false) FAILED at index %d: "
                    "rollback=%.6f fresh=%.6f diff=%.6f\n",
                    i, outputA[i], outputB[i], diff);
                return false;
            }
        }
        MNN_PRINT("LinearAttention Rollback (no snapshot, state zeroed) PASSED\n");
        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionRollbackTest, "op/linear_attention_rollback");

// ─── Chunked prefix-cache layer_index drift test ───
//
// Background: prefix-cache file naming uses mMeta->layer_index as a counter
// that each layer's onExecute advances by 1, wrapping mod layer_nums. The
// counter is shared between LinearAttention and CPUKVCacheManager (Full
// Attention). CPUKVCacheManager advances it ONLY inside onAlloc, which fires
// on chunk 1 (when mMeta->previous == mMeta->remove); chunks 2..N go through
// onRealloc, which does NOT touch layer_index.
//
// Before the fix, CPULinearAttention advanced layer_index inside its
// PendingWrite/PendingRead branches on EVERY chunk's onExecute. In hybrid
// models (attention_type="mix"), this caused LinearAttention's counter to
// drift past Full Attention's layer positions on chunks 2..N — LA would
// compute the wrong file index and overwrite Full Attention's prefix cache
// .k/.v files, corrupting their live mmap regions and triggering SIGBUS on
// subsequent FA access.
//
// The fix captures layer_index ONCE per session (when previous == remove)
// into mStateCache->mPrefixLayerIndex; subsequent chunks reuse the cached
// value and do NOT touch mMeta->layer_index. This mirrors CPUKVCacheManager's
// once-per-session advancement semantics so the two co-exist correctly.
//
// This test exercises the layer_index lifecycle directly on a single
// LinearAttention op (no FA dependency needed to expose the regression):
//   chunk 1 (previous == remove == 0): expect layer_index to advance by 1.
//   chunk 2 (previous > 0, remove == 0): expect layer_index UNCHANGED.
// A failure here means chunks 2..N would clobber some other layer's file.
class LinearAttentionChunkedLayerIndexTest : public MNNTestCase {
public:
    LinearAttentionChunkedLayerIndexTest() = default;
    virtual ~LinearAttentionChunkedLayerIndexTest() = default;

    virtual bool run(int precision) {
        const int B = 1, numKHeads = 2, numVHeads = 2;
        const int headKDim = 4, headVDim = 4, K_conv = 4;
        const int key_dim = numKHeads * headKDim;
        const int val_dim = numVHeads * headVDim;
        const int D = 2 * key_dim + val_dim;
        const int prefillLen = 4;
        // Starting layer_index value chosen to be non-zero so we can
        // distinguish "no advance" from "reset to zero".
        const int kInitialLayerIndex = 5;
        const int kLayerNums = 24;

        // Shared conv weight across both chunks.
        auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

        // Simulate the meta state at the start of a chunked prefix-cache
        // write session:
        //   - file_name + file_flag=PendingWrite trigger the prefix-cache
        //     write branch in CPULinearAttention::onExecute.
        //   - layer_index = 5 simulates this op being not-first in a multi-
        //     layer forward pass (previous layers' onExecute have already
        //     advanced the counter).
        //   - previous = 0, remove = 0 marks "first chunk" — the per-session
        //     capture-and-advance block should fire on this call only.
        LATestKVMeta meta;
        meta.file_name = "test_chunked_layer_index";
        meta.file_flag = LATestKVMeta::PendingWrite;
        meta.layer_index = kInitialLayerIndex;
        meta.layer_nums = kLayerNums;
        meta.previous = 0;
        meta.remove = 0;

        auto module = _makeLinearAttentionModuleWithMeta(numKHeads, numVHeads, headKDim, headVDim, true, &meta);
        if (!module) {
            MNN_PRINT("ChunkedLayerIndexTest: failed to create module\n");
            return false;
        }

        auto runChunk = [&](float seed_offset) -> bool {
            auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
            auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
            fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.07f, seed_offset);
            fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
            fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);
            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                return false;
            }
            // Force evaluation so onExecute (and its meta-state mutations) runs.
            (void)outputs[0]->readMap<float>();
            return true;
        };

        // ─── Chunk 1: should capture layer_index=5 and advance to 6 ───
        if (!runChunk(0.0f)) {
            MNN_PRINT("ChunkedLayerIndexTest: chunk 1 forward failed\n");
            return false;
        }
        if (meta.layer_index != kInitialLayerIndex + 1) {
            MNN_PRINT(
                "ChunkedLayerIndexTest FAIL: after chunk 1, layer_index = %d, "
                "expected %d (capture-and-advance must bump once on the first "
                "PendingWrite call of a session)\n",
                meta.layer_index, kInitialLayerIndex + 1);
            return false;
        }

        // ─── Between chunks: simulate the meta updates Llm performs ───
        //   sync() at end of forwardRaw: previous += add (= prefillLen), remove resets
        // layer_index is intentionally NOT touched here — in real hybrid
        // models, FA layers' onRealloc on chunks 2..N also does NOT touch it.
        meta.previous = prefillLen;
        meta.remove = 0;
        int layer_index_before_chunk2 = meta.layer_index;

        // ─── Chunk 2: must NOT re-advance layer_index ───
        if (!runChunk(0.1f)) {
            MNN_PRINT("ChunkedLayerIndexTest: chunk 2 forward failed\n");
            return false;
        }
        if (meta.layer_index != layer_index_before_chunk2) {
            MNN_PRINT(
                "ChunkedLayerIndexTest FAIL: after chunk 2, layer_index = %d, "
                "expected %d (chunks 2..N must reuse mStateCache->mPrefixLayerIndex "
                "without re-advancing mMeta->layer_index — the drift was the "
                "root cause of LA overwriting FA's prefix cache files in hybrid "
                "models, manifesting as SIGBUS on subsequent FA mmap access)\n",
                meta.layer_index, layer_index_before_chunk2);
            return false;
        }

        // Cleanup: PendingWrite branch writes the per-layer prefix cache files
        // as a side effect. Default prefix cache dir relative to CWD is
        // "prefixcache/". Remove them so we don't leave artifacts behind.
        ::remove("prefixcache/test_chunked_layer_index_5.k");
        ::remove("prefixcache/test_chunked_layer_index_5.v");
        // (leave the empty prefixcache/ dir behind; harmless and cross-platform-friendly)

        MNN_PRINT("LinearAttention Chunked LayerIndex (per-session capture) PASSED\n");
        return true;
    }
};
MNNTestSuiteRegister(LinearAttentionChunkedLayerIndexTest, "op/linear_attention_chunked_layer_index");

// ─── Edge case: PendingWrite when previous != remove (capture must be skipped) ───
//
// The capture-and-advance block in CPULinearAttention::onExecute only fires
// when (file_name set, file_flag in {PendingWrite, PendingRead}, previous ==
// remove). The `previous == remove` predicate identifies "first call of a
// fresh-or-fully-rolled-back session" (chunk 1 of a new write, or chunk 1
// after eraseHistory(0, previous)).
//
// If something triggers PendingWrite/PendingRead outside that entry path
// (e.g. partial eraseHistory(begin>0, end) followed by a forced cache write
// while `mMeta->remove < mMeta->previous`), the capture block is skipped and
// mStateCache->mPrefixLayerIndex stays at its initial sentinel -1.
//
// The PendingWrite branch then constructs a file path using -1 as the layer
// index, writing junk to "<dir>/<name>_-1.k". This corrupts the prefix cache
// directory layout — silent on success but reads as a phantom layer to any
// future PendingRead pass.
//
// This test pins down the desired behavior on that mismatched-meta path:
//   (a) mMeta->layer_index must NOT advance (consistent with all advancement
//       being moved into the capture block), and
//   (b) no junk "_-1.{k,v}" file should be created.
//
// Failure on (b) means production code needs either a fallback (use
// mMeta->layer_index when mPrefixLayerIndex == -1) or an early-out guard
// inside the PendingWrite/PendingRead branches. The test cleans up any junk
// it may have produced so subsequent runs are not affected by today's bug.
class LinearAttentionPendingWriteUnsyncedTest : public MNNTestCase {
public:
    LinearAttentionPendingWriteUnsyncedTest() = default;
    virtual ~LinearAttentionPendingWriteUnsyncedTest() = default;

    virtual bool run(int precision) {
        const int B = 1, numKHeads = 2, numVHeads = 2;
        const int headKDim = 4, headVDim = 4, K_conv = 4;
        const int key_dim = numKHeads * headKDim;
        const int val_dim = numVHeads * headVDim;
        const int D = 2 * key_dim + val_dim;
        const int prefillLen = 4;
        const int kInitialLayerIndex = 5;
        const int kLayerNums = 24;
        const std::string cacheName = "test_pending_write_unsynced";
        const std::string junkK = "prefixcache/" + cacheName + "_-1.k";
        const std::string junkV = "prefixcache/" + cacheName + "_-1.v";

        // Defensive: remove any pre-existing junk from a previous failing run
        // so we measure THIS run's behavior.
        ::remove(junkK.c_str());
        ::remove(junkV.c_str());

        auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), D * 1 * K_conv);

        // Construct meta where PendingWrite fires but the capture-and-advance
        // condition fails (previous != remove). 4/2 mimics a partial
        // eraseHistory(begin=2, end=4) followed by a forced cache write.
        LATestKVMeta meta;
        meta.file_name = cacheName;
        meta.file_flag = LATestKVMeta::PendingWrite;
        meta.layer_index = kInitialLayerIndex;
        meta.layer_nums = kLayerNums;
        meta.previous = 4;
        meta.remove = 2;

        auto module = _makeLinearAttentionModuleWithMeta(numKHeads, numVHeads, headKDim, headVDim, true, &meta);
        if (!module) {
            MNN_PRINT("PendingWriteUnsyncedTest: failed to create module\n");
            return false;
        }

        auto qkvVar = _Input({B, D, prefillLen}, NCHW, halide_type_of<float>());
        auto gateVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
        auto betaVar = _Input({B, prefillLen, numVHeads}, NCHW, halide_type_of<float>());
        fillDeterministic(qkvVar->writeMap<float>(), B * D * prefillLen, 0.07f, 0.0f);
        fillGate(gateVar->writeMap<float>(), B * prefillLen * numVHeads);
        fillBeta(betaVar->writeMap<float>(), B * prefillLen * numVHeads);

        auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
        if (outputs.empty()) {
            MNN_PRINT("PendingWriteUnsyncedTest: forward failed\n");
            return false;
        }
        (void)outputs[0]->readMap<float>();

        // (a) layer_index must NOT have advanced
        bool layerIndexOk = (meta.layer_index == kInitialLayerIndex);
        if (!layerIndexOk) {
            MNN_PRINT(
                "PendingWriteUnsyncedTest FAIL (a): layer_index = %d, expected %d "
                "(capture-and-advance must not fire when previous != remove)\n",
                meta.layer_index, kInitialLayerIndex);
        }

        // (b) no junk "_-1.{k,v}" file should be written
        struct stat st;
        bool junkKExists = (::stat(junkK.c_str(), &st) == 0);
        bool junkVExists = (::stat(junkV.c_str(), &st) == 0);
        bool junkOk = !junkKExists && !junkVExists;
        if (!junkOk) {
            MNN_PRINT(
                "PendingWriteUnsyncedTest FAIL (b): junk files written at sentinel "
                "index -1: %s=%d %s=%d. Production code should either skip the disk "
                "write or fall back to mMeta->layer_index when mPrefixLayerIndex is -1.\n",
                junkK.c_str(), (int)junkKExists, junkV.c_str(), (int)junkVExists);
        }

        // Cleanup regardless of pass/fail so subsequent runs start clean.
        ::remove(junkK.c_str());
        ::remove(junkV.c_str());

        bool ok = layerIndexOk && junkOk;
        if (ok) {
            MNN_PRINT("LinearAttention PendingWrite-Unsynced (no capture, no junk write) PASSED\n");
        }
        return ok;
    }
};
MNNTestSuiteRegister(LinearAttentionPendingWriteUnsyncedTest, "op/linear_attention_pending_write_unsynced");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
