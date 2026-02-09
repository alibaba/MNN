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
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>

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

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
