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
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "core/KVMeta.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>
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
    const std::string& attnType = "gated_delta_rule", bool forceOpenCLBuffer = false,
    bool gateFold = false, const std::vector<float>& gateCoef = std::vector<float>(),
    const std::vector<float>& gateBias = std::vector<float>())
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
    param->gate_fold    = gateFold;
    param->gate_coef    = gateCoef;
    param->gate_bias    = gateBias;

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
    config.numThread = forceOpenCLBuffer && status.forwardType == MNN_FORWARD_OPENCL
                           ? MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE
                           : 1;

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

        // ─── Test 4: Chunk-64 prefill (Qwen3.5 shape) ───
        // dk=dv=128 with L >= 64 is the only shape that reaches the chunked
        // prefill kernels; the cases above all stay on the scalar/decode paths.
        // L=192 is an exact chunk multiple, L=100 leaves a partial tail chunk
        // whose padded lanes still carry the chunk's gate cumsum.
        {
            const int numHeads = 16;
            const int wideKDim = 128, wideVDim = 128;
            const int wideKeyDim = numHeads * wideKDim;
            const int wideD = 2 * wideKeyDim + numHeads * wideVDim;
            // Relative to the output scale (see below).  The deviation from the
            // sequential reference measures 0.16~0.23% at both fp16 and fp32
            // storage, i.e. it comes from the chunked reassociation rather than
            // precision, so one 1% band covers both with ~4x headroom.
            const float wideTolerance = 0.01f;

            for (int L : {192, 100}) {
                auto module = _makeLinearAttentionModule(numHeads, numHeads, wideKDim, wideVDim, useL2Norm);
                if (!module) {
                    MNN_PRINT("Error: Failed to create chunk64 LinearAttention module\n");
                    return false;
                }

                auto qkvVar   = _Input({B, wideD, L}, NCHW, halide_type_of<float>());
                auto gateVar  = _Input({B, L, numHeads}, NCHW, halide_type_of<float>());
                auto betaVar  = _Input({B, L, numHeads}, NCHW, halide_type_of<float>());
                auto convWVar = _Input({wideD, 1, K_conv}, NCHW, halide_type_of<float>());

                fillDeterministic(qkvVar->writeMap<float>(), B * wideD * L, 0.03f);
                fillGate(gateVar->writeMap<float>(), B * L * numHeads);
                fillBeta(betaVar->writeMap<float>(), B * L * numHeads);
                fillConvWeight(convWVar->writeMap<float>(), wideD * K_conv);

                NaiveLinearAttention naive;
                naive.init(B, wideD, K_conv, numHeads, wideKDim, wideVDim);
                auto expected = naive.forward(
                    qkvVar->readMap<float>(), gateVar->readMap<float>(),
                    betaVar->readMap<float>(), convWVar->readMap<float>(),
                    B, L, wideD, K_conv, numHeads, numHeads, wideKDim, wideVDim, useL2Norm);

                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (outputs.empty()) {
                    MNN_PRINT("Error: chunk64 prefill (L=%d) returned empty output\n", L);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outSize = B * L * numHeads * wideVDim;

                // An absolute tolerance is vacuous here: the reference outputs
                // are far below 1.0, so scale the comparison by the observed
                // magnitude and refuse a degenerate scale outright.
                float scale = 0.0f;
                for (int i = 0; i < outSize; ++i) {
                    scale = std::max(scale, fabsf(expected[i]));
                }
                if (scale < 1e-3f) {
                    MNN_PRINT("Chunk64 Prefill (L=%d) reference is degenerate (max=%g)\n", L, scale);
                    return false;
                }
                float maxDiff = 0.0f;
                int worst = 0;
                for (int i = 0; i < outSize; ++i) {
                    float diff = fabs(resultPtr[i] - expected[i]);
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        worst = i;
                    }
                }
                if (maxDiff > wideTolerance * scale) {
                    MNN_PRINT("Chunk64 Prefill FAILED (L=%d) at index %d: expected %.6f, got %.6f (diff=%.6f, scale=%.6f, rel=%.4f%%)\n",
                              L, worst, expected[worst], resultPtr[worst], maxDiff, scale,
                              100.0f * maxDiff / scale);
                    return false;
                }
                MNN_PRINT("LinearAttention Chunk64 Prefill (L=%d, dk=dv=128, H=16) PASSED (max rel dev %.4f%%)\n",
                          L, 100.0f * maxDiff / scale);
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionTest, "op/linear_attention");

static VARP makeC4TokenChannelInput(const std::vector<float>& logical, int tokenCount, int channels,
                                    bool channelMajor) {
    std::vector<float> packed(UP_DIV(channels, 4) * tokenCount * 4, 0.0f);
    for (int t = 0; t < tokenCount; ++t) {
        for (int c = 0; c < channels; ++c) {
            int logicalIndex = channelMajor ? c * tokenCount + t : t * channels + c;
            int packedIndex = ((c / 4) * tokenCount + t) * 4 + c % 4;
            packed[packedIndex] = logical[logicalIndex];
        }
    }
    auto input = _Input({tokenCount, channels, 1, 1}, NC4HW4, halide_type_of<float>());
    ::memcpy(input->writeMap<float>(), packed.data(), packed.size() * sizeof(float));
    input->unMap();
    return input;
}

class LinearAttentionC4TailTest : public MNNTestCase {
public:
    virtual ~LinearAttentionC4TailTest() = default;

    virtual bool run(int precision) {
        const int B = 1;
        const int L = 3;
        const int numKHeads = 1;
        const int numVHeads = 3;
        const int headKDim = 4;
        const int headVDim = 5;
        const int kernelSize = 4;
        const int keyDim = numKHeads * headKDim;
        const int valueDim = numVHeads * headVDim;
        const int convDim = 2 * keyDim + valueDim;

        std::vector<float> qkv(B * convDim * L);
        std::vector<float> gate(B * L * numVHeads);
        std::vector<float> beta(B * L * numVHeads);
        std::vector<float> convWeight(convDim * kernelSize);
        fillDeterministic(qkv.data(), qkv.size(), 0.05f);
        fillGate(gate.data(), gate.size());
        fillBeta(beta.data(), beta.size());
        fillConvWeight(convWeight.data(), convWeight.size());

        NaiveLinearAttention naive;
        naive.init(B, convDim, kernelSize, numVHeads, headKDim, headVDim);
        auto expected = naive.forward(qkv.data(), gate.data(), beta.data(), convWeight.data(), B, L, convDim,
                                      kernelSize, numKHeads, numVHeads, headKDim, headVDim, true);

        auto qkvVar = makeC4TokenChannelInput(qkv, L, convDim, true);
        auto gateVar = makeC4TokenChannelInput(gate, L, numVHeads, false);
        auto betaVar = makeC4TokenChannelInput(beta, L, numVHeads, false);
        auto convWeightVar = _Input({convDim, 1, kernelSize}, NCHW, halide_type_of<float>());
        ::memcpy(convWeightVar->writeMap<float>(), convWeight.data(), convWeight.size() * sizeof(float));
        convWeightVar->unMap();

        auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true);
        if (!module) {
            return false;
        }
        auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWeightVar});
        if (outputs.empty()) {
            return false;
        }
        const float* result = outputs[0]->readMap<float>();
        const int outputTokens = L * numVHeads;
        for (int token = 0; token < outputTokens; ++token) {
            for (int d = 0; d < headVDim; ++d) {
                int packedIndex = ((d / 4) * outputTokens + token) * 4 + d % 4;
                int logicalIndex = token * headVDim + d;
                if (fabs(result[packedIndex] - expected[logicalIndex]) > 0.02f) {
                    MNN_ERROR("LinearAttention C4 tail failed at token=%d, channel=%d: expected=%f, actual=%f\n",
                              token, d, expected[logicalIndex], result[packedIndex]);
                    return false;
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionC4TailTest, "op/linear_attention_c4_tail");

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
            const int headKDim = 4, headVDim = 4, K_conv = 1;
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

        // Qwen3.5 uses d_v=128 and a 6144-channel conv. Multiple C4 decode
        // steps exercise conv-state shifts across Metal threadgroup boundaries.
        {
            const int B = 1, numKHeads = 16, numVHeads = 16;
            const int headKDim = 128, headVDim = 128, K_conv = 4;
            const int key_dim = numKHeads * headKDim;
            const int val_dim = numVHeads * headVDim;
            const int D = 2 * key_dim + val_dim;
            const int L = 1;
            const int decodeSteps = 256;
            const float qwenTolerance = precision == MNN::BackendConfig::Precision_Low ? 0.015f : 0.002f;

            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, true,
                                                     "gated_delta_rule", true);
            if (!module) {
                MNN_PRINT("Error: Failed to create Qwen3.5 LinearAttention module\n");
                return false;
            }

            auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
            fillConvWeight(convWVar->writeMap<float>(), D * K_conv);

            NaiveLinearAttention naive;
            naive.init(B, D, K_conv, numVHeads, headKDim, headVDim);
            const float* convWeight = convWVar->readMap<float>();
            auto checkOutput = [&](const std::vector<VARP>& outputs, const std::vector<float>& expected,
                                   int tokenCount, int step) {
                if (outputs.empty()) {
                    MNN_PRINT("Error: Qwen3.5 LinearAttention step %d returned empty output\n", step);
                    return false;
                }
                const float* resultPtr = outputs[0]->readMap<float>();
                const int outputTokens = tokenCount * numVHeads;
                for (int token = 0; token < outputTokens; ++token) {
                    for (int d = 0; d < headVDim; ++d) {
                        int packedIndex = ((d / 4) * outputTokens + token) * 4 + d % 4;
                        int logicalIndex = token * headVDim + d;
                        float diff = fabs(resultPtr[packedIndex] - expected[logicalIndex]);
                        if (!std::isfinite(resultPtr[packedIndex]) ||
                            diff > qwenTolerance + 0.02f * fabs(expected[logicalIndex])) {
                            MNN_PRINT("Qwen3.5 C4 step %d FAILED at %d: expected %.6f, got %.6f (diff=%.6f)\n",
                                      step, logicalIndex, expected[logicalIndex], resultPtr[packedIndex], diff);
                            return false;
                        }
                    }
                }
                return true;
            };

            const int prefillLength = 14;
            std::vector<float> prefillQkv(B * D * prefillLength);
            std::vector<float> prefillGate(B * prefillLength * numVHeads);
            std::vector<float> prefillBeta(B * prefillLength * numVHeads);
            fillDeterministic(prefillQkv.data(), prefillQkv.size(), 0.08f, 0.01f);
            fillGate(prefillGate.data(), prefillGate.size());
            fillBeta(prefillBeta.data(), prefillBeta.size());
            auto prefillExpected = naive.forward(prefillQkv.data(), prefillGate.data(), prefillBeta.data(), convWeight,
                                                 B, prefillLength, D, K_conv, numKHeads, numVHeads, headKDim,
                                                 headVDim, true);
            auto prefillOutputs = module->onForward({makeC4TokenChannelInput(prefillQkv, prefillLength, D, true),
                                                     makeC4TokenChannelInput(prefillGate, prefillLength, numVHeads,
                                                                             false),
                                                     makeC4TokenChannelInput(prefillBeta, prefillLength, numVHeads,
                                                                             false),
                                                     convWVar});
            if (!checkOutput(prefillOutputs, prefillExpected, prefillLength, -1)) {
                return false;
            }

            for (int step = 0; step < decodeSteps; ++step) {
                std::vector<float> qkv(B * D * L);
                std::vector<float> gate(B * L * numVHeads);
                std::vector<float> beta(B * L * numVHeads);
                fillDeterministic(qkv.data(), qkv.size(), 0.08f, 0.01f * (step + 1));
                fillGate(gate.data(), gate.size());
                fillBeta(beta.data(), beta.size());

                auto expected = naive.forward(qkv.data(), gate.data(), beta.data(), convWeight, B, L, D, K_conv,
                                              numKHeads, numVHeads, headKDim, headVDim, true);
                auto qkvVar = makeC4TokenChannelInput(qkv, L, D, true);
                auto gateVar = makeC4TokenChannelInput(gate, L, numVHeads, false);
                auto betaVar = makeC4TokenChannelInput(beta, L, numVHeads, false);
                auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
                if (!checkOutput(outputs, expected, L, step)) {
                    return false;
                }
            }
            MNN_PRINT("LinearAttention Qwen3.5 C4 prefill(%d)+decode(%d) layout (H=%d, dv=%d) PASSED\n",
                      prefillLength, decodeSteps, numVHeads, headVDim);
        }

        return true;
    }
};

MNNTestSuiteRegister(LinearAttentionDecodeTest, "op/linear_attention_decode");

// Wires a caller-owned KVMeta through the KVCACHE_INFO hint; must be the real
// MNN::KVMeta, since a local mirror would be read past its end.
static std::shared_ptr<Module> _makeLinearAttentionModuleWithMeta(int numKHeads, int numVHeads, int headKDim,
                                                                  int headVDim, bool useL2Norm, MNN::KVMeta* meta) {
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
        MNN::KVMeta metaA;
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
        MNN::KVMeta metaB;
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
        // Only CPU (CPULinearAttention.cpp:594-603) and OpenCL
        // (LinearAttentionBufExecution.cpp:1042-1050) implement the linear-attention
        // prefix cache. MetalLinearAttention never persists LA state — it only reads
        // the PendingRead flag to decide whether to keep the recurrent state — so it
        // has no layer_index to capture or advance. Asserting the counter there would
        // report a missing feature as a wrong result.
        if (MNNTestSuite::get()->pStaus.forwardType == MNN_FORWARD_METAL) {
            return true;
        }
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
        MNN::KVMeta meta;
        meta.file_name = "test_chunked_layer_index";
        meta.file_flag = MNN::KVMeta::PendingWrite;
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
        MNN::KVMeta meta;
        meta.file_name = cacheName;
        meta.file_flag = MNN::KVMeta::PendingWrite;
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

// ─── gate_fold equivalence test ───
// With gate_fold the exporter stops emitting the gate/beta pre-computation as
// separate elementwise ops: inputs 1/2 then carry the raw a/b projections and the
// op itself applies
//   gate = gate_coef[h] * softplus(a + gate_bias[h])
//   beta = sigmoid(b)
// Shapes are identical either way, so a backend that ignores the flag produces
// silently wrong numbers. This drives one set of random a/b through both paths and
// requires them to agree, which isolates the fold arithmetic from the recurrence.
class LinearAttentionGateFoldTest : public MNNTestCase {
    static float softplus(float x) {
        return logf(1.0f + expf(x));
    }

    // Backends branch on (L, headKDim, headVDim):
    //   L == 1                  -> decode kernel
    //   L > 1 && dk != dv       -> sequential prefill
    //   L > 1 && dk == dv       -> chunked prefill (separate cumsum / attn kernels)
    // Every one of them reads gate/beta, so every one has to fold.
    bool runCase(const char* name, int numKHeads, int numVHeads, int headKDim, int headVDim, int L,
                 float tolerance) {
        const int B = 1;
        const int kernelSize = 4;
        const int keyDim = numKHeads * headKDim;
        const int valDim = numVHeads * headVDim;
        const int D = 2 * keyDim + valDim;
        const bool useL2Norm = true;
        const int gateCount = B * L * numVHeads;
        const int outSize = B * L * numVHeads * headVDim;

        std::vector<float> gateCoef(numVHeads), gateBias(numVHeads);
        for (int h = 0; h < numVHeads; ++h) {
            gateCoef[h] = -(0.5f + 0.25f * h); // -exp(A_log) is always negative
            gateBias[h] = -0.3f + 0.2f * h;
        }

        std::vector<float> qkv(B * D * L), rawA(gateCount), rawB(gateCount), convWeight(D * kernelSize);
        fillDeterministic(qkv.data(), (int)qkv.size(), 0.05f);
        fillConvWeight(convWeight.data(), (int)convWeight.size());
        for (int i = 0; i < gateCount; ++i) {
            rawA[i] = -0.4f + 0.15f * (i % 7);
            rawB[i] = -0.6f + 0.30f * (i % 5);
        }

        // Host-side fold: exactly the elementwise chain the exporter used to emit.
        std::vector<float> gate(gateCount), beta(gateCount);
        for (int t = 0; t < B * L; ++t) {
            for (int h = 0; h < numVHeads; ++h) {
                const int i = t * numVHeads + h;
                gate[i] = gateCoef[h] * softplus(rawA[i] + gateBias[h]);
                beta[i] = 1.0f / (1.0f + expf(-rawB[i]));
            }
        }

        auto runOnce = [&](bool fold, const std::vector<float>& coef, const std::vector<float>& bias,
                           const std::vector<float>& in1, const std::vector<float>& in2,
                           std::vector<float>& out) -> bool {
            // Buffer memory is what actually decides whether OpenCL picks up this op
            // (see run()); the flag here only covers the case where this module is
            // the one creating the runtime.
            auto module = _makeLinearAttentionModule(numKHeads, numVHeads, headKDim, headVDim, useL2Norm,
                                                     "gated_delta_rule", true, fold, coef, bias);
            if (!module) {
                return false;
            }
            auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
            auto in1Var = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto in2Var = _Input({B, L, numVHeads}, NCHW, halide_type_of<float>());
            auto wVar = _Input({D, 1, kernelSize}, NCHW, halide_type_of<float>());
            ::memcpy(qkvVar->writeMap<float>(), qkv.data(), qkv.size() * sizeof(float));
            ::memcpy(in1Var->writeMap<float>(), in1.data(), in1.size() * sizeof(float));
            ::memcpy(in2Var->writeMap<float>(), in2.data(), in2.size() * sizeof(float));
            ::memcpy(wVar->writeMap<float>(), convWeight.data(), convWeight.size() * sizeof(float));
            auto outputs = module->onForward({qkvVar, in1Var, in2Var, wVar});
            if (outputs.empty()) {
                return false;
            }
            const float* p = outputs[0]->readMap<float>();
            if (p == nullptr) {
                return false;
            }
            out.assign(p, p + outSize);
            return true;
        };

        auto compare = [&](const char* label, const std::vector<float>& ref, const std::vector<float>& got) {
            for (int i = 0; i < outSize; ++i) {
                const float diff = fabs(ref[i] - got[i]);
                if (diff > tolerance) {
                    MNN_PRINT("GateFold %s [%s] FAILED at index %d: reference %.6f, got %.6f (diff=%.6f)\n",
                              name, label, i, ref[i], got[i], diff);
                    return false;
                }
            }
            return true;
        };

        std::vector<float> refOut, foldOut;
        if (!runOnce(false, gateCoef, gateBias, gate, beta, refOut)) {
            MNN_PRINT("GateFold %s: reference (fold off) forward failed\n", name);
            return false;
        }
        // Anchor the non-fold path against the host reference first. Without this the
        // fold/non-fold comparison below is self-referential: both runs share the same
        // backend, so a bug in the recurrence itself would cancel out and still pass.
        NaiveLinearAttention naive;
        naive.init(B, D, kernelSize, numVHeads, headKDim, headVDim);
        auto naiveOut = naive.forward(qkv.data(), gate.data(), beta.data(), convWeight.data(), B, L, D, kernelSize,
                                      numKHeads, numVHeads, headKDim, headVDim, useL2Norm);
        if (!compare("naive", naiveOut, refOut)) {
            return false;
        }
        if (!runOnce(true, gateCoef, gateBias, rawA, rawB, foldOut)) {
            MNN_PRINT("GateFold %s: folded forward failed\n", name);
            return false;
        }
        if (!compare("fold", refOut, foldOut)) {
            return false;
        }

        // gate_fold set but the constants are malformed. A folded graph has no gate
        // chain left, so there is nothing to fall back to: the op must refuse to run
        // rather than consume inputs 1/2 as if they were gate/beta.
        std::vector<float> shortCoef(gateCoef.begin(), gateCoef.end() - 1);
        std::vector<float> rejectedOut;
        if (runOnce(true, shortCoef, gateBias, gate, beta, rejectedOut)) {
            MNN_PRINT("GateFold %s: malformed gate_coef was accepted, expected rejection\n", name);
            return false;
        }

        MNN_PRINT("LinearAttention GateFold %s PASSED\n", name);
        return true;
    }

public:
    virtual ~LinearAttentionGateFoldTest() = default;

    virtual bool run(int precision) {
        const float tolerance = 0.001f;
        // The shared executor creates its OpenCL runtime before any test runs, and
        // RuntimeManager reuses that runtime without re-applying numThread, so the
        // per-module MNN_GPU_MEMORY_BUFFER request is dropped and the memory mode
        // stays AUTO (IMAGE on most GPUs). LinearAttention only registers a BUFFER
        // creator, so on OpenCL run under a private executor whose runtime really is
        // built in buffer mode; otherwise this silently exercises CPU.
        auto status = MNNTestSuite::get()->pStaus;
        std::shared_ptr<Executor> privateExe;
        std::shared_ptr<ExecutorScope> privateScope;
        if (status.forwardType == MNN_FORWARD_OPENCL) {
            MNN::BackendConfig bnConfig;
            bnConfig.memory    = (MNN::BackendConfig::MemoryMode)status.memory;
            bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
            bnConfig.power     = (MNN::BackendConfig::PowerMode)status.power;
            privateExe = Executor::newExecutor(MNN_FORWARD_OPENCL, bnConfig,
                                               MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE);
            privateScope.reset(new ExecutorScope(privateExe));
        }
        // L=20 on the chunked path spans two chunks, so the tail chunk is partly
        // padding — padding must contribute an exact 0 to the gate cumsum even though
        // fold(0) != 0.
        return runCase("decode", 2, 2, 4, 4, 1, tolerance) &&
               runCase("prefill-seq", 2, 2, 4, 8, 5, tolerance) &&
               runCase("prefill-chunked", 2, 2, 4, 4, 20, tolerance);
    }
};
MNNTestSuiteRegister(LinearAttentionGateFoldTest, "op/linear_attention_gate_fold");

// Metal stores fp16 for every precision except Precision_High; CPU only for
// Precision_Low. Tolerances must follow the storage actually in use.
static bool linearAttnRunsFp16() {
    auto status = MNNTestSuite::get()->pStaus;
    if (status.forwardType == MNN_FORWARD_METAL) {
        return status.precision != MNN::BackendConfig::Precision_High;
    }
    return status.precision == MNN::BackendConfig::Precision_Low;
}

// ─── Metal kernel-path coverage for the gated-delta-rule attention ───
//
// Path map of MetalLinearAttention::onEncode (B=1, L=seqLen):
//   L < 16        : fused decode kernels (fused_sg_tg when H<16 && L==1, else
//                   fused_sg_align, else fused_sg); conv fuses state_update at L==1
//   L >= 16       : dk==128 -> register-scan prefill (qkv_prep_sg + delta_rule_sg_v4)
//                   dk==64  -> register-scan prefill (qkv_prep_sg + delta_rule_sg_v2)
//                   L>=32 && dv%4==0 -> fused_chunk_sg (any other dk)
//                   else -> qkv_prep_sg + delta_rule_sg (generic)
//   spec_block>0  : verify_fused_sg lazy path (covered by the spec-verify test below)
// Paths not reachable on M4-class devices (no tensor API) and therefore not
// cased here: chunk64 flash (tensor ops, M5+), flash_chunk_sgmm (dk==128 is
// shadowed by the sg_v4 scan), and the scalar fallbacks (devices without
// simdgroup reduce). Each case runs a prefill followed by decode steps so the
// persistent conv + recurrent state is exercised end to end; on CPU the same
// cases run through the decode / sequential-prefill kernels, so CI also guards
// the CPU backend.
class LinearAttentionMetalPathsTest : public MNNTestCase {
public:
    LinearAttentionMetalPathsTest() = default;
    virtual ~LinearAttentionMetalPathsTest() = default;

    bool runGatedCase(int numK, int numV, int dk, int dv, int prefillL, int decodeSteps, bool c4, const char* tag) {
        const int B = 1, K_conv = 4;
        const int keyDim = numK * dk, valDim = numV * dv;
        const int D = 2 * keyDim + valDim;
        const float tol = linearAttnRunsFp16() ? 0.02f : 0.002f;

        auto module = _makeLinearAttentionModule(numK, numV, dk, dv, true);
        if (!module) {
            MNN_PRINT("Error: failed to create module for %s\n", tag);
            return false;
        }
        NaiveLinearAttention naive;
        naive.init(B, D, K_conv, numV, dk, dv);

        auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), D * K_conv);

        auto runOne = [&](int L, int step, const char* phase) -> bool {
            std::vector<float> qkv(B * D * L), gate(B * L * numV), beta(B * L * numV);
            fillDeterministic(qkv.data(), (int)qkv.size(), 0.08f, 0.03f * step);
            fillGate(gate.data(), (int)gate.size());
            fillBeta(beta.data(), (int)beta.size());

            VARP qkvVar, gateVar, betaVar;
            if (c4) {
                qkvVar  = makeC4TokenChannelInput(qkv, L, D, true);
                gateVar = makeC4TokenChannelInput(gate, L, numV, false);
                betaVar = makeC4TokenChannelInput(beta, L, numV, false);
            } else {
                qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                ::memcpy(qkvVar->writeMap<float>(), qkv.data(), qkv.size() * sizeof(float));
                qkvVar->unMap();
                gateVar = _Input({B, L, numV}, NCHW, halide_type_of<float>());
                ::memcpy(gateVar->writeMap<float>(), gate.data(), gate.size() * sizeof(float));
                gateVar->unMap();
                betaVar = _Input({B, L, numV}, NCHW, halide_type_of<float>());
                ::memcpy(betaVar->writeMap<float>(), beta.data(), beta.size() * sizeof(float));
                betaVar->unMap();
            }

            auto expected = naive.forward(qkv.data(), gate.data(), beta.data(), convWVar->readMap<float>(), B, L, D,
                                          K_conv, numK, numV, dk, dv, true);
            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("%s: %s returned empty output\n", tag, phase);
                return false;
            }
            const float* result = outputs[0]->readMap<float>();
            const int tokens = B * L * numV;
            for (int token = 0; token < tokens; ++token) {
                for (int d = 0; d < dv; ++d) {
                    int resultIdx = c4 ? (((d / 4) * tokens + token) * 4 + d % 4) : (token * dv + d);
                    float exp = expected[token * dv + d];
                    float diff = fabs(result[resultIdx] - exp);
                    if (diff > tol + 0.02f * fabs(exp)) {
                        MNN_PRINT("%s: %s mismatch at token=%d dim=%d: expected=%f actual=%f diff=%f\n", tag, phase,
                                  token, d, exp, result[resultIdx], diff);
                        return false;
                    }
                }
            }
            return true;
        };

        if (prefillL > 0 && !runOne(prefillL, 0, "prefill")) {
            return false;
        }
        for (int step = 1; step <= decodeSteps; ++step) {
            if (!runOne(1, step, "decode")) {
                return false;
            }
        }
        MNN_PRINT("LinearAttention %s PASSED\n", tag);
        return true;
    }

    virtual bool run(int precision) {
        bool ok = true;
        // Decode kernels: fused_sg_tg needs H<16 && L==1; fused_sg_align covers H>=16.
        ok &= runGatedCase(4, 8, 64, 64, 0, 5, false, "decode fused_sg_tg (H=8, dk=64)");
        ok &= runGatedCase(8, 16, 64, 64, 0, 4, false, "decode fused_sg_align (H=16, dk=64)");
        // 2 <= L < 16 rides the fused decode kernels with an internal L loop.
        ok &= runGatedCase(2, 2, 64, 64, 8, 3, false, "short prefill L=8 (fused multi-step)");
        // Register-scan prefill: dk==128 selects delta_rule_sg_v4, dk==64 sg_v2.
        ok &= runGatedCase(2, 4, 128, 64, 48, 2, false, "scan prefill sg_v4 (dk=128, L=48)");
        ok &= runGatedCase(4, 8, 64, 64, 48, 2, false, "scan prefill sg_v2 (dk=64, L=48)");
        // dk outside {64,128}: L<32 falls back to qkv_prep + delta_rule_sg, L>=32
        // (dv%4==0) to fused_chunk_sg.
        ok &= runGatedCase(2, 4, 96, 64, 20, 2, false, "fallback qkv_prep+delta_rule_sg (dk=96, L=20)");
        ok &= runGatedCase(2, 4, 96, 64, 192, 2, false, "fused_chunk_sg (dk=96, L=192)");
        // NC4HW4 input runs the same kernels through their C4 offset paths.
        ok &= runGatedCase(2, 4, 128, 64, 48, 2, true, "scan prefill sg_v4 C4 (dk=128, L=48)");
        ok &= runGatedCase(2, 4, 96, 64, 192, 2, true, "fused_chunk_sg C4 (dk=96, L=192)");
        return ok;
    }
};

MNNTestSuiteRegister(LinearAttentionMetalPathsTest, "op/linear_attention_metal_paths");

// ─── short_conv path coverage ───
//
// attn_type="short_conv": qkv [B, 3H, L] carries the b / c / x projections.
// Per (b, h): depthwise conv over b*x with NO SiLU, then y = c * conv_out.
// Persistent state [B, H, K-1] holds the last positions of b*x. The output is
// [B, L, 1, H] (num_v_heads=1, head_v_dim=H). Kernels: short_conv_nosilu +
// short_conv_state_update + short_conv_output on Metal; the same math on CPU.
struct NaiveShortConv {
    std::vector<float> state;  // [B, H, css]
    int B, H, css;

    void init(int batch, int hidden, int kernel) {
        B = batch;
        H = hidden;
        css = kernel - 1;
        state.assign(B * H * css, 0.0f);
    }

    float inputVal(const float* qkv, int b, int h, int pos, int L) const {
        return qkv[b * 3 * H * L + h * L + (pos - css)] * qkv[b * 3 * H * L + (2 * H + h) * L + (pos - css)];
    }

    // qkv [B, 3H, L], w [H, K]; returns [B, L, H]
    std::vector<float> forward(const float* qkv, const float* w, int L) {
        const int K = css + 1;
        std::vector<float> out(B * L * H, 0.0f);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int l = 0; l < L; ++l) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        int pos = l + k;
                        float v = pos < css ? state[(b * H + h) * css + pos] : inputVal(qkv, b, h, pos, L);
                        sum += v * w[h * K + k];
                    }
                    out[(b * L + l) * H + h] = qkv[b * 3 * H * L + (H + h) * L + l] * sum;
                }
                // In-place shift, ascending i: reads state[L+i] (i < L+i) before it is written.
                for (int i = 0; i < css; ++i) {
                    int pos = L + i;
                    state[(b * H + h) * css + i] =
                        pos < css ? state[(b * H + h) * css + pos] : inputVal(qkv, b, h, pos, L);
                }
            }
        }
        return out;
    }
};

class LinearAttentionShortConvTest : public MNNTestCase {
public:
    LinearAttentionShortConvTest() = default;
    virtual ~LinearAttentionShortConvTest() = default;

    bool runShortConvCase(int H, int K, int prefillL, int decodeSteps, bool c4, const char* tag) {
        const int B = 1, D = 3 * H;
        const float tol = linearAttnRunsFp16() ? 0.02f : 0.002f;

        auto module = _makeLinearAttentionModule(1, 1, H, H, false, "short_conv");
        if (!module) {
            MNN_PRINT("Error: failed to create module for %s\n", tag);
            return false;
        }
        NaiveShortConv naive;
        naive.init(B, H, K);

        auto convWVar = _Input({H, 1, K}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), H * K);

        auto runOne = [&](int L, int step, const char* phase) -> bool {
            std::vector<float> qkv(B * D * L), gate(B * L, 0.0f), beta(B * L, 0.0f);
            fillDeterministic(qkv.data(), (int)qkv.size(), 0.1f, 0.02f * step);

            VARP qkvVar, gateVar, betaVar;
            if (c4) {
                qkvVar  = makeC4TokenChannelInput(qkv, L, D, true);
                gateVar = makeC4TokenChannelInput(gate, L, 1, false);
                betaVar = makeC4TokenChannelInput(beta, L, 1, false);
            } else {
                qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
                ::memcpy(qkvVar->writeMap<float>(), qkv.data(), qkv.size() * sizeof(float));
                qkvVar->unMap();
                gateVar = _Input({B, L, 1}, NCHW, halide_type_of<float>());
                ::memcpy(gateVar->writeMap<float>(), gate.data(), gate.size() * sizeof(float));
                gateVar->unMap();
                betaVar = _Input({B, L, 1}, NCHW, halide_type_of<float>());
                ::memcpy(betaVar->writeMap<float>(), beta.data(), beta.size() * sizeof(float));
                betaVar->unMap();
            }

            auto expected = naive.forward(qkv.data(), convWVar->readMap<float>(), L);
            auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
            if (outputs.empty()) {
                MNN_PRINT("%s: %s returned empty output\n", tag, phase);
                return false;
            }
            const float* result = outputs[0]->readMap<float>();
            const int tokens = B * L;
            for (int token = 0; token < tokens; ++token) {
                for (int h = 0; h < H; ++h) {
                    int resultIdx = c4 ? (((h / 4) * tokens + token) * 4 + h % 4) : (token * H + h);
                    float exp = expected[token * H + h];
                    float diff = fabs(result[resultIdx] - exp);
                    if (diff > tol + 0.02f * fabs(exp)) {
                        MNN_PRINT("%s: %s mismatch at token=%d head=%d: expected=%f actual=%f diff=%f\n", tag, phase,
                                  token, h, exp, result[resultIdx], diff);
                        return false;
                    }
                }
            }
            return true;
        };

        if (prefillL > 0 && !runOne(prefillL, 0, "prefill")) {
            return false;
        }
        for (int step = 1; step <= decodeSteps; ++step) {
            if (!runOne(1, step, "decode")) {
                return false;
            }
        }
        MNN_PRINT("LinearAttention %s PASSED\n", tag);
        return true;
    }

    virtual bool run(int precision) {
        bool ok = true;
        ok &= runShortConvCase(8, 4, 5, 3, false, "short_conv (H=8, K=4)");
        ok &= runShortConvCase(8, 1, 3, 2, false, "short_conv (K=1, no state)");
        ok &= runShortConvCase(8, 4, 5, 2, true, "short_conv C4 (H=8, K=4)");
        return ok;
    }
};

MNNTestSuiteRegister(LinearAttentionShortConvTest, "op/linear_attention_short_conv");

// ─── Speculative-decode verify path (Metal-only) ───
//
// KVMeta::spec_block tags a verify forward. MetalLinearAttention defers the
// recurrent-state update: the verify kernel replays the accepted prefix of the
// previous pending block (meta.remove holds the rejected count), computes the
// new block against that state, and saves the new block as pending without
// persisting its effect. The next non-verify forward flushes the pending
// block's accepted prefix before running normally. CPU has no spec_block
// support (its recurrent state cannot roll back), so this test is Metal-only.
struct NaiveSpecVerify {
    NaiveLinearAttention core;
    int B, D, K, numK, H, dk, dv;
    std::vector<float> pendRaw, pendK, pendV, pendGate, pendBeta;
    int pendLen = 0;

    void init(int batch, int convDim, int convKernel, int numKHeads, int numVHeads, int headK, int headV) {
        B = batch;
        D = convDim;
        K = convKernel;
        numK = numKHeads;
        H = numVHeads;
        dk = headK;
        dv = headV;
        core.init(batch, convDim, convKernel, numVHeads, headK, headV);
    }

    // Replay commitLen tokens of the pending block into conv + recurrent state
    // (Metal conv_commit + the verify kernel's replay prologue).
    void commitPending(int commitLen) {
        if (commitLen <= 0) {
            return;
        }
        const int css = K - 1;
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int i = 0; i < css; ++i) {
                    int pos = commitLen + i;
                    float v = pos < css ? core.convState[(b * D + d) * css + pos]
                                        : pendRaw[(b * D + d) * pendLen + (pos - css)];
                    core.convState[(b * D + d) * css + i] = v;
                }
            }
        }
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                float* S = core.rnnState.data() + (b * H + h) * dk * dv;
                for (int t = 0; t < commitLen; ++t) {
                    float decay = expf(pendGate[(b * pendLen + t) * H + h]);
                    float beta = pendBeta[(b * pendLen + t) * H + h];
                    const float* k_t = pendK.data() + ((b * pendLen + t) * H + h) * dk;
                    const float* v_t = pendV.data() + ((b * pendLen + t) * H + h) * dv;
                    for (int i = 0; i < dk * dv; ++i) {
                        S[i] *= decay;
                    }
                    for (int j = 0; j < dv; ++j) {
                        float vPred = 0.0f;
                        for (int i = 0; i < dk; ++i) {
                            vPred += S[i * dv + j] * k_t[i];
                        }
                        float delta = beta * (v_t[j] - vPred);
                        for (int i = 0; i < dk; ++i) {
                            S[i * dv + j] += k_t[i] * delta;
                        }
                    }
                }
            }
        }
    }

    // Conv1D + SiLU over the current conv state, without updating it (lazy).
    std::vector<float> convOnly(const float* qkv, const float* convW, int L) {
        const int css = K - 1;
        std::vector<float> out(B * D * L, 0.0f);
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int l = 0; l < L; ++l) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        int pos = l + k;
                        float v = pos < css ? core.convState[(b * D + d) * css + pos]
                                            : qkv[b * D * L + d * L + (pos - css)];
                        sum += v * convW[d * K + k];
                    }
                    float sig = 1.0f / (1.0f + expf(-sum));
                    out[(b * D + d) * L + l] = sum * sig;
                }
            }
        }
        return out;
    }

    // Lazy verify forward: commit the accepted prefix of the previous pending
    // block, compute the new block against the committed state, and save the
    // new block as pending without persisting its effect.
    std::vector<float> lazyForward(const float* qkv, const float* gate, const float* beta, const float* convW, int L,
                                   int commitLen) {
        commitPending(commitLen);

        const int gqa = (H > numK) ? (H / numK) : 1;
        const int keyDim = numK * dk;
        const float qScale = 1.0f / sqrtf((float)dk);
        const float eps = 1e-6f;

        std::vector<float> convOut = convOnly(qkv, convW, L);
        pendRaw.assign(qkv, qkv + B * D * L);
        pendK.assign(B * L * H * dk, 0.0f);
        pendV.assign(B * L * H * dv, 0.0f);
        pendGate.assign(gate, gate + B * L * H);
        pendBeta.assign(beta, beta + B * L * H);
        pendLen = L;

        std::vector<float> out(B * L * H * dv, 0.0f);
        std::vector<float> S = core.rnnState;  // work copy; committed state already persisted
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < L; ++t) {
                for (int h = 0; h < H; ++h) {
                    float* state = S.data() + (b * H + h) * dk * dv;
                    const int kh = h / gqa;

                    std::vector<float> q_t(dk), k_t(dk), v_t(dv);
                    for (int i = 0; i < dk; ++i) {
                        q_t[i] = convOut[(b * D + kh * dk + i) * L + t];
                        k_t[i] = convOut[(b * D + keyDim + kh * dk + i) * L + t];
                    }
                    for (int j = 0; j < dv; ++j) {
                        v_t[j] = convOut[(b * D + 2 * keyDim + h * dv + j) * L + t];
                    }
                    float invQ = qScale, invK = 1.0f;
                    {
                        float sqQ = 0.0f, sqK = 0.0f;
                        for (int i = 0; i < dk; ++i) {
                            sqQ += q_t[i] * q_t[i];
                            sqK += k_t[i] * k_t[i];
                        }
                        invQ = 1.0f / sqrtf(sqQ + eps) * qScale;
                        invK = 1.0f / sqrtf(sqK + eps);
                    }
                    for (int i = 0; i < dk; ++i) {
                        q_t[i] *= invQ;
                        k_t[i] *= invK;
                    }
                    // Pending save uses the post-norm k (the replay feeds it in as-is).
                    ::memcpy(pendK.data() + ((b * L + t) * H + h) * dk, k_t.data(), dk * sizeof(float));
                    ::memcpy(pendV.data() + ((b * L + t) * H + h) * dv, v_t.data(), dv * sizeof(float));

                    float decay = expf(gate[(b * L + t) * H + h]);
                    float betaT = beta[(b * L + t) * H + h];
                    for (int i = 0; i < dk * dv; ++i) {
                        state[i] *= decay;
                    }
                    std::vector<float> delta(dv);
                    for (int j = 0; j < dv; ++j) {
                        float vPred = 0.0f;
                        for (int i = 0; i < dk; ++i) {
                            vPred += state[i * dv + j] * k_t[i];
                        }
                        delta[j] = betaT * (v_t[j] - vPred);
                        for (int i = 0; i < dk; ++i) {
                            state[i * dv + j] += k_t[i] * delta[j];
                        }
                    }
                    for (int j = 0; j < dv; ++j) {
                        float o = 0.0f;
                        for (int i = 0; i < dk; ++i) {
                            o += state[i * dv + j] * q_t[i];
                        }
                        out[(b * L + t) * H * dv + h * dv + j] = o;
                    }
                }
            }
        }
        return out;
    }

    // Flush: commit the pending block's accepted prefix, then a normal forward.
    std::vector<float> flushForward(const float* qkv, const float* gate, const float* beta, const float* convW, int L,
                                    int commitLen) {
        commitPending(commitLen);
        pendLen = 0;
        return core.forward(qkv, gate, beta, convW, B, L, D, K, numK, H, dk, dv, true);
    }
};

class LinearAttentionSpecVerifyTest : public MNNTestCase {
public:
    LinearAttentionSpecVerifyTest() = default;
    virtual ~LinearAttentionSpecVerifyTest() = default;

    virtual bool run(int precision) {
        if ((MNNForwardType)MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            MNN_PRINT("skip: LinearAttention spec-verify is Metal-only (CPU has no spec_block support)\n");
            return true;
        }
        const int B = 1, numK = 4, numV = 8, dk = 64, dv = 64, K_conv = 4, specBlock = 8;
        const int keyDim = numK * dk, valDim = numV * dv;
        const int D = 2 * keyDim + valDim;
        const float tol = linearAttnRunsFp16() ? 0.02f : 0.002f;

        MNN::KVMeta meta;
        auto module = _makeLinearAttentionModuleWithMeta(numK, numV, dk, dv, true, &meta);
        if (!module) {
            MNN_PRINT("SpecVerifyTest: failed to create module\n");
            return false;
        }
        NaiveSpecVerify naive;
        naive.init(B, D, K_conv, numK, numV, dk, dv);

        auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
        fillConvWeight(convWVar->writeMap<float>(), D * K_conv);

        int inputSeed = 0;
        auto makeInputs = [&](int L) {
            VARP qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
            VARP gateVar = _Input({B, L, numV}, NCHW, halide_type_of<float>());
            VARP betaVar = _Input({B, L, numV}, NCHW, halide_type_of<float>());
            fillDeterministic(qkvVar->writeMap<float>(), B * D * L, 0.08f, 0.03f * inputSeed);
            fillGate(gateVar->writeMap<float>(), B * L * numV);
            fillBeta(betaVar->writeMap<float>(), B * L * numV);
            inputSeed++;
            std::vector<float> qkv(B * D * L), gate(B * L * numV), beta(B * L * numV);
            ::memcpy(qkv.data(), qkvVar->readMap<float>(), qkv.size() * sizeof(float));
            ::memcpy(gate.data(), gateVar->readMap<float>(), gate.size() * sizeof(float));
            ::memcpy(beta.data(), betaVar->readMap<float>(), beta.size() * sizeof(float));
            return std::make_tuple(qkvVar, gateVar, betaVar, qkv, gate, beta);
        };

        // verify(tag, remove, expectedCommit): run one spec-tagged forward of
        // specBlock tokens and compare against the naive lazy reference.
        auto verifyBlock = [&](const char* tag, int remove, int expectedCommit) -> bool {
            meta.spec_block = specBlock;
            meta.remove = remove;
            auto in = makeInputs(specBlock);
            auto expected = naive.lazyForward(std::get<3>(in).data(), std::get<4>(in).data(), std::get<5>(in).data(),
                                             convWVar->readMap<float>(), specBlock, expectedCommit);
            auto outputs = module->onForward({std::get<0>(in), std::get<1>(in), std::get<2>(in), convWVar});
            meta.spec_block = 0;
            if (outputs.empty()) {
                MNN_PRINT("SpecVerifyTest: %s returned empty output\n", tag);
                return false;
            }
            const float* result = outputs[0]->readMap<float>();
            const int outSize = B * specBlock * numV * dv;
            for (int i = 0; i < outSize; ++i) {
                float diff = fabs(result[i] - expected[i]);
                if (diff > tol + 0.02f * fabs(expected[i])) {
                    MNN_PRINT("SpecVerifyTest: %s mismatch at index %d: expected=%f actual=%f diff=%f\n", tag, i,
                              expected[i], result[i], diff);
                    return false;
                }
            }
            return true;
        };
        // flush(tag, remove, expectedCommit): one ordinary decode forward that
        // must first flush the pending block's accepted prefix.
        auto flushBlock = [&](const char* tag, int remove, int expectedCommit) -> bool {
            meta.spec_block = 0;
            meta.remove = remove;
            auto in = makeInputs(1);
            auto expected = naive.flushForward(std::get<3>(in).data(), std::get<4>(in).data(), std::get<5>(in).data(),
                                              convWVar->readMap<float>(), 1, expectedCommit);
            auto outputs = module->onForward({std::get<0>(in), std::get<1>(in), std::get<2>(in), convWVar});
            if (outputs.empty()) {
                MNN_PRINT("SpecVerifyTest: %s returned empty output\n", tag);
                return false;
            }
            const float* result = outputs[0]->readMap<float>();
            const int outSize = B * numV * dv;
            for (int i = 0; i < outSize; ++i) {
                float diff = fabs(result[i] - expected[i]);
                if (diff > tol + 0.02f * fabs(expected[i])) {
                    MNN_PRINT("SpecVerifyTest: %s mismatch at index %d: expected=%f actual=%f diff=%f\n", tag, i,
                              expected[i], result[i], diff);
                    return false;
                }
            }
            meta.sync();
            return true;
        };

        // Cycle 1: first verify block (nothing pending, commit 0), second verify
        // block commits the first block's accepted prefix (3 of 8), then a
        // decode flushes the rest of the second block's accepted prefix (3).
        meta.previous = 0;
        meta.remove = 0;
        bool ok = verifyBlock("verify-A (no pending)", 0, 0);
        meta.previous = 100;
        ok &= verifyBlock("verify-B (commit 3)", specBlock - 3, 3);
        ok &= flushBlock("flush-C (commit 3)", specBlock - 3, 3);
        // Cycle 2: full reject (remove == specBlock -> commit 0).
        meta.previous = 100;
        ok &= verifyBlock("verify-D (no pending)", 0, 0);
        ok &= verifyBlock("verify-E (full reject)", specBlock, 0);
        ok &= flushBlock("flush-F (commit 0)", specBlock, 0);
        // Cycle 3: full accept (remove == 0 -> commit all specBlock tokens).
        meta.previous = 100;
        ok &= verifyBlock("verify-G (no pending)", 0, 0);
        ok &= verifyBlock("verify-H (full accept)", 0, specBlock);
        ok &= flushBlock("flush-I (commit all)", 0, specBlock);
        if (ok) {
            MNN_PRINT("LinearAttention spec-verify (commit 3 / reject-all / accept-all) PASSED\n");
        }
        return ok;
    }
};

MNNTestSuiteRegister(LinearAttentionSpecVerifyTest, "op/linear_attention_spec_verify");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
