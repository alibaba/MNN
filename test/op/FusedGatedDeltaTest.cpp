//
//  FusedGatedDeltaTest.cpp
//  MNNTests
//
//  End-to-end test for the MNNFusedGatedDelta kernel. Runs the
//  LinearAttention op via Module::load with Qwen3-Next-style head
//  dimensions (d_k=d_v=128) and Mamba-style (d_k=d_v=64) — these
//  exercise the SIMD chunk path of the fused kernel that the existing
//  LinearAttentionTest (d_v=4/16) does not.
//
//  Compares the runtime output against a scalar Python-style reference
//  that decomposes the gated_delta_rule recurrence step by step.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

// Reference implementation of one decode step of gated_delta_rule for a
// single attention head. Mirrors the math in CPULinearAttention.cpp:
//   out_k = S^T @ k
//   delta = beta * (v - decay * out_k)
//   out   = decay * (S^T @ q) + dot(k,q) * delta
//   S     = decay * S + k ⊗ delta
static void refOneStep(float* S, const float* k, const float* q, const float* v, float* out, float decay, float beta,
                       int dk, int dv) {
    std::vector<float> outK(dv, 0.0f), outQ(dv, 0.0f), delta(dv, 0.0f);
    for (int i = 0; i < dk; ++i) {
        const float* row = S + i * dv;
        for (int j = 0; j < dv; ++j) {
            outK[j] += row[j] * k[i];
            outQ[j] += row[j] * q[i];
        }
    }
    float kq = 0.0f;
    for (int i = 0; i < dk; ++i)
        kq += k[i] * q[i];
    for (int j = 0; j < dv; ++j) {
        float vPred = decay * outK[j];
        delta[j] = beta * (v[j] - vPred);
        out[j] = decay * outQ[j] + kq * delta[j];
    }
    for (int i = 0; i < dk; ++i) {
        float ki = k[i];
        float* row = S + i * dv;
        for (int j = 0; j < dv; ++j) {
            row[j] = decay * row[j] + ki * delta[j];
        }
    }
}

// Apply L2-norm + scale (Q is also scaled by 1/sqrt(d_k)) — matches the
// useL2Norm=true path inside CPULinearAttention.
static void applyL2NormAndScale(float* q, float* k, int dk) {
    const float eps = 1e-6f;
    const float qScale = 1.0f / std::sqrt((float)dk);
    float qSq = 0.0f, kSq = 0.0f;
    for (int i = 0; i < dk; ++i) {
        qSq += q[i] * q[i];
        kSq += k[i] * k[i];
    }
    float qNS = qScale / std::sqrt(qSq + eps);
    float kIN = 1.0f / std::sqrt(kSq + eps);
    for (int i = 0; i < dk; ++i) {
        q[i] *= qNS;
        k[i] *= kIN;
    }
}

// Build a LinearAttention op as a Module so we can run forward(). Uses the
// same FlatBuffers-based construction as LinearAttentionTest.
static std::shared_ptr<Module> makeModule(int numKHeads, int numVHeads, int dk, int dv) {
    auto qkv = _Input();
    auto gate = _Input();
    auto beta = _Input();
    auto convW = _Input();

    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    auto* p = op->main.AsLinearAttentionParam();
    p->attn_type = "gated_delta_rule";
    p->num_k_heads = numKHeads;
    p->num_v_heads = numVHeads;
    p->head_k_dim = dk;
    p->head_v_dim = dv;
    p->use_qk_l2norm = true;

    auto out = Variable::create(Expr::create(op.get(), {qkv, gate, beta, convW}));
    auto buffer = Variable::save({out});

    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bn;
    bn.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bn.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bn.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bn;
    config.numThread = 1;

    std::shared_ptr<Executor::RuntimeManager> rt(Executor::RuntimeManager::createRuntimeManager(config));
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rt));
}

struct Case {
    const char* name;
    int Hk;
    int Hv;
    int dk;
    int dv;
    int L;        // 1 → exercises gated_delta_rule_decode; >1 → gated_delta_rule_mnn (prefill)
    int numSteps; // number of forward() invocations; state accumulates across steps
};

// Run one Case: drives the LinearAttention module forward `numSteps` times,
// each invocation feeding a fresh random qkv tensor with seqLen=L. State (both
// the conv state and the recurrent S) accumulates across calls. The reference
// path replays the same conv1D + L2norm + gated_delta_rule pipeline as
// CPULinearAttention and compares element-wise.
static bool runCase(const Case& cs, std::mt19937& rng, float tolerance) {
    const int B = 1;
    const int Hk = cs.Hk;
    const int Hv = cs.Hv;
    const int dk = cs.dk;
    const int dv = cs.dv;
    const int L = cs.L;
    const int K_conv = 4;
    const int convStateSize = K_conv - 1;
    const int key_dim = Hk * dk;
    const int val_dim = Hv * dv;
    const int D = 2 * key_dim + val_dim;
    const int gqa_factor = (Hv > Hk) ? (Hv / Hk) : 1;

    auto module = makeModule(Hk, Hv, dk, dv);
    if (!module) {
        MNN_PRINT("FusedGatedDeltaTest[%s]: module creation failed\n", cs.name);
        return false;
    }

    auto convWVar = _Input({D, 1, K_conv}, NCHW, halide_type_of<float>());
    {
        float* w = convWVar->writeMap<float>();
        std::uniform_real_distribution<float> dist(-0.15f, 0.15f);
        for (int i = 0; i < D * K_conv; ++i)
            w[i] = dist(rng);
    }

    // Reference state — module starts from zero on first call.
    std::vector<float> refConvState(B * D * convStateSize, 0.0f);
    std::vector<float> refS(B * Hv * dk * dv, 0.0f);

    for (int step = 0; step < cs.numSteps; ++step) {
        auto qkvVar = _Input({B, D, L}, NCHW, halide_type_of<float>());
        auto gateVar = _Input({B, L, Hv}, NCHW, halide_type_of<float>());
        auto betaVar = _Input({B, L, Hv}, NCHW, halide_type_of<float>());
        std::uniform_real_distribution<float> qkvDist(-0.3f, 0.3f);
        {
            float* qkv = qkvVar->writeMap<float>();
            for (int i = 0; i < B * D * L; ++i)
                qkv[i] = qkvDist(rng);
            float* g = gateVar->writeMap<float>();
            for (int i = 0; i < B * L * Hv; ++i)
                g[i] = -0.1f - 0.05f * (i % 3);
            float* b = betaVar->writeMap<float>();
            for (int i = 0; i < B * L * Hv; ++i)
                b[i] = 0.4f + 0.1f * (i % 4);
        }

        // ── Reference path ──
        std::vector<float> refOut(B * L * Hv * dv, 0.0f);
        const float* convWPtr = convWVar->readMap<float>();
        const float* qkvPtr = qkvVar->readMap<float>();
        const float* gPtr = gateVar->readMap<float>();
        const float* bPtr = betaVar->readMap<float>();

        // Conv1D + SiLU across all L tokens, channel-by-channel.
        // convOut layout matches the runtime: [B, D, L] (channel-major within batch).
        std::vector<float> convOut(B * D * L, 0.0f);
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < D; ++d) {
                float* mState = refConvState.data() + (b * D + d) * convStateSize;
                const float* w = convWPtr + d * K_conv;
                for (int l = 0; l < L; ++l) {
                    float xnew = qkvPtr[(b * D + d) * L + l];
                    float sum = xnew * w[convStateSize];
                    for (int kk = 0; kk < convStateSize; ++kk)
                        sum += mState[kk] * w[kk];
                    float sig = 1.0f / (1.0f + std::exp(-sum));
                    convOut[(b * D + d) * L + l] = sum * sig;
                    // Shift state and append xnew.
                    for (int kk = 0; kk < convStateSize - 1; ++kk)
                        mState[kk] = mState[kk + 1];
                    mState[convStateSize - 1] = xnew;
                }
            }
        }

        // gated_delta_rule across all timesteps and heads (GQA: k_head = h / gqa_factor).
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < L; ++t) {
                for (int h = 0; h < Hv; ++h) {
                    const int k_head = h / gqa_factor;
                    std::vector<float> qLocal(dk), kLocal(dk), vLocal(dv);
                    for (int i = 0; i < dk; ++i) {
                        qLocal[i] = convOut[(b * D + k_head * dk + i) * L + t];
                        kLocal[i] = convOut[(b * D + key_dim + k_head * dk + i) * L + t];
                    }
                    for (int i = 0; i < dv; ++i) {
                        vLocal[i] = convOut[(b * D + 2 * key_dim + h * dv + i) * L + t];
                    }
                    applyL2NormAndScale(qLocal.data(), kLocal.data(), dk);
                    float decay = std::exp(gPtr[b * L * Hv + t * Hv + h]);
                    float beta_t = bPtr[b * L * Hv + t * Hv + h];
                    float* state = refS.data() + (b * Hv + h) * dk * dv;
                    float* outSlot = refOut.data() + ((b * L + t) * Hv + h) * dv;
                    refOneStep(state, kLocal.data(), qLocal.data(), vLocal.data(), outSlot, decay, beta_t, dk, dv);
                }
            }
        }

        // ── Module path ──
        auto outputs = module->onForward({qkvVar, gateVar, betaVar, convWVar});
        if (outputs.empty()) {
            MNN_PRINT("FusedGatedDeltaTest[%s]: empty output at step %d\n", cs.name, step);
            return false;
        }
        const float* res = outputs[0]->readMap<float>();
        const int N = B * L * Hv * dv;
        for (int i = 0; i < N; ++i) {
            float diff = std::fabs(res[i] - refOut[i]);
            if (diff > tolerance) {
                MNN_PRINT(
                    "FusedGatedDeltaTest[%s] step %d MISMATCH idx=%d "
                    "ref=%.6f got=%.6f diff=%.4e (tol=%.4e)\n",
                    cs.name, step, i, refOut[i], res[i], diff, tolerance);
                return false;
            }
        }
    }
    MNN_PRINT("FusedGatedDeltaTest[%s] Hk=%d Hv=%d dk=%d dv=%d L=%d × %d steps PASSED\n", cs.name, Hk, Hv, dk, dv, L,
              cs.numSteps);
    return true;
}

} // anonymous namespace

class FusedGatedDeltaTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::mt19937 rng(0x5A17u);
        // Coverage matrix:
        //  - decode (L=1): single-head, multi-head, GQA, all production dk/dv
        //  - prefill (L>1): multi-head + GQA at dk=128/dv=128 to exercise the
        //    gated_delta_rule_mnn path with the shared per-thread buffer
        std::vector<Case> cases = {
            // {name,                    Hk, Hv,  dk,  dv,  L, steps}
            {"decode_1h_dk64_dv64", 1, 1, 64, 64, 1, 4},
            {"decode_1h_dk64_dv128", 1, 1, 64, 128, 1, 4},
            {"decode_1h_dk128_dv64", 1, 1, 128, 64, 1, 4},
            {"decode_1h_dk128_dv128", 1, 1, 128, 128, 1, 4},    // ← Qwen3-Next shape
            {"decode_4h_dk128_dv128", 4, 4, 128, 128, 1, 3},    // multi-head decode
            {"decode_gqa2_dk128_dv128", 2, 4, 128, 128, 1, 3},  // GQA 2:1 decode
            {"prefill_1h_dk128_dv128", 1, 1, 128, 128, 8, 2},   // L>1 single head
            {"prefill_4h_dk128_dv128", 4, 4, 128, 128, 8, 2},   // L>1 multi-head
            {"prefill_gqa2_dk128_dv128", 2, 4, 128, 128, 8, 2}, // L>1 + GQA
        };

        // fp16 path accumulates more round-off — loosen tolerance for low-precision.
        float tol = (precision == BackendConfig::Precision_Low) ? 6e-2f : 5e-3f;
        for (auto& cs : cases) {
            if (!runCase(cs, rng, tol))
                return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(FusedGatedDeltaTest, "op/fused_gated_delta");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
