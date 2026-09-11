//
//  LlmAttentionSpeed.cpp
//  MNNTests
//
//  Attention-op-only speed benchmark at Qwen3-0.6B prefill shapes, for
//  attributing the long-prefill gap against an external reference
//  implementation. Runs the production op
//  configuration: fused Attention op (kv_cache=true, output_c4=true), GQA
//  16 q-heads / 8 kv-heads / headDim 128, and the scalar (empty) mask that
//  Llm::gen_attention_mask feeds on Metal to signal a causal prefill.
//
//  Usage:
//    ./run_test.out speed/LlmAttn 1 2   # Metal, precision Low (fp16)
//    MNN_QWEN3_MODEL=4b ./run_test.out speed/LlmAttn 1 2
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cstring>
#include <limits>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/OpCommonUtils.hpp"

using namespace MNN::Express;
using MNN::KVMeta;

namespace {

// Shape set, selected by MNN_QWEN3_MODEL (default 0.6b). 0.6B and 4B share
// head_dim 128 and 8 kv-heads and differ only in q-head count and depth (GQA
// group 2 vs 4, which is what changes the tc kernel's per-kv-tile reuse);
// Qwen3.5 moves to head_dim 256 with 8 q / 2 kv heads.
static const char* kModelName = "Qwen3-0.6B";
static int kNumHead = 16;
static int kKvNumHead = 8;
static int kHeadDim = 128;
static int kLayerNum = 28;

static void selectModel() {
    if (auto env = getenv("MNN_QWEN3_MODEL")) {
        if (0 == strcmp(env, "4b")) {
            kModelName = "Qwen3-4B";
            kNumHead = 32;
            kKvNumHead = 8;
            kHeadDim = 128;
            kLayerNum = 36;
        } else if (0 == strcmp(env, "minicpm")) {
            // GQA group 8 with only 16 q heads: the shape where the decode SDPA's
            // KV-sharing win (8x fewer KV read requests) and its threadgroup-count
            // loss (16 -> 8 TGs at qh=2) pull hardest against each other.
            kModelName = "MiniCPM5-1B";
            kNumHead = 16;
            kKvNumHead = 2;
            kHeadDim = 128;
            kLayerNum = 24;
        } else if (0 == strcmp(env, "3.5")) {
            // Qwen3.5-2B's 6 full-attention layers (the other 18 are linear
            // attention). head_dim 256, so this is the only shape that reaches
            // the tc kernel's 8-output-tile configuration.
            kModelName = "Qwen3.5-2B full-attn";
            kNumHead = 8;
            kKvNumHead = 2;
            kHeadDim = 256;
            kLayerNum = 6;
        }
    }
    // Per-dimension overrides, so a knob's dependence on one shape dimension can
    // be isolated: the 3.5 preset moves q-head count and head_dim together, and
    // a preset-only sweep cannot say which of the two a tuning default should
    // key on. Applied after the presets, and independently of them.
    if (auto v = getenv("MNN_ATTN_QHEAD")) {
        kModelName = "custom";
        kNumHead = atoi(v);
    }
    if (auto v = getenv("MNN_ATTN_KVHEAD")) {
        kModelName = "custom";
        kKvNumHead = atoi(v);
    }
    if (auto v = getenv("MNN_ATTN_HEADDIM")) {
        kModelName = "custom";
        kHeadDim = atoi(v);
    }
    if (auto v = getenv("MNN_ATTN_LAYERS")) {
        kLayerNum = atoi(v);
    }
}

static KVMeta gAttnMeta;

// `layers` independent Attention ops sharing one forward. A single-op module
// pays the whole Express/Session per-forward cost (~45 us measured) on one op,
// which swamps a decode step's few microseconds of GPU work; the real model
// pays it once for all 28 layers, so the decode benchmark reproduces that
// structure. Each op owns its own KV cache.
static std::shared_ptr<Module> makeAttentionModule(bool kvCache, int layers = 1) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = kvCache;
    attention->main.AsAttentionParam()->output_c4 = true;
    std::vector<VARP> outs;
    for (int i = 0; i < layers; ++i) {
        // Scaling Q per layer keeps the ops from collapsing into one under
        // common-subexpression elimination, and mirrors layers seeing
        // different activations.
        auto qi = layers == 1 ? Q : Q * _Scalar<float>(1.0f + 0.001f * (float)i);
        outs.emplace_back(Variable::create(Expr::create(attention.get(), {qi, K, V, mask})));
    }
    auto buffer = Variable::save(outs);

    auto status = MNNTestSuite::get()->pStaus;
    MNN::ScheduleConfig config;
    config.type = (MNNForwardType)status.forwardType;
    config.numThread = 1;
    MNN::BackendConfig bnConfig;
    bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gAttnMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, 8); // float qkv, no kv quant
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
}

// [1, seq, head, dim], filled with small deterministic values
static VARP makeQkv(int seq, int head) {
    auto var = _Input({1, seq, head, kHeadDim}, NCHW, halide_type_of<float>());
    auto ptr = var->writeMap<float>();
    for (int i = 0; i < seq * head * kHeadDim; ++i) {
        ptr[i] = ((float)(i % 17) - 8.0f) * 0.02f;
    }
    var->unMap();
    if (nullptr == getenv("MNN_ATTN_HOSTIN")) {
        // A host _Input var is re-uploaded (and fp32->fp16 converted) on every
        // onForward: Q+K+V is ~67 MB per call at seq=4096, which lands inside
        // the timed loop. The external baseline feeds arrays that are already
        // device-resident, so counting the transfer makes the comparison
        // unfair. Multiplying by 1 yields a computed var cached on the GPU
        // backend, so onForward sees a device tensor and skips the copy.
        // MNN_ATTN_HOSTIN=1 restores the host path.
        var = var * _Scalar<float>(1.0f);
    }
    return var;
}

static float benchOne(int seq, int loop, int round) {
    auto Q = makeQkv(seq, kNumHead);
    auto K = makeQkv(seq, kKvNumHead);
    auto V = makeQkv(seq, kKvNumHead);
    // Production Metal prefill mask: scalar 0 => "causal, build it yourself".
    auto Mask = _Input({}, NCHW, halide_type_of<float>());
    Mask->writeMap<float>()[0] = 0.0f;

    auto attn = makeAttentionModule(true);
    auto forwardOnce = [&]() {
        gAttnMeta.previous = 0;
        gAttnMeta.remove = 0;
        gAttnMeta.add = seq;
        return attn->onForward({Q, K, V, Mask})[0];
    };
    for (int i = 0; i < 3; ++i) {
        auto out = forwardOnce();
        out->readMap<float>();
        out->unMap();
    }
    // Time enqueue-only, then sync once: mapping the [seq, 2048] output every
    // iteration costs more than the attention itself at short seq.
    MNN::Timer timer;
    VARP last;
    for (int i = 0; i < loop; ++i) {
        last = forwardOnce();
    }
    last->readMap<float>();
    float msPerLayer = (float)timer.durationInUs() / 1000.0f / (float)loop;
    last->unMap();
    // Causal work only: QK^T and PV each need heads*dim*seq*(seq+1)/2 MACs.
    double causalFlops = 2.0 * 2.0 * kNumHead * kHeadDim * ((double)seq * (seq + 1) / 2.0);
    double tflops = causalFlops / (msPerLayer * 1e6) / 1000.0;
    if (round < 0) {
        return msPerLayer;
    }
    MNN_PRINT("r%d seq=%-5d  per-layer=%8.3f ms (%6.2f TFLOPS causal)  x%d layers=%8.2f ms\n", round, seq, msPerLayer,
              tflops, kLayerNum, msPerLayer * kLayerNum);
    return msPerLayer;
}

// One decode step against a ctx-long KV cache. Returns ms per layer.
//
// Decode attention is bandwidth-bound on the cache, not compute-bound: a step
// reads 2 * ctx * kKvNumHead * kHeadDim fp16 elements and does only ctx MACs
// per (head, dim), so the useful figure of merit is KV bytes/s.
static float benchDecode(int ctx, int loop, int round) {
    auto Qp = makeQkv(ctx, kNumHead);
    auto Kp = makeQkv(ctx, kKvNumHead);
    auto Vp = makeQkv(ctx, kKvNumHead);
    auto Q1 = makeQkv(1, kNumHead);
    auto K1 = makeQkv(1, kKvNumHead);
    auto V1 = makeQkv(1, kKvNumHead);
    auto Mask = _Input({}, NCHW, halide_type_of<float>());
    Mask->writeMap<float>()[0] = 0.0f;

    auto attn = makeAttentionModule(true, kLayerNum);
    // Fill the cache in one prefill, exactly as generation does before its
    // first decode step.
    auto fill = [&](int n) {
        gAttnMeta.previous = 0;
        gAttnMeta.remove = 0;
        gAttnMeta.add = n;
        auto outs = attn->onForward({Qp, Kp, Vp, Mask});
        outs.back()->readMap<float>();
        outs.back()->unMap();
        gAttnMeta.sync();
    };
    auto decodeOnce = [&]() {
        gAttnMeta.remove = 0;
        gAttnMeta.add = 1;
        auto outs = attn->onForward({Q1, K1, V1, Mask});
        gAttnMeta.sync();
        return outs.back();
    };
    fill(ctx);
    for (int i = 0; i < 3; ++i) {
        auto out = decodeOnce();
        out->readMap<float>();
        out->unMap();
    }
    MNN::Timer timer;
    VARP last;
    for (int i = 0; i < loop; ++i) {
        last = decodeOnce();
    }
    last->readMap<float>();
    float ms = (float)timer.durationInUs() / 1000.0f / (float)loop / (float)kLayerNum;
    last->unMap();
    // The cache grew by the warmup + timed steps, so report the midpoint length
    // the timed window actually saw.
    const double kvLen = ctx + 3 + loop / 2.0;
    const double kvGBps = 2.0 * kvLen * kKvNumHead * kHeadDim * 2.0 / (ms * 1e-3) / 1e9;
    if (round < 0) {
        return ms;
    }
    MNN_PRINT("r%d ctx=%-5d  per-layer=%8.4f ms (%6.1f GB/s KV)  x%d layers=%7.3f ms\n", round, ctx, ms, kvGBps,
              kLayerNum, ms * kLayerNum);
    return ms;
}

} // namespace

class LlmAttentionSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        selectModel();
        MNN_PRINT("\n===== %s prefill Attention op (q=%d kv=%d dim=%d, causal) =====\n", kModelName, kNumHead,
                  kKvNumHead, kHeadDim);
        MNN_PRINT("forwardType=%d precision=%d\n", st.forwardType, st.precision);
        std::vector<int> seqs = {128, 256, 512, 1024, 2048, 4096};
        // A full sweep heats the device before the shape under test runs, and
        // run-to-run spread at seq4096 swamps a 5-10% kernel change.
        // MNN_ATTN_SEQ pins the sweep to one length, MNN_ATTN_ROUNDS raises the
        // repeat count, and the summary reports the minimum: contention only
        // ever adds time.
        if (auto seqEnv = getenv("MNN_ATTN_SEQ")) {
            seqs = {atoi(seqEnv)};
        }
        int rounds = 3;
        if (auto roundEnv = getenv("MNN_ATTN_ROUNDS")) {
            rounds = std::max(1, atoi(roundEnv));
        }
        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        for (int seq : seqs) {
            benchOne(seq, 2, -1);
        }
        std::vector<float> best(seqs.size(), 1e30f);
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < seqs.size(); ++i) {
                best[i] = std::min(best[i], benchOne(seqs[i], seqs[i] >= 2048 ? 5 : 10, round));
            }
        }
        MNN_PRINT("--- best of %d ---\n", rounds);
        for (size_t i = 0; i < seqs.size(); ++i) {
            const int seq = seqs[i];
            double causalFlops = 2.0 * 2.0 * kNumHead * kHeadDim * ((double)seq * (seq + 1) / 2.0);
            MNN_PRINT("BEST seq=%-5d  per-layer=%8.3f ms (%6.2f TFLOPS causal)\n", seq, best[i],
                      causalFlops / (best[i] * 1e6) / 1000.0);
        }
        return true;
    }
};

class LlmAttentionDecodeSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        selectModel();
        MNN_PRINT("\n===== %s decode Attention op (q=1 kv=%d dim=%d) =====\n", kModelName, kKvNumHead, kHeadDim);
        MNN_PRINT("forwardType=%d precision=%d\n", st.forwardType, st.precision);
        std::vector<int> ctxs = {128, 512, 1024, 2048, 4096};
        if (auto ctxEnv = getenv("MNN_ATTN_CTX")) {
            ctxs = {atoi(ctxEnv)};
        }
        int rounds = 3;
        if (auto roundEnv = getenv("MNN_ATTN_ROUNDS")) {
            rounds = std::max(1, atoi(roundEnv));
        }
        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        for (int ctx : ctxs) {
            benchDecode(ctx, 8, -1);
        }
        std::vector<float> best(ctxs.size(), 1e30f);
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < ctxs.size(); ++i) {
                best[i] = std::min(best[i], benchDecode(ctxs[i], 32, round));
            }
        }
        MNN_PRINT("--- best of %d ---\n", rounds);
        for (size_t i = 0; i < ctxs.size(); ++i) {
            const double kvLen = ctxs[i] + 3 + 16.0;
            const double kvGBps = 2.0 * kvLen * kKvNumHead * kHeadDim * 2.0 / (best[i] * 1e-3) / 1e9;
            MNN_PRINT("BEST ctx=%-5d  per-layer=%8.4f ms (%6.1f GB/s KV)  x%d layers=%7.3f ms\n", ctxs[i], best[i],
                      kvGBps, kLayerNum, best[i] * kLayerNum);
        }
        return true;
    }
};

MNNTestSuiteRegister(LlmAttentionSpeedTest, "speed/LlmAttn");
MNNTestSuiteRegister(LlmAttentionDecodeSpeedTest, "speed/LlmAttnDecode");
#endif
