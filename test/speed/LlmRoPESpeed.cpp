//
//  LlmRoPESpeed.cpp
//  MNNTests
//
//  RoPE-op-only speed benchmark at Qwen3 prefill shapes, for attribution
//  against an external reference implementation. The op under test is the
//  production one: the exporter's
//  LlmExporter::FusedRoPE lowers to OpType_RoPE with q_norm/k_norm RMSNorm
//  children (mnn_converter.py rebuild_rope), so one dispatch covers the whole
//  head-to-tail span
//
//    C4-packed q/k  ->  per-head RMSNorm(gamma)  ->  rotary half-split
//                   ->  NHWC [1, seq, head, dim]
//
//  which is what the reference mirror (qwen3_rope_mlx.py) has to reproduce:
//  reshape, a fused RMSNorm on q and k, then a fused rope on both.
//
//  The op is pure bandwidth: it touches q+k once each for the norm pass, once
//  more for the rotation, and reads a precomputed cos/sin table, so the figure
//  of merit is GB/s and not FLOPS.
//
//  Usage:
//    ./run_test.out speed/LlmRoPE 1 2   # Metal, precision Low (fp16)
//    MNN_QWEN3_MODEL=4b MNN_ROPE_SEQ=4096 ./run_test.out speed/LlmRoPE 1 2
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

// POD table so no dynamic initializer is emitted.
struct ModelShape {
    const char* name;
    int numHead;
    int kvNumHead;
    int headDim;
    int layers;
};
constexpr ModelShape kModels[] = {
    {"Qwen3-0.6B", 16, 8, 128, 28},
    {"Qwen3-1.7B", 16, 8, 128, 28},
    {"Qwen3-4B", 32, 8, 128, 36},
    {"Qwen3-8B", 32, 8, 128, 36},
};
constexpr const char* kModelKeys[] = {"0.6b", "1.7b", "4b", "8b"};

static const ModelShape* gModel = &kModels[0];

static void selectModel() {
    auto env = getenv("MNN_QWEN3_MODEL");
    if (nullptr == env) {
        return;
    }
    for (int i = 0; i < (int)(sizeof(kModelKeys) / sizeof(kModelKeys[0])); ++i) {
        if (0 == strcmp(env, kModelKeys[i])) {
            gModel = &kModels[i];
            return;
        }
    }
}

// How many independent RoPE ops share one onForward. A single-op module pays the
// whole Express/Session per-forward cost on one op; the real model pays it once
// for all layers, so MNN_ROPE_LAYERS reproduces that structure.
static int gLayers = 1;

static std::shared_ptr<Module> makeRopeModule(int layers) {
    const int numHead = gModel->numHead;
    const int kvNumHead = gModel->kvNumHead;
    const int headDim = gModel->headDim;

    auto q = _Input({1, numHead * headDim, 1, 1}, NC4HW4);
    auto k = _Input({1, kvNumHead * headDim, 1, 1}, NC4HW4);
    auto cos = _Input({1, 1, headDim}, NCHW);
    auto sin = _Input({1, 1, headDim}, NCHW);

    std::vector<VARP> outputs;
    for (int l = 0; l < layers; ++l) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_RoPE;
        op->main.type = OpParameter_RoPEParam;
        op->main.value = new RoPEParamT;
        auto param = op->main.AsRoPEParam();
        param->rope_cut_head_dim = headDim;
        param->num_head = numHead;
        param->kv_num_head = kvNumHead;
        param->head_dim = headDim;
        // Qwen3 normalizes each head before rotating it; both norms are RMSNorm
        // over head_dim. Absent them the kernel takes a different (non-simdgroup)
        // branch entirely, so they are not optional for a production reading.
        // Per-layer gamma keeps the ops from collapsing into one under
        // common-subexpression elimination. Perturbing an input instead would
        // cost a full extra read+write pass over q, which at seq=4096 is as
        // large as the op being measured.
        const float gamma = 1.0f + 0.001f * (float)l;
        param->q_norm.reset(new LayerNormT);
        param->q_norm->epsilon = 1e-6f;
        param->q_norm->gamma = std::vector<float>(headDim, gamma);
        param->q_norm->axis = {-1};
        param->q_norm->useRMSNorm = true;
        param->k_norm.reset(new LayerNormT);
        param->k_norm->epsilon = 1e-6f;
        param->k_norm->gamma = std::vector<float>(headDim, gamma);
        param->k_norm->axis = {-1};
        param->k_norm->useRMSNorm = true;
        auto expr = Expr::create(std::move(op), {q, k, cos, sin}, 2);
        outputs.push_back(Variable::create(expr, 0));
        outputs.push_back(Variable::create(expr, 1));
    }
    auto buffer = Variable::save(outputs);

    auto status = MNNTestSuite::get()->pStaus;
    ScheduleConfig config;
    config.type = (MNNForwardType)status.forwardType;
    config.numThread = 1;
    BackendConfig bnConfig;
    bnConfig.memory = BackendConfig::Memory_Low;
    bnConfig.precision = (BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
}

// A host _Input var is re-uploaded (and fp32->fp16 converted) on every
// onForward, which lands inside the timed loop. The external baseline feeds
// arrays that are already device-resident, so counting the transfer makes the
// comparison unfair. Multiplying by 1 yields a computed var cached on the GPU
// backend. MNN_ROPE_HOSTIN=1 restores the host path.
static VARP deviceResident(VARP var) {
    if (nullptr == getenv("MNN_ROPE_HOSTIN")) {
        return var * _Scalar<float>(1.0f);
    }
    return var;
}

// C4-packed activation, laid out the way the qkv FusedLinear op emits it.
static VARP makeQkv(int seq, int channel, float scale) {
    auto var = _Input({seq, channel, 1, 1}, NC4HW4);
    auto ptr = var->writeMap<float>();
    for (int i = 0; i < seq * channel; ++i) {
        ptr[i] = ((float)(i % 13) - 6.0f) * scale;
    }
    var->unMap();
    return deviceResident(var);
}

// Precomputed rotary table, [1, seq, headDim], both halves duplicated exactly as
// Llm's rope embedding feeds it.
static VARP makeTrig(int seq, bool isCos) {
    const int headDim = gModel->headDim;
    const int half = headDim / 2;
    auto var = _Input({1, seq, headDim}, NCHW);
    auto ptr = var->writeMap<float>();
    for (int t = 0; t < seq; ++t) {
        for (int i = 0; i < half; ++i) {
            float angle = (float)t / std::pow(1000000.0f, (float)(2 * i) / (float)headDim);
            float v = isCos ? std::cos(angle) : std::sin(angle);
            ptr[t * headDim + i] = v;
            ptr[t * headDim + i + half] = v;
        }
    }
    var->unMap();
    return deviceResident(var);
}

static float benchOne(int seq, int loop, int round) {
    const int numHead = gModel->numHead;
    const int kvNumHead = gModel->kvNumHead;
    const int headDim = gModel->headDim;

    auto rope = makeRopeModule(gLayers);
    auto q = makeQkv(seq, numHead * headDim, 0.11f);
    auto k = makeQkv(seq, kvNumHead * headDim, -0.07f);
    auto cos = makeTrig(seq, true);
    auto sin = makeTrig(seq, false);

    auto forwardOnce = [&]() { return rope->onForward({q, k, cos, sin}); };
    for (int i = 0; i < 3; ++i) {
        auto out = forwardOnce();
        out[0]->readMap<float>();
        out[0]->unMap();
    }
    // Enqueue-only timing with one sync after the loop: mapping the outputs
    // every iteration costs more than the kernel itself at short seq.
    Timer timer;
    std::vector<VARP> last;
    for (int i = 0; i < loop; ++i) {
        last = forwardOnce();
    }
    last[0]->readMap<float>();
    float ms = (float)timer.durationInUs() / 1000.0f / (float)loop / (float)gLayers;
    last[0]->unMap();
    // fp16 storage: q and k are read once and written once, and the cos/sin
    // tables are read once. This is the kernel's whole traffic, so it is also
    // its floor.
    double bytes = 2.0 * 2.0 * (double)seq * (numHead + kvNumHead) * headDim + 2.0 * 2.0 * (double)seq * headDim;
    if (round < 0) {
        return ms;
    }
    MNN_PRINT("r%d seq=%-5d  per-layer=%8.4f ms (%6.1f GB/s)  x%d layers=%8.3f ms\n", round, seq, ms,
              bytes / (ms * 1e6), gModel->layers, ms * gModel->layers);
    return ms;
}

} // namespace

class LlmRoPESpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        selectModel();
        MNN_PRINT("\n===== %s RoPE op (q=%d kv=%d dim=%d, q_norm+k_norm) =====\n", gModel->name, gModel->numHead,
                  gModel->kvNumHead, gModel->headDim);
        MNN_PRINT("forwardType=%d precision=%d memory=Low\n", st.forwardType, st.precision);
        std::vector<int> seqs = {128, 256, 512, 1024, 2048, 4096};
        if (auto seqEnv = getenv("MNN_ROPE_SEQ")) {
            seqs = {atoi(seqEnv)};
        }
        int rounds = 3;
        if (auto roundEnv = getenv("MNN_ROPE_ROUNDS")) {
            rounds = std::max(1, atoi(roundEnv));
        }
        if (auto layerEnv = getenv("MNN_ROPE_LAYERS")) {
            gLayers = std::max(1, atoi(layerEnv));
        }
        MNN_PRINT("layers-per-forward=%d\n", gLayers);
        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        for (int seq : seqs) {
            benchOne(seq, 2, -1);
        }
        std::vector<float> best(seqs.size(), 1e30f);
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < seqs.size(); ++i) {
                best[i] = std::min(best[i], benchOne(seqs[i], seqs[i] >= 2048 ? 10 : 20, round));
            }
        }
        MNN_PRINT("--- best of %d ---\n", rounds);
        const int heads = gModel->numHead + gModel->kvNumHead;
        for (size_t i = 0; i < seqs.size(); ++i) {
            double bytes =
                2.0 * 2.0 * (double)seqs[i] * heads * gModel->headDim + 2.0 * 2.0 * (double)seqs[i] * gModel->headDim;
            MNN_PRINT("BEST seq=%-5d  per-layer=%8.4f ms (%6.1f GB/s)  x%d layers=%8.3f ms\n", seqs[i], best[i],
                      bytes / (best[i] * 1e6), gModel->layers, best[i] * gModel->layers);
        }
        return true;
    }
};

MNNTestSuiteRegister(LlmRoPESpeedTest, "speed/LlmRoPE");
#endif
