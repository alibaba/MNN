//
//  LlmLinearSpeed.cpp
//  MNNTests
//
//  Every linear-layer speed benchmark of a decoder LLM, over one shared model
//  shape table. Four families, all reading the same `kModels` entry so a new
//  model only has to be described once:
//
//    speed/LlmPrefill          prefill GEMM per projection (float + int4 block)
//    speed/LlmPrefillAttn      attention QK^T / PV as batch MatMul
//    speed/LlmPrefillDownDiag  K/N/M isolation sweeps for a slow GEMM shape
//    speed/LlmFusedLinear      OpType_FusedLinear seq sweep (decode + prefill)
//    speed/LlmConvDecode       plain int4 Convolution decode GEMV
//    speed/LlmDecodeGemv       whole per-token linear budget, grouped by layer
//
//  Shapes come from each converted model's llm.mnn.json, not from config.json,
//  because the exporter's fusion decides what the GPU actually dispatches. Two
//  things only the dump reveals:
//    - Qwen3.5's attention qkv carries a fourth 2048-wide slice (the output
//      gate), so it is a 4-member FusedLinear, not 3.
//    - Qwen3.5's linear-attention layers project through one FusedLinear with
//      slices [conv_dim, z, b, a] = [6144, 2048, 16, 16]; those two 16-wide
//      members are real dispatches at a shape nothing else in the suite covers.
//
//  Weights are 4-bit block-quantized and the executor runs Memory_Low, which is
//  what gates Metal's fused decode-GEMV path (is2sgDecodePipeline). Fusion only
//  exists at decode: every prefill shape dispatches per member, so the seq
//  sweep of speed/LlmFusedLinear crosses two different regimes.
//
//  Usage:
//    ./run_test.out speed/LlmDecodeGemv 1 2        # Metal, precision Low (fp16)
//    MNN_QWEN3_MODEL=4b ./run_test.out speed/LlmFusedLinear 1 2
//
//  Precision must be 2 (Low/fp16). Precision 1 silently falls back off every
//  fp16-only Metal kernel these benchmarks exist to measure.
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "CommonOpCreator.hpp"
#include "MNNTestSuite.h"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

// ---------------------------------------------------------------------------
// Shared: model shape table and runtime setup
// ---------------------------------------------------------------------------

// MNN_DECODE_GEMV_BITS selects the weight quant width (2/3/4/8, default 4). At a
// fixed byte footprint a wider weight means fewer MACs per byte, so sweeping it
// separates a memory roof from a kernel-side limit.
int bits() {
    static const int sBits = [] {
        const char* env = getenv("MNN_DECODE_GEMV_BITS");
        if (nullptr != env) {
            const int n = atoi(env);
            if (2 == n || 3 == n || 4 == n || 8 == n) {
                return n;
            }
        }
        return 4;
    }();
    return sBits;
}

// MNN_DECODE_GEMV_BLOCK selects the quant block size. It drives the Metal W16
// decode-GEMV's compile-time quads-per-block (block/4 = 8/16/32/64), so it is
// the axis along which that wide-load specialization gets priced. Every ic in
// the model table is a multiple of 256, so all four values stay eligible.
int blockSize() {
    static const int sSize = [] {
        const char* env = getenv("MNN_DECODE_GEMV_BLOCK");
        if (nullptr != env) {
            const int n = atoi(env);
            if (32 == n || 64 == n || 128 == n || 256 == n) {
                return n;
            }
        }
        return 64;
    }();
    return sSize;
}

// A FusedLinear's member projections. Fixed-size POD so the model table stays
// in .rodata (AGENTS.md forbids namespace-scope dynamic initialization).
struct SliceList {
    int n;
    int oc[4];
};

// `attnLayers` counts the layers carrying `qkv`; the rest carry `linearIn`
// (hybrid models only, n == 0 elsewhere). o_proj's input width is qkv.oc[0],
// the q projection, so it needs no field of its own.
struct ModelShape {
    const char* name;
    const char* key;
    int hidden;
    int layers;
    int attnLayers;
    int inter;  // ffn intermediate; gate and up are each this wide
    int vocab;
    SliceList qkv;
    SliceList linearIn;
};

constexpr ModelShape kModels[] = {
    {"Qwen3-0.6B", "0.6b", 1024, 28, 28, 3072, 151936, {3, {2048, 1024, 1024}}, {0, {0}}},
    {"Qwen3-1.7B", "1.7b", 2048, 28, 28, 6144, 151936, {3, {2048, 1024, 1024}}, {0, {0}}},
    {"Qwen3-4B", "4b", 2560, 36, 36, 9728, 151936, {3, {4096, 1024, 1024}}, {0, {0}}},
    {"Qwen3-8B", "8b", 4096, 36, 36, 12288, 151936, {3, {4096, 1024, 1024}}, {0, {0}}},
    // Qwen3.5 is hybrid: full_attention_interval 4 => 6 of 24 layers attend.
    {"Qwen3.5-0.8B", "3.5-0.8b", 1024, 24, 6, 3584, 248320, {4, {2048, 512, 512, 2048}}, {4, {6144, 2048, 16, 16}}},
    {"Qwen3.5-2B", "3.5-2b", 2048, 24, 6, 6144, 248320, {4, {2048, 512, 512, 2048}}, {4, {6144, 2048, 16, 16}}},
};

const ModelShape& model() {
    const char* env = getenv("MNN_QWEN3_MODEL");
    if (nullptr != env) {
        if (0 == strcmp(env, "3.5")) {
            env = "3.5-2b"; // alias kept for the existing A/B scripts
        }
        for (const auto& m : kModels) {
            if (0 == strcmp(env, m.key)) {
                return m;
            }
        }
    }
    return kModels[0];
}

std::vector<int> toVec(const SliceList& s) {
    return std::vector<int>(s.oc, s.oc + s.n);
}

int sumOc(const SliceList& s) {
    int total = 0;
    for (int i = 0; i < s.n; ++i) {
        total += s.oc[i];
    }
    return total;
}

// Memory_Low is what keeps the weights quantized, and that is what gates
// Metal's quantized decode-GEMV pipeline. Production LLM runs low-memory too.
std::shared_ptr<Executor::RuntimeManager> makeRuntime(int thread = 1) {
    auto status = MNNTestSuite::get()->pStaus;
    ScheduleConfig config;
    config.type      = (MNNForwardType)status.forwardType;
    config.numThread = thread > 0 ? thread : 1;
    BackendConfig bnConfig;
    bnConfig.precision   = (BackendConfig::PrecisionMode)status.precision;
    bnConfig.power       = (BackendConfig::PowerMode)status.power;
    bnConfig.memory      = BackendConfig::Memory_Low;
    config.backendConfig = &bnConfig;
    return std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(config));
}

// A host _Input var is re-uploaded (and fp32->fp16 converted) on every
// onForward: 32 MB per call at seq=4096, measured at 0.6-1.2 ms, which lands
// inside the timed loop. Cross-framework baselines feed device-resident
// arrays, so counting the transfer makes the comparison unfair. Multiplying by
// 1 yields a computed var cached on the GPU backend, so onForward sees a device
// tensor and skips the upload.
VARP makeActivation(int seq, int channel, float scale, bool deviceResident) {
    auto var = _Input({seq, channel, 1, 1}, NC4HW4);
    auto ptr = var->writeMap<float>();
    for (int i = 0; i < seq * channel; ++i) {
        ptr[i] = ((float)(i % 13) - 6.0f) * scale;
    }
    var->unMap();
    if (deviceResident) {
        var = var * _Scalar<float>(1.0f);
    }
    return var;
}

// int4 groups of `blockSize()` carry one fp16 scale and one fp16 bias, so a
// group of 64 weights costs 32 + 2 + 2 bytes. At seq=1 this traffic, not flops,
// sets the floor.
double weightBytes(double params) {
    return params * bits() / 8.0 + params / blockSize() * 4.0;
}

} // namespace

// ---------------------------------------------------------------------------
// Prefill GEMM shapes (Conv1x1), float and int4-block arms
// ---------------------------------------------------------------------------

namespace {

VARP buildFloatConv1x1(VARP x, int ic, int oc) {
    std::vector<float> weight(oc * ic);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = ((float)(i % 127) - 63.0f) / 1000.0f;
    }
    std::vector<float> bias(oc, 0.0f);
    return _Conv(std::move(weight), std::move(bias), x, {ic, oc}, {1, 1}, PaddingMode::VALID, {1, 1}, {1, 1}, 1,
                 {0, 0}, false, false);
}

VARP buildQuantConv1x1(VARP x, int ic, int oc) {
    const int block = blockSize();
    MNN_ASSERT(ic % block == 0);
    const int blockNum = ic / block;

    std::vector<float> weightFp32(oc * ic);
    for (size_t i = 0; i < weightFp32.size(); ++i) {
        weightFp32[i] = ((float)(i % 127) - 63.0f) / 1000.0f;
    }
    std::vector<float> wScale(2 * oc * blockNum);
    for (int k = 0; k < oc; ++k) {
        for (int b = 0; b < blockNum; ++b) {
            wScale[2 * (k * blockNum + b)]     = -0.5f;
            wScale[2 * (k * blockNum + b) + 1] = 0.01f;
        }
    }
    std::vector<float> bias(oc, 0.0f);
    return _HybridConv(weightFp32, std::move(bias), wScale, x, {ic, oc}, {1, 1}, PaddingMode::CAFFE, {1, 1}, {1, 1}, 1,
                       {0, 0}, false, false, bits(), true);
}

const char* quantArmName() {
    static char sName[16] = {0};
    if (0 == sName[0]) {
        snprintf(sName, sizeof(sName), "w%db%d", bits(), blockSize());
    }
    return sName;
}

// mode: 0 = float dense, 1 = int4 block-quantized
void benchLinear(const char* name, int M, int K, int N, int mode, int thread) {
    // Graph is wrapped in a Module so every onForward recomputes; an Express
    // VARP caches its result and would need a host writeMap to dirty the input,
    // which drags a 159 MB upload into the measurement.
    auto x = _Input({1, K, 1, M}, NC4HW4, halide_type_of<float>());
    VARP y = (mode == 0) ? buildFloatConv1x1(x, K, N) : buildQuantConv1x1(x, K, N);
    x.fix(VARP::INPUT);
    auto buffer = Variable::save({y});
    std::shared_ptr<Module> mod(
        Module::load({}, {}, (const uint8_t*)buffer.data(), buffer.size(), makeRuntime(thread)));

    auto input = _Input({1, K, 1, M}, NC4HW4, halide_type_of<float>());
    auto iPtr  = input->writeMap<float>();
    ::memset(iPtr, 0, input->getInfo()->size * sizeof(float));
    input->unMap();
    input = input * _Scalar<float>(1.0f); // park the activation on the GPU

    for (int w = 0; w < 3; ++w) {
        auto out = mod->onForward({input});
        out[0]->readMap<float>();
        out[0]->unMap();
    }

    const int LOOP = 10;
    MNN::Timer _t;
    std::vector<VARP> last;
    for (int i = 0; i < LOOP; ++i) {
        last = mod->onForward({input});
    }
    last[0]->readMap<float>();
    float avgMs = (float)_t.durationInUs() / 1000.0f / (float)LOOP;
    last[0]->unMap();
    double flops  = 2.0 * (double)M * (double)K * (double)N;
    double tflops = flops / (avgMs * 1e6) / 1000.0;
    MNN_PRINT("%-10s %-6s M=%-5d K=%-5d N=%-6d  wall=%8.3f ms (%6.2f TFLOPS)\n", name,
              mode == 0 ? "float" : quantArmName(), M, K, N, avgMs, tflops);
    fflush(stdout);
}

// {name, M, K, N}; M is the prompt length.
struct LinearShape {
    const char* name;
    int M, K, N;
};

int prefillSeq() {
    if (auto env = getenv("MNN_PREFILL_SEQ")) {
        return std::max(1, atoi(env));
    }
    return 2048;
}

// o_proj and down are the two shapes speed/LlmFusedLinear cannot reach: they
// are plain Conv1x1 ops in the model (no rms_norm to fuse), though they still
// dispatch the same fused-int4 M64 GEMM kernel. At 4B they carry 290 of the
// 827 GFLOP/layer of prefill linear work.
std::vector<LinearShape> prefillShapes() {
    const int S   = prefillSeq();
    const auto& m = model();
    std::vector<LinearShape> shapes = {{"qkv", S, m.hidden, sumOc(m.qkv)}};
    if (0 != m.linearIn.n) {
        shapes.push_back({"linear_in", S, m.hidden, sumOc(m.linearIn)});
    }
    shapes.push_back({"o_proj", S, m.qkv.oc[0], m.hidden});
    shapes.push_back({"gate_up", S, m.hidden, 2 * m.inter});
    shapes.push_back({"down", S, m.inter, m.hidden});
    shapes.push_back({"lm_head", 1, m.hidden, m.vocab}); // production: last token only
    if (nullptr != getenv("MNN_PREFILL_LMHEADS")) {
        // [S, vocab] fp32 is 2.5 GB at S=4096 and will exhaust memory.
        shapes.push_back({"lm_headS", S, m.hidden, m.vocab});
    }
    return shapes;
}

} // namespace

class LlmPrefillLinearTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== %s prefill S=%d linear shapes (Conv1x1 GEMM) =====\n", model().name, prefillSeq());
        MNN_PRINT("forwardType=%d precision=%d thread=%d block=%d\n", st.forwardType, st.precision, st.thread,
                  blockSize());
        for (auto& s : prefillShapes()) {
            // The float arm materializes dense fp32 weights on the host: at
            // Qwen3.5-2B's 2048 x 248320 lm_head that is 2 GB and the process
            // gets OOM-killed before the quantized row prints. Production
            // lm_head is quantized, so the float reading there is not worth it.
            if ((double)s.K * (double)s.N * 4.0 > 1e9) {
                MNN_PRINT("%-10s float  M=%-5d K=%-5d N=%-6d  skipped (dense fp32 weights >1 GB)\n", s.name, s.M, s.K,
                          s.N);
            } else {
                benchLinear(s.name, s.M, s.K, s.N, 0, st.thread);
            }
            benchLinear(s.name, s.M, s.K, s.N, 1, st.thread);
        }
        return true;
    }
};

// Attention inner matmuls, expressed as batch MatMul (batch = 16 Q heads).
// These are NOT the production path (MNN runs fused flash attention in
// MetalAttention); they quantify the raw batched-matmul cost of the shapes.
class LlmPrefillAttnTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== prefill S=2048 attention matmul shapes, 16 heads (batch MatMul) =====\n");
        MNN_PRINT("forwardType=%d precision=%d thread=%d\n", st.forwardType, st.precision, st.thread);

        struct AttnShape {
            const char* name;
            int batch, e, l, h; // [batch,e,l] x [batch,l,h]
        };
        std::vector<AttnShape> shapes = {
            {"qk", 16, 2048, 128, 2048},
            {"pv", 16, 2048, 2048, 128},
        };
        for (auto& s : shapes) {
            BackendConfig bnConfig;
            bnConfig.precision = (BackendConfig::PrecisionMode)st.precision;
            bnConfig.memory    = BackendConfig::Memory_Low;
            auto exe           = Executor::newExecutor((MNNForwardType)st.forwardType, bnConfig, st.thread);
            ExecutorScope scope(exe);

            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type        = MNN::OpType_MatMul;
            op->main.type   = MNN::OpParameter_MatMul;
            op->main.value  = new MNN::MatMulT;
            auto param      = op->main.AsMatMul();
            param->transposeA = false;
            param->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({s.batch, s.e, s.l});
            x1->resize({s.batch, s.l, s.h});
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            Variable::prepareCompute({y});
            x0.fix(VARP::INPUT);
            x1.fix(VARP::INPUT);

            for (int w = 0; w < 3; ++w) {
                ::memset(x0->writeMap<float>(), 0, x0->getInfo()->size * sizeof(float));
                ::memset(x1->writeMap<float>(), 0, x1->getInfo()->size * sizeof(float));
                y->readMap<float>();
            }
            const int LOOP = 10;
            MNN::Timer _t;
            for (int i = 0; i < LOOP; ++i) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
            float avgMs   = (float)_t.durationInUs() / 1000.0f / (float)LOOP;
            double flops  = 2.0 * (double)s.batch * (double)s.e * (double)s.l * (double)s.h;
            double tflops = flops / (avgMs * 1e6) / 1000.0;
            MNN_PRINT("%-12s [%d,%d,%d]x[%d,%d,%d]  wall=%8.3f ms (%6.2f TFLOPS)\n", s.name, s.batch, s.e, s.l, s.batch,
                      s.l, s.h, avgMs, tflops);
        }
        return true;
    }
};

// Root-cause diagnostics for the slow down_proj shape (K=3072, N=1024):
// K/N isolation variants + M sweep. Quantized arm only by default. Untimed
// warmup pass first, then 3 timed rounds (GPU clock ramp makes first-dispatch
// numbers unusable).
class LlmPrefillDownDiagTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== down-shape diagnosis (%s) =====\n", quantArmName());
        MNN_PRINT("forwardType=%d precision=%d thread=%d\n", st.forwardType, st.precision, st.thread);
        struct Shape {
            const char* name;
            int M, K, N;
        };
        std::vector<Shape> shapes = {
            {"down", 2048, 3072, 1024},    // the slow one
            {"down_t", 2048, 1024, 3072},  // same FLOPs, swapped K/N
            {"o_proj", 2048, 2048, 1024},  // small-N family
            {"gate_up", 2048, 1024, 6144}, // reference fast shape
            {"qkv", 2048, 1024, 4096},     // reference
            {"K3072N2048", 2048, 3072, 2048},
            {"K3072N3072", 2048, 3072, 3072},
            {"K3072N4096", 2048, 3072, 4096},
            {"K3072N6144", 2048, 3072, 6144},
            {"K1536N1024", 2048, 1536, 1024},
            {"K2048N1024", 2048, 2048, 1024},
            {"K4096N1024", 2048, 4096, 1024},
            {"K6144N1024", 2048, 6144, 1024},
            {"down_M256", 256, 3072, 1024},
            {"down_M512", 512, 3072, 1024},
            {"down_M1024", 1024, 3072, 1024},
            {"down_M4096", 4096, 3072, 1024},
            {"gup_M1024", 1024, 1024, 6144},
        };
        if (auto sweep = getenv("MNN_PREFILL_NSWEEP")) {
            if (0 == strcmp(sweep, "gu")) {
                // Verification arm: the two 4B shapes the M-sweep flagged as
                // behind the external baseline.
                shapes = {
                    {"qkv_M1024", 1024, 2560, 6144},
                    {"gup_M1024", 1024, 2560, 19456},
                    {"qkv_M4096", 4096, 2560, 6144},
                    {"gup_M4096", 4096, 2560, 19456},
                };
            } else if (0 == strcmp(sweep, "m")) {
                // Fine M sweep at the 4B qkv K/N, to localize the dense-float
                // cliff seen between M=2048 and M=4096. Run with
                // MNN_PREFILL_DIAG_MODE=0.
                shapes = {
                    {"M2048", 2048, 2560, 6144}, {"M2112", 2112, 2560, 6144}, {"M2176", 2176, 2560, 6144},
                    {"M2304", 2304, 2560, 6144}, {"M2560", 2560, 2560, 6144}, {"M4096", 4096, 2560, 6144},
                    {"M4224", 4224, 2560, 6144}, {"M4352", 4352, 2560, 6144}, {"M4480", 4480, 2560, 6144},
                    {"M4608", 4608, 2560, 6144}, {"M5120", 5120, 2560, 6144}, {"M6144", 6144, 2560, 6144},
                    {"M8192", 8192, 2560, 6144},
                };
            } else {
                // 4B gate_up is M=4096 K=2560 N=19456. Fix M/K and sweep N to see
                // whether MNN loses ground as N grows (the external baseline
                // gains ~2% from 6144 to 19456).
                shapes = {
                    {"N2560", 4096, 2560, 2560},   {"N5120", 4096, 2560, 5120},
                    {"N6144", 4096, 2560, 6144},   {"N10240", 4096, 2560, 10240},
                    {"N19456", 4096, 2560, 19456}, {"N30720", 4096, 2560, 30720},
                };
            }
        }
        int mode = 1;
        if (auto modeEnv = getenv("MNN_PREFILL_DIAG_MODE")) {
            mode = atoi(modeEnv);
        }
        MNN_PRINT("--- warmup pass (untimed) ---\n");
        for (auto& s : shapes) {
            benchLinear("warm", s.M, s.K, s.N, mode, st.thread);
        }
        for (int round = 0; round < 3; ++round) {
            MNN_PRINT("--- round %d ---\n", round);
            for (auto& s : shapes) {
                benchLinear(s.name, s.M, s.K, s.N, mode, st.thread);
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(LlmPrefillLinearTest, "speed/LlmPrefill");
MNNTestSuiteRegister(LlmPrefillAttnTest, "speed/LlmPrefillAttn");
MNNTestSuiteRegister(LlmPrefillDownDiagTest, "speed/LlmPrefillDownDiag");

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

// ---------------------------------------------------------------------------
// FusedLinear seq sweep: decode (fused GEMV) and prefill (per-member dispatch)
// ---------------------------------------------------------------------------

namespace {

std::shared_ptr<Module> makeFusedLinearModule(const std::vector<int>& ocs, bool gateUp, int layers) {
    const int hidden = model().hidden;
    auto residual    = _Input({1, hidden, 1, 1}, NC4HW4);
    auto hiddenIn    = _Input({1, hidden, 1, 1}, NC4HW4);

    std::vector<VARP> outputs;
    for (int l = 0; l < layers; ++l) {
        std::unique_ptr<OpT> op(new OpT);
        op->type                   = OpType_FusedLinear;
        op->main.type              = OpParameter_FusedLinearParam;
        op->main.value             = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param                 = op->main.AsFusedLinearParam();
        param->act_silu_mul        = gateUp;
        param->has_ln              = true;
        param->ln.reset(new LayerNormT);
        param->ln->epsilon    = 1e-6f;
        param->ln->gamma      = std::vector<float>(hidden, 1.0f);
        param->ln->beta       = std::vector<float>(hidden, 0.0f);
        param->ln->axis       = {-1};
        param->ln->useRMSNorm = true;
        for (size_t m = 0; m < ocs.size(); ++m) {
            // Distinct weights per layer: at decode the kernel is bound by
            // weight DRAM traffic, and sharing one weight set across layers
            // would let the cache serve them and inflate the result.
            param->convs.push_back(_blockQuantConv1x1(hidden, ocs[m], bits(), blockSize(), (int)m + 1 + l * 97));
        }
        // gate/up collapse to one output; qkv keeps one per member. Plus residual_out.
        const int numOut = (gateUp ? 1 : (int)ocs.size()) + 1;
        auto expr        = Expr::create(std::move(op), {residual, hiddenIn}, numOut);
        for (int i = 0; i < numOut; ++i) {
            outputs.push_back(Variable::create(expr, i));
        }
    }
    auto buffer = Variable::save(outputs);
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), makeRuntime()));
}

// How many independent FusedLinear ops share one onForward. A single-op module
// pays the whole Express/Session per-forward cost (~45 us measured) on one op,
// which swamps a decode step's few microseconds of GPU work; the real model pays
// it once for all layers. MNN_FL_LAYERS=28 reproduces that structure so seq=1
// readings measure the kernel instead of the harness.
int gLayers = 1;

float benchFusedLinear(const char* tag, const std::vector<int>& ocs, bool gateUp, int seq, int loop, int round,
                       int layerCount) {
    const int hidden = model().hidden;
    auto attn        = makeFusedLinearModule(ocs, gateUp, gLayers);
    // MNN_FL_HOSTIN=1 restores the per-call host upload (see makeActivation).
    const bool deviceResident = (nullptr == getenv("MNN_FL_HOSTIN"));
    auto residual             = makeActivation(seq, hidden, 0.11f, deviceResident);
    auto hiddenIn             = makeActivation(seq, hidden, -0.07f, deviceResident);

    auto forwardOnce = [&]() { return attn->onForward({residual, hiddenIn}); };
    for (int i = 0; i < 3; ++i) {
        auto out = forwardOnce();
        out[0]->readMap<float>();
        out[0]->unMap();
    }
    // Enqueue-only timing, one sync after the loop: mapping the output every
    // iteration costs more than the GEMM at decode shapes.
    Timer timer;
    std::vector<VARP> last;
    for (int i = 0; i < loop; ++i) {
        last = forwardOnce();
    }
    last[0]->readMap<float>();
    float ms = (float)timer.durationInUs() / 1000.0f / (float)loop / (float)gLayers;
    last[0]->unMap();
    if (round < 0) {
        return ms;
    }
    int totalOc = 0;
    for (int oc : ocs) {
        totalOc += oc;
    }
    double flops  = 2.0 * (double)seq * hidden * totalOc;
    double wBytes = weightBytes((double)totalOc * hidden);
    MNN_PRINT("%-9s r%d seq=%-5d  per-layer=%8.4f ms (%7.1f GFLOPS, %6.1f GB/s W)  x%d layers=%8.2f ms\n", tag, round,
              seq, ms, flops / (ms * 1e6), wBytes / (ms * 1e6), layerCount, ms * layerCount);
    return ms;
}

} // namespace

class LlmFusedLinearSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st      = MNNTestSuite::get()->pStaus;
        const auto& m = model();
        MNN_PRINT("\n===== %s FusedLinear (int%d block%d, hidden=%d) =====\n", m.name, bits(), blockSize(), m.hidden);
        MNN_PRINT("forwardType=%d precision=%d memory=Low\n", st.forwardType, st.precision);
        const std::vector<int> qkvOcs      = toVec(m.qkv);
        const std::vector<int> linearInOcs = toVec(m.linearIn);
        const std::vector<int> gateUpOcs   = {m.inter, m.inter};
        // Attribution shapes. Shape inference only accepts 3..4 convs without
        // act_silu_mul, so the epilogue is isolated by matching gateup's total
        // width with three members instead of two: nosilu6144 has the same
        // flops as gateup but no silu-mul. wide9216 probes whether per-output
        // cost keeps scaling once the N grid grows past gateup's.
        const bool diag                   = nullptr != getenv("MNN_FL_DIAG");
        const std::vector<int> nosilu6144 = {2048, 2048, 2048};
        const std::vector<int> wide9216   = {3072, 3072, 3072};
        // Near-zero GEMM width: isolates the shape-independent prologue
        // (rms-norm, plus any per-call input transfer).
        const std::vector<int> tiny192 = {64, 64, 64};
        // seq 1 is decode (fused GEMV); the rest are prefill (per-member dispatch).
        std::vector<int> seqs = {1, 128, 256, 512, 1024, 2048, 4096};
        // The GPU is noisy enough at seq4096 (+-40% run to run) that a full
        // sweep cannot resolve a 5-10% kernel change: the earlier shapes heat
        // the device before the shape under test runs. MNN_FL_SEQ pins the
        // sweep to one length and MNN_FL_ROUNDS raises the repeat count, so an
        // A/B compares best-of-N on the shape being optimised only.
        if (auto seqEnv = getenv("MNN_FL_SEQ")) {
            seqs = {atoi(seqEnv)};
        }
        int rounds = 3;
        if (auto roundEnv = getenv("MNN_FL_ROUNDS")) {
            rounds = std::max(1, atoi(roundEnv));
        }
        if (auto layerEnv = getenv("MNN_FL_LAYERS")) {
            gLayers = std::max(1, atoi(layerEnv));
        }
        MNN_PRINT("layers-per-forward=%d\n", gLayers);

        struct Shape {
            const char* tag;
            const std::vector<int>* ocs;
            bool gateUp;
            int layerCount;
        };
        std::vector<Shape> shapes = {{"qkv", &qkvOcs, false, m.attnLayers}};
        if (!linearInOcs.empty()) {
            shapes.push_back({"linear_in", &linearInOcs, false, m.layers - m.attnLayers});
        }
        shapes.push_back({"gateup", &gateUpOcs, true, m.layers});
        if (diag) {
            shapes.push_back({"nosilu6144", &nosilu6144, false, m.layers});
            shapes.push_back({"wide9216", &wide9216, false, m.layers});
            shapes.push_back({"tiny192", &tiny192, false, m.layers});
        }

        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        for (int seq : seqs) {
            for (const auto& s : shapes) {
                benchFusedLinear(s.tag, *s.ocs, s.gateUp, seq, 2, -1, s.layerCount);
            }
        }
        // best[shape][seq]
        std::vector<std::vector<float>> best(shapes.size(), std::vector<float>(seqs.size(), 1e30f));
        for (int round = 0; round < rounds; ++round) {
            for (size_t si = 0; si < seqs.size(); ++si) {
                const int seq  = seqs[si];
                const int loop = seq >= 2048 ? 5 : (seq == 1 ? 50 : 10);
                for (size_t i = 0; i < shapes.size(); ++i) {
                    float ms = benchFusedLinear(shapes[i].tag, *shapes[i].ocs, shapes[i].gateUp, seq, loop, round,
                                                shapes[i].layerCount);
                    best[i][si] = std::min(best[i][si], ms);
                }
            }
        }
        MNN_PRINT("--- best of %d ---\n", rounds);
        for (size_t i = 0; i < shapes.size(); ++i) {
            int totalOc = 0;
            for (int oc : *shapes[i].ocs) {
                totalOc += oc;
            }
            for (size_t si = 0; si < seqs.size(); ++si) {
                double flops  = 2.0 * (double)seqs[si] * m.hidden * totalOc;
                double wBytes = weightBytes((double)totalOc * m.hidden);
                MNN_PRINT("BEST %-11s seq=%-5d  per-layer=%8.4f ms (%7.1f GFLOPS, %6.1f GB/s W)  x%d layers=%8.2f ms\n",
                          shapes[i].tag, seqs[si], best[i][si], flops / (best[i][si] * 1e6),
                          wBytes / (best[i][si] * 1e6), shapes[i].layerCount, best[i][si] * shapes[i].layerCount);
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(LlmFusedLinearSpeedTest, "speed/LlmFusedLinear");

// ---------------------------------------------------------------------------
// Plain (non-fused) int4 Convolution decode GEMV
// ---------------------------------------------------------------------------

namespace {

std::shared_ptr<Module> makeConvModule(int ic, int oc, int layers) {
    auto input = _Input({1, ic, 1, 1}, NC4HW4);
    std::vector<VARP> outputs;
    for (int l = 0; l < layers; ++l) {
        std::unique_ptr<OpT> op(new OpT);
        op->type                   = OpType_Convolution;
        op->main.type              = OpParameter_Convolution2D;
        op->main.value             = _blockQuantConv1x1(ic, oc, bits(), blockSize(), l + 1).release();
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        outputs.emplace_back(Variable::create(Expr::create(std::move(op), {input})));
    }
    auto buffer = Variable::save(outputs);
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), makeRuntime()));
}

float benchConvModule(const std::shared_ptr<Module>& module, VARP input, const std::string& tag, int ic, int oc,
                      int layers, int loop, int round) {
    auto forwardOnce = [&]() { return module->onForward({input}); };
    for (int i = 0; i < 3; ++i) {
        auto out = forwardOnce();
        out[0]->readMap<float>();
        out[0]->unMap();
    }
    // Enqueue-only timing, one sync after the loop.
    Timer timer;
    std::vector<VARP> last;
    for (int i = 0; i < loop; ++i) {
        last = forwardOnce();
    }
    last[0]->readMap<float>();
    float ms = (float)timer.durationInUs() / 1000.0f / (float)loop / (float)layers;
    last[0]->unMap();
    if (round < 0) {
        return ms;
    }
    double wBytes = weightBytes((double)oc * ic);
    double gflops = 2.0 * (double)oc * ic / (ms * 1e6);
    MNN_PRINT("%-7s r%d  per-layer=%8.4f ms (%7.1f GFLOPS, %6.1f GB/s W)  x%d layers=%8.3f ms\n", tag.c_str(),
              round, ms,
              gflops, wBytes / (ms * 1e6), layers, ms * layers);
    return ms;
}

float benchConv(const std::string& tag, int ic, int oc, int layers, int loop, int round) {
    auto module = makeConvModule(ic, oc, layers);
    auto input  = makeActivation(1, ic, 0.07f, true);
    return benchConvModule(module, input, tag, ic, oc, layers, loop, round);
}

} // namespace

class LlmConvDecodeSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st      = MNNTestSuite::get()->pStaus;
        const auto& m = model();
        MNN_PRINT("\n===== %s plain conv decode GEMV (int%d block%d) =====\n", m.name, bits(), blockSize());
        MNN_PRINT("forwardType=%d precision=%d memory=Low\n", st.forwardType, st.precision);
        int rounds = 3;
        if (auto roundEnv = getenv("MNN_CD_ROUNDS")) {
            rounds = std::max(1, atoi(roundEnv));
        }
        struct Shape {
            std::string tag;
            int ic;
            int oc;
            int layers;
        };
        // The three shapes the model runs outside FusedLinear: o_proj
        // (heads*dim -> hidden), down_proj (inter -> hidden), lm_head
        // (hidden -> vocab). lm_head is a single op but inherently DRAM-bound.
        std::vector<Shape> shapes = {
            {"o_proj", m.qkv.oc[0], m.hidden, m.layers},
            {"down", m.inter, m.hidden, m.layers},
            {"lm_head", m.hidden, m.vocab, 1},
        };
        // MNN_CD_SHAPES="ic:oc;ic:oc;..." overrides the model table with an
        // arbitrary ic sweep, tagged by block count; MNN_CD_LAYERS stacks that
        // many copies per shape to amortize per-forward host overhead.
        int cdLayers = 1;
        if (auto lEnv = getenv("MNN_CD_LAYERS")) {
            cdLayers = std::max(1, atoi(lEnv));
        }
        if (auto sweepEnv = getenv("MNN_CD_SHAPES")) {
            shapes.clear();
            std::string spec(sweepEnv);
            size_t pos = 0;
            while (pos < spec.size()) {
                size_t sep = spec.find(';', pos);
                if (sep == std::string::npos) {
                    sep = spec.size();
                }
                auto pair = spec.substr(pos, sep - pos);
                auto colon = pair.find(':');
                if (colon != std::string::npos) {
                    int ic = atoi(pair.substr(0, colon).c_str());
                    int oc = atoi(pair.substr(colon + 1).c_str());
                    if (ic > 0 && oc > 0) {
                        char buf[16];
                        snprintf(buf, sizeof(buf), "b%d", ic / blockSize());
                        shapes.push_back({buf, ic, oc, cdLayers});
                    }
                }
                pos = sep + 1;
            }
        }
        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        // The env-sweep path prebuilds all modules so timed rounds run
        // back-to-back instead of idling the GPU on rebuilds between rounds.
        std::vector<std::shared_ptr<Module>> mods;
        std::vector<VARP> inputs;
        if (nullptr != getenv("MNN_CD_SHAPES")) {
            mods.resize(shapes.size());
            inputs.resize(shapes.size());
            for (size_t i = 0; i < shapes.size(); ++i) {
                mods[i]   = makeConvModule(shapes[i].ic, shapes[i].oc, shapes[i].layers);
                inputs[i] = makeActivation(1, shapes[i].ic, 0.07f, true);
            }
        }
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (mods.empty()) {
                benchConv(shapes[i].tag, shapes[i].ic, shapes[i].oc, shapes[i].layers, 2, -1);
            } else {
                benchConvModule(mods[i], inputs[i], shapes[i].tag, shapes[i].ic, shapes[i].oc, shapes[i].layers, 2,
                                -1);
            }
        }
        std::vector<float> best(shapes.size(), 1e30f);
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < shapes.size(); ++i) {
                const int loop = shapes[i].layers == 1 ? 20 : 50;
                const float ms =
                    mods.empty()
                        ? benchConv(shapes[i].tag, shapes[i].ic, shapes[i].oc, shapes[i].layers, loop, round)
                        : benchConvModule(mods[i], inputs[i], shapes[i].tag, shapes[i].ic, shapes[i].oc,
                                          shapes[i].layers, loop, round);
                best[i] = std::min(best[i], ms);
            }
        }
        MNN_PRINT("--- best of %d ---\n", rounds);
        for (size_t i = 0; i < shapes.size(); ++i) {
            double wBytes = weightBytes((double)shapes[i].oc * shapes[i].ic);
            MNN_PRINT("BEST %-7s  per-layer=%8.4f ms (%6.1f GB/s W)  x%d layers=%8.3f ms\n", shapes[i].tag.c_str(),
                      best[i], wBytes / (best[i] * 1e6), shapes[i].layers, best[i] * shapes[i].layers);
        }
        return true;
    }
};

MNNTestSuiteRegister(LlmConvDecodeSpeedTest, "speed/LlmConvDecode");

#ifdef MNN_LOW_MEMORY

// ---------------------------------------------------------------------------
// Whole per-token decode linear budget, grouped the way the model dispatches
// ---------------------------------------------------------------------------
//
// Dumping llm.mnn.json for Qwen3-0.6B gives exactly five signatures:
//
//    28x FusedLinear  qkv     has_ln(RMSNorm over residual+hidden), 1024 -> 2048/1024/1024
//    28x Convolution  o_proj  2048 -> 1024
//    28x FusedLinear  gateup  has_ln + act_silu_mul, 1024 -> 3072 x2
//    28x Convolution  down    3072 -> 1024
//     1x Convolution  lm_head 1024 -> 151936
//
// So the two GEMV clusters per layer are *not* reachable as bare matmuls: they
// enter through FusedLinear, which folds the RMSNorm / residual add / SiLU-mul
// into the same dispatch. Benchmarking bare convs there would measure a kernel
// the model never runs. Each group holds `count` layers with distinct weights
// (~335 MB total at 0.6B), so weights stream from DRAM exactly like real decode
// instead of sitting in cache.

namespace {

enum Entry { kFusedQkv, kFusedQkvNoLn, kFusedGateUp, kConv, kConvSplit, kConvChain };

struct DecodeGroup {
    const char* name;
    Entry entry;
    int ic;
    std::vector<int> ocs;
    int count;
};

std::vector<DecodeGroup> decodeGroups() {
    const auto& m = model();
    std::vector<DecodeGroup> groups = {
        {"qkv(FL)", kFusedQkv, m.hidden, toVec(m.qkv), m.attnLayers},
    };
    if (m.linearIn.n > 0) {
        groups.push_back({"linear_in(FL)", kFusedQkv, m.hidden, toVec(m.linearIn), m.layers - m.attnLayers});
    }
    groups.push_back({"o_proj", kConv, m.qkv.oc[0], {m.hidden}, m.layers});
    groups.push_back({"gateup(FL)", kFusedGateUp, m.hidden, {m.inter, m.inter}, m.layers});
    groups.push_back({"down_proj", kConv, m.inter, {m.hidden}, m.layers});
    groups.push_back({"lm_head", kConv, m.hidden, {m.vocab}, 1});
    if (getenv("MNN_DECODE_GEMV_DIAG") != nullptr) {
        const int hidden = m.hidden;
        const int layers = m.layers;
        // Same byte budget as qkv(FL) but without RMSNorm folding or QKV fusion,
        // so the group's cost can be split into raw GEMV vs fusion overhead.
        groups.push_back({"diag_oc4096", kConv, hidden, {4096}, layers});
        // qkv(FL) shapes with the QKV fusion but no RMSNorm folding: splits the
        // qkv(FL) overhead into its LN half and its fusion half.
        groups.push_back({"diag_qkvfused_noln", kFusedQkvNoLn, hidden, toVec(m.qkv), layers});
        groups.push_back({"diag_q", kConv, hidden, {2048}, layers});
        groups.push_back({"diag_k", kConv, hidden, {1024}, layers});
        // Same kernel (g4m1, not lm_head's g16) at growing oc, constant total
        // bytes: separates the kernel's own bandwidth roof from the
        // threadgroup-count ramp.
        groups.push_back({"diag_oc8192", kConv, hidden, {8192}, 14});
        groups.push_back({"diag_oc16384", kConv, hidden, {16384}, 7});
        // Control for the above: same oc as diag_oc4096, same dispatch count as
        // diag_oc16384. Isolates per-dispatch cost from per-oc bandwidth.
        groups.push_back({"diag_oc4096_x7", kConv, hidden, {4096}, 7});
        // Footprint ladder: one identical oc=4096 dispatch, only the layer count
        // varies, so the dispatch geometry is fixed and the resident weight
        // footprint sweeps 8 MB -> 264 MB. If the apparent bandwidth falls as the
        // footprint grows, the small arms are reading out of the system-level
        // cache and their GB/s is not a DRAM-peak figure to compare against.
        groups.push_back({"diag_fp4", kConv, hidden, {4096}, 4});
        groups.push_back({"diag_fp14", kConv, hidden, {4096}, 14});
        groups.push_back({"diag_fp56", kConv, hidden, {4096}, 56});
        groups.push_back({"diag_fp112", kConv, hidden, {4096}, 112});
        // Same 264 MB as diag_fp112, reached with 28 wide weight buffers instead
        // of 112 narrow ones. The layer count moves footprint and allocation
        // count together, so without this arm the two cannot be told apart.
        groups.push_back({"diag_wide28", kConv, hidden, {16384}, 28});
        // Dependency pair at identical shape, layer count and bytes: indep feeds
        // every layer the same activation, chain threads each output into the next
        // (which is why oc == ic here). Anything indep gains over chain is
        // adjacent-dispatch overlap, which a real decode graph never gets.
        groups.push_back({"diag_indep64", kConv, hidden, {hidden}, 64});
        groups.push_back({"diag_chain64", kConvChain, hidden, {hidden}, 64});
        // Fused QKV at wider hidden, i.e. more quant blocks per row: 2048/64 = 32
        // blocks (Qwen3-1.7B: 16 q-heads / 8 kv-heads x 128) and 2560/64 = 40
        // (Qwen3-4B: 32 q-heads / 8 kv-heads). qkv(FL) above only reaches 16, so
        // without these the block-count axis of the fused split-K is untestable.
        // Every projection's slice count stays divisible by 4, so the wide split
        // remains eligible and the wide/plain/no-split arms are all reachable.
        groups.push_back({"diag_qkv_b32", kFusedQkv, 2048, {2048, 1024, 1024}, layers});
        groups.push_back({"diag_qkv_b40", kFusedQkv, 2560, {4096, 1024, 1024}, layers});
        // Prices QKV_PACKED_GRID on its own. mQKVPackedGrid only turns on when the
        // members differ in outputChannel, so these two arms hold everything else
        // equal -- 4 members, 4096 total oc, and (at 2 quads/TG) 512 threadgroups
        // each -- and differ only in whether the shader derives the projection
        // from x or takes it from z. No LN on either, so the norm is out of scope.
        groups.push_back({"diag_qkv_grid_eq", kFusedQkvNoLn, hidden, {1024, 1024, 1024, 1024}, layers});
        groups.push_back({"diag_qkv_grid_un", kFusedQkvNoLn, hidden, {2048, 1024, 512, 512}, layers});
        // Byte- and dispatch-matched ladder for the fused-vs-plain bandwidth gap.
        // gap6144 and gapf4 both carry hidden*6144 weight bytes per layer and one
        // dispatch per layer, so they differ only in the shader body and in how
        // many weight buffers the layer streams from, 1 vs 4. gap3072 is the
        // half-width plain control, which tells whether a 3072-wide dispatch
        // alone already saturates before the fused arm is read. gapf3 carries the
        // same bytes and dispatch count with 3 members, which is the shape the
        // merged weight+scale buffer accepts by default (see setupQKVFusion), so
        // gapf3-vs-gapf4 separates the stream count from the member count. There
        // is no 2-member arm: ShapeFusedProj only accepts 3 or 4 convs unless
        // act_silu_mul is set, and setting it would add the epilogue back.
        groups.push_back({"diag_gap6144", kConv, hidden, {6144}, layers});
        groups.push_back({"diag_gap3072", kConv, hidden, {3072}, layers});
        groups.push_back({"diag_gapf3", kFusedQkvNoLn, hidden, {2048, 2048, 2048}, layers});
        groups.push_back({"diag_gapf4", kFusedQkvNoLn, hidden, {1536, 1536, 1536, 1536}, layers});
        // Unfused controls: same weight bytes and layer count as gapf3/gapf4, but
        // one plain dispatch per projection instead of one fused dispatch per
        // group. This is what fusion actually replaces, so it prices fusion
        // itself rather than the shader body -- the fused arm only pays off if it
        // beats these plus the AddRMSNorm dispatch it folds in.
        groups.push_back({"diag_unf3", kConvSplit, hidden, {2048, 2048, 2048}, layers});
        groups.push_back({"diag_unf4", kConvSplit, hidden, {1536, 1536, 1536, 1536}, layers});
        // Deep-footprint twins of gap6144/gapf4 at ~198 MB, past the cliff the
        // footprint ladder above locates. Below the cliff part of the weight
        // stream is served from cache, and the plain and fused arms are not
        // equally helped by it, so only these two arms compare the shader bodies
        // at a traffic mix that matches a real model's resident weights.
        groups.push_back({"diag_deep6144", kConv, hidden, {6144}, 56});
        groups.push_back({"diag_deepf4", kFusedQkvNoLn, hidden, {1536, 1536, 1536, 1536}, 56});
    }
    return groups;
}

// One FusedLinear layer. With has_ln: inputs [residual, hidden] ->
// residual_out = residual + hidden, normalized = rmsnorm(residual_out), then
// the members project `normalized`. Without it (kFusedQkvNoLn) the single input
// is projected directly. gateup collapses its two members into
// out = up * silu(gate).
std::vector<VARP> makeDecodeFusedLayer(const DecodeGroup& group, VARP residual, VARP hidden, int seed) {
    const bool hasLn = (group.entry != kFusedQkvNoLn);
    std::unique_ptr<OpT> op(new OpT);
    op->type                   = OpType_FusedLinear;
    op->main.type              = OpParameter_FusedLinearParam;
    op->main.value             = new FusedLinearParamT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto param                 = op->main.AsFusedLinearParam();
    param->act_silu_mul        = (group.entry == kFusedGateUp);
    param->has_ln              = hasLn;
    if (hasLn) {
        param->ln.reset(new LayerNormT);
        param->ln->epsilon    = 1e-6f;
        param->ln->gamma      = std::vector<float>(group.ic, 1.0f);
        param->ln->beta       = std::vector<float>(group.ic, 0.0f);
        param->ln->axis       = {-1};
        param->ln->useRMSNorm = true;
    }
    for (size_t m = 0; m < group.ocs.size(); ++m) {
        param->convs.push_back(_blockQuantConv1x1(group.ic, group.ocs[m], bits(), blockSize(), seed + (int)m + 1));
    }
    const int numOut = (param->act_silu_mul ? 1 : (int)group.ocs.size()) + (hasLn ? 1 : 0); // + residual_out
    std::vector<VARP> inputs;
    if (hasLn) {
        inputs = {residual, hidden};
    } else {
        inputs = {hidden};
    }
    auto expr = Expr::create(std::move(op), inputs, numOut);
    std::vector<VARP> outputs;
    for (int i = 0; i < numOut; ++i) {
        outputs.push_back(Variable::create(expr, i));
    }
    return outputs;
}

// A whole group as one module: `count` layers with distinct weights. kConv,
// kConvSplit and the fused entries feed every layer from the same activation, so
// the layers are independent and the GPU is free to run adjacent dispatches
// concurrently -- a real decode graph is a dependency chain and cannot. kConvChain
// is the control for exactly that: it threads each layer's output into the next,
// which needs ic == oc.
std::shared_ptr<Module> makeGroupModule(const DecodeGroup& group) {
    std::vector<VARP> outputs;
    if (group.entry == kConvChain) {
        auto x = makeActivation(1, group.ic, 0.02f, false);
        for (int l = 0; l < group.count; ++l) {
            std::unique_ptr<OpT> op(new OpT);
            op->type                   = OpType_Convolution;
            op->main.type              = OpParameter_Convolution2D;
            op->main.value             = _blockQuantConv1x1(group.ic, group.ocs[0], bits(), blockSize(), l + 1).release();
            op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            x                          = Variable::create(Expr::create(std::move(op), {x}));
        }
        outputs.emplace_back(x);
    } else if (group.entry == kConv || group.entry == kConvSplit) {
        auto x = makeActivation(1, group.ic, 0.02f, false);
        // defaultDimentionFormat has to be set explicitly (the schema default is
        // NHWC). Going through _blockQuantConv1x1 keeps the quant tables here
        // identical to the FusedLinear members above and to LlmConvDecode, so
        // the two harnesses time the same graph.
        // kConvSplit walks every oc: that is the unfused control for the fused
        // arms, one dispatch per projection instead of one for the group.
        const size_t projs = (group.entry == kConvSplit) ? group.ocs.size() : 1;
        for (int l = 0; l < group.count; ++l) {
            for (size_t p = 0; p < projs; ++p) {
                std::unique_ptr<OpT> op(new OpT);
                op->type       = OpType_Convolution;
                op->main.type  = OpParameter_Convolution2D;
                op->main.value = _blockQuantConv1x1(group.ic, group.ocs[p], bits(), blockSize(),
                                                    l * 8 + (int)p + 1)
                                     .release();
                op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                outputs.emplace_back(Variable::create(Expr::create(std::move(op), {x})));
            }
        }
    } else {
        auto residual = makeActivation(1, group.ic, 0.11f, false);
        auto hidden   = makeActivation(1, group.ic, -0.07f, false);
        for (int l = 0; l < group.count; ++l) {
            auto layerOut = makeDecodeFusedLayer(group, residual, hidden, l * 97);
            outputs.insert(outputs.end(), layerOut.begin(), layerOut.end());
        }
    }
    auto buffer = Variable::save(outputs);
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), makeRuntime()));
}

double groupWeightBytes(const DecodeGroup& group) {
    double params = 0.0;
    for (int oc : group.ocs) {
        params += (double)group.ic * oc;
    }
    return weightBytes(params * group.count);
}

} // namespace

class LlmDecodeGemvSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st      = MNNTestSuite::get()->pStaus;
        const auto& m = model();
        MNN_PRINT("\n===== %s decode linear budget (W%d async block%d, 1 token) =====\n", m.name, bits(), blockSize());
        MNN_PRINT("forwardType=%d precision=%d memory=Low layers=%d (attn %d + linear %d) hidden=%d\n", st.forwardType,
                  st.precision, m.layers, m.attnLayers, m.layers - m.attnLayers, m.hidden);

        auto groups      = decodeGroups();
        const char* only = getenv("MNN_DECODE_GEMV_GROUP");
        if (only != nullptr) {
            std::vector<DecodeGroup> picked;
            for (auto& g : groups) {
                if (nullptr != strstr(g.name, only)) {
                    picked.push_back(g);
                }
            }
            groups = picked;
            if (groups.empty()) {
                MNN_ERROR("LlmDecodeGemv: MNN_DECODE_GEMV_GROUP=%s matched nothing\n", only);
                return false;
            }
        }
        std::vector<std::shared_ptr<Module>> modules(groups.size());
        std::vector<std::vector<VARP>> inputs(groups.size());
        // The external reference mirror reuses device-resident arrays across
        // every loop iteration, so a host-resident input would charge MNN a
        // re-upload the baseline never pays (worth ~3% on lm_head, noise on
        // the rest). MNN_DECODE_GEMV_HOSTIN=1 restores the host upload (matches
        // MNN_FL_HOSTIN on LlmFusedLinear).
        const bool deviceResident = (nullptr == getenv("MNN_DECODE_GEMV_HOSTIN"));
        for (size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].entry == kConv || groups[i].entry == kConvSplit || groups[i].entry == kConvChain) {
                inputs[i] = {makeActivation(1, groups[i].ic, 0.02f, deviceResident)};
            } else if (groups[i].entry == kFusedQkvNoLn) {
                inputs[i] = {makeActivation(1, groups[i].ic, -0.07f, deviceResident)};
            } else {
                inputs[i] = {makeActivation(1, groups[i].ic, 0.11f, deviceResident),
                             makeActivation(1, groups[i].ic, -0.07f, deviceResident)};
            }
            modules[i] = makeGroupModule(groups[i]);
            if (nullptr == modules[i]) {
                MNN_ERROR("LlmDecodeGemv: module build failed for %s\n", groups[i].name);
                return false;
            }
        }

        // Enqueue the whole loop, sync once: mapping the output every iteration
        // costs more than the GEMVs themselves at decode shapes.
        // The whole output vector has to stay alive, not just the tail.
        // StaticModule::_resize only re-acquires an output buffer when the caller
        // still references it, so keeping just the tail makes all 28 layers reuse
        // one output buffer every iteration and costs 40% (o_proj 0.214 -> 0.297
        // ms per 28 layers). The penalty is entirely GPU-side -- enqueue time is
        // unchanged (0.024 vs 0.020 ms/iter), the drain grows 0.189 -> 0.277 --
        // so it is the aliased output, not the release, that is being measured.
        auto bench = [&](size_t i, int loop) {
            Timer timer;
            std::vector<VARP> last;
            for (int n = 0; n < loop; ++n) {
                last = modules[i]->onForward(inputs[i]);
            }
            const double enqueueMs = (double)timer.durationInUs() / 1000.0 / loop;
            last.back()->readMap<float>();
            double ms = (double)timer.durationInUs() / 1000.0 / loop;
            if (getenv("MNN_DECODE_GEMV_SPLIT") != nullptr) {
                MNN_PRINT("    [split] %-16s enqueue %7.4f  drain %7.4f ms/iter\n", groups[i].name, enqueueMs,
                          ms - enqueueMs);
            }
            last.back()->unMap();
            return ms;
        };
        for (size_t i = 0; i < groups.size(); ++i) {
            bench(i, 5); // warmup: pipeline build + GPU clock ramp
        }

        const int rounds = 5;
        const int loop   = 50;
        std::vector<double> best(groups.size(), 1e30);
        for (int round = 0; round < rounds; ++round) {
            double totalMs = 0.0, totalBytes = 0.0;
            for (size_t i = 0; i < groups.size(); ++i) {
                const double ms    = bench(i, loop);
                const double bytes = groupWeightBytes(groups[i]);
                best[i]            = std::min(best[i], ms);
                totalMs += ms;
                totalBytes += bytes;
                MNN_PRINT("r%d %-11s ic=%-5d oc=%-6d x%-3d %8.3f ms %7.1f MB %6.1f GB/s\n", round, groups[i].name,
                          groups[i].ic, groups[i].ocs[0], groups[i].count, ms, bytes / 1e6, bytes / (ms * 1e6));
            }
            MNN_PRINT("r%d TOTAL       %8.3f ms/token %7.1f MB %6.1f GB/s  (linear-only %6.1f tok/s)\n", round, totalMs,
                      totalBytes / 1e6, totalBytes / (totalMs * 1e6), 1000.0 / totalMs);
        }
        // Thermal drift only ever makes a latency-bound GEMV slower, so the
        // per-group minimum over rounds is the stable statistic to A/B on.
        double bestTotal = 0.0, totalBytes = 0.0;
        for (size_t i = 0; i < groups.size(); ++i) {
            const double bytes = groupWeightBytes(groups[i]);
            bestTotal += best[i];
            totalBytes += bytes;
            MNN_PRINT("MIN %-11s %8.3f ms %6.1f GB/s\n", groups[i].name, best[i], bytes / (best[i] * 1e6));
        }
        MNN_PRINT("MIN TOTAL       %8.3f ms/token %6.1f GB/s  (linear-only %6.1f tok/s)\n", bestTotal,
                  totalBytes / (bestTotal * 1e6), 1000.0 / bestTotal);
        return true;
    }
};

MNNTestSuiteRegister(LlmDecodeGemvSpeedTest, "speed/LlmDecodeGemv");

#endif // MNN_LOW_MEMORY
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
