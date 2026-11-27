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
//    speed/LlmChain            full transformer layer chain with KV cache
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
//  what gates Metal's fused decode-GEMV path (decodeFusionEligible).
//  fp16 prefill area>=64 takes the device-tensor GEMM, so the fused-quant M64 tile needs fp32 (precision 1).
//
//  speed/LlmChain knobs: MNN_CHAIN_LAYERS / ROUNDS / SEQ / CTX / HEADDIM / WSCALE.
//  Its sums gate finiteness only; MNN_CHAIN_SWITCH=1 is the reproducibility gate, on one module.
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
#include "core/KVMeta.hpp"

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

// MNN_DECODE_GEMV_BLOCK selects the quant block size. It sets the C4 quads per
// quant block the Metal decode GEMV walks (block/4 = 8/16/32/64) and gates the
// block64-only calibrated routes, so it is the axis along which the decode-GEMV
// block shape gets priced. Every ic in the model table is a multiple of 256, so
// all four values stay eligible.
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
    bool splitQkv;
};

constexpr ModelShape kModels[] = {
    {"Qwen3-0.6B", "0.6b", 1024, 28, 28, 3072, 151936, {3, {2048, 1024, 1024}}, {0, {0}}, false},
    {"Qwen3-1.7B", "1.7b", 2048, 28, 28, 6144, 151936, {3, {2048, 1024, 1024}}, {0, {0}}, false},
    {"Qwen3-4B", "4b", 2560, 36, 36, 9728, 151936, {3, {4096, 1024, 1024}}, {0, {0}}, false},
    {"Qwen3-8B", "8b", 4096, 36, 36, 12288, 151936, {3, {4096, 1024, 1024}}, {0, {0}}, false},
    // Qwen3.5 is hybrid: full_attention_interval 4 => 6 of 24 layers attend.
    {"Qwen3.5-0.8B", "3.5-0.8b", 1024, 24, 6, 3584, 248320, {4, {2048, 512, 512, 2048}}, {4, {6144, 2048, 16, 16}}, false},
    {"Qwen3.5-2B", "3.5-2b", 2048, 24, 6, 6144, 248320, {4, {2048, 512, 512, 2048}}, {4, {6144, 2048, 16, 16}}, false},
    {"Qwen3.5-4B", "3.5-4b", 2560, 32, 8, 9216, 248320, {4, {4096, 1024, 1024, 4096}}, {4, {8192, 4096, 32, 32}}, false},
    {"Qwen3.6-35B-A3B", "3.6-35b", 2048, 40, 10, 512, 248320, {4, {4096, 512, 512, 4096}}, {4, {8192, 4096, 32, 32}}, false},
    {"MobileMoE-S", "mmoe", 768, 20, 20, 1536, 128256, {3, {768, 256, 256}}, {0, {0}}, true},
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

VARP buildFloatConv1x1(VARP x, int ic, int oc, int seed = 0) {
    std::vector<float> weight(oc * ic);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = ((float)((i + seed * 7919) % 127) - 63.0f) / 1000.0f;
    }
    std::vector<float> bias(oc, 0.0f);
    return _Conv(std::move(weight), std::move(bias), x, {ic, oc}, {1, 1}, PaddingMode::VALID, {1, 1}, {1, 1}, 1,
                 {0, 0}, false, false);
}

VARP buildQuantConv1x1(VARP x, int ic, int oc, int seed = 0) {
    const int block = blockSize();
    MNN_ASSERT(ic % block == 0);
    const int blockNum = ic / block;

    std::vector<float> weightFp32(oc * ic);
    for (size_t i = 0; i < weightFp32.size(); ++i) {
        weightFp32[i] = ((float)((i + seed * 7919) % 127) - 63.0f) / 1000.0f;
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
void benchLinear(const char* name, int M, int K, int N, int mode, int thread, int layers = 1) {
    // Graph is wrapped in a Module so every onForward recomputes; an Express
    // VARP caches its result and would need a host writeMap to dirty the input,
    // which drags a 159 MB upload into the measurement.
    auto x = _Input({1, K, 1, M}, NC4HW4, halide_type_of<float>());
    std::vector<VARP> ys;
    for (int l = 0; l < layers; ++l) {
        ys.push_back((mode == 0) ? buildFloatConv1x1(x, K, N, l) : buildQuantConv1x1(x, K, N, l));
    }
    x.fix(VARP::INPUT);
    auto buffer = Variable::save(ys);
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
    float avgMs = (float)_t.durationInUs() / 1000.0f / (float)LOOP / (float)layers;
    last[0]->unMap();
    double flops  = 2.0 * (double)M * (double)K * (double)N;
    double tflops = flops / (avgMs * 1e6) / 1000.0;
    MNN_PRINT("%-10s %-6s M=%-5d K=%-5d N=%-6d  wall=%8.3f ms (%6.2f TFLOPS) x%d\n", name,
              mode == 0 ? "float" : quantArmName(), M, K, N, avgMs, tflops, layers);
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
    std::vector<LinearShape> shapes;
    if (m.splitQkv) {
        shapes.push_back({"q", S, m.hidden, m.qkv.oc[0]});
        shapes.push_back({"k", S, m.hidden, m.qkv.oc[1]});
        shapes.push_back({"v", S, m.hidden, m.qkv.oc[2]});
        shapes.push_back({"router", S, m.hidden, 60});
    } else {
        shapes.push_back({"qkv", S, m.hidden, sumOc(m.qkv)});
    }
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
            const int layers = (model().splitQkv && 0 != strcmp(s.name, "lm_head")) ? model().layers : 1;
            // The float arm materializes dense fp32 weights on the host: at
            // Qwen3.5-2B's 2048 x 248320 lm_head that is 2 GB and the process
            // gets OOM-killed before the quantized row prints. Production
            // lm_head is quantized, so the float reading there is not worth it.
            // MNN_PREFILL_NOFLOAT drops the float arm everywhere: its allocation perturbs the next quantized row.
            const bool noFloat = nullptr != getenv("MNN_PREFILL_NOFLOAT");
            if (noFloat || (double)s.K * (double)s.N * 4.0 * layers > 1e9) {
                if (!noFloat) {
                    MNN_PRINT("%-10s float  M=%-5d K=%-5d N=%-6d  skipped (dense fp32 weights >1 GB)\n", s.name, s.M,
                              s.K, s.N);
                }
            } else {
                benchLinear(s.name, s.M, s.K, s.N, 0, st.thread, layers);
            }
            benchLinear(s.name, s.M, s.K, s.N, 1, st.thread, layers);
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
        param->has_ln              = nullptr == getenv("MNN_FL_NOLN") && !model().splitQkv;
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
        const bool hasLn = param->has_ln;
        const int numOut = (gateUp ? 1 : (int)ocs.size()) + (hasLn ? 1 : 0);
        auto expr        = Expr::create(std::move(op),
                                        hasLn ? std::vector<VARP>{residual, hiddenIn}
                                              : std::vector<VARP>{hiddenIn}, numOut);
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

    const bool noLn = nullptr != getenv("MNN_FL_NOLN") || model().splitQkv;
    auto forwardOnce = [&]() {
        return noLn ? attn->onForward({hiddenIn}) : attn->onForward({residual, hiddenIn});
    };
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
        const std::vector<int> linbig     = {8192, 4096};
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
        std::vector<Shape> shapes;
        if (!m.splitQkv) {
            shapes.push_back({"qkv", &qkvOcs, false, m.attnLayers});
        }
        if (!linearInOcs.empty()) {
            shapes.push_back({"linear_in", &linearInOcs, false, m.layers - m.attnLayers});
        }
        shapes.push_back({"gateup", &gateUpOcs, true, m.layers});
        if (diag) {
            shapes.push_back({"nosilu6144", &nosilu6144, false, m.layers});
            shapes.push_back({"wide9216", &wide9216, false, m.layers});
            shapes.push_back({"linbig", &linbig, false, m.layers});
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
                      int layers, int loop, int round, int area = 1) {
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
    double gflops = 2.0 * (double)area * oc * ic / (ms * 1e6);
    MNN_PRINT("%-7s r%d  per-layer=%8.4f ms (%7.1f GFLOPS, %6.1f GB/s W)  x%d layers=%8.3f ms\n", tag.c_str(),
              round, ms,
              gflops, wBytes / (ms * 1e6), layers, ms * layers);
    return ms;
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
            int area;
        };
        // The three shapes the model runs outside FusedLinear: o_proj
        // (heads*dim -> hidden), down_proj (inter -> hidden), lm_head
        // (hidden -> vocab). lm_head is a single op but inherently DRAM-bound.
        std::vector<Shape> shapes;
        if (m.splitQkv) {
            shapes.push_back({"q", m.hidden, m.qkv.oc[0], m.layers, 1});
            shapes.push_back({"k", m.hidden, m.qkv.oc[1], m.layers, 1});
            shapes.push_back({"v", m.hidden, m.qkv.oc[2], m.layers, 1});
            shapes.push_back({"router", m.hidden, 60, m.layers, 1});
        }
        shapes.push_back({"o_proj", m.qkv.oc[0], m.hidden, m.layers, 1});
        shapes.push_back({"down", m.inter, m.hidden, m.layers, 1});
        shapes.push_back({"lm_head", m.hidden, m.vocab, 1, 1});
        // MNN_CD_SHAPES="ic:oc[:area];..." overrides the model table with an
        // arbitrary ic sweep, tagged by block count; MNN_CD_LAYERS stacks that
        // many copies per shape to amortize per-forward host overhead.
        // Several areas in ONE process share warmup and allocator state; Conv1x1 area is ob*ow*oh, so N counts.
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
                size_t colon = pair.find(':');
                if (colon != std::string::npos) {
                    int ic = atoi(pair.substr(0, colon).c_str());
                    size_t colon2 = pair.find(':', colon + 1);
                    int oc = atoi(pair.substr(colon + 1, colon2 == std::string::npos ? colon2 : colon2 - colon - 1).c_str());
                    int area = colon2 == std::string::npos ? 1 : std::max(1, atoi(pair.substr(colon2 + 1).c_str()));
                    if (ic > 0 && oc > 0) {
                        char buf[24];
                        if (area == 1) {
                            snprintf(buf, sizeof(buf), "b%d", ic / blockSize());
                        } else {
                            snprintf(buf, sizeof(buf), "M%dN%d", area, oc);
                        }
                        shapes.push_back({buf, ic, oc, cdLayers, area});
                    }
                }
                pos = sep + 1;
            }
        }
        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        // Prebuild all modules so timed rounds run back-to-back instead of
        // idling the GPU on rebuilds between rounds. Rebuilding every round
        // also accumulates allocator pressure on memory-constrained devices:
        // lm_head alone is a ~286 MB buffer.
        std::vector<std::shared_ptr<Module>> mods(shapes.size());
        std::vector<VARP> inputs(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            mods[i]   = makeConvModule(shapes[i].ic, shapes[i].oc, shapes[i].layers);
            inputs[i] = makeActivation(shapes[i].area, shapes[i].ic, 0.07f, true);
        }
        for (size_t i = 0; i < shapes.size(); ++i) {
            benchConvModule(mods[i], inputs[i], shapes[i].tag, shapes[i].ic, shapes[i].oc, shapes[i].layers, 2, -1,
                            shapes[i].area);
        }
        std::vector<float> best(shapes.size(), 1e30f);
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < shapes.size(); ++i) {
                const int loop = shapes[i].layers == 1 ? 20 : 50;
                const float ms = benchConvModule(mods[i], inputs[i], shapes[i].tag, shapes[i].ic, shapes[i].oc,
                                                 shapes[i].layers, loop, round, shapes[i].area);
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

enum Entry { kFusedQkv, kFusedQkvNoLn, kFusedGateUp, kFusedGateUpNoLn, kConv, kConvSplit, kConvChain };

struct DecodeGroup {
    const char* name;
    Entry entry;
    int ic;
    std::vector<int> ocs;
    int count;
};

std::vector<DecodeGroup> decodeGroups() {
    const auto& m = model();
    std::vector<DecodeGroup> groups;
    if (m.splitQkv) {
        groups.push_back({"q", kConv, m.hidden, {m.qkv.oc[0]}, m.attnLayers});
        groups.push_back({"k", kConv, m.hidden, {m.qkv.oc[1]}, m.attnLayers});
        groups.push_back({"v", kConv, m.hidden, {m.qkv.oc[2]}, m.attnLayers});
        groups.push_back({"router", kConv, m.hidden, {60}, m.attnLayers});
    } else {
        groups.push_back({"qkv(FL)", kFusedQkv, m.hidden, toVec(m.qkv), m.attnLayers});
    }
    if (m.linearIn.n > 0) {
        groups.push_back({"linear_in(FL)", kFusedQkv, m.hidden, toVec(m.linearIn), m.layers - m.attnLayers - 1});
        groups.push_back({"linear_in0(FL)", kFusedQkvNoLn, m.hidden, toVec(m.linearIn), 1});
    }
    groups.push_back({"o_proj", kConv, m.qkv.oc[0], {m.hidden}, m.layers});
    groups.push_back({"gateup(FL)", m.splitQkv ? kFusedGateUpNoLn : kFusedGateUp, m.hidden, {m.inter, m.inter},
                      m.layers});
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
        // merged weight+scale buffer accepts by default (see prepareQKVWeights), so
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
    const bool hasLn = (group.entry != kFusedQkvNoLn && group.entry != kFusedGateUpNoLn);
    std::unique_ptr<OpT> op(new OpT);
    op->type                   = OpType_FusedLinear;
    op->main.type              = OpParameter_FusedLinearParam;
    op->main.value             = new FusedLinearParamT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto param                 = op->main.AsFusedLinearParam();
    param->act_silu_mul        = (group.entry == kFusedGateUp || group.entry == kFusedGateUpNoLn);
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
            } else if (groups[i].entry == kFusedQkvNoLn || groups[i].entry == kFusedGateUpNoLn) {
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

// ---------------------------------------------------------------------------
// speed/LlmChain: dense Qwen3 Transformer layers with synthetic weights and a KV cache.
// Measures per-layer decode/prefill latency and Encode Replay, not model accuracy.
// ---------------------------------------------------------------------------

namespace {

// Independent projections let the GPU overlap and never recycle a live buffer; a real chain serializes
// dispatch and keeps intermediates alive, which is what Encode Replay and the fusion decisions see.

// qkv.oc[0]/[1] are the packed q/kv widths, so headDim divides them out (128 for Qwen3 dense).
int chainHeadDim() {
    if (auto env = getenv("MNN_CHAIN_HEADDIM")) {
        return std::max(1, atoi(env));
    }
    return 128;
}

// KVMeta holds a string and a vector, so it must not be a namespace-scope object (AGENTS.md).
KVMeta& chainMeta() {
    static KVMeta sMeta;
    return sMeta;
}

// The meta outlives each module: leftover capacity or previous length changes the cache alloc and kernel.
void resetChainMeta() {
    auto& meta      = chainMeta();
    meta.previous   = 0;
    meta.remove     = 0;
    meta.add        = 0;
    meta.capacity   = 0;
    meta.n_reserve  = 0;
    meta.reserve    = nullptr;
}

// CPURoPE reads the first half as the even partners and the second half as the odd ones at the same angle,
// so both halves hold identical values.
VARP ropeTable(int seq, int pos, int headDim, bool isSin) {
    auto var = _Input({seq, headDim}, NCHW, halide_type_of<float>());
    auto ptr = var->writeMap<float>();
    const int half = headDim / 2;
    for (int t = 0; t < seq; ++t) {
        for (int d = 0; d < half; ++d) {
            const float angle = (float)(pos + t) / powf(1e6f, 2.0f * (float)d / (float)headDim);
            const float v     = isSin ? sinf(angle) : cosf(angle);
            ptr[t * headDim + d]        = v;
            ptr[t * headDim + half + d] = v;
        }
    }
    var->unMap();
    return var;
}

std::unique_ptr<OpT> chainNormOp(int hidden) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                   = OpType_LayerNorm;
    op->main.type              = OpParameter_LayerNorm;
    op->main.value             = new LayerNormT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto ln                    = op->main.AsLayerNorm();
    ln->epsilon                = 1e-6f;
    ln->gamma                  = std::vector<float>(hidden, 1.0f);
    ln->beta                   = std::vector<float>(hidden, 0.0f);
    ln->axis                   = {-1};
    ln->useRMSNorm             = true;
    return op;
}

// Weights are hash-scattered: a periodic pattern accumulates coherently and overflows fp16 within a layer.
// The scale sets residual growth and logit range; fp16 only survives a narrow window around this default.
float chainWeightScale() {
    if (auto env = getenv("MNN_CHAIN_WSCALE")) {
        return (float)atof(env);
    }
    return 0.005f;
}

std::unique_ptr<Convolution2DT> chainQuantConv(int ic, int oc, int seed) {
    const float threshold = (float)(1 << (bits() - 1)) - 1.0f;
    const float clampMin  = -threshold - 1.0f;
    const int blocknum    = ic / blockSize();
    std::vector<float> weight((size_t)oc * ic), alpha((size_t)2 * oc * blocknum), bias(oc, 0.0f);
    const float wscale = chainWeightScale();
    for (size_t i = 0; i < weight.size(); ++i) {
        uint32_t h = (uint32_t)(i * 2654435761u + (uint32_t)seed * 40503u);
        h ^= h >> 15;
        weight[i] = ((float)(h % 17) - 8.0f) * wscale;
    }
    for (int o = 0; o < oc; ++o) {
        for (int b = 0; b < blocknum; ++b) {
            const float* w = weight.data() + (size_t)o * ic + b * blockSize();
            float mn = w[0], mx = w[0];
            for (int u = 1; u < blockSize(); ++u) {
                mn = std::min(mn, w[u]);
                mx = std::max(mx, w[u]);
            }
            alpha[2 * ((size_t)o * blocknum + b)]     = mn;
            alpha[2 * ((size_t)o * blocknum + b) + 1] = (mx - mn) / (threshold - clampMin);
        }
    }
    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->common.reset(new Convolution2DCommonT);
    conv->common->kernelX    = 1;
    conv->common->kernelY    = 1;
    conv->common->inputCount = ic;
    conv->common->outputCount = oc;
    conv->quanParameter = IDSTEncoder::encode(weight.data(), alpha, blockSize(), oc * blocknum, true, nullptr,
                                              (int)clampMin, {bits(), false});
    return conv;
}

std::unique_ptr<OpT> chainFusedOp(const std::vector<int>& ocs, bool gateUp, bool hasLn, int hidden, int seed) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                   = OpType_FusedLinear;
    op->main.type              = OpParameter_FusedLinearParam;
    op->main.value             = new FusedLinearParamT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto param                 = op->main.AsFusedLinearParam();
    param->act_silu_mul        = gateUp;
    param->has_ln              = hasLn;
    if (hasLn) {
        param->ln.reset(new LayerNormT);
        param->ln->epsilon    = 1e-6f;
        param->ln->gamma      = std::vector<float>(hidden, 1.0f);
        param->ln->beta       = std::vector<float>(hidden, 0.0f);
        param->ln->axis       = {-1};
        param->ln->useRMSNorm = true;
    }
    for (size_t i = 0; i < ocs.size(); ++i) {
        // Distinct weights per member: a shared set would let the cache serve decode's weight DRAM traffic.
        param->convs.push_back(chainQuantConv(hidden, ocs[i], seed + (int)i + 1));
    }
    return op;
}

std::unique_ptr<OpT> chainConvOp(int ic, int oc, int seed) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                   = OpType_Convolution;
    op->main.type              = OpParameter_Convolution2D;
    op->main.value             = chainQuantConv(ic, oc, seed).release();
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    return op;
}

// Inputs {residual, mask, cos, sin}; output {residual after `layers` layers}.
// The converter only folds a *binary* add&norm into FusedLinear, so the attention norm stays a unary RMSNorm.
std::shared_ptr<Module> makeChainModule(int layers) {
    const auto& m      = model();
    const int hidden   = m.hidden;
    const int headDim  = chainHeadDim();
    const int numHead  = m.qkv.oc[0] / headDim;
    const int kvHead   = m.qkv.oc[1] / headDim;
    auto h             = _Input({1, hidden, 1, 1}, NC4HW4);
    auto mask          = _Input();
    auto cos           = _Input();
    auto sin           = _Input();

    auto norm = chainNormOp(hidden);
    std::unique_ptr<OpT> rope(new OpT);
    rope->type                   = OpType_RoPE;
    rope->main.type              = OpParameter_RoPEParam;
    rope->main.value             = new RoPEParamT;
    rope->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto rp                      = rope->main.AsRoPEParam();
    rp->num_head                 = numHead;
    rp->kv_num_head              = kvHead;
    rp->head_dim                 = headDim;

    std::unique_ptr<OpT> attn(new OpT);
    attn->type                   = OpType_Attention;
    attn->main.type              = OpParameter_AttentionParam;
    attn->main.value             = new AttentionParamT;
    attn->main.AsAttentionParam()->kv_cache  = true;
    attn->main.AsAttentionParam()->output_c4 = true;

    const std::vector<int> qkvOcs    = toVec(m.qkv);
    const std::vector<int> gateUpOcs = {m.inter, m.inter};
    VARP cur                         = h;
    for (int l = 0; l < layers; ++l) {
        const int seed = l * 97 + 1;
        auto n1        = Variable::create(Expr::create(norm.get(), {cur}));
        auto qkvOp     = chainFusedOp(qkvOcs, false, false, hidden, seed);
        auto qkvExpr   = Expr::create(std::move(qkvOp), {n1}, (int)qkvOcs.size());
        std::vector<VARP> proj;
        for (size_t i = 0; i < qkvOcs.size(); ++i) {
            proj.push_back(Variable::create(qkvExpr, (int)i));
        }
        // RoPE bridges q/k from C4 to NHWC; Attention takes v as C4 directly (value_c4).
        auto ropeExpr = Expr::create(rope.get(), {proj[0], proj[1], cos, sin}, 2);
        auto qr       = Variable::create(ropeExpr, 0);
        auto kr       = Variable::create(ropeExpr, 1);
        auto a        = Variable::create(Expr::create(attn.get(), {qr, kr, proj[2], mask}));
        auto o        = Variable::create(Expr::create(chainConvOp(m.qkv.oc[0], hidden, seed + 11), {a}));
        auto guOp     = chainFusedOp(gateUpOcs, true, true, hidden, seed + 23);
        auto guExpr   = Expr::create(std::move(guOp), {cur, o}, 2);
        auto ffn      = Variable::create(guExpr, 0);
        auto sum      = Variable::create(guExpr, 1);
        auto d        = Variable::create(Expr::create(chainConvOp(m.inter, hidden, seed + 37), {ffn}));
        cur           = _Add(sum, d);
    }
    auto buffer = Variable::save({cur});

    auto status = MNNTestSuite::get()->pStaus;
    ScheduleConfig config;
    config.type      = (MNNForwardType)status.forwardType;
    config.numThread = 1;
    BackendConfig bnConfig;
    bnConfig.precision   = (BackendConfig::PrecisionMode)status.precision;
    bnConfig.power       = (BackendConfig::PowerMode)status.power;
    bnConfig.memory      = BackendConfig::Memory_Low;
    config.backendConfig = &bnConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(Interpreter::KVCACHE_INFO, &chainMeta());
    rtmgr->setHint(Interpreter::ATTENTION_OPTION, 8); // float qkv, no kv quant
    return std::shared_ptr<Module>(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
}

double chainChecksum(const VARP& var) {
    auto ptr = var->readMap<float>();
    if (nullptr == ptr) {
        return -1.0;
    }
    const int n = var->getInfo()->size;
    double sum  = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double)ptr[i];
    }
    var->unMap();
    return sum;
}

bool varFinite(const VARP& var) {
    auto ptr = var->readMap<float>();
    if (nullptr == ptr) {
        return false;
    }
    const int n = var->getInfo()->size;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        ok = ok && std::isfinite(ptr[i]);
    }
    var->unMap();
    return ok;
}

struct ChainTiming {
    float enqueueMs = 0.f; // CPU encode+commit per forward, GPU runs ahead
    float wallMs    = 0.f; // enqueue loop + one drain, per forward
    float serialMs  = 0.f; // sync every forward: CPU + GPU, no overlap
    double checksum = 0.0;
    bool finite     = true;
    bool shapeOk    = true;
};

// seq==1 fills `ctx` once and then times decode steps against that cache; seq>1 times prefill with a reset.
ChainTiming benchChain(int seq, int ctx, int loop, int layers, bool report) {
    resetChainMeta();
    const int hidden          = model().hidden;
    const int headDim         = chainHeadDim();
    auto mod                  = makeChainModule(layers);
    const bool deviceResident = (nullptr == getenv("MNN_FL_HOSTIN"));
    const bool decode         = (seq == 1);
    // The fill must feed exactly the tokens it adds to the KV cache, or the Conv area never switches.
    auto hPrefill             = makeActivation(seq, hidden, 0.11f, deviceResident);
    auto hDecode              = makeActivation(1, hidden, 0.11f, deviceResident);
    auto hCtx                 = decode ? makeActivation(ctx, hidden, 0.11f, deviceResident) : hDecode;
    auto mask                 = _Input({}, NCHW, halide_type_of<float>());
    mask->writeMap<float>()[0] = 0.0f;
    auto& meta                = chainMeta();
    ChainTiming t;
    auto checkTokens = [&](const VARP& v, int want, const char* site) {
        const int got = (nullptr == v->getInfo()) ? -1 : v->getInfo()->dim[0];
        if (got != want) {
            MNN_ERROR("LlmChain: %s activation carries %d tokens, want %d.\n", site, got, want);
            t.shapeOk = false;
        }
    };
    // Rebuilding RoPE tables per step would churn the input addresses Encode Replay fingerprints.
    // Slot 0 is the fill/prefill table; decode step i uses slot i+1.
    const int ropeCount = decode ? loop + 5 : 1;
    std::vector<std::pair<VARP, VARP>> rope;
    rope.reserve(ropeCount);
    const int fillTokens = decode ? ctx : seq;
    rope.emplace_back(ropeTable(fillTokens, 0, headDim, false), ropeTable(fillTokens, 0, headDim, true));
    for (int i = 1; i < ropeCount; ++i) {
        rope.emplace_back(ropeTable(1, ctx + i - 1, headDim, false), ropeTable(1, ctx + i - 1, headDim, true));
    }

    auto fill = [&](int n) {
        checkTokens(hCtx, n, "fill");
        meta.previous = 0;
        meta.remove   = 0;
        meta.add      = n;
        auto out      = mod->onForward({hCtx, mask, rope[0].first, rope[0].second});
        out[0]->readMap<float>();
        out[0]->unMap();
        meta.sync();
    };
    auto prefillOnce = [&]() {
        checkTokens(hPrefill, seq, "prefill");
        meta.previous = 0;
        meta.remove   = 0;
        meta.add      = seq;
        return mod->onForward({hPrefill, mask, rope[0].first, rope[0].second});
    };
    auto decodeOnce = [&](int i) {
        checkTokens(hDecode, 1, "decode");
        meta.remove = 0;
        meta.add    = 1;
        auto out    = mod->onForward({hDecode, mask, rope[i + 1].first, rope[i + 1].second});
        meta.sync();
        return out;
    };

    // Finiteness gate only: buffer alignment changes accumulation order, so the untrained fp16 chain's sum
    // differs across module instantiations. Reproducibility is gated by MNN_CHAIN_SWITCH on one module.
    double checksum = 0.0;
    if (decode) {
        fill(ctx);
        checksum = chainChecksum(decodeOnce(0)[0]);
    } else {
        checksum = chainChecksum(prefillOnce()[0]);
    }
    if (!std::isfinite(checksum)) {
        MNN_ERROR("LlmChain: seq=%d produced a non-finite sum (%g).\n", seq, checksum);
    }
    auto forwardOnce = [&](int i) { return decode ? decodeOnce(i) : prefillOnce(); };
    for (int i = 0; i < 3; ++i) {
        auto out = forwardOnce(i);
        out[0]->readMap<float>();
        out[0]->unMap();
    }

    t.finite = std::isfinite(checksum);
    {
        Timer timer;
        std::vector<VARP> last;
        for (int i = 0; i < loop; ++i) {
            last = forwardOnce(i);
        }
        t.enqueueMs = (float)timer.durationInUs() / 1000.0f / (float)loop;
        last[0]->readMap<float>();
        t.wallMs = (float)timer.durationInUs() / 1000.0f / (float)loop;
        last[0]->unMap();
    }
    {
        Timer timer;
        std::vector<VARP> last;
        for (int i = 0; i < loop; ++i) {
            last = forwardOnce(i);
            last[0]->readMap<float>();
            last[0]->unMap();
        }
        t.serialMs = (float)timer.durationInUs() / 1000.0f / (float)loop;
        // The pre-timing checksum alone would pass a round that overflowed mid-loop.
        t.finite = t.finite && varFinite(last[0]);
    }
    t.checksum = checksum;
    t.enqueueMs /= (float)layers;
    t.wallMs /= (float)layers;
    t.serialMs /= (float)layers;
    if (report) {
        // serial - enqueue is not GPU time: it is the sync cost plus whatever GPU time the CPU cannot hide.
        // serial - wall is what overlapping CPU encode with GPU execution buys.
        MNN_PRINT("seq=%-5d ctx=%-5d per-layer cpu=%7.4f wall=%7.4f serial=%7.4f ms "
                  "(serial_minus_enqueue=%7.4f overlap=%7.4f) sum=%.6g\n",
                  seq, decode ? ctx : 0, t.enqueueMs, t.wallMs, t.serialMs, t.serialMs - t.enqueueMs,
                  t.serialMs - t.wallMs, t.checksum);
        fflush(stdout);
    }
    return t;
}

} // namespace

class LlmChainSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st      = MNNTestSuite::get()->pStaus;
        const auto& m = model();
        if (0 != m.linearIn.n || 3 != m.qkv.n) {
            MNN_PRINT("speed/LlmChain: skipped (%s is hybrid; the chain covers dense Qwen3).\n", m.name);
            return true;
        }
        const int headDim = chainHeadDim();
        if (m.qkv.oc[0] % headDim != 0 || m.qkv.oc[1] % headDim != 0) {
            MNN_PRINT("speed/LlmChain: headDim=%d does not divide qkv widths.\n", headDim);
            return false;
        }
        int layers = m.layers;
        if (auto env = getenv("MNN_CHAIN_LAYERS")) {
            layers = std::max(1, atoi(env));
        }
        int rounds = 3;
        if (auto env = getenv("MNN_CHAIN_ROUNDS")) {
            rounds = std::max(1, atoi(env));
        }
        std::vector<int> seqs = {1, 128, 512};
        if (auto env = getenv("MNN_CHAIN_SEQ")) {
            seqs = {atoi(env)};
        }
        int ctx = 512;
        if (auto env = getenv("MNN_CHAIN_CTX")) {
            ctx = std::max(1, atoi(env));
        }
        MNN_PRINT("\n===== %s layer chain (int%d block%d, hidden=%d inter=%d, q=%d kv=%d dim=%d) =====\n", m.name,
                  bits(), blockSize(), m.hidden, m.inter, m.qkv.oc[0] / headDim, m.qkv.oc[1] / headDim, headDim);
        MNN_PRINT("forwardType=%d precision=%d layers-per-forward=%d rounds=%d\n", st.forwardType, st.precision, layers,
                  rounds);

        MNN_PRINT("--- warmup pass (untimed, GPU clock ramp) ---\n");
        bool shapeOk = true;
        for (int seq : seqs) {
            auto w = benchChain(seq, ctx, 2, layers, false);
            shapeOk = shapeOk && w.shapeOk;
            MNN_PRINT("warmup seq=%d sum=%.6g\n", seq, w.checksum);
        }
        std::vector<ChainTiming> best(seqs.size());
        std::vector<bool> have(seqs.size(), false);
        int prefillNonFinite = 0;
        for (int round = 0; round < rounds; ++round) {
            for (size_t i = 0; i < seqs.size(); ++i) {
                const int seq  = seqs[i];
                const int loop = seq == 1 ? 32 : (seq >= 512 ? 5 : 10);
                auto t         = benchChain(seq, ctx, loop, layers, true);
                shapeOk = shapeOk && t.shapeOk;
                if (seq != 1 && !t.finite) {
                    ++prefillNonFinite;
                }
                // Prefer a finite round: the untrained fp16 chain can overflow on some instantiations.
                const bool better = !have[i] || (t.finite && !best[i].finite) ||
                                    (t.finite == best[i].finite && t.wallMs < best[i].wallMs);
                if (better) {
                    best[i] = t;
                    have[i] = true;
                }
            }
        }
        MNN_PRINT("--- best of %d (per layer) ---\n", rounds);
        for (size_t i = 0; i < seqs.size(); ++i) {
            const auto& t = best[i];
            MNN_PRINT("BEST seq=%-5d cpu=%7.4f wall=%7.4f serial=%7.4f ms  x%d layers=%8.3f ms  sum=%.6g\n", seqs[i],
                      t.enqueueMs, t.wallMs, t.serialMs, layers, t.wallMs * layers, t.checksum);
        }
        if (prefillNonFinite > 0) {
            MNN_PRINT("note: %d prefill round(s) overflowed fp16 and were excluded from BEST; "
                      "the untrained chain has a narrow range window (MNN_CHAIN_WSCALE).\n", prefillNonFinite);
        }
        bool allFinite = true;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (!best[i].finite) {
                MNN_ERROR("LlmChain: seq=%d has no finite round -- narrow the range with MNN_CHAIN_WSCALE.\n", seqs[i]);
                allFinite = false;
            }
        }
        if (!shapeOk) {
            MNN_ERROR("LlmChain: activation token counts did not match the declared KV add -- "
                      "the measurements above are not the shapes they claim.\n");
        }

        // Prefill -> decode -> prefill on one module arms, invalidates and re-arms Encode Replay while the
        // allocator recycles buffers; a drifting or non-finite sum means a stale recording was replayed.
        if (nullptr != getenv("MNN_CHAIN_SWITCH")) {
            MNN_PRINT("--- shape switch (prefill %d -> decode x8 -> prefill %d, one module) ---\n", seqs.back(),
                      seqs.back());
            const int S    = seqs.back();
            auto mod       = makeChainModule(layers);
            const bool dev = (nullptr == getenv("MNN_FL_HOSTIN"));
            // The decode leg must feed one token, or the Conv area never drops to 1.
            auto hS        = makeActivation(S, m.hidden, 0.11f, dev);
            auto h1        = makeActivation(1, m.hidden, 0.11f, dev);
            if (hS->getInfo()->dim[0] != S || h1->getInfo()->dim[0] != 1) {
                MNN_ERROR("LlmChain: switch activations carry the wrong token counts.\n");
                return false;
            }
            auto mask      = _Input({}, NCHW, halide_type_of<float>());
            mask->writeMap<float>()[0] = 0.0f;
            auto& meta     = chainMeta();
            auto runAt     = [&](int seq, int pos) {
                meta.previous = 0;
                meta.remove   = 0;
                meta.add      = seq;
                auto cos      = ropeTable(seq, pos, headDim, false);
                auto sin      = ropeTable(seq, pos, headDim, true);
                auto out      = mod->onForward({hS, mask, cos, sin});
                double sum    = chainChecksum(out[0]);
                meta.sync();
                return sum;
            };
            bool ok       = true;
            double first  = 0.0;
            for (int round = 0; round < 3; ++round) {
                double pre = runAt(S, 0);
                if (0 == round) {
                    first = pre;
                }
                // Decode against the cache the prefill just wrote.
                meta.previous = S;
                for (int i = 0; i < 8; ++i) {
                    meta.remove = 0;
                    meta.add    = 1;
                    auto cos    = ropeTable(1, S + i, headDim, false);
                    auto sin    = ropeTable(1, S + i, headDim, true);
                    auto out    = mod->onForward({h1, mask, cos, sin});
                    double sum  = chainChecksum(out[0]);
                    meta.sync();
                    if (!std::isfinite(sum)) {
                        MNN_ERROR("LlmChain: decode step %d produced a non-finite sum.\n", i);
                        ok = false;
                    }
                }
                double again = runAt(S, 0);
                if (!std::isfinite(pre) || !std::isfinite(again)) {
                    MNN_ERROR("LlmChain: non-finite prefill sum (%g / %g).\n", pre, again);
                    ok = false;
                } else if (fabs(again - first) > 1e-3 * (fabs(first) + 1.0)) {
                    MNN_ERROR("LlmChain: prefill sum drifted across the switch: %g -> %g.\n", first, again);
                    ok = false;
                }
                MNN_PRINT("switch r%d prefill=%.6g again=%.6g %s\n", round, pre, again, ok ? "ok" : "MISMATCH");
            }
            if (!ok) {
                return false;
            }
        }
        return allFinite && shapeOk;
    }
};

MNNTestSuiteRegister(LlmChainSpeedTest, "speed/LlmChain");

#endif // MNN_LOW_MEMORY
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
