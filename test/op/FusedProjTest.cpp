//
//  FusedProjTest.cpp
//  MNNTests
//
//  Tests for the export-time fused projection op (OpType_FusedLinear);
//  act_silu_mul selects the gate/up flavour.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/MNNFileUtils.h"

using namespace MNN;
using namespace MNN::Express;

static std::vector<float> packC4(const std::vector<float>& input, int seqLen, int channel) {
    std::vector<float> output(((channel + 3) / 4) * seqLen * 4, 0.0f);
    for (int t = 0; t < seqLen; ++t) {
        for (int c = 0; c < channel; ++c) {
            output[(c / 4) * seqLen * 4 + t * 4 + (c % 4)] = input[t * channel + c];
        }
    }
    return output;
}

static std::unique_ptr<Convolution2DT> makeConv(int ic, int oc, int seed) {
    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->common.reset(new Convolution2DCommonT);
    conv->common->kernelX     = 1;
    conv->common->kernelY     = 1;
    conv->common->inputCount  = ic;
    conv->common->outputCount = oc;
    conv->weight.resize(oc * ic);
    conv->bias.resize(oc);
    for (int i = 0; i < oc * ic; ++i) {
        conv->weight[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.021f;
    }
    for (int i = 0; i < oc; ++i) {
        conv->bias[i] = (float)(((i * 5 + seed * 3) % 11) - 5) * 0.013f;
    }
    return conv;
}

// x: [seq, ic] plain; w: [oc, ic]; b: [oc] → out: [seq, oc]
static void convRef(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
                    std::vector<float>& out, int seq, int ic, int oc) {
    out.assign(seq * oc, 0.0f);
    for (int t = 0; t < seq; ++t) {
        for (int o = 0; o < oc; ++o) {
            float sum = b[o];
            for (int i = 0; i < ic; ++i) {
                sum += w[o * ic + i] * x[t * ic + i];
            }
            out[t * oc + o] = sum;
        }
    }
}

static bool checkOut(VARP out, const std::vector<float>& expectedPlain, int seq, int channel, const char* tag) {
    auto expectedC4 = packC4(expectedPlain, seq, channel);
    auto info = out->getInfo();
    if (info == nullptr) {
        MNN_ERROR("FusedProj %s: null info\n", tag);
        return false;
    }
    if (info->dim.size() != 4 || info->dim[0] != seq || info->dim[1] != channel) {
        MNN_ERROR("FusedProj %s: bad shape\n", tag);
        return false;
    }
    if (!checkVector<float>(out->readMap<float>(), expectedC4.data(), expectedC4.size(), 0.002f)) {
        MNN_ERROR("FusedProj %s: value mismatch\n", tag);
        return false;
    }
    return true;
}

class FusedQKVTest : public MNNTestCase {
public:
    virtual ~FusedQKVTest() = default;
    virtual bool run(int precision) {
        const int seq = 3, ic = 8;
        const int ocs[3] = {8, 4, 4};
        std::vector<float> xData(seq * ic);
        for (int i = 0; i < seq * ic; ++i) {
            xData[i] = (float)((i % 13) - 6) * 0.11f;
        }
        auto x = _Input({seq, ic, 1, 1}, NC4HW4);
        auto xC4 = packC4(xData, seq, ic);
        ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
        x->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        for (int m = 0; m < 3; ++m) {
            param->convs.push_back(makeConv(ic, ocs[m], m + 1));
        }
        auto expr = Expr::create(std::move(op), {x}, 3);
        // Reference computed from the same weights.
        auto& convs = op->main.AsFusedLinearParam()->convs;
        bool ok = true;
        for (int m = 0; m < 3; ++m) {
            std::vector<float> expected;
            convRef(xData, convs[m]->weight, convs[m]->bias, expected, seq, ic, ocs[m]);
            ok = ok && checkOut(Variable::create(expr, m), expected, seq, ocs[m], "qkv");
        }
        return ok;
    }
};
MNNTestSuiteRegister(FusedQKVTest, "op/fused_qkv");

// 2 members without act_silu_mul: a gate/up projection whose SiLU-mul tail was
// materialized as plain ops, which the split-LoRA patcher produces to inject
// LoRA between the GEMMs and the activation.
class FusedGateUpSplitTest : public MNNTestCase {
public:
    virtual ~FusedGateUpSplitTest() = default;
    virtual bool run(int precision) {
        const int seq = 3, ic = 8;
        const int ocs[2] = {12, 8};
        std::vector<float> xData(seq * ic);
        for (int i = 0; i < seq * ic; ++i) {
            xData[i] = (float)((i % 13) - 6) * 0.11f;
        }
        auto x = _Input({seq, ic, 1, 1}, NC4HW4);
        auto xC4 = packC4(xData, seq, ic);
        ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
        x->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        for (int m = 0; m < 2; ++m) {
            param->convs.push_back(makeConv(ic, ocs[m], m + 1));
        }
        auto expr = Expr::create(std::move(op), {x}, 2);
        auto& convs = op->main.AsFusedLinearParam()->convs;
        bool ok = true;
        for (int m = 0; m < 2; ++m) {
            std::vector<float> expected;
            convRef(xData, convs[m]->weight, convs[m]->bias, expected, seq, ic, ocs[m]);
            ok = ok && checkOut(Variable::create(expr, m), expected, seq, ocs[m], "gateup_split");
        }
        return ok;
    }
};
MNNTestSuiteRegister(FusedGateUpSplitTest, "op/fused_gateup_split");

static std::vector<int8_t> makeSharedFusedProjModel(int seed) {
    const int ic = 8;
    auto x = _Input({2, ic, 1, 1}, NC4HW4);
    x->setName("x");
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_FusedLinear;
    op->name = "shared_fused_proj";
    op->main.type = OpParameter_FusedLinearParam;
    op->main.value = new FusedLinearParamT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto param = op->main.AsFusedLinearParam();
    param->convs.push_back(makeConv(ic, 8, seed));
    param->convs.push_back(makeConv(ic, 4, seed + 1));
    auto expr = Expr::create(std::move(op), {x}, 2);
    auto y0 = Variable::create(expr, 0);
    auto y1 = Variable::create(expr, 1);
    y0->setName("y0");
    y1->setName("y1");
    return Variable::save({y0, y1});
}

class FusedProjCPUBaseCloneTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto baseBuffer = makeSharedFusedProjModel(1);
        auto adapterBuffer = makeSharedFusedProjModel(101);
        BackendConfig backendConfig;
        backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        backendConfig.memory = static_cast<BackendConfig::MemoryMode>(MNNTestSuite::get()->pStaus.memory);
        ScheduleConfig scheduleConfig;
        scheduleConfig.type = MNN_FORWARD_CPU;
        scheduleConfig.backendConfig = &backendConfig;
        scheduleConfig.numThread = 1;
        std::shared_ptr<Executor::RuntimeManager> runtime(
            Executor::RuntimeManager::createRuntimeManager(scheduleConfig), Executor::RuntimeManager::destroy);
        Module::Config baseConfig;
        baseConfig.rearrange = true;
        std::shared_ptr<Module> base(Module::load({"x"}, {"y0", "y1"}, (const uint8_t*)baseBuffer.data(),
                                                  baseBuffer.size(), runtime, &baseConfig));
        if (!base) {
            return false;
        }
        Module::Config adapterConfig;
        adapterConfig.rearrange = true;
        adapterConfig.base = base.get();
        std::shared_ptr<Module> adapter(Module::load({"x"}, {"y0", "y1"}, (const uint8_t*)adapterBuffer.data(),
                                                     adapterBuffer.size(), runtime, &adapterConfig));
        if (!adapter) {
            return false;
        }
        const int seq = 2, ic = 8;
        std::vector<float> xData(seq * ic);
        for (int i = 0; i < seq * ic; ++i) {
            xData[i] = (float)((i % 9) - 4) * 0.17f;
        }
        auto x = _Input({seq, ic, 1, 1}, NC4HW4);
        auto xC4 = packC4(xData, seq, ic);
        ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
        x->unMap();
        auto outputs = adapter->onForward({x});
        if (outputs.size() != 2) {
            return false;
        }
        const int ocs[2] = {8, 4};
        bool ok = true;
        for (int m = 0; m < 2; ++m) {
            auto conv = makeConv(ic, ocs[m], m + 1);
            std::vector<float> expected;
            convRef(xData, conv->weight, conv->bias, expected, seq, ic, ocs[m]);
            ok = ok && checkOut(outputs[m], expected, seq, ocs[m], "cpu_base_clone");
        }
        return ok;
    }
};
MNNTestSuiteRegister(FusedProjCPUBaseCloneTest, "op/fused_proj_cpu_base_clone");

// Folded LN with external gamma/beta under a reused mmap weight cache: the
// second load creates from the original op (gamma empty) and rearrange clones
// the composite onto the repacked op, so the clone must not rebuild its LN.
class FusedProjMmapLNTest : public MNNTestCase {
protected:
    bool runFusedProjMmapLN(int precision, MNNForwardType forwardType, const char* tag) {
        const int seq = 2, ic = 8;
        const int ocs[2] = {8, 4};
        const float eps = 1e-5f;
        const bool openCL = forwardType == MNN_FORWARD_OPENCL;
        std::vector<float> gamma(ic), beta(ic, 0.0f);
        for (int c = 0; c < ic; ++c) {
            gamma[c] = 0.85f + 0.04f * c;
        }
        const std::string dir = openCL ? "fused_proj_opencl_mmap_cache" : "fused_proj_cpu_mmap_cache";
        const std::string weightFile = MNNFilePathConcat(dir, "ln.weight");
        if (!MNNDirExist(dir.c_str()) && !MNNCreateDir(dir.c_str())) {
            return false;
        }
        auto pStatus = MNNTestSuite::get()->pStaus;
        std::string prefix = openCL ? "1_0_0_0_" : "0_0_0_0_";
        const std::string postfix = openCL ? "opencl.weight" : "static";
        if (!openCL) {
            prefix[2] += precision;
            prefix[4] += pStatus.memory;
            prefix[6] += pStatus.power;
        }
        const std::string syncFile = MNNFilePathConcat(dir, prefix + "sync." + postfix);
        auto cleanup = [&]() {
            for (int i = 0; i < 4; ++i) {
                auto name = MNNFilePathConcat(dir, prefix + std::to_string(i) + "." + postfix);
                if (MNNFileExist(name.c_str())) {
                    MNNRemoveFile(name.c_str());
                }
            }
            if (MNNFileExist(syncFile.c_str())) {
                MNNRemoveFile(syncFile.c_str());
            }
            if (MNNFileExist(weightFile.c_str())) {
                MNNRemoveFile(weightFile.c_str());
            }
        };
        cleanup();
        {
            FILE* f = fopen(weightFile.c_str(), "wb");
            if (f == nullptr) {
                return false;
            }
            fwrite(gamma.data(), sizeof(float), ic, f);
            fwrite(beta.data(), sizeof(float), ic, f);
            fclose(f);
        }

        std::vector<int8_t> model;
        {
            auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
            residual->setName("residual");
            auto hidden = _Input({seq, ic, 1, 1}, NC4HW4);
            hidden->setName("hidden");
            std::unique_ptr<OpT> op(new OpT);
            op->type = OpType_FusedLinear;
            op->name = "fused_proj_ln";
            op->main.type = OpParameter_FusedLinearParam;
            op->main.value = new FusedLinearParamT;
            op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
            auto param = op->main.AsFusedLinearParam();
            param->has_ln = true;
            param->ln.reset(new LayerNormT);
            param->ln->epsilon = eps;
            param->ln->axis = {-1};
            param->ln->useRMSNorm = true;
            param->ln->external = {0, (int64_t)(ic * sizeof(float)), (int64_t)(ic * sizeof(float))};
            for (int m = 0; m < 2; ++m) {
                param->convs.push_back(makeConv(ic, ocs[m], m + 1));
            }
            auto expr = Expr::create(std::move(op), {residual, hidden}, 3);
            std::vector<VARP> outs;
            for (int i = 0; i < 3; ++i) {
                outs.emplace_back(Variable::create(expr, i));
                outs.back()->setName("y" + std::to_string(i));
            }
            model = Variable::save(outs);
        }

        std::vector<float> resData(seq * ic), hidData(seq * ic);
        for (int i = 0; i < seq * ic; ++i) {
            resData[i] = (float)((i % 7) - 3) * 0.19f;
            hidData[i] = (float)((i % 9) - 4) * -0.07f;
        }
        std::vector<float> d(seq * ic), norm(seq * ic);
        for (int t = 0; t < seq; ++t) {
            float ss = 0.0f;
            for (int c = 0; c < ic; ++c) {
                d[t * ic + c] = resData[t * ic + c] + hidData[t * ic + c];
                ss += d[t * ic + c] * d[t * ic + c];
            }
            float inv = 1.0f / sqrtf(ss / ic + eps);
            for (int c = 0; c < ic; ++c) {
                norm[t * ic + c] = d[t * ic + c] * inv * gamma[c];
            }
        }

        // Phase 0 writes the weight cache, phase 1 reuses it.
        bool ok = true;
        for (int phase = 0; phase < 2 && ok; ++phase) {
            BackendConfig backendConfig;
            backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
            backendConfig.memory = static_cast<BackendConfig::MemoryMode>(pStatus.memory);
            backendConfig.power = static_cast<BackendConfig::PowerMode>(pStatus.power);
            const int runtimeMode = openCL ? MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE : 1;
            ScheduleConfig scheduleConfig;
            scheduleConfig.type = forwardType;
            scheduleConfig.backendConfig = &backendConfig;
            scheduleConfig.numThread = runtimeMode;
            auto executor = Executor::newExecutor(forwardType, backendConfig, runtimeMode);
            if (!executor) {
                ok = false;
                break;
            }
            ExecutorScope scope(executor);
            std::shared_ptr<Executor::RuntimeManager> runtime(
                Executor::RuntimeManager::createRuntimeManager(scheduleConfig), Executor::RuntimeManager::destroy);
            runtime->setHint(Interpreter::USE_CACHED_MMAP, 1);
            runtime->setExternalPath(dir, Interpreter::EXTERNAL_WEIGHT_DIR);
            runtime->setExternalFile(weightFile);
            Module::Config config;
            config.rearrange = true;
            std::shared_ptr<Module> module(Module::load({"residual", "hidden"}, {"y0", "y1", "y2"},
                                                        (const uint8_t*)model.data(), model.size(), runtime, &config),
                                           Module::destroy);
            if (!module) {
                ok = false;
                break;
            }
            auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
            auto hidden = _Input({seq, ic, 1, 1}, NC4HW4);
            auto resC4 = packC4(resData, seq, ic);
            auto hidC4 = packC4(hidData, seq, ic);
            ::memcpy(residual->writeMap<float>(), resC4.data(), resC4.size() * sizeof(float));
            ::memcpy(hidden->writeMap<float>(), hidC4.data(), hidC4.size() * sizeof(float));
            residual->unMap();
            hidden->unMap();
            auto outputs = module->onForward({residual, hidden});
            if (outputs.size() != 3) {
                ok = false;
                break;
            }
            for (int m = 0; m < 2; ++m) {
                auto conv = makeConv(ic, ocs[m], m + 1);
                std::vector<float> expected;
                convRef(norm, conv->weight, conv->bias, expected, seq, ic, ocs[m]);
                ok = ok && checkOut(outputs[m], expected, seq, ocs[m], tag);
            }
            ok = ok && checkOut(outputs[2], d, seq, ic, tag);
            if (phase == 0 && !MNNFileExist(syncFile.c_str())) {
                MNN_ERROR("FusedProj %s: weight cache was not written\n", tag);
                ok = false;
            }
        }
        cleanup();
        return ok;
    }
};

class FusedProjCPUMmapLNTest : public FusedProjMmapLNTest {
public:
    virtual bool run(int precision) { return runFusedProjMmapLN(precision, MNN_FORWARD_CPU, "cpu_mmap_ln"); }
};
MNNTestSuiteRegister(FusedProjCPUMmapLNTest, "op/fused_proj_cpu_mmap_ln");

class FusedProjOpenCLMmapLNTest : public FusedProjMmapLNTest {
public:
    virtual bool run(int precision) {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_OPENCL) {
            return true;
        }
        return runFusedProjMmapLN(precision, MNN_FORWARD_OPENCL, "opencl_mmap_ln");
    }
};
MNNTestSuiteRegister(FusedProjOpenCLMmapLNTest, "op/fused_proj_opencl_mmap_ln");

class FusedGateUpTest : public MNNTestCase {
public:
    virtual ~FusedGateUpTest() = default;
    virtual bool run(int precision) {
        const int seq = 2, ic = 8, oc = 8;
        std::vector<float> xData(seq * ic);
        for (int i = 0; i < seq * ic; ++i) {
            xData[i] = (float)((i % 11) - 5) * 0.13f;
        }
        auto x = _Input({seq, ic, 1, 1}, NC4HW4);
        auto xC4 = packC4(xData, seq, ic);
        ::memcpy(x->writeMap<float>(), xC4.data(), xC4.size() * sizeof(float));
        x->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        param->act_silu_mul = true;
        param->convs.push_back(makeConv(ic, oc, 1)); // gate
        param->convs.push_back(makeConv(ic, oc, 2)); // up
        auto expr = Expr::create(std::move(op), {x}, 1);

        auto& convs = op->main.AsFusedLinearParam()->convs;
        std::vector<float> gate, up;
        convRef(xData, convs[0]->weight, convs[0]->bias, gate, seq, ic, oc);
        convRef(xData, convs[1]->weight, convs[1]->bias, up, seq, ic, oc);
        std::vector<float> expected(seq * oc);
        for (int i = 0; i < seq * oc; ++i) {
            expected[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
        }
        return checkOut(Variable::create(expr, 0), expected, seq, oc, "gateup");
    }
};
MNNTestSuiteRegister(FusedGateUpTest, "op/fused_gateup");

// gate/up MLP with the block-input RMSNorm folded in: in [residual, hidden],
// out [silu_mul, residual_out]. seq == 1 is the decode shape, the only one a
// backend's fused GEMV path can take.
class FusedGateUpLNTest : public MNNTestCase {
public:
    FusedGateUpLNTest(int seq) : mSeq(seq) {
    }
    virtual ~FusedGateUpLNTest() = default;
    virtual bool run(int precision) {
        const int seq = mSeq, ic = 8, oc = 8;
        const float eps = 1e-5f;
        std::vector<float> resData(seq * ic), hidData(seq * ic), gamma(ic);
        for (int i = 0; i < seq * ic; ++i) {
            resData[i] = (float)((i % 7) - 3) * 0.17f;
            hidData[i] = (float)((i % 5) - 2) * -0.11f;
        }
        for (int c = 0; c < ic; ++c) {
            gamma[c] = 0.9f + 0.03f * c;
        }
        auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
        auto hidden   = _Input({seq, ic, 1, 1}, NC4HW4);
        auto resC4 = packC4(resData, seq, ic);
        auto hidC4 = packC4(hidData, seq, ic);
        ::memcpy(residual->writeMap<float>(), resC4.data(), resC4.size() * sizeof(float));
        ::memcpy(hidden->writeMap<float>(), hidC4.data(), hidC4.size() * sizeof(float));
        residual->unMap();
        hidden->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        param->act_silu_mul = true;
        param->has_ln = true;
        param->ln.reset(new LayerNormT);
        param->ln->epsilon = eps;
        param->ln->gamma = gamma;
        param->ln->beta = std::vector<float>(ic, 0.0f);
        param->ln->axis = {-1};
        param->ln->useRMSNorm = true;
        param->convs.push_back(makeConv(ic, oc, 1)); // gate
        param->convs.push_back(makeConv(ic, oc, 2)); // up
        auto expr = Expr::create(std::move(op), {residual, hidden}, 2);

        std::vector<float> d(seq * ic), norm(seq * ic);
        for (int t = 0; t < seq; ++t) {
            float ss = 0.0f;
            for (int c = 0; c < ic; ++c) {
                d[t * ic + c] = resData[t * ic + c] + hidData[t * ic + c];
                ss += d[t * ic + c] * d[t * ic + c];
            }
            float inv = 1.0f / sqrtf(ss / ic + eps);
            for (int c = 0; c < ic; ++c) {
                norm[t * ic + c] = d[t * ic + c] * inv * gamma[c];
            }
        }
        auto& convs = op->main.AsFusedLinearParam()->convs;
        std::vector<float> gate, up;
        convRef(norm, convs[0]->weight, convs[0]->bias, gate, seq, ic, oc);
        convRef(norm, convs[1]->weight, convs[1]->bias, up, seq, ic, oc);
        std::vector<float> expected(seq * oc);
        for (int i = 0; i < seq * oc; ++i) {
            expected[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
        }
        bool ok = checkOut(Variable::create(expr, 0), expected, seq, oc, "gateup_ln_proj");
        ok = ok && checkOut(Variable::create(expr, 1), d, seq, ic, "gateup_ln_residual");
        return ok;
    }

private:
    int mSeq;
};
class FusedGateUpLNPrefillTest : public FusedGateUpLNTest {
public:
    FusedGateUpLNPrefillTest() : FusedGateUpLNTest(2) {
    }
};
class FusedGateUpLNDecodeTest : public FusedGateUpLNTest {
public:
    FusedGateUpLNDecodeTest() : FusedGateUpLNTest(1) {
    }
};
MNNTestSuiteRegister(FusedGateUpLNPrefillTest, "op/fused_gateup_ln");
MNNTestSuiteRegister(FusedGateUpLNDecodeTest, "op/fused_gateup_ln_decode");

class FusedQKVLNTest : public MNNTestCase {
public:
    virtual ~FusedQKVLNTest() = default;
    virtual bool run(int precision) {
        const int seq = 2, ic = 8;
        const int ocs[3] = {8, 4, 4};
        const float eps = 1e-5f;
        std::vector<float> resData(seq * ic), hidData(seq * ic), gamma(ic);
        for (int i = 0; i < seq * ic; ++i) {
            resData[i] = (float)((i % 7) - 3) * 0.19f;
            hidData[i] = (float)((i % 9) - 4) * -0.07f;
        }
        for (int c = 0; c < ic; ++c) {
            gamma[c] = 0.8f + 0.05f * c;
        }
        auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
        auto hidden   = _Input({seq, ic, 1, 1}, NC4HW4);
        auto resC4 = packC4(resData, seq, ic);
        auto hidC4 = packC4(hidData, seq, ic);
        ::memcpy(residual->writeMap<float>(), resC4.data(), resC4.size() * sizeof(float));
        ::memcpy(hidden->writeMap<float>(), hidC4.data(), hidC4.size() * sizeof(float));
        residual->unMap();
        hidden->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        param->has_ln = true;
        param->ln.reset(new LayerNormT);
        param->ln->epsilon = eps;
        param->ln->gamma = gamma;
        param->ln->beta = std::vector<float>(ic, 0.0f);
        param->ln->axis = {-1};
        param->ln->useRMSNorm = true;
        for (int m = 0; m < 3; ++m) {
            param->convs.push_back(makeConv(ic, ocs[m], m + 1));
        }
        // in [residual, hidden], out [q, k, v, residual_out]
        auto expr = Expr::create(std::move(op), {residual, hidden}, 4);

        // Reference: d = residual + hidden; n = rmsnorm(d) * gamma; residual_out = d.
        std::vector<float> d(seq * ic), norm(seq * ic);
        for (int t = 0; t < seq; ++t) {
            float ss = 0.0f;
            for (int c = 0; c < ic; ++c) {
                d[t * ic + c] = resData[t * ic + c] + hidData[t * ic + c];
                ss += d[t * ic + c] * d[t * ic + c];
            }
            float inv = 1.0f / sqrtf(ss / ic + eps);
            for (int c = 0; c < ic; ++c) {
                norm[t * ic + c] = d[t * ic + c] * inv * gamma[c];
            }
        }
        bool ok = checkOut(Variable::create(expr, 3), d, seq, ic, "qkv_ln_residual");
        auto& convs = op->main.AsFusedLinearParam()->convs;
        for (int m = 0; m < 3; ++m) {
            std::vector<float> expected;
            convRef(norm, convs[m]->weight, convs[m]->bias, expected, seq, ic, ocs[m]);
            ok = ok && checkOut(Variable::create(expr, m), expected, seq, ocs[m], "qkv_ln_proj");
        }
        return ok;
    }
};
MNNTestSuiteRegister(FusedQKVLNTest, "op/fused_qkv_ln");

// Qwen3.5 linear attention exports four shared-input projections
// (in_proj_qkv/z/b/a) as one FusedLinear, with the block-input RMSNorm folded in.
class FusedQKVP4LNTest : public MNNTestCase {
public:
    virtual ~FusedQKVP4LNTest() = default;
    virtual bool run(int precision) {
        const int seq = 2, ic = 8;
        const int ocs[4] = {12, 8, 4, 4};
        const float eps = 1e-5f;
        std::vector<float> resData(seq * ic), hidData(seq * ic), gamma(ic);
        for (int i = 0; i < seq * ic; ++i) {
            resData[i] = (float)((i % 5) - 2) * 0.23f;
            hidData[i] = (float)((i % 11) - 5) * -0.09f;
        }
        for (int c = 0; c < ic; ++c) {
            gamma[c] = 1.1f - 0.04f * c;
        }
        auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
        auto hidden   = _Input({seq, ic, 1, 1}, NC4HW4);
        auto resC4 = packC4(resData, seq, ic);
        auto hidC4 = packC4(hidData, seq, ic);
        ::memcpy(residual->writeMap<float>(), resC4.data(), resC4.size() * sizeof(float));
        ::memcpy(hidden->writeMap<float>(), hidC4.data(), hidC4.size() * sizeof(float));
        residual->unMap();
        hidden->unMap();

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_FusedLinear;
        op->main.type = OpParameter_FusedLinearParam;
        op->main.value = new FusedLinearParamT;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto param = op->main.AsFusedLinearParam();
        param->has_ln = true;
        param->ln.reset(new LayerNormT);
        param->ln->epsilon = eps;
        param->ln->gamma = gamma;
        param->ln->beta = std::vector<float>(ic, 0.0f);
        param->ln->axis = {-1};
        param->ln->useRMSNorm = true;
        for (int m = 0; m < 4; ++m) {
            param->convs.push_back(makeConv(ic, ocs[m], m + 1));
        }
        // in [residual, hidden], out [qkv, z, b, a, residual_out]
        auto expr = Expr::create(std::move(op), {residual, hidden}, 5);

        std::vector<float> d(seq * ic), norm(seq * ic);
        for (int t = 0; t < seq; ++t) {
            float ss = 0.0f;
            for (int c = 0; c < ic; ++c) {
                d[t * ic + c] = resData[t * ic + c] + hidData[t * ic + c];
                ss += d[t * ic + c] * d[t * ic + c];
            }
            float inv = 1.0f / sqrtf(ss / ic + eps);
            for (int c = 0; c < ic; ++c) {
                norm[t * ic + c] = d[t * ic + c] * inv * gamma[c];
            }
        }
        bool ok = checkOut(Variable::create(expr, 4), d, seq, ic, "qkv_p4_ln_residual");
        auto& convs = op->main.AsFusedLinearParam()->convs;
        for (int m = 0; m < 4; ++m) {
            std::vector<float> expected;
            convRef(norm, convs[m]->weight, convs[m]->bias, expected, seq, ic, ocs[m]);
            ok = ok && checkOut(Variable::create(expr, m), expected, seq, ocs[m], "qkv_p4_ln_proj");
        }
        return ok;
    }
};
MNNTestSuiteRegister(FusedQKVP4LNTest, "op/fused_qkv_p4_ln");

// --- Quantized decode-shape coverage -----------------------------------------
//
// The float cases above never reach a backend's fused GEMV path: Metal's
// mIs2sgDecode is only set for quantized convs under Memory_Low. These cases
// run at seq == 1 with 4/8-bit weights under their own Memory_Low executor so
// Metal actually exercises setupQKVFusion / setupGateUpFusion / setupLNFusion.
// The mixed-bit case makes the projection fusion fail on purpose (quant layout
// mismatch): the LayerNorm must then stay a separate dispatch — the regression
// guard for MetalFusedProj::setupFusion ignoring the projection-fusion result.

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/IDSTEncoder.hpp"

// Asymmetric block quant, mirroring _HybridConv: alpha carries (min, scale)
// pairs; recon receives the dequantized weights for the fp32 reference.
static std::unique_ptr<Convolution2DT> makeQuantConv(int ic, int oc, int nbits, int blocksize, int seed,
                                                     std::vector<float>& recon, std::vector<float>& bias) {
    const float threshold = (float)(1 << (nbits - 1)) - 1.0f;
    const float clampMin  = -threshold - 1.0f;
    const int blocknum    = ic / blocksize;
    std::vector<float> weight(oc * ic), alpha(2 * oc * blocknum);
    bias.resize(oc);
    recon.resize(oc * ic);
    for (int o = 0; o < oc; ++o) {
        bias[o] = (float)(((o * 5 + seed * 3) % 11) - 5) * 0.013f;
        for (int i = 0; i < ic; ++i) {
            weight[o * ic + i] = (float)(((o * ic + i) * 7 + seed * 13) % 17 - 8) * 0.021f;
        }
    }
    for (int o = 0; o < oc; ++o) {
        for (int b = 0; b < blocknum; ++b) {
            const float* w = weight.data() + o * ic + b * blocksize;
            float mn = w[0], mx = w[0];
            for (int u = 1; u < blocksize; ++u) {
                mn = std::min(mn, w[u]);
                mx = std::max(mx, w[u]);
            }
            const float scale        = (mx - mn) / (threshold - clampMin);
            alpha[2 * (o * blocknum + b)]     = mn;
            alpha[2 * (o * blocknum + b) + 1] = scale;
            for (int u = 0; u < blocksize; ++u) {
                int q = (int)((w[u] - mn) * (threshold - clampMin) / (mx - mn) + clampMin);
                recon[o * ic + b * blocksize + u] = (q - (int)clampMin) * scale + mn;
            }
        }
    }
    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->common.reset(new Convolution2DCommonT);
    conv->common->kernelX     = 1;
    conv->common->kernelY     = 1;
    conv->common->inputCount  = ic;
    conv->common->outputCount = oc;
    conv->quanParameter = IDSTEncoder::encode(weight.data(), alpha, blocksize, oc * blocknum,
                                              /*async=*/true, nullptr, (int)clampMin, {nbits, false});
    conv->bias = bias;
    return conv;
}

// Input dynamic quant + fp16 storage make byte-exactness impossible; same
// tolerance shape as QKVFusionTest.
static bool checkOutLoose(VARP out, const std::vector<float>& expectedPlain, int seq, int channel,
                          const char* tag) {
    auto expectedC4 = packC4(expectedPlain, seq, channel);
    auto info = out->getInfo();
    if (info == nullptr || info->dim.size() != 4 || info->dim[0] != seq || info->dim[1] != channel) {
        MNN_ERROR("FusedProj %s: bad shape\n", tag);
        return false;
    }
    auto ptr = out->readMap<float>();
    if (ptr == nullptr) {
        MNN_ERROR("FusedProj %s: null output\n", tag);
        return false;
    }
    float maxVal = 0.001f;
    for (auto v : expectedC4) {
        maxVal = std::max(maxVal, fabsf(v));
    }
    for (size_t i = 0; i < expectedC4.size(); ++i) {
        float err = fabsf(ptr[i] - expectedC4[i]);
        if (err > 0.5f && err / maxVal > 0.1f) {
            MNN_ERROR("FusedProj %s[%d]: got %f, expect %f\n", tag, (int)i, ptr[i], expectedC4[i]);
            return false;
        }
    }
    return true;
}

// Quantized FusedLinear with folded LN. seq 1 is the decode shape; seq >= 256
// reaches the prefill M64 tile and its gate/up dual epilogue.
// bitsPerConv.size() selects the flavour: 2 = gate/up (act_silu_mul), else qkv.
static bool runQuantLNCase(const std::vector<int>& bitsPerConv, const std::vector<int>& ocs, const char* tag,
                           int seq = 1, int ic = 128, int blocksize = 32) {
    const float eps = 1e-5f;
    const bool gateUp = bitsPerConv.size() == 2;

    auto status = MNNTestSuite::get()->pStaus;
    BackendConfig config;
    config.precision = (BackendConfig::PrecisionMode)status.precision;
    config.memory    = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor((MNNForwardType)status.forwardType, config, 1);
    ExecutorScope scope(exe);

    std::vector<float> resData(seq * ic), hidData(seq * ic), gamma(ic);
    for (int i = 0; i < seq * ic; ++i) {
        resData[i] = (float)((i % 7) - 3) * 0.17f;
        hidData[i] = (float)((i % 5) - 2) * -0.11f;
    }
    for (int c = 0; c < ic; ++c) {
        gamma[c] = 0.9f + 0.002f * c;
    }
    auto residual = _Input({seq, ic, 1, 1}, NC4HW4);
    auto hidden   = _Input({seq, ic, 1, 1}, NC4HW4);
    auto resC4 = packC4(resData, seq, ic);
    auto hidC4 = packC4(hidData, seq, ic);
    ::memcpy(residual->writeMap<float>(), resC4.data(), resC4.size() * sizeof(float));
    ::memcpy(hidden->writeMap<float>(), hidC4.data(), hidC4.size() * sizeof(float));
    residual->unMap();
    hidden->unMap();

    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_FusedLinear;
    op->main.type = OpParameter_FusedLinearParam;
    op->main.value = new FusedLinearParamT;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    auto param = op->main.AsFusedLinearParam();
    param->act_silu_mul = gateUp;
    param->has_ln = true;
    param->ln.reset(new LayerNormT);
    param->ln->epsilon = eps;
    param->ln->gamma = gamma;
    param->ln->beta = std::vector<float>(ic, 0.0f);
    param->ln->axis = {-1};
    param->ln->useRMSNorm = true;
    std::vector<std::vector<float>> recon(bitsPerConv.size()), bias(bitsPerConv.size());
    for (size_t m = 0; m < bitsPerConv.size(); ++m) {
        param->convs.push_back(makeQuantConv(ic, ocs[m], bitsPerConv[m], blocksize, (int)m + 1,
                                             recon[m], bias[m]));
    }
    const int numProjOut = gateUp ? 1 : (int)bitsPerConv.size();
    auto expr = Expr::create(std::move(op), {residual, hidden}, numProjOut + 1);

    std::vector<float> d(seq * ic), norm(seq * ic);
    for (int t = 0; t < seq; ++t) {
        float ss = 0.0f;
        for (int c = 0; c < ic; ++c) {
            d[t * ic + c] = resData[t * ic + c] + hidData[t * ic + c];
            ss += d[t * ic + c] * d[t * ic + c];
        }
        float inv = 1.0f / sqrtf(ss / ic + eps);
        for (int c = 0; c < ic; ++c) {
            norm[t * ic + c] = d[t * ic + c] * inv * gamma[c];
        }
    }
    bool ok = checkOutLoose(Variable::create(expr, numProjOut), d, seq, ic,
                            (std::string(tag) + "_residual").c_str());
    if (gateUp) {
        std::vector<float> gate, up, expected(seq * ocs[0]);
        convRef(norm, recon[0], bias[0], gate, seq, ic, ocs[0]);
        convRef(norm, recon[1], bias[1], up, seq, ic, ocs[0]);
        for (int i = 0; i < seq * ocs[0]; ++i) {
            expected[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
        }
        ok = ok && checkOutLoose(Variable::create(expr, 0), expected, seq, ocs[0], tag);
    } else {
        for (size_t m = 0; m < bitsPerConv.size(); ++m) {
            std::vector<float> expected;
            convRef(norm, recon[m], bias[m], expected, seq, ic, ocs[m]);
            ok = ok && checkOutLoose(Variable::create(expr, (int)m), expected, seq, ocs[m], tag);
        }
    }
    if (!ok) {
        MNN_ERROR("FusedProj quant case %s FAILED\n", tag);
    }
    return ok;
}

// Uniform 4-bit q/k/v + LN: projection fusion succeeds, the LN folds into the
// leader — the actual fused dispatch.
class FusedQKVLNQuantTest : public MNNTestCase {
public:
    virtual ~FusedQKVLNQuantTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4, 4}, {64, 16, 16}, "qkv_ln_quant");
    }
};
MNNTestSuiteRegister(FusedQKVLNQuantTest, "op/fused_qkv_ln_quant");

// Uniform 4-bit q/k/v + LN at the 0.6b decode shape: 16 quant blocks, above
// the narrow split-K ceiling, so the fused QKV GEMV routes to the unsplit
// single-stream body (the path the ceiling default change flipped).
class FusedQKVLNQuant16BlockTest : public MNNTestCase {
public:
    virtual ~FusedQKVLNQuant16BlockTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4, 4}, {2048, 1024, 1024}, "qkv_ln_quant_b16", 1, 1024, 64);
    }
};
MNNTestSuiteRegister(FusedQKVLNQuant16BlockTest, "op/fused_qkv_ln_quant_b16");

// Uniform 4-bit q/k/v + LN at a prefill sequence length: the non-dual M64
// fused-quant tile (QKV fusion is decode-only, so each projection dispatches
// its own kernel), exercising the shared staging loop without the gate/up
// dual accumulator.
class FusedQKVLNQuantPrefillTest : public MNNTestCase {
public:
    virtual ~FusedQKVLNQuantPrefillTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4, 4}, {64, 16, 16}, "qkv_ln_quant_prefill", 256);
    }
};
MNNTestSuiteRegister(FusedQKVLNQuantPrefillTest, "op/fused_qkv_ln_quant_prefill");

// 4-bit gate/up + LN: the GATE_UP_FUSED × LN_FUSED pipeline.
class FusedGateUpLNQuantTest : public MNNTestCase {
public:
    virtual ~FusedGateUpLNQuantTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4}, {64, 64}, "gateup_ln_quant");
    }
};
MNNTestSuiteRegister(FusedGateUpLNQuantTest, "op/fused_gateup_ln_quant");

// 4-bit gate/up + LN at decode shapes whose K splits: blockCount 16 drives
// GEMV_2OCQUAD_PER_SG_SPLIT_K, and the blocksize-64 variant additionally
// reaches the dual-stream W16 uint4 branch (quadsPerBlock 16).
class FusedGateUpLNQuantSplitKTest : public MNNTestCase {
public:
    virtual ~FusedGateUpLNQuantSplitKTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4}, {64, 64}, "gateup_ln_quant_splitk", 1, 512, 32);
    }
};
MNNTestSuiteRegister(FusedGateUpLNQuantSplitKTest, "op/fused_gateup_ln_quant_splitk");

class FusedGateUpLNQuantSplitKW16DsTest : public MNNTestCase {
public:
    virtual ~FusedGateUpLNQuantSplitKW16DsTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4}, {64, 64}, "gateup_ln_quant_splitk_w16ds", 1, 1024, 64);
    }
};
MNNTestSuiteRegister(FusedGateUpLNQuantSplitKW16DsTest, "op/fused_gateup_ln_quant_splitk_w16ds");

// Same pipeline at a prefill sequence length: the M64 fused-quant tile, where
// one dispatch accumulates both projections' tiles and applies silu-mul in
// registers instead of materializing the gate tensor.
class FusedGateUpLNQuantPrefillTest : public MNNTestCase {
public:
    virtual ~FusedGateUpLNQuantPrefillTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4}, {64, 64}, "gateup_ln_quant_prefill", 256);
    }
};
MNNTestSuiteRegister(FusedGateUpLNQuantPrefillTest, "op/fused_gateup_ln_quant_prefill");

// Mixed 4/8-bit q/k/v + LN: setupQKVFusion rejects the quant-layout mismatch,
// so the LN fold must be skipped and the LayerNorm dispatched separately —
// wrong output here means the projection-fusion result was ignored again.
class FusedQKVLNQuantMixedTest : public MNNTestCase {
public:
    virtual ~FusedQKVLNQuantMixedTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 8, 4}, {64, 16, 16}, "qkv_ln_quant_mixed");
    }
};
MNNTestSuiteRegister(FusedQKVLNQuantMixedTest, "op/fused_qkv_ln_quant_mixed");

// Uniform 4-bit 4-projection group + LN: the QKV_FUSED_P4 × LN_FUSED pipeline
// that Qwen3.5 linear-attention layers use (qkv/z/b/a share one LN input).
// The 2/3-bit P4 variants are separate shader builds but have no case here:
// makeQuantConv's reference dequantization only has 4 levels at 2 bits and
// drifts past checkOutLoose's tolerance on CPU as well.
class FusedQKVP4LNQuantTest : public MNNTestCase {
public:
    virtual ~FusedQKVP4LNQuantTest() = default;
    virtual bool run(int precision) {
        return runQuantLNCase({4, 4, 4, 4}, {64, 16, 16, 16}, "qkv_p4_ln_quant");
    }
};
MNNTestSuiteRegister(FusedQKVP4LNQuantTest, "op/fused_qkv_p4_ln_quant");

#endif
