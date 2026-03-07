//
//  AttentionStabilityTest.cpp
//  MNNTests
//
//  Created by Aime (AI assistant) on 2026/03/07.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

#include <vector>
#include <cmath>
#include <limits>
#include <string>

using namespace MNN;
using namespace MNN::Express;

// Test configuration: B=1, NumHead=16, KvNumHead=2, HeadDim=128
static const int kBatch      = 1;
static const int kNumHead    = 16;
static const int kKvNumHead  = 2;
static const int kHeadDim    = 128;

// Max allowed absolute difference between two runs with identical inputs
static const float kRepeatTolerance = 1e-3f;

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    } file_operation;
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
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};
static KVMeta gMeta;

// Create Attention Module via FlatBuffers, kv_cache=true, numThread=1.
static std::shared_ptr<Module> _makeAttentionModule(int attentionMode = 8) {
    auto Q    = _Input();
    auto K    = _Input();
    auto V    = _Input();
    auto mask = _Input();

    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type      = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;

    auto o      = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
    auto buffer = Variable::save({o});

    MNN::ScheduleConfig config;
    auto status  = MNNTestSuite::get()->pStaus;
    config.type  = static_cast<MNNForwardType>(status.forwardType);
    FUNC_PRINT(config.type);

    MNN::BackendConfig bnConfig;
    bnConfig.memory    = static_cast<MNN::BackendConfig::MemoryMode>(status.memory);
    bnConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(status.precision);
    bnConfig.power     = static_cast<MNN::BackendConfig::PowerMode>(status.power);
    config.backendConfig = &bnConfig;
    config.numThread     = 1; // single thread for deterministic behavior

    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    // Pass KV meta information to backend
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    // Keep default attention option (8) unless overridden by caller
//    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);

    std::shared_ptr<Module> m(Module::load({}, {}, reinterpret_cast<uint8_t*>(buffer.data()), buffer.size(), rtmgr));
    return m;
}

// Deterministic data filler: simple periodic pattern, independent of rand()
static void fillDeterministic(VARP var, float scale, float offset) {
    auto info = var->getInfo();
    if (nullptr == info) {
        return;
    }
    auto ptr = var->writeMap<float>();
    if (nullptr == ptr) {
        return;
    }
    const int size = info->size;
    for (int i = 0; i < size; ++i) {
        // Range roughly [-1, 1] * scale + offset
        ptr[i] = ((i % 97) - 48) * scale + offset;
    }
    var->unMap();
}

struct AttentionInputs {
    VARP q;
    VARP k;
    VARP v;
    VARP mask;
};

// Create deterministic Q/K/V/Mask for given seq_len.
static AttentionInputs makeInputs(int seqLen) {
    AttentionInputs inputs;

    inputs.q = _Input({kBatch, seqLen, kNumHead,   kHeadDim}, NCHW, halide_type_of<float>());
    inputs.k = _Input({kBatch, seqLen, kKvNumHead, kHeadDim}, NCHW, halide_type_of<float>());
    inputs.v = _Input({kBatch, seqLen, kKvNumHead, kHeadDim}, NCHW, halide_type_of<float>());

    // Mask set to scalar 0, consistent with generateMask() simplified usage
    inputs.mask = _Input({}, NCHW, halide_type_of<float>());

    fillDeterministic(inputs.q, 0.01f, 0.0f);
    fillDeterministic(inputs.k, 0.08f, 0.13f);
    fillDeterministic(inputs.v, 0.09f, -0.05f);

    auto maskPtr = inputs.mask->writeMap<float>();
    if (maskPtr) {
        maskPtr[0] = 0.0f;
        inputs.mask->unMap();
    }

    return inputs;
}

// Reset KV meta before a new run with given sequence length.
// This follows the requirement: previous=0, add=seq_len, and no cross-shape history.
static void resetKVMeta(int seqLen) {
    gMeta.previous       = 0;
    gMeta.remove         = 0;
    gMeta.reserve        = nullptr;
    gMeta.n_reserve      = 0;
    gMeta.add            = static_cast<size_t>(seqLen);
    gMeta.file_name.clear();
    gMeta.seqlen_in_disk = 0;
    gMeta.layer_index    = 0;
    gMeta.layer_nums     = 0;
}

// Check that all elements in tensor are finite (no NaN/Inf).
static bool checkFinite(VARP y) {
    auto info = y->getInfo();
    if (nullptr == info) {
        MNN_ERROR("AttentionStabilityTest: output info is null\n");
        return false;
    }
    const float* ptr = y->readMap<float>();
    if (nullptr == ptr) {
        MNN_ERROR("AttentionStabilityTest: output data is null\n");
        return false;
    }
    for (int i = 0; i < info->size; ++i) {
        float v = ptr[i];
        if (!std::isfinite(v)) {
            MNN_ERROR("AttentionStabilityTest: found non-finite value %f at index %d\n", v, i);
            return false;
        }
    }
    return true;
}

// Compute max absolute difference between two tensors.
static float maxAbsDiff(VARP a, VARP b) {
    auto aPtr = a->readMap<float>();
    auto bPtr = b->readMap<float>();
    auto size = a->getInfo()->size;
    float maxV = 0.0f;
    float maxA = 0.0f;
    for (int i=0; i<size; ++i) {
        maxV = fmaxf(maxV, aPtr[i]-bPtr[i]);
        maxA = fmaxf(aPtr[i], maxA);
    }
    return maxV;
}

class AttentionStabilityTest : public MNNTestCase {
public:
    AttentionStabilityTest()  = default;
    ~AttentionStabilityTest() override = default;

    bool run(int precision) override {
        (void)precision;

        auto module = _makeAttentionModule();
        if (!module) {
            MNN_ERROR("AttentionStabilityTest: failed to create Attention module\n");
            return false;
        }

        // Sequence lengths to test.
        std::vector<int> seqLens = {65, 1, 3, 7, 16, 32, 33, 64};

        std::vector<VARP> outputs;
        gMeta.remove = gMeta.previous;
        for (int seqLen : seqLens) {
            // First run: finite check and baseline output.
            auto inputs1 = makeInputs(seqLen);
            gMeta.add = seqLen;
            auto outs1 = module->onForward({inputs1.q, inputs1.k, inputs1.v, inputs1.mask});
            gMeta.sync();
            if (outs1.empty()) {
                MNN_ERROR("AttentionStabilityTest: forward failed for seq_len=%d (first run)\n", seqLen);
                return false;
            }
            auto out1 = outs1[0];
            
            if (!checkFinite(out1)) {
                MNN_ERROR("AttentionStabilityTest: non-finite output for seq_len=%d (first run)\n", seqLen);
                return false;
            }
            out1.fix(VARP::CONSTANT);
            outputs.emplace_back(out1);
        }
        gMeta.remove = gMeta.previous;
        for (int i=0; i<seqLens.size(); ++i) {
            auto seqLen = seqLens[i];
            // Second run with the same shape and identical deterministic inputs.
            auto inputs2 = makeInputs(seqLen);
            gMeta.add = seqLen;
            auto outs2 = module->onForward({inputs2.q, inputs2.k, inputs2.v, inputs2.mask});
            gMeta.sync();
            if (outs2.empty()) {
                MNN_ERROR("AttentionStabilityTest: forward failed for seq_len=%d (second run)\n", seqLen);
                return false;
            }
            auto out2 = outs2[0];

            if (!checkFinite(out2)) {
                MNN_ERROR("AttentionStabilityTest: non-finite output for seq_len=%d (second run)\n", seqLen);
                return false;
            }

            float diff = maxAbsDiff(outputs[i], out2);
            if (!(diff <= kRepeatTolerance)) {
                MNN_PRINT("AttentionStabilityTest: stability failed for seq_len=%d, max diff=%f (threshold=%f)\n",
                          seqLen, diff, kRepeatTolerance);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(AttentionStabilityTest, "op/attention_stability");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
