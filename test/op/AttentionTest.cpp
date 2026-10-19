//
//  AttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2024/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "core/MNNFileUtils.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <vector>
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;
using MNN::KVMeta;

int NumHead = 16;
int KvNumHead = 2;
int HeadDim = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;
// Additive-mask "masked out" bias. Must stay representable in fp16: the float
// minimum becomes -inf in half, and the `(1 - m) * bias` masks below then evaluate
// visible entries as 0 * -inf = NaN, which silently poisons every GPU fp16 result.
const float kMaskNegative = -30000.0f;
const int pastLength = 101;
#define GENERATE_TOKENS 128

static KVMeta gMeta;
static std::shared_ptr<Module> _makeAttentionModule(int attentionMode = 8, bool outputC4 = false,
                                                    bool forceOpenCLBuffer = false, int numThread = -1,
                                                    KVMeta* meta = &gMeta,
                                                    const std::string& prefixCacheDir = std::string()) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
    attention->main.AsAttentionParam()->output_c4 = outputC4;
    auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
    auto buffer = Variable::save({o});
    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bnConfig;
    bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    if (numThread < 0) {
        numThread = status.thread > 0 ? status.thread : 1;
    }
    config.numThread = forceOpenCLBuffer && status.forwardType == MNN_FORWARD_OPENCL
                           ? MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE
                           : numThread;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, meta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    if (!prefixCacheDir.empty()) {
        rtmgr->setExternalPath(prefixCacheDir, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
    }
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

// The shared executor creates its OpenCL runtime before any test runs, and a
// RuntimeManager reuses that runtime without re-applying its numThread mode bits,
// so per-module MNN_GPU_MEMORY_BUFFER requests are dropped and memory mode stays
// AUTO (IMAGE on Adreno). The OpenCL Attention op only registers a BUFFER creator,
// so under the shared runtime every case below silently runs on CPU instead. Hold
// one of these for the duration of a run() to execute on a private buffer-mode
// runtime. No-op on other backends.
struct OpenCLBufferScope {
    std::shared_ptr<Executor> mExe;
    std::shared_ptr<ExecutorScope> mScope;
    OpenCLBufferScope() {
        auto status = MNNTestSuite::get()->pStaus;
        if (MNN_FORWARD_OPENCL != (MNNForwardType)status.forwardType) {
            return;
        }
        MNN::BackendConfig bnConfig;
        bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
        bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
        bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
        mExe = Executor::newExecutor(MNN_FORWARD_OPENCL, bnConfig, MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE);
        mScope.reset(new ExecutorScope(mExe));
    }
};

// The OpenCL attention op stores Q/K/V as half whenever precision != High, while
// the expr decomposition keeps fp32 storage in Normal mode; the op-vs-expr runs
// then disagree by the half rounding of the inputs, so Normal needs a wider
// tolerance than the Low (both half) and High (both float) runs.
static float _opExprDiffThreshold(int precision) {
    return precision == (int)MNN::BackendConfig::Precision_Normal ? 0.05f : 0.01f;
}

struct KVCache {
    VARP pastK;
    VARP pastV;
    VARP pastMask;
    int current = 0;
    KVCache() {
        pastK = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastV = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastMask = _Input({pastLength}, NCHW);
        ::memset(pastK->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        ::memset(pastV->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        for (int v = 0; v < pastLength; ++v) {
            pastMask->writeMap<float>()[v] = kMaskNegative;
        }
    }
};

static VARP _computeAttentionExpr(VARP Q, VARP K, VARP V, VARP mask, KVCache cache) {
    auto qinfo = Q->getInfo();
    auto kinfo = K->getInfo();
    auto vinfo = V->getInfo();
    auto seqLength = qinfo->dim[1];
    auto numHead = qinfo->dim[2];
    auto headDim = qinfo->dim[3];
    auto kvNumHead = kinfo->dim[2];
    auto batch = qinfo->dim[0];
    auto group = numHead / kvNumHead;
    if (mask->getInfo()->type.code == halide_type_int) {
        mask = (_Scalar<float>(1.0) - _Cast<float>(mask)) * _Scalar<float>(kMaskNegative);
    }

    Q = _Reshape(Q, {batch, seqLength, kvNumHead, group, headDim});
    Q = _Transpose(Q, {0, 2, 3, 1, 4});
    K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
    K = _Transpose(K, {0, 2, 3, 1, 4});

    auto scale = 1.0f / sqrtf(headDim);
    K = K * _Scalar<float>(scale);
    K.fix(VARP::CONSTANT);
    auto QK = _MatMul(Q, K, false, true); // [batch, kvNumHead, group , seq_len, seq_len]
    QK = QK + mask;
    auto QKPast = _MatMul(Q, cache.pastK, false, true);
    QKPast = QKPast + cache.pastMask;
    QK = _Concat({QKPast, QK}, -1);
    QK = _Softmax(QK, -1);
    V = _Reshape(V, {batch, seqLength, kvNumHead, 1, headDim});
    V = _Transpose(V, {0, 2, 3, 1, 4});
    V.fix(VARP::CONSTANT);
    auto totalV = _Concat({cache.pastV, V}, 3);
    auto QKV = _MatMul(QK, totalV, false, false);
    auto info = QKV->getInfo();
    auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
    O = _Reshape(O, {batch, seqLength, -1});
    O.fix(VARP::CONSTANT);
    // Update KVCache
    for (int y = 0; y < kvNumHead; ++y) {
        ::memcpy(cache.pastK->writeMap<float>() + y * pastLength * headDim + cache.current * headDim,
                 K->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
        ::memcpy(cache.pastV->writeMap<float>() + y * pastLength * headDim + cache.current * headDim,
                 V->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
    }
    for (int i = 0; i < seqLength; ++i) {
        cache.pastMask->writeMap<float>()[i + cache.current] = 0.0f;
    }
    cache.current += seqLength;
    return O;
}

static std::vector<std::vector<std::vector<float>>> generateRandTensor(int C, int H, int W, int precision) {
    std::vector<std::vector<std::vector<float>>> a;
    a.resize(C);
    for (int i = 0; i < C; i++) {
        a[i].resize(H);
        for (int j = 0; j < H; j++) {
            a[i][j].resize(W);
            for (int k = 0; k < W; k++) {
                if (precision == 2) {
                    a[i][j][k] = ((i + j + k) % 10) * 0.002;
                } else {
                    a[i][j][k] = ((i + j + k) % 10) * 0.16 - 5.6;
                }
            }
        }
    }
    return a;
}

VARP vector_to_var(std::vector<std::vector<std::vector<float>>>& a) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    VARP var = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    float* ptr = var->writeMap<float>();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                ptr[i * H * W + j * W + k] = a[i][j][k];
            }
        }
    }
    var->unMap();
    return var;
}

VARP vector_to_c4_value(std::vector<std::vector<std::vector<float>>>& a) {
    int seqLen = a.size();
    int kvNumHead = a[0].size();
    int headDim = a[0][0].size();
    int channel = kvNumHead * headDim;
    VARP var = _Input({seqLen, channel, 1, 1}, NCHW, halide_type_of<float>());
    auto ptr = var->writeMap<float>();
    for (int s = 0; s < seqLen; ++s) {
        for (int h = 0; h < kvNumHead; ++h) {
            for (int d = 0; d < headDim; ++d) {
                int c = h * headDim + d;
                ptr[s * channel + c] = a[s][h][d];
            }
        }
    }
    var->unMap();
    return _Convert(var, NC4HW4);
}

VARP vector_to_var(std::vector<std::vector<int>>& a) {
    int H = a.size();
    int W = a[0].size();
    VARP var = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
    int* ptr = var->writeMap<int>();
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            ptr[i * W + j] = a[i][j];
        }
    }
    var->unMap();
    return var;
}

static std::vector<std::vector<std::vector<float>>>
computeAttention(std::vector<std::vector<std::vector<float>>>& query, std::vector<std::vector<std::vector<float>>>& key,
                 std::vector<std::vector<std::vector<float>>>& value, std::vector<std::vector<int>>& mask, int seq_len,
                 int kv_seq_len) {
    int group_size = NumHead / KvNumHead;
    std::vector<std::vector<std::vector<float>>> output(seq_len);
    for (int i = 0; i < seq_len; i++) {
        output[i].resize(NumHead);
        for (int j = 0; j < NumHead; j++) {
            output[i][j].resize(HeadDim);
        }
    }
    for (int h = 0; h < NumHead; h++) {
        int kv_h = h / group_size;
        /*---- Q * K ----*/
        std::vector<std::vector<float>> qk(seq_len, std::vector<float>(kv_seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = 0.0f;
                for (int k = 0; k < HeadDim; k++) {
                    qk[i][j] += query[i][h][k] * key[j][kv_h][k];
                }
            }
        }
        /*---- Mask QK ----*/
        if (mask.size() > 0) {
            float scale = 1.0 / sqrt(HeadDim);
            if (mask[0].size() == seq_len) {
                auto diff = kv_seq_len - seq_len;
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        qk[i][j + diff] = qk[i][j + diff] * scale + (1.f - mask[i][j]) * kMaskNegative;
                    }
                }
            } else {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < kv_seq_len; j++) {
                        qk[i][j] = qk[i][j] * scale + (1.f - mask[i][j]) * kMaskNegative;
                    }
                }
            }
        } else {
            float scale = 1.0 / sqrt(HeadDim);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    qk[i][j] *= scale;
                }
            }
        }
        /*---- Softmax QK ----*/
        for (int i = 0; i < seq_len; i++) {
            float maxValue = qk[i][0];
            for (int j = 1; j < kv_seq_len; j++) {
                maxValue = ALIMAX(maxValue, qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] -= maxValue;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_seq_len; j++) {
                sum += exp(qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = exp(qk[i][j]) / sum;
            }
        }
        /*---- QK * V ----*/
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HeadDim; j++) {
                output[i][h][j] = 0.0f;
                for (int k = 0; k < kv_seq_len; k++) {
                    output[i][h][j] += qk[i][k] * value[k][kv_h][j];
                }
            }
        }
    }
    return output;
}

class NaiveAttention {
private:
    std::vector<std::vector<std::vector<float>>> mPastKey, mPastValue;
    int mPastLen;

public:
    NaiveAttention() : mPastLen(0) {}
    ~NaiveAttention() = default;
    // Push prefill K/V into history WITHOUT computing attention. The wide-KV-block boundary
    // test needs kv cache filled past 2048 rows; running onExecute for that prefill would cost
    // O(kv^2) scalar work (~10 GFLOP at kv=2040). Only the decode steps need a reference.
    void appendHistory(std::vector<std::vector<std::vector<float>>>& key,
                       std::vector<std::vector<std::vector<float>>>& value, int seq_len) {
        for (int i = 0; i < seq_len; i++) {
            mPastKey.push_back(key[i]);
            mPastValue.push_back(value[i]);
        }
        mPastLen += seq_len;
    }
    int pastLen() const { return mPastLen; }
    std::vector<std::vector<std::vector<float>>> onExecute(std::vector<std::vector<std::vector<float>>>& query,
                                                           std::vector<std::vector<std::vector<float>>>& key,
                                                           std::vector<std::vector<std::vector<float>>>& value,
                                                           std::vector<std::vector<int>>& mask, int seq_len) {
        for (int i = 0; i < seq_len; i++) {
            mPastKey.push_back(key[i]);
            mPastValue.push_back(value[i]);
        }
        mPastLen += seq_len;
        return computeAttention(query, mPastKey, mPastValue, mask, seq_len, mPastLen);
    }
};

class AttentionTest : public MNNTestCase {
protected:
    std::vector<std::vector<std::vector<float>>> query;
    std::vector<std::vector<std::vector<float>>> key;
    std::vector<std::vector<std::vector<float>>> value;
    std::vector<std::vector<int>> mask;
    std::vector<std::vector<std::vector<float>>> expected_result;
    VARP Query, Key, Value, Mask, Output;
    VARP Query1, Key1, Value1, Mask1;

public:
    AttentionTest() = default;
    virtual ~AttentionTest() = default;
    void generateInput(int seq_len, int precision, bool genDecodeInput = false) {
        query = generateRandTensor(seq_len, NumHead, HeadDim, precision);
        key = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        value = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        Query = vector_to_var(query);
        Key = vector_to_var(key);
        Value = vector_to_var(value);
        if (genDecodeInput) {
            auto vecquery = generateRandTensor(1, NumHead, HeadDim, precision);
            auto veckey = generateRandTensor(1, KvNumHead, HeadDim, precision);
            auto vecvalue = generateRandTensor(1, KvNumHead, HeadDim, precision);
            Query1 = vector_to_var(vecquery);
            Key1 = vector_to_var(veckey);
            Value1 = vector_to_var(vecvalue);
        }
    }
    void generateChunkMask(int seq_len, int kv_seq_len, int chunk_size, bool genDecodeInput = false) {
        // 防止除以0
        if (chunk_size <= 0)
            chunk_size = 1;

        mask.resize(seq_len);

        // 计算历史长度 (Gap)，用于处理 KV 长度大于 Seq 长度的情况 (Right Alignment)
        // j < gap 的部分通常被视为 History，默认可见
        int gap = kv_seq_len - seq_len;

        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);

            // --- 核心逻辑对应 ---
            // MNN Expr: auto N = _Divide(i, rankVar) * rankVar + rankVar;
            // i 是当前行 (Query)，计算当前块的右边界 (不包含)
            // 比如 rank=2, i=0, block_end_rel=2; i=2, block_end_rel=4
            int block_end_rel = (i / chunk_size) * chunk_size + chunk_size;

            for (int j = 0; j < kv_seq_len; j++) {
                // 将 j 转换为相对于当前 seq_len 的坐标
                int j_rel = j - gap;

                if (j_rel < 0) {
                    // 情况 1: j 在 Gap 区域 (历史 KV Cache)
                    // 通常历史数据对当前所有 Token 都是可见的
                    mask[i][j] = 1;
                } else {
                    // 情况 2: j 在当前处理的序列范围内
                    // 对应 MNN Expr: _Less(j, N)
                    if (j_rel < block_end_rel) {
                        mask[i][j] = 1;
                    } else {
                        mask[i][j] = 0;
                    }
                }
            }
        }

        // 转为 VARP 并处理成 -inf / 0.0 格式
        Mask = vector_to_var(mask);
        Mask = (_Scalar<float>(1.0) - _Cast<float>(Mask)) * _Scalar<float>(kMaskNegative);

        // Decode Input 部分通常保持全 1 (即看清所有历史)，或者根据需求修改
        if (genDecodeInput) {
            std::vector<std::vector<int>> vecmask;
            vecmask.resize(1);
            vecmask[0].resize(gMeta.previous + 1);
            for (int i = 0; i < gMeta.previous + 1; ++i) {
                vecmask[0][i] = 1;
            }
            Mask1 = vector_to_var(vecmask);
            Mask1 = (_Scalar<float>(1.0) - _Cast<float>(Mask1)) * _Scalar<float>(kMaskNegative);
        }
    }

    void generateMask(int seq_len, int kv_seq_len, bool genDecodeInput = false) {
        mask.resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);
            for (int j = 0; j < kv_seq_len; j++) {
                if (j - i <= kv_seq_len - seq_len) {
                    mask[i][j] = 1;
                } else {
                    mask[i][j] = 0;
                }
            }
        }
        Mask = _Input({}, NCHW, halide_type_of<float>());
        Mask1 = _Input({}, NCHW, halide_type_of<float>());
        Mask->writeMap<float>()[0] = 0.0f;
        Mask1->writeMap<float>()[0] = 0.0f;
    }

    bool compareResult(int seq_len) {
        const float* resultPtr = Output->readMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < NumHead; j++) {
                for (int k = 0; k < HeadDim; k++) {
                    float got = resultPtr[i * NumHead * HeadDim + j * HeadDim + k];
                    float diff = fabs(got - expected_result[i][j][k]);
                    float diff_percent = fabs(diff / expected_result[i][j][k]);
                    // `diff > threshold` is false for NaN, so NaN output would pass silently.
                    if (got != got) {
                        printf("Result NaN: expected %lf but got nan in Attention Test\n", expected_result[i][j][k]);
                        printf("Error Position: Output[%d][%d][%d]\n", i, j, k);
                        return false;
                    }
                    if (diff > diff_threshold && diff_percent > diff_percent_threshold) {
                        printf("Result Mismatch: expected %lf but got %lf in CPU Attention Test\n",
                               expected_result[i][j][k], resultPtr[i * NumHead * HeadDim + j * HeadDim + k]);
                        printf("Error Position: Output[%d][%d][%d]\n", i, j, k);
                        return false;
                    }
                }
            }
        }
        Output->unMap();
        return true;
    }

    virtual bool run(int precision) {
        OpenCLBufferScope clBufferScope;
        srand(2024);
        // unit test 1
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateMask(seq_len, seq_len);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }

            /* generate mask expr */
            /* generate mask expr */
            auto MaskExpr = vector_to_var(mask);
            MaskExpr = (_Scalar<float>(1.0) - _Cast<float>(MaskExpr)) * _Scalar<float>(kMaskNegative);
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= _opExprDiffThreshold(precision)) {
                FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // test2
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateChunkMask(seq_len, seq_len, 2);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.previous = 0;
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Not LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= _opExprDiffThreshold(precision)) {
                FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // unit test 3
        // Skipping must not return from run(): the long-prefill cases below have to
        // keep running on every backend.
        bool skipNoKvCache = false;
        {
            auto rtInfo = ExecutorScope::Current()->getRuntime().first;
            bool cpuInfer = true;
            for (auto& rt : rtInfo) {
                if (rt.first != MNN_FORWARD_CPU) {
                    cpuInfer = false;
                    break;
                }
            }
            // TODO: CPU support kv_cache == false
            skipNoKvCache = cpuInfer;
            // MNN: kv_cache=false also falls back to CPU on OpenCL with
            // MNN_GPU_MEMORY_IMAGE (no IMAGE-memtype Attention creator) and
            // on Vulkan, so it hits the same CPUAttention "kv_cache == false"
            // TODO and crashes. Skip until the CPU fallback is completed.
            for (auto& rt : rtInfo) {
                if (rt.first == MNN_FORWARD_OPENCL || rt.first == MNN_FORWARD_VULKAN) {
                    skipNoKvCache = true;
                }
            }
        }
        if (!skipNoKvCache) {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = false;
            int seq_len = 128;
            generateInput(seq_len, precision);
            mask.clear();
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            Output = Variable::create(Expr::create(attention.get(), {Query, Key, Value}));
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Attention without kv_cacheunit test failed!\n");
                return false;
            }
        }
        // Long causal prefill: exercises the tiled prefill kernels (three-stage
        // and fused flash-attn variants) past their 32/16-wide tile boundaries.
        // 100 is a multiple of neither, so it covers both q and kv tail blocks.
        {
            for (int seq_len : {64, 100, 192, 512}) {
                std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                generateInput(seq_len, precision);
                generateMask(seq_len, seq_len);
                expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                auto attn = _makeAttentionModule();
                gMeta.previous = 0;
                gMeta.add = seq_len;
                Output = attn->onForward({Query, Key, Value, Mask})[0];
                gMeta.sync();
                if (!compareResult(seq_len)) {
                    printf("Error: long causal prefill (seq_len=%d) unit test failed!\n", seq_len);
                    return false;
                }
            }
        }
        // Materialized mask: the operator gets a real mask plane instead of the causal sentinel.
        // generateMask only ever hands the operator the scalar sentinel, so the backends' additive
        // mask paths had no coverage at all -- on OpenCL that is the whole -DADD_MASK variant of
        // the prefill kernels. seq_len 16 sits on the boundary where a backend may start tuning
        // between branches, 64 is past it.
        {
            for (int seq_len : {16, 64}) {
                std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                generateInput(seq_len, precision);
                generateMask(seq_len, seq_len);
                expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                // 1 where attended, 0 where masked -> 0.0f / kMaskNegative, i.e. an additive mask.
                auto maskPlane = vector_to_var(mask);
                maskPlane = (_Scalar<float>(1.0) - _Cast<float>(maskPlane)) * _Scalar<float>(kMaskNegative);
                auto attn = _makeAttentionModule();
                gMeta.previous = 0;
                gMeta.add = seq_len;
                Output = attn->onForward({Query, Key, Value, maskPlane})[0];
                gMeta.sync();
                if (!compareResult(seq_len)) {
                    printf("Error: materialized mask (seq_len=%d) unit test failed!\n", seq_len);
                    return false;
                }
            }
        }
        // Non-causal materialized mask (chunked attention / sliding window). The case above is
        // lower-triangular, so a backend that quietly assumes causality still passes it. test2
        // does hand over a non-causal plane, but at seq_len=10 -- under the threshold where
        // OpenCL starts measuring prefill paths against each other, so only one of them ever
        // ran. Past the threshold both do.
        {
            for (int seq_len : {64}) {
                std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                generateInput(seq_len, precision);
                generateChunkMask(seq_len, seq_len, 16);
                expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                auto attn = _makeAttentionModule();
                gMeta.previous = 0;
                gMeta.add = seq_len;
                Output = attn->onForward({Query, Key, Value, Mask})[0];
                gMeta.sync();
                if (!compareResult(seq_len)) {
                    printf("Error: non-causal materialized mask (seq_len=%d) unit test failed!\n", seq_len);
                    return false;
                }
            }
        }
        // Chunked prefill: a second prefill on top of a non-empty kv cache, handed a mask plane
        // that spans the whole kv axis. This is what an LLM does when it splits a long prompt,
        // and nothing here covered it -- test2 also runs a second prefill, but its plane covers
        // only the new tokens, which is the right-aligned shape instead.
        {
            const int chunk = 64;
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            auto attn = _makeAttentionModule();
            gMeta.previous = 0;
            for (int step = 0; step < 2; ++step) {
                generateInput(chunk, precision);
                generateChunkMask(chunk, chunk * (step + 1), 16);
                expected_result = naiveAttention->onExecute(query, key, value, mask, chunk);
                gMeta.add = chunk;
                Output = attn->onForward({Query, Key, Value, Mask})[0];
                gMeta.sync();
                if (!compareResult(chunk)) {
                    printf("Error: chunked prefill (step=%d) unit test failed!\n", step);
                    return false;
                }
            }
        }
        // Integer mask plane: 1 where attended, 0 where masked, and the backend applies the
        // -inf itself rather than being handed a bias. A distinct kernel variant from the
        // additive plane above (-DSET_MASK vs -DADD_MASK on OpenCL), and until now an untested
        // one. Only OpenCL and Metal implement it -- CPU and Vulkan read the mask as float and
        // would reinterpret the integers -- so the case is scoped to those two.
        {
            auto forwardType = (MNNForwardType)MNNTestSuite::get()->pStaus.forwardType;
            if (MNN_FORWARD_OPENCL == forwardType || MNN_FORWARD_METAL == forwardType) {
                for (int seq_len : {16, 64}) {
                    std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                    generateInput(seq_len, precision);
                    generateMask(seq_len, seq_len);
                    expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                    auto maskPlane = vector_to_var(mask);
                    auto attn = _makeAttentionModule();
                    gMeta.previous = 0;
                    gMeta.add = seq_len;
                    Output = attn->onForward({Query, Key, Value, maskPlane})[0];
                    gMeta.sync();
                    if (!compareResult(seq_len)) {
                        printf("Error: integer mask (seq_len=%d) unit test failed!\n", seq_len);
                        return false;
                    }
                }
            }
        }
        // Mask plane with fewer rows than the query. The trailing query rows it does not reach are
        // left unmasked, mirroring what the kv axis already does for the history columns it does
        // not reach -- the convention Vulkan's attention_fused.comp applies with `q < maskQlen`.
        // OpenCL used to refuse the shape outright; short prefill claimed to take it but rearranged
        // the plane with a maskQlen stride and read it back with a seqlen one.
        {
            auto forwardType = (MNNForwardType)MNNTestSuite::get()->pStaus.forwardType;
            if (MNN_FORWARD_OPENCL == forwardType) {
                for (int seq_len : {16, 64}) {
                    const int maskQlen = seq_len / 2;
                    std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                    generateInput(seq_len, precision);
                    generateMask(seq_len, seq_len);
                    std::vector<std::vector<int>> shortMask(mask.begin(), mask.begin() + maskQlen);
                    // The reference sees the rows the plane omits as fully visible.
                    for (int i = maskQlen; i < seq_len; ++i) {
                        std::fill(mask[i].begin(), mask[i].end(), 1);
                    }
                    expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                    auto maskPlane = vector_to_var(shortMask);
                    maskPlane = (_Scalar<float>(1.0) - _Cast<float>(maskPlane)) * _Scalar<float>(kMaskNegative);
                    auto attn = _makeAttentionModule();
                    gMeta.previous = 0;
                    gMeta.add = seq_len;
                    Output = attn->onForward({Query, Key, Value, maskPlane})[0];
                    gMeta.sync();
                    if (!compareResult(seq_len)) {
                        printf("Error: short mask plane (seq_len=%d maskQlen=%d) unit test failed!\n", seq_len,
                               maskQlen);
                        return false;
                    }
                }
            }
        }
        return true;
    }
};

class SpeedAttentionTest : public AttentionTest {
protected:
    std::vector<std::vector<std::vector<float>>> query;
    std::vector<std::vector<std::vector<float>>> key;
    std::vector<std::vector<std::vector<float>>> value;
    std::vector<std::vector<int>> mask;
    std::vector<std::vector<std::vector<float>>> expected_result;

public:
    SpeedAttentionTest() = default;
    virtual ~SpeedAttentionTest() = default;

    virtual bool run(int precision) {
        OpenCLBufferScope clBufferScope;
        std::vector<int> seqs = {4096};
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = true;
        /* 3 attention module */
        std::vector<int> quantQKV = {8, 9, 10};
        std::vector<std::string> testNames = {"float qkv", "quant qk", "quant qkv"};
        for (int n = 0; n < seqs.size(); ++n) {
            int seq_len = seqs[n];
            MNN_PRINT(">>> seq_len=%d, decode_len=%d\n", seq_len, GENERATE_TOKENS);
            generateInput(seqs[n], precision, true);
            generateMask(seqs[n], seq_len, true);
            for (int m = 0; m < testNames.size(); ++m) {
                gMeta.previous = 0;
                gMeta.add = seq_len;
                auto _module = _makeAttentionModule(quantQKV[m]);
                MNN::Timer t1;
                for (int x = 0; x < 5; ++x) {
                    Output = _module->onForward({Query, Key, Value, Mask})[0];
                }
                auto time = (float)t1.durationInUs() / 1000.0f / 5.f;
                MNN_PRINT("%s: prefill cost = %.2f\n", testNames[m].c_str(), time);
                gMeta.sync();
                MNN::Timer t2;
                for (int x = 0; x < GENERATE_TOKENS; ++x) {
                    gMeta.add = 1;
                    auto output2 = _module->onForward({Query1, Key1, Value1, Mask1})[0];
                    gMeta.sync();
                }
                time = (float)t2.durationInUs() / 1000.0f;
                MNN_PRINT("%s: decode cost = %f\n", testNames[m].c_str(), time);
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(AttentionTest, "op/attention");

namespace {
const char* kAttentionPrefixCacheDir = "prefixcache";

struct AttentionPrefixCacheFiles {
    explicit AttentionPrefixCacheFiles(const std::string& cacheName) : name(cacheName) { clear(); }
    ~AttentionPrefixCacheFiles() { clear(); }
    std::string path(const char* suffix) const {
        return MNNFilePathConcat(kAttentionPrefixCacheDir, name) + "_0" + suffix;
    }
    void clear() const {
        for (auto suffix : {".k", ".v", "_sync.k", "_sync.v"}) {
            ::remove(path(suffix).c_str());
        }
    }
    size_t size(const char* suffix) const {
        auto fd = MNNOpenFile(path(suffix).c_str(), MNN_FILE_READ);
        if (fd == INVALID_FILE) {
            return INVALID_SIZE;
        }
        auto result = MNNGetFileSize(fd);
        MNNCloseFile(fd);
        return result;
    }
    bool resize(const char* suffix, size_t bytes) const {
        auto fd = MNNCreateFile(path(suffix).c_str());
        if (fd == INVALID_FILE) {
            return false;
        }
        bool result = MNNSetFileSize(fd, bytes) == NO_ERROR;
        MNNCloseFile(fd);
        return result;
    }
    std::string name;
};

static bool readAttentionOutput(const std::shared_ptr<Module>& module, const VARPS& inputs,
                                std::vector<float>& output) {
    auto outputs = module->onForward(inputs);
    if (outputs.empty() || outputs[0] == nullptr || outputs[0]->getInfo() == nullptr) {
        return false;
    }
    auto ptr = outputs[0]->readMap<float>();
    if (ptr == nullptr) {
        return false;
    }
    output.assign(ptr, ptr + outputs[0]->getInfo()->size);
    return true;
}

static bool compareAttentionOutputs(const std::vector<float>& expected, const std::vector<float>& actual,
                                    float tolerance, const char* testName) {
    if (expected.size() != actual.size()) {
        MNN_ERROR("%s: output size mismatch, expected %zu, got %zu\n", testName, expected.size(), actual.size());
        return false;
    }
    for (int i = 0; i < expected.size(); ++i) {
        if (fabsf(expected[i] - actual[i]) > tolerance) {
            MNN_ERROR("%s: output mismatch at %d, expected %.6f, got %.6f\n", testName, i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

static size_t prefixCacheBytesPerElement() {
    return MNNTestSuite::get()->pStaus.precision == MNN::BackendConfig::Precision_High ? sizeof(float)
                                                                                       : sizeof(uint16_t);
}
} // namespace

class AttentionPrefixCacheChunkedWriteTest : public AttentionTest {
public:
    virtual bool run(int precision) {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_OPENCL) {
            return true;
        }
        OpenCLBufferScope clBufferScope;
        const int chunk = 8;
        AttentionPrefixCacheFiles files("attention_chunked_write");
        KVMeta meta;
        meta.file_name = files.name;
        meta.file_flag = KVMeta::PendingWrite;
        meta.layer_nums = 1;
        meta.prefix_session_id = 1;
        auto module = _makeAttentionModule(8, false, true, 1, &meta, kAttentionPrefixCacheDir);
        if (!module) {
            return false;
        }
        size_t firstKeyBytes = 0;
        size_t firstValueBytes = 0;
        for (int step = 0; step < 2; ++step) {
            srand(2024 + step);
            generateInput(chunk, precision);
            generateMask(chunk, chunk);
            meta.add = chunk;
            std::vector<float> output;
            if (!readAttentionOutput(module, {Query, Key, Value, Mask}, output)) {
                return false;
            }
            meta.sync();
            if (step == 0) {
                firstKeyBytes = files.size(".k");
                firstValueBytes = files.size(".v");
                if (firstKeyBytes == 0 || firstValueBytes == 0) {
                    return false;
                }
            }
        }
        size_t keyBytes = files.size(".k");
        size_t valueBytes = files.size(".v");
        if (keyBytes != firstKeyBytes * 2 || valueBytes != firstValueBytes * 2) {
            MNN_ERROR("Chunked prefix cache size mismatch: first key=%zu, value=%zu; final key=%zu, value=%zu\n",
                      firstKeyBytes, firstValueBytes, keyBytes, valueBytes);
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(AttentionPrefixCacheChunkedWriteTest, "op/attention_prefix_cache_chunked_write");

class AttentionPrefixCachePartialLoadTest : public AttentionTest {
public:
    virtual bool run(int precision) {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_OPENCL) {
            return true;
        }
        OpenCLBufferScope clBufferScope;
        const int seqLen = 8;
        AttentionPrefixCacheFiles files("attention_partial_load");
        if (!MNNCreateDir(kAttentionPrefixCacheDir)) {
            return false;
        }
        size_t fullBytes = (size_t)KvNumHead * HeadDim * seqLen * prefixCacheBytesPerElement();
        if (!files.resize(".k", fullBytes / 2) || !files.resize(".v", fullBytes / 2)) {
            return false;
        }

        srand(2024);
        generateInput(seqLen, precision);
        generateMask(seqLen, seqLen);
        VARPS inputs{Query, Key, Value, Mask};
        std::vector<float> expected;
        {
            KVMeta baselineMeta;
            baselineMeta.add = seqLen;
            auto baseline = _makeAttentionModule(8, false, true, 1, &baselineMeta);
            if (!baseline || !readAttentionOutput(baseline, inputs, expected)) {
                return false;
            }
        }

        KVMeta meta;
        meta.add = seqLen;
        meta.file_name = files.name;
        meta.file_flag = KVMeta::PendingRead;
        meta.seqlen_in_disk = seqLen;
        meta.layer_nums = 1;
        meta.prefix_session_id = 1;
        auto module = _makeAttentionModule(8, false, true, 1, &meta, kAttentionPrefixCacheDir);
        std::vector<float> actual;
        if (!module || !readAttentionOutput(module, inputs, actual)) {
            return false;
        }
        return compareAttentionOutputs(expected, actual, _opExprDiffThreshold(precision), "partial prefix load");
    }
};
MNNTestSuiteRegister(AttentionPrefixCachePartialLoadTest, "op/attention_prefix_cache_partial_load");

class AttentionPrefixCacheContinuousSessionTest : public AttentionTest {
public:
    virtual bool run(int precision) {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_OPENCL) {
            return true;
        }
        OpenCLBufferScope clBufferScope;
        const int seqLen = 8;
        AttentionPrefixCacheFiles files("attention_continuous_session");

        srand(2024);
        generateInput(seqLen, precision);
        generateMask(seqLen, seqLen);
        VARPS prefixInputs{Query, Key, Value, Mask};
        srand(2025);
        generateInput(seqLen, precision);
        generateMask(seqLen, seqLen);
        VARPS suffixInputs{Query, Key, Value, Mask};

        std::vector<float> expected;
        {
            KVMeta baselineMeta;
            auto baseline = _makeAttentionModule(8, false, true, 1, &baselineMeta);
            std::vector<float> ignored;
            baselineMeta.add = seqLen;
            if (!baseline || !readAttentionOutput(baseline, prefixInputs, ignored)) {
                return false;
            }
            baselineMeta.sync();
            baselineMeta.add = seqLen;
            if (!readAttentionOutput(baseline, suffixInputs, expected)) {
                return false;
            }
        }

        KVMeta meta;
        meta.add = seqLen;
        meta.file_name = files.name;
        meta.file_flag = KVMeta::PendingWrite;
        meta.layer_nums = 1;
        meta.prefix_session_id = 1;
        auto module = _makeAttentionModule(8, false, true, 1, &meta, kAttentionPrefixCacheDir);
        std::vector<float> ignored;
        if (!module || !readAttentionOutput(module, prefixInputs, ignored)) {
            return false;
        }
        meta.sync();

        meta.previous = 0;
        meta.add = seqLen;
        meta.file_name = files.name;
        meta.file_flag = KVMeta::PendingRead;
        meta.seqlen_in_disk = seqLen;
        meta.layer_index = 0;
        ++meta.prefix_session_id;
        std::vector<float> actual;
        if (!readAttentionOutput(module, suffixInputs, actual)) {
            return false;
        }
        return compareAttentionOutputs(expected, actual, _opExprDiffThreshold(precision), "continuous prefix session");
    }
};
MNNTestSuiteRegister(AttentionPrefixCacheContinuousSessionTest, "op/attention_prefix_cache_continuous_session");

// Non-causal attention with kv_cache=false driven by an explicit tensor mask --
// the shape a ViT / vision-encoder export emits. AttentionTest's unit test 3
// pairs kv_cache=false with *no* mask input, so this combination was previously
// uncovered. Covers both an all-visible (all-zero ADD) mask and a row-varying
// (causal ADD) mask, at mask rank 3 and 4.
class AttentionNoCacheMaskTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        const float tol = (precision == 2) ? 0.05f : 0.01f;
        bool pass = true;
        for (int seqLen : {64, 100, 128, 660}) {
            float vis3 = maxRelError(seqLen, 12, 12, 64, 3, false);
            float vis4 = maxRelError(seqLen, 12, 12, 64, 4, false);
            float row3 = maxRelError(seqLen, 12, 12, 64, 3, true);
            float row4 = maxRelError(seqLen, 12, 12, 64, 4, true);
            MNN_PRINT(
                "[attention_nocache_mask] seq=%4d allvisible(3d/4d)=%.6f/%.6f rowvarying(3d/4d)=%.6f/%.6f "
                "(tol %.3f)\n",
                seqLen, vis3, vis4, row3, row4, tol);
            if (!(vis3 < tol) || !(vis4 < tol) || !(row3 < tol) || !(row4 < tol)) {
                pass = false;
            }
        }
        return pass;
    }

private:
    // maskRank: 3 = [1,seq,seq], 4 = [1,1,seq,seq]. The no-mask form is covered by
    // AttentionTest unit test 3; CPUAttention does not support it yet.
    // rowVarying: false = all-zero (fully visible) ADD mask; true = causal ADD mask,
    // which only matches if the kernel reads the mask row belonging to each query.
    static float maxRelError(int seqLen, int numHead, int kvNumHead, int headDim, int maskRank, bool rowVarying) {
        const int group = numHead / kvNumHead;
        const float scale = 1.0f / sqrtf((float)headDim);
        const float kMaskNegative = -1e9f;

        uint32_t state = 12345;
        auto next = [&state]() {
            state = state * 1103515245u + 12345u;
            return (float)((state >> 16) % 2000) / 1000.0f - 1.0f;
        };

        auto Q = _Input({1, seqLen, numHead, headDim}, NCHW, halide_type_of<float>());
        auto K = _Input({1, seqLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
        auto V = _Input({1, seqLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());

        std::vector<float> q(seqLen * numHead * headDim), k(seqLen * kvNumHead * headDim),
            v(seqLen * kvNumHead * headDim);
        for (auto& x : q)
            x = next();
        for (auto& x : k)
            x = next();
        for (auto& x : v)
            x = next();
        ::memcpy(Q->writeMap<float>(), q.data(), q.size() * sizeof(float));
        ::memcpy(K->writeMap<float>(), k.data(), k.size() * sizeof(float));
        ::memcpy(V->writeMap<float>(), v.data(), v.size() * sizeof(float));

        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = false;

        VARP Output;
        {
            auto Mask = (3 == maskRank) ? _Input({1, seqLen, seqLen}, NCHW, halide_type_of<float>())
                                        : _Input({1, 1, seqLen, seqLen}, NCHW, halide_type_of<float>());
            auto maskPtr = Mask->writeMap<float>();
            for (int i = 0; i < seqLen; ++i) {
                for (int j = 0; j < seqLen; ++j) {
                    maskPtr[i * seqLen + j] = (rowVarying && j > i) ? kMaskNegative : 0.0f;
                }
            }
            Output = Variable::create(Expr::create(attention.get(), {Q, K, V, Mask}));
        }
        auto got = Output->readMap<float>();
        if (nullptr == got) {
            MNN_ERROR("attention_nocache_mask: failed to map output\n");
            return std::numeric_limits<float>::max();
        }

        std::vector<float> scores(seqLen);
        float maxRel = 0.0f;
        for (int h = 0; h < numHead; ++h) {
            const int kvh = h / group;
            for (int i = 0; i < seqLen; ++i) {
                const int kEnd = rowVarying ? (i + 1) : seqLen;
                float maxScore = -std::numeric_limits<float>::max();
                for (int j = 0; j < kEnd; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; ++d) {
                        dot += q[(i * numHead + h) * headDim + d] * k[(j * kvNumHead + kvh) * headDim + d];
                    }
                    scores[j] = dot * scale;
                    maxScore = std::max(maxScore, scores[j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < kEnd; ++j) {
                    scores[j] = expf(scores[j] - maxScore);
                    sum += scores[j];
                }
                for (int d = 0; d < headDim; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < kEnd; ++j) {
                        acc += scores[j] * v[(j * kvNumHead + kvh) * headDim + d];
                    }
                    acc /= sum;
                    float out = got[(i * numHead + h) * headDim + d];
                    float denom = std::max(fabsf(acc), 0.05f);
                    maxRel = std::max(maxRel, fabsf(out - acc) / denom);
                }
            }
        }
        return maxRel;
    }
};

MNNTestSuiteRegister(AttentionNoCacheMaskTest, "op/attention_nocache_mask");

// Decode-phase attention scaling at the Qwen3-0.6B shape: 16 Q heads / 8 KV heads / head_dim 128,
// one query token per step over kv lengths 512/1024/2048, 1 vs 4 threads.
class AttentionDecodeThreadScaleTest : public AttentionTest {
public:
    AttentionDecodeThreadScaleTest() = default;
    virtual ~AttentionDecodeThreadScaleTest() = default;

    virtual bool run(int precision) {
        const int savedNumHead = NumHead, savedKvNumHead = KvNumHead, savedHeadDim = HeadDim;
        NumHead = 16;
        KvNumHead = 8;
        HeadDim = 128;
        srand(2024);
        const int warmup = 8;
        const int threadCfgs[2] = {1, 4};
        for (int kvLen : {512, 1024, 2048}) {
            generateInput(kvLen, precision, true);
            generateMask(kvLen, kvLen, true);
            float ms[2] = {0.f, 0.f};
            for (int ti = 0; ti < 2; ++ti) {
                gMeta.previous = 0;
                gMeta.add = kvLen;
                auto module = _makeAttentionModule(8, false, false, threadCfgs[ti]);
                module->onForward({Query, Key, Value, Mask});
                gMeta.sync();
                for (int x = 0; x < warmup; ++x) {
                    gMeta.add = 1;
                    module->onForward({Query1, Key1, Value1, Mask1});
                    gMeta.sync();
                }
                MNN::Timer timer;
                for (int x = 0; x < GENERATE_TOKENS; ++x) {
                    gMeta.add = 1;
                    module->onForward({Query1, Key1, Value1, Mask1});
                    gMeta.sync();
                }
                ms[ti] = (float)timer.durationInUs() / 1000.0f / GENERATE_TOKENS;
            }
            MNN_PRINT("kvLen=%d decode: t1=%.3f ms/token, t4=%.3f ms/token, speedup=%.2fx\n", kvLen, ms[0], ms[1],
                      ms[1] > 0.f ? ms[0] / ms[1] : 0.f);
        }
        NumHead = savedNumHead;
        KvNumHead = savedKvNumHead;
        HeadDim = savedHeadDim;
        return true;
    }
};
MNNTestSuiteRegister(AttentionDecodeThreadScaleTest, "speed/attention_threads");

// ---- Wide KV-block / chunked V-cache boundary coverage ----------------------------------------
//
// Decode-phase flash attention derives its logical KV block width and the physical V-cache chunk
// size from the thread count and quant mode (CPUAttention.cpp:468-481 and
// CPUKVCacheManager.hpp:109-114). A logical block that indexes a differently-chunked physical
// layout only reads wrong rows *past the first chunk*, so nothing is observable until kv grows
// past 64 / 256 / 2048. op/attention never decodes past kv=101, which is why the original
// occurrence of exactly this bug had to be caught by an llm_demo long-prompt canary.
//
// Input sensitivity is the other half of the problem. The shared generateRandTensor pattern
// (((i+j+k)%10)*0.002 in the fp16 tier) makes every logit nearly equal, so softmax over ~2050
// rows degenerates into a mean and one mis-addressed row moves the output by only ~1/2050 --
// far below diff_percent_threshold. Instead, K rows here are +-1 sign vectors from a
// deterministic hash and each decode query is an exact copy of one K row: QK peaks at
// HeadDim/sqrt(HeadDim) ~ 11.3 against ~+-1 elsewhere, so the output is ~93% of that single V
// row and reading the wrong row is an O(1) error.
static inline uint32_t _kvbHash(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 2654435761u + b * 2246822519u + c * 3266489917u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

class AttentionKvBlockBoundaryTest : public AttentionTest {
private:
    struct ShapeGuard {
        int n, kv, d;
        ShapeGuard() : n(NumHead), kv(KvNumHead), d(HeadDim) {}
        ~ShapeGuard() {
            NumHead = n;
            KvNumHead = kv;
            HeadDim = d;
        }
    };
    typedef std::vector<std::vector<std::vector<float>>> Tensor3;

    static Tensor3 genKeyRows(int len) {
        Tensor3 k(len);
        for (int j = 0; j < len; ++j) {
            k[j].resize(KvNumHead);
            for (int h = 0; h < KvNumHead; ++h) {
                k[j][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    k[j][h][d] = (_kvbHash(j, h, d) & 1u) ? 1.0f : -1.0f;
                }
            }
        }
        return k;
    }
    static Tensor3 genValueRows(int len) {
        Tensor3 v(len);
        for (int j = 0; j < len; ++j) {
            v[j].resize(KvNumHead);
            for (int h = 0; h < KvNumHead; ++h) {
                v[j][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    v[j][h][d] = (float)(_kvbHash(j + 7919u, h + 31u, d) % 2001u) / 1000.0f - 1.0f;
                }
            }
        }
        return v;
    }
    // Small-amplitude queries for the prefill segment; its output is never checked, only the
    // resulting KV cache contents matter.
    static Tensor3 genPrefillQuery(int len) {
        Tensor3 q(len);
        for (int i = 0; i < len; ++i) {
            q[i].resize(NumHead);
            for (int h = 0; h < NumHead; ++h) {
                q[i][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    q[i][h][d] = (float)(_kvbHash(i + 104729u, h, d) % 101u) * 0.002f - 0.1f;
                }
            }
        }
        return q;
    }
    // Decode query that peaks on kv row `target`.
    static Tensor3 genProbeQuery(const Tensor3& key, int target) {
        const int group = NumHead / KvNumHead;
        Tensor3 q(1);
        q[0].resize(NumHead);
        for (int h = 0; h < NumHead; ++h) {
            q[0][h] = key[target][h / group];
        }
        return q;
    }
    static Tensor3 sliceRow(const Tensor3& src, int row) {
        Tensor3 out(1);
        out[0] = src[row];
        return out;
    }
    static Tensor3 sliceHead(const Tensor3& src, int len) { return Tensor3(src.begin(), src.begin() + len); }
    static VARP scalarMask() {
        auto m = _Input({}, NCHW, halide_type_of<float>());
        m->writeMap<float>()[0] = 0.0f;
        return m;
    }
    // Probe positions chosen to land on and around every chunk / block boundary.
    static int probeTarget(int step, int kvLen) {
        static const int kProbes[] = {63, 64, 65, 255, 256, 257, 319, 320, 2047, 2048, 2049, 0};
        const int n = (int)(sizeof(kProbes) / sizeof(kProbes[0]));
        int t = kProbes[step % n];
        if (t >= kvLen) {
            t = kvLen - 1;
        }
        return t;
    }

    // float-KV configs: compare every decode step against the scalar fp32 reference.
    bool runAgainstReference(int hint, int numThread, int prefill, int steps, const char* tag) {
        const int total = prefill + steps;
        auto key = genKeyRows(total);
        auto value = genValueRows(total);
        auto pq = genPrefillQuery(prefill);
        auto prefillKey = sliceHead(key, prefill);
        auto prefillValue = sliceHead(value, prefill);

        std::shared_ptr<NaiveAttention> ref(new NaiveAttention);
        ref->appendHistory(prefillKey, prefillValue, prefill);

        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = prefill;
        auto module = _makeAttentionModule(hint, false, false, numThread);
        {
            auto Qp = vector_to_var(pq);
            auto Kp = vector_to_var(prefillKey);
            auto Vp = vector_to_var(prefillValue);
            module->onForward({Qp, Kp, Vp, scalarMask()});
        }
        gMeta.sync();

        std::vector<std::vector<int>> noMask;
        for (int s = 0; s < steps; ++s) {
            const int kvLen = prefill + s + 1;
            auto q1 = genProbeQuery(key, probeTarget(s, kvLen));
            auto k1 = sliceRow(key, prefill + s);
            auto v1 = sliceRow(value, prefill + s);
            expected_result = ref->onExecute(q1, k1, v1, noMask, 1);
            gMeta.add = 1;
            Output = module->onForward({vector_to_var(q1), vector_to_var(k1), vector_to_var(v1), scalarMask()})[0];
            gMeta.sync();
            if (!compareResult(1)) {
                MNN_PRINT("Error: %s failed at decode step %d (kvLen=%d, probe=%d)\n", tag, s, kvLen,
                          probeTarget(s, kvLen));
                return false;
            }
        }
        return true;
    }

    // quant-KV configs: the scalar fp32 reference cannot model int8 KV error tightly, so compare
    // flash ON (wide block + chunked V) against flash OFF (single block) of the SAME quant mode.
    // Sound for quantMode 0/1 only -- quantMode 2 downgrades V to float when flash is off
    // (CPUAttention.cpp:188-190), which is a genuinely different numeric path.
    bool runFlashOnOffDiff(int quantMode, int numThread, int prefill, int steps, const char* tag) {
        const int total = prefill + steps;
        auto key = genKeyRows(total);
        auto value = genValueRows(total);
        auto pq = genPrefillQuery(prefill);
        auto prefillKey = sliceHead(key, prefill);
        auto prefillValue = sliceHead(value, prefill);
        const int outSize = NumHead * HeadDim;
        std::vector<std::vector<float>> captured(steps);

        for (int pass = 0; pass < 2; ++pass) {
            const int hint = (pass == 0 ? 0 : 8) + quantMode; // pass0 = flash off, pass1 = flash on
            gMeta.previous = 0;
            gMeta.remove = 0;
            gMeta.add = prefill;
            auto module = _makeAttentionModule(hint, false, false, numThread);
            {
                auto Qp = vector_to_var(pq);
                auto Kp = vector_to_var(prefillKey);
                auto Vp = vector_to_var(prefillValue);
                module->onForward({Qp, Kp, Vp, scalarMask()});
            }
            gMeta.sync();
            for (int s = 0; s < steps; ++s) {
                const int kvLen = prefill + s + 1;
                auto q1 = genProbeQuery(key, probeTarget(s, kvLen));
                auto k1 = sliceRow(key, prefill + s);
                auto v1 = sliceRow(value, prefill + s);
                gMeta.add = 1;
                auto out =
                    module->onForward({vector_to_var(q1), vector_to_var(k1), vector_to_var(v1), scalarMask()})[0];
                gMeta.sync();
                const float* ptr = out->readMap<float>();
                if (pass == 0) {
                    captured[s].assign(ptr, ptr + outSize);
                } else {
                    for (int i = 0; i < outSize; ++i) {
                        float diff = fabsf(ptr[i] - captured[s][i]);
                        float rel = fabsf(diff / (captured[s][i] == 0.f ? 1e-20f : captured[s][i]));
                        if (diff > diff_threshold && rel > diff_percent_threshold) {
                            MNN_PRINT(
                                "Error: %s flash-on/off mismatch at step %d (kvLen=%d), "
                                "elem %d: off=%f on=%f\n",
                                tag, s, kvLen, i, captured[s][i], ptr[i]);
                            return false;
                        }
                    }
                }
                out->unMap();
            }
        }
        return true;
    }

public:
    AttentionKvBlockBoundaryTest() = default;
    virtual ~AttentionKvBlockBoundaryTest() = default;

    virtual bool run(int precision) {
        // The block/chunk tiering under test is CPU-only.
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_CPU) {
            return true;
        }
        ShapeGuard guard;
        // Qwen3-0.6B decode shape: GQA group = 2, 8 kv heads -> numUnits = 8.
        NumHead = 16;
        KvNumHead = 8;
        HeadDim = 128;

        // Single thread: physical V chunk 2048, logical block ALIMIN(2048, kvLen).
        if (!runAgainstReference(8, 1, 250, 10, "t1 short kv"))
            return false;
        if (!runAgainstReference(8, 1, 2040, 12, "t1 kv crossing 2048"))
            return false;
        // Prefill past the physical chunk boundary: the chunk gate has no insertLen term while the
        // logical-block gate does, so this prefills with 64-row blocks into 2048-row chunks and the
        // following decode must still read both chunks correctly.
        if (!runAgainstReference(8, 1, 2100, 10, "t1 prefill crossing chunk"))
            return false;

        // Multi thread: physical V chunk 64, logical block ALIMIN(256, kvLen) + sub-chunk addTile.
        if (!runAgainstReference(8, 4, 60, 10, "t4 kv crossing 64")) return false;
        if (!runAgainstReference(8, 4, 250, 12, "t4 kv crossing 256")) return false;
        if (!runAgainstReference(8, 4, 512, 128, "t4 pg512 decode128")) return false;
        if (!runAgainstReference(8, 4, 2040, 12, "t4 wide kv")) return false;
        if (!runAgainstReference(8, 8, 2040, 12, "t8 wide kv")) return false;
        HeadDim = 48;
        if (!runAgainstReference(8, 4, 60, 10, "t4 half-width tile fallback")) return false;
        HeadDim = 96;
        if (!runAgainstReference(8, 4, 60, 10, "t4 three wide tiles")) return false;
        HeadDim = 70;
        if (!runAgainstReference(8, 4, 250, 12, "t4 head dimension tail")) return false;
        HeadDim = 128;

        // K-int8 KV cache: wide block is gated separately, use the flash on/off differential.
        if (!runFlashOnOffDiff(1, 1, 2040, 10, "quantK t1 kv crossing 2048"))
            return false;
        if (!runFlashOnOffDiff(1, 4, 250, 10, "quantK t4 kv crossing 256"))
            return false;

        NumHead = 8; HeadDim = 70;
        if (!runAgainstReference(8, 4, 250, 12, "t4 single query head tail")) return false;

        // kvSplit > 1 needs few kv heads: numUnits = 2 gives kvSplit = 2 at 2 threads.
        NumHead = 8;
        KvNumHead = 2;
        HeadDim = 128;
        if (!runAgainstReference(8, 2, 250, 12, "t2 kvSplit merge"))
            return false;
        if (!runAgainstReference(8, 4, 2040, 10, "t4 kvSplit merge wide kv"))
            return false;
        return true;
    }
};
MNNTestSuiteRegister(AttentionKvBlockBoundaryTest, "op/attention_kvblock");

class AttentionC4Test : public AttentionTest {
public:
    AttentionC4Test() = default;
    virtual ~AttentionC4Test() = default;

    bool compareC4Result(int seqLen, const char* caseName) {
        auto outputInfo = Output->getInfo();
        if (outputInfo == nullptr) {
            MNN_ERROR("AttentionC4Test failed to get output info\n");
            return false;
        }
        auto logicalOutput = _Convert(Output, NCHW);
        const float* resultPtr = logicalOutput->readMap<float>();
        if (resultPtr == nullptr) {
            MNN_ERROR("AttentionC4Test failed to map output, expected seqLen=%d, output size=%zu\n", seqLen,
                      outputInfo->size);
            return false;
        }
        if (expected_result.size() != seqLen) {
            MNN_ERROR("AttentionC4Test expected result size mismatch: expected=%d, actual=%zu\n", seqLen,
                      expected_result.size());
            return false;
        }
        const int hidden = NumHead * HeadDim;
        std::vector<float> actual(seqLen * hidden);
        std::vector<float> expected(seqLen * hidden);
        for (int i = 0; i < seqLen; ++i) {
            for (int h = 0; h < NumHead; ++h) {
                for (int d = 0; d < HeadDim; ++d) {
                    int c = h * HeadDim + d;
                    int logicalIndex = i * hidden + c;
                    actual[logicalIndex] = resultPtr[logicalIndex];
                    expected[logicalIndex] = expected_result[i][h][d];
                }
            }
        }
        if (!checkVectorByRelativeError<float>(actual.data(), expected.data(), actual.size(), 0.02f)) {
            MNN_ERROR("AttentionC4Test failed for %s\n", caseName);
            return false;
        }
        return true;
    }

    bool runOne(int seqLen, int precision) {
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        generateInput(seqLen, precision);
        generateMask(seqLen, seqLen);
        expected_result = naiveAttention->onExecute(query, key, value, mask, seqLen);

        auto decodeQuery = generateRandTensor(1, NumHead, HeadDim, precision);
        auto decodeKey = generateRandTensor(1, KvNumHead, HeadDim, precision);
        auto decodeValue = generateRandTensor(1, KvNumHead, HeadDim, precision);
        auto decodeQueryVar = vector_to_var(decodeQuery);
        auto decodeKeyVar = vector_to_var(decodeKey);
        auto decodeValueVar = vector_to_var(decodeValue);
        std::vector<std::vector<int>> decodeMask;

        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = seqLen;
        auto attn = _makeAttentionModule(8, true, true);
        Output = attn->onForward({Query, Key, Value, Mask})[0];
        if (!compareC4Result(seqLen, "NCHW Q/K/V prefill")) {
            return false;
        }
        gMeta.sync();
        expected_result = naiveAttention->onExecute(decodeQuery, decodeKey, decodeValue, decodeMask, 1);
        gMeta.add = 1;
        Output = attn->onForward({decodeQueryVar, decodeKeyVar, decodeValueVar, Mask})[0];
        if (!compareC4Result(1, "NCHW Q/K/V decode")) {
            return false;
        }
        gMeta.sync();

        std::shared_ptr<NaiveAttention> naiveAttentionValueC4(new NaiveAttention);
        expected_result = naiveAttentionValueC4->onExecute(query, key, value, mask, seqLen);
        auto valueC4 = vector_to_c4_value(value);
        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = seqLen;
        auto attnValueC4 = _makeAttentionModule(8, true, true);
        Output = attnValueC4->onForward({Query, Key, valueC4, Mask})[0];
        if (!compareC4Result(seqLen, "NCHW Q/K and C4 V prefill")) {
            return false;
        }
        gMeta.sync();

        auto decodeValueC4 = vector_to_c4_value(decodeValue);
        expected_result = naiveAttentionValueC4->onExecute(decodeQuery, decodeKey, decodeValue, decodeMask, 1);
        gMeta.add = 1;
        Output = attnValueC4->onForward({decodeQueryVar, decodeKeyVar, decodeValueC4, Mask})[0];
        if (!compareC4Result(1, "NCHW Q/K and C4 V decode")) {
            return false;
        }
        gMeta.sync();

        return true;
    }

    bool runTail(int seqLen, int precision) {
        generateInput(seqLen, precision);
        generateMask(seqLen, seqLen);
        auto valueC4 = vector_to_c4_value(value);

        std::shared_ptr<NaiveAttention> outputTailNaive(new NaiveAttention);
        expected_result = outputTailNaive->onExecute(query, key, value, mask, seqLen);
        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = seqLen;
        auto outputTailAttn = _makeAttentionModule(8, true, true);
        Output = outputTailAttn->onForward({Query, Key, Value, Mask})[0];
        if (!compareC4Result(seqLen, "NCHW Q/K/V with C4 tail output")) {
            return false;
        }
        gMeta.sync();

        std::shared_ptr<NaiveAttention> valueTailNaive(new NaiveAttention);
        expected_result = valueTailNaive->onExecute(query, key, value, mask, seqLen);
        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = seqLen;
        auto valueTailAttn = _makeAttentionModule(8, true, true);
        Output = valueTailAttn->onForward({Query, Key, valueC4, Mask})[0];
        if (!compareC4Result(seqLen, "NCHW Q/K with C4 tail V/output")) {
            return false;
        }
        gMeta.sync();

        return true;
    }

    virtual bool run(int precision) {
        OpenCLBufferScope clBufferScope;
        srand(2024);
        return runOne(10, precision) && runOne(32, precision);
    }
};

class AttentionC4TailTest : public AttentionC4Test {
public:
    virtual bool run(int precision) {
        OpenCLBufferScope clBufferScope;
        const int originalNumHead = NumHead;
        const int originalKvNumHead = KvNumHead;
        const int originalHeadDim = HeadDim;
        NumHead = 3;
        KvNumHead = 1;
        HeadDim = 4;
        srand(2024);
        bool tailPass = runTail(10, precision);
        NumHead = 2;
        tailPass = tailPass && runTail(10, precision);
        NumHead = originalNumHead;
        KvNumHead = originalKvNumHead;
        HeadDim = originalHeadDim;
        return tailPass;
    }
};

// head_dim 256 causal prefill, at Qwen3.5's full-attention shape (8 q heads,
// 2 kv heads, group 4). generateMask() emits the scalar sentinel that signals a
// kv-cache causal prefill, which is what selects the fused prefill kernels, so
// this is the correctness gate for the 8-output-tile tc path. 100 is a multiple
// of neither the 64-wide q tile nor the 32-wide kv tile.
class AttentionHeadDim256Test : public AttentionTest {
public:
    virtual bool run(int precision) {
        const int originalNumHead = NumHead;
        const int originalKvNumHead = KvNumHead;
        const int originalHeadDim = HeadDim;
        NumHead = 8;
        KvNumHead = 2;
        HeadDim = 256;
        srand(2024);
        bool pass = true;
        for (int seq_len : {64, 100, 192, 512}) {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            generateInput(seq_len, precision);
            generateMask(seq_len, seq_len);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.previous = 0;
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            if (!compareResult(seq_len)) {
                printf("Error: head_dim 256 causal prefill (seq_len=%d) unit test failed!\n", seq_len);
                pass = false;
                break;
            }
        }
        NumHead = originalNumHead;
        KvNumHead = originalKvNumHead;
        HeadDim = originalHeadDim;
        return pass;
    }
};

// The Metal fused prefill kernels (prefill_flash_attn_sg / prefill_flash_attn_tc)
// engage at seq>=1024 with the scalar causal sentinel. head_dim 64 and 128
// select different kv tile widths (32 / 16) and 1090 exercises the q-tile tail.
// head_dim 256 stays on the three-stage path, covering the q-sequence split that
// bounds its scratch.
class AttentionCausalPrefillTest : public AttentionTest {
public:
    virtual bool run(int precision) {
        const int originalNumHead = NumHead;
        const int originalKvNumHead = KvNumHead;
        const int originalHeadDim = HeadDim;
        NumHead = 2;
        KvNumHead = 1;
        srand(2024);
        bool pass = true;
        for (int head_dim : {64, 128, 256}) {
            HeadDim = head_dim;
            for (int seq_len : {1024, 1090}) {
                std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
                generateInput(seq_len, precision);
                generateMask(seq_len, seq_len);
                expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
                auto attn = _makeAttentionModule();
                gMeta.previous = 0;
                gMeta.add = seq_len;
                Output = attn->onForward({Query, Key, Value, Mask})[0];
                gMeta.sync();
                if (!compareResult(seq_len)) {
                    printf("Error: causal prefill (head_dim=%d, seq_len=%d) unit test failed!\n", head_dim, seq_len);
                    pass = false;
                    break;
                }
            }
            if (!pass) {
                break;
            }
        }
        NumHead = originalNumHead;
        KvNumHead = originalKvNumHead;
        HeadDim = originalHeadDim;
        return pass;
    }
};

MNNTestSuiteRegister(AttentionC4Test, "op/attention_c4");
MNNTestSuiteRegister(AttentionC4TailTest, "op/attention_c4_tail");
MNNTestSuiteRegister(AttentionHeadDim256Test, "op/attention_hd256");
MNNTestSuiteRegister(AttentionCausalPrefillTest, "op/attention_prefill");
MNNTestSuiteRegister(SpeedAttentionTest, "speed/attention");

// ============================================================================
// Metal decode-path shape sweep + prefill-path coverage.
//
// Motivation: the Metal single-pass decode (decode_splitkv) auto-selects its
// simdgroups-per-threadgroup count as NSG = clamp(product / tgCount, 4, 32),
// where tgCount = batch * numHead / qhPerTg and product is the device bandwidth
// tier (256 / 512). For most head counts that is NOT a power of two, and the
// kernel's transposed cross-simdgroup reduce (MetalAttentionShader.hpp) only
// publishes output components owned by lanes sgitg + rep*NSG with
// rep < 32/NSG (integer division): e.g. NSG=12 covers lanes 0..23, so with
// DPT=4 the output dims 96..127 are never written -- stale garbage for ANY
// input data. MR 30076167 rounds NSG down to a power of two. The cases below
// sweep head counts whose auto NSG is non-pow2 on both bandwidth tiers and
// must therefore FAIL on the unfixed kernel and pass after the fix, while the
// pow2 controls (16/32/64 heads, narrow-cap shapes) pass on both.
//
// Input sensitivity: the shared generateRandTensor pattern makes every logit
// nearly equal, so softmax degenerates into a mean and one mis-addressed or
// unwritten slice moves the output by only ~1/kv. K rows here are +-1 sign
// vectors from a deterministic hash and each decode query is an exact copy of
// one K row: QK peaks at HeadDim/sqrt(HeadDim) ~ 11.3 against ~+-1 elsewhere,
// so the output is ~93% of that single V row and the unwritten tail is an
// O(1) error.
// ============================================================================

static int _envInt(const char* name, int defValue) {
    const char* v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return defValue;
    }
    return atoi(v);
}

// [batch, len, heads, dim] built from a single-batch tensor, every batch
// receiving identical rows: each batch's KV cache then holds the same
// content, so the single-batch NaiveAttention reference applies to all.
static VARP _rowsToVar(const std::vector<std::vector<std::vector<float>>>& a, int batch) {
    const int len = (int)a.size();
    const int heads = (int)a[0].size();
    const int dim = (int)a[0][0].size();
    VARP var = _Input({batch, len, heads, dim}, NCHW, halide_type_of<float>());
    float* ptr = var->writeMap<float>();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < len; ++i) {
            for (int h = 0; h < heads; ++h) {
                ::memcpy(ptr + ((b * len + i) * heads + h) * dim, a[i][h].data(), dim * sizeof(float));
            }
        }
    }
    var->unMap();
    return var;
}

static VARP _scalarMask() {
    auto m = _Input({}, NCHW, halide_type_of<float>());
    m->writeMap<float>()[0] = 0.0f;
    return m;
}

// Static KV quantization via AttentionParam.mhq_quant (4 scales:
// q, k, qk, v). kScale/vScale != 0 turns on int8 KV storage.
static std::shared_ptr<Module> _makeAttentionModuleQuant(int attentionMode, bool outputC4, float qScale,
                                                         float kScale, float qkScale, float vScale) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
    attention->main.AsAttentionParam()->output_c4 = outputC4;
    const float scales[4] = {qScale, kScale, qkScale, vScale};
    for (int i = 0; i < 4; ++i) {
        attention->main.AsAttentionParam()->mhq_quant.emplace_back(new MNN::TensorQuantInfoT);
        attention->main.AsAttentionParam()->mhq_quant.back()->scale = scales[i];
    }
    auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
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
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

class AttentionDecodeTest : public AttentionTest {
protected:
    struct ShapeGuard {
        int n, kv, d;
        ShapeGuard() : n(NumHead), kv(KvNumHead), d(HeadDim) {}
        ~ShapeGuard() { NumHead = n; KvNumHead = kv; HeadDim = d; }
    };
    typedef std::vector<std::vector<std::vector<float>>> Tensor3;

    static Tensor3 genKeyRows(int len) {
        Tensor3 k(len);
        for (int j = 0; j < len; ++j) {
            k[j].resize(KvNumHead);
            for (int h = 0; h < KvNumHead; ++h) {
                k[j][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    k[j][h][d] = (_kvbHash(j, h, d) & 1u) ? 1.0f : -1.0f;
                }
            }
        }
        return k;
    }
    static Tensor3 genValueRows(int len) {
        Tensor3 v(len);
        for (int j = 0; j < len; ++j) {
            v[j].resize(KvNumHead);
            for (int h = 0; h < KvNumHead; ++h) {
                v[j][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    v[j][h][d] = (float)(_kvbHash(j + 7919u, h + 31u, d) % 2001u) / 1000.0f - 1.0f;
                }
            }
        }
        return v;
    }
    // Small-amplitude queries for the prefill segment; its output is never
    // checked, only the resulting KV cache contents matter.
    static Tensor3 genPrefillQuery(int len) {
        Tensor3 q(len);
        for (int i = 0; i < len; ++i) {
            q[i].resize(NumHead);
            for (int h = 0; h < NumHead; ++h) {
                q[i][h].resize(HeadDim);
                for (int d = 0; d < HeadDim; ++d) {
                    q[i][h][d] = (float)(_kvbHash(i + 104729u, h, d) % 101u) * 0.002f - 0.1f;
                }
            }
        }
        return q;
    }
    // Decode query that peaks on kv row `target`.
    static Tensor3 genProbeQuery(const Tensor3& key, int target) {
        const int group = NumHead / KvNumHead;
        Tensor3 q(1);
        q[0].resize(NumHead);
        for (int h = 0; h < NumHead; ++h) {
            q[0][h] = key[target][h / group];
        }
        return q;
    }
    static Tensor3 sliceRow(const Tensor3& src, int row) {
        Tensor3 out(1);
        out[0] = src[row];
        return out;
    }
    static Tensor3 sliceHead(const Tensor3& src, int len) {
        return Tensor3(src.begin(), src.begin() + len);
    }
    // Probe positions spread across the head dim and typical tile boundaries;
    // kvLen caps them for short cases.
    static int probeTarget(int step, int kvLen) {
        static const int kProbes[] = {0, 1, 30, 31, 63, 7, 15, 23, 47, 55, 2, 29, 40, 48, 56, 62};
        const int n = (int)(sizeof(kProbes) / sizeof(kProbes[0]));
        int t = kProbes[step % n];
        if (t >= kvLen) {
            t = kvLen - 1;
        }
        return t;
    }

    bool compareDecodeBatch(int batch, float tolAbs, float tolRel) {
        VARP logical = Output;
        if (mOutputC4) {
            logical = _Convert(Output, NCHW);
        }
        const float* resultPtr = logical->readMap<float>();
        if (nullptr == resultPtr) {
            MNN_ERROR("AttentionDecodeTest failed to map output\n");
            return false;
        }
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < NumHead; ++h) {
                for (int d = 0; d < HeadDim; ++d) {
                    const int idx = (b * NumHead + h) * HeadDim + d;
                    const float got = resultPtr[idx];
                    const float exp = expected_result[0][h][d];
                    if (got != got) {
                        MNN_PRINT("AttentionDecodeTest: NaN at batch=%d head=%d dim=%d\n", b, h, d);
                        return false;
                    }
                    const float diff = fabsf(got - exp);
                    if (diff > tolAbs && diff > tolRel * fabsf(exp)) {
                        MNN_PRINT("AttentionDecodeTest: mismatch at batch=%d head=%d dim=%d: got=%f exp=%f\n",
                                  b, h, d, got, exp);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // quantMode: 0 float, 9 dynamic int8 K, 10 dynamic int8 K+V, 11 static int8 K+V.
    // maskMode: 0 scalar causal sentinel, 1 trivial [1,1,1,1] ADD mask,
    //           2 materialized [1,1,1,kv] ADD mask plane.
    bool runDecodeCase(int numHead, int kvNumHead, int headDim, int batch, int prefill, int steps, int quantMode,
                       bool outputC4, int maskMode, bool chunked, const char* tag) {
        NumHead = numHead;
        KvNumHead = kvNumHead;
        HeadDim = headDim;
        mOutputC4 = outputC4;
        const int total = prefill + steps;
        auto key = genKeyRows(total);
        auto value = genValueRows(total);
        auto pq = genPrefillQuery(prefill);
        auto prefillKey = sliceHead(key, prefill);
        auto prefillValue = sliceHead(value, prefill);

        std::shared_ptr<NaiveAttention> ref(new NaiveAttention);
        ref->appendHistory(prefillKey, prefillValue, prefill);

        std::shared_ptr<Module> module;
        if (quantMode == 11) {
            // Static scales: k/v = 1/127 reconstructs +-1 K exactly and keeps
            // V in [-1,1] within half an int8 step; q/qk scales are unused by
            // the Metal kernels, set to identity.
            module = _makeAttentionModuleQuant(8, outputC4, 1.0f, 1.0f / 127.0f, 1.0f, 1.0f / 127.0f);
        } else {
            module = _makeAttentionModule(quantMode, outputC4);
        }
        gMeta.previous = 0;
        gMeta.remove = 0;
        if (chunked) {
            // Two prefills of prefill/2 each, exercising a second onResize on
            // top of a non-empty KV cache (the llm prompt-splitting pattern).
            // No GPU sync between them on purpose: on KV growth the Metal
            // cache manager memcpys old rows on the CPU, racing the first
            // prefill's in-flight copy kernel. gMeta.sync() already rolls add
            // into previous, so the second prefill must not touch previous.
            const int half = prefill / 2;
            gMeta.add = half;
            module->onForward({_rowsToVar(sliceHead(pq, half), batch), _rowsToVar(sliceHead(prefillKey, half), batch),
                               _rowsToVar(sliceHead(prefillValue, half), batch), _scalarMask()});
            gMeta.sync();
            Tensor3 prefillKeyTail(prefillKey.begin() + half, prefillKey.end());
            Tensor3 prefillValueTail(prefillValue.begin() + half, prefillValue.end());
            gMeta.add = prefill - half;
            module->onForward({_rowsToVar(sliceHead(pq, half), batch), _rowsToVar(prefillKeyTail, batch),
                               _rowsToVar(prefillValueTail, batch), _scalarMask()});
            gMeta.sync();
        } else if (prefill > 0) {
            gMeta.add = prefill;
            module->onForward({_rowsToVar(pq, batch), _rowsToVar(prefillKey, batch),
                               _rowsToVar(prefillValue, batch), _scalarMask()});
            gMeta.sync();
        }

        std::vector<std::vector<int>> noMask;
        const float tolAbs = (quantMode != 0) ? 0.03f : 0.005f;
        const float tolRel = (quantMode != 0) ? 0.12f : 0.05f;
        for (int s = 0; s < steps; ++s) {
            const int kvLen = prefill + s + 1;
            auto q1 = genProbeQuery(key, probeTarget(s, kvLen));
            auto k1 = sliceRow(key, prefill + s);
            auto v1 = sliceRow(value, prefill + s);
            expected_result = ref->onExecute(q1, k1, v1, noMask, 1);
            gMeta.add = 1;
            VARP maskVar;
            if (maskMode == 1) {
                // Non-scalar but single-element all-zero ADD mask: the
                // trivialFloatMask branch of the decode dispatch.
                maskVar = _Input({1, 1, 1, 1}, NCHW, halide_type_of<float>());
                maskVar->writeMap<float>()[0] = 0.0f;
            } else if (maskMode == 2) {
                // Real mask plane over the whole kv axis: neither causal nor
                // trivial, so decode falls back to the three-stage path.
                maskVar = _Input({1, 1, 1, kvLen}, NCHW, halide_type_of<float>());
                ::memset(maskVar->writeMap<float>(), 0, kvLen * sizeof(float));
            } else {
                maskVar = _scalarMask();
            }
            Output = module->onForward({_rowsToVar(q1, batch), _rowsToVar(k1, batch), _rowsToVar(v1, batch),
                                        maskVar})[0];
            gMeta.sync();
            if (!compareDecodeBatch(batch, tolAbs, tolRel)) {
                MNN_PRINT("Error: %s failed at decode step %d (kvLen=%d, probe=%d)\n", tag, s, kvLen,
                          probeTarget(s, kvLen));
                return false;
            }
        }
        return true;
    }

public:
    AttentionDecodeTest() = default;
    virtual ~AttentionDecodeTest() = default;

    virtual bool run(int precision) {
        ShapeGuard guard;
        srand(2024);
        bool pass = true;
        // Fused single-pass decode sweep at group=2, hd=128, batch=1.
        // Auto NSG on the 256-product tier / 512-product tier:
        //   16q->16/32 (pow2 control)        20q->12/25 (bug)
        //   24q->10/21 (bug)                 28q->9/18  (bug)
        //   32q->16/32 (qh=2, control)       40q->12/25 (bug, qh=2)
        //   48q->10/21 (bug, qh=2)           64q->8/16  (control)
        for (int numHead : {16, 20, 24, 28, 32, 40, 48, 64}) {
            pass &= runDecodeCase(numHead, numHead / 2, 128, 1, 64, 4, 0, false, 0, false,
                                  "decode nsg sweep");
        }
        // batch>1 is not covered: attention KV caches (CPU and Metal) are
        // allocated per-token without a batch dimension, so multi-batch runs
        // write past the cache and fail on both backends. The engine only
        // runs attention with batch=1.
        // The bug is kv-length independent: it must also fire right at the
        // single-pass threshold and past the narrow-cap cutoff.
        pass &= runDecodeCase(48, 24, 128, 1, 2, 4, 0, false, 0, false, "decode tiny kv");
        pass &= runDecodeCase(48, 24, 128, 1, 511, 4, 0, false, 0, false, "decode long kv");
        // Narrow-cap shapes (tgCount < 16): NSG capped at 16. prefill=510
        // crosses the totalKv<512 cap boundary mid-stream (NSG 16 -> 32).
        pass &= runDecodeCase(8, 4, 128, 1, 64, 4, 0, false, 0, false, "decode narrow cap");
        pass &= runDecodeCase(8, 4, 128, 1, 510, 6, 0, false, 0, false, "decode cap crossing");
        // GQA group 4 and 8 shapes (Qwen3.5-class).
        pass &= runDecodeCase(8, 2, 128, 1, 64, 4, 0, false, 0, false, "decode group4");
        pass &= runDecodeCase(16, 2, 128, 1, 64, 4, 0, false, 0, false, "decode group8");
        // MQA group=1 in the single-pass kernel.
        pass &= runDecodeCase(16, 16, 128, 1, 64, 4, 0, false, 0, false, "decode mqa");
        // kv==1 at the first step stays on the fused qk_softmax kernel
        // (single-pass needs totalKv >= 2); step 1 crosses over to splitkv.
        pass &= runDecodeCase(16, 8, 128, 1, 0, 4, 0, false, 0, false, "decode kv1 fused");
        // head_dim 64 / 256 (DPT 2 / 8) and 96 (hd%32 != 0 -> qk_softmax).
        pass &= runDecodeCase(16, 8, 64, 1, 64, 4, 0, false, 0, false, "decode hd64");
        pass &= runDecodeCase(16, 8, 256, 1, 64, 4, 0, false, 0, false, "decode hd256");
        pass &= runDecodeCase(16, 8, 96, 1, 64, 4, 0, false, 0, false, "decode hd96");
        // Mask variants: trivial single-element tensor mask (single-pass) and a
        // real [1,1,1,kv] plane (three-stage decode).
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, false, 1, false, "decode trivial mask");
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, false, 2, false, "decode plane mask");
        // C4 output (ATTENTION_C4 kernel variant).
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, true, 0, false, "decode out c4");
        // Chunked double prefill, then decode.
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, false, 0, true, "decode chunked");
        // KV quantization: dynamic int8 K / K+V and static mhq_quant scales.
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 9, false, 0, false, "decode quant k");
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 10, false, 0, false, "decode quant kv");
        pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 11, false, 0, false, "decode static quant");
        return pass;
    }

private:
    bool mOutputC4 = false;
};

// Env-gated decode paths that the default flags never select:
//   MNN_METAL_DECODE_SDPA=0   -> fused qk_softmax / three-stage decode
//   MNN_METAL_DECODE_SDPA=N   -> qk_softmax up to kv=N (QK_QSPLIT at kv>=512)
//   MNN_METAL_DECODE_SDPA_NTG=2 -> 2-pass split-KV reduce
// MetalEnv parses env vars once per process, so the cases run only when the
// matching var was set before this process started.
class AttentionDecodeEnvTest : public AttentionDecodeTest {
public:
    virtual bool run(int precision) {
        ShapeGuard guard;
        if ((MNNForwardType)MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_METAL) {
            return true;
        }
        bool pass = true;
        const int sdpaEnv = _envInt("MNN_METAL_DECODE_SDPA", 1);
        if (sdpaEnv == 0) {
            // mDecodeQkSoftmax requires group >= 2, so MQA is forced onto the
            // three-stage decode at every kv length; group=8 caps the fused
            // path at kv<=512, so kv 513+ also lands on three-stage.
            pass &= runDecodeCase(16, 16, 128, 1, 64, 4, 0, false, 0, false,
                                  "decodeSdpa=0 mqa three-stage");
            pass &= runDecodeCase(16, 2, 128, 1, 513, 4, 0, false, 0, false,
                                  "decodeSdpa=0 three-stage");
            pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, false, 0, false,
                                  "decodeSdpa=0 qk_softmax");
        } else if (sdpaEnv > 1) {
            pass &= runDecodeCase(16, 8, 128, 1, 64, 4, 0, false, 0, false,
                                  "decodeSdpa=N qk_softmax short kv");
            pass &= runDecodeCase(16, 8, 128, 1, 511, 4, 0, false, 0, false,
                                  "decodeSdpa=N qk_softmax qsplit");
        }
        if (_envInt("MNN_METAL_DECODE_SDPA_NTG", 0) > 1) {
            pass &= runDecodeCase(16, 8, 128, 1, 64, 6, 0, false, 0, false, "split-kv 2-pass");
        }
        return pass;
    }
};

// Prefill-path coverage: three-stage vs fused FA variants at the shapes the
// path flags distinguish. hint=0 keeps the three-stage pipeline on every
// backend; hint=8 enables the legacy FA kernel where eligible (M2+, fp16,
// hd in {64,128,256}, group in {1,2,4,8}, seq>=128; M4-class devices demote
// back to three-stage). FA-SG / FA-TC engage automatically at seq>=1024 on
// their tiers, already covered by op/attention_prefill.
class AttentionPrefillPathTest : public AttentionC4Test {
public:
    virtual bool run(int precision) {
        const int savedNumHead = NumHead;
        const int savedKvNumHead = KvNumHead;
        const int savedHeadDim = HeadDim;
        mPrecision = precision;
        srand(2024);
        bool pass = true;
        // hint=8: FA eligible (group 1/2/4/8, hd 64/128/256). batch>1 is not
        // covered: the KV cache is allocated per-token without a batch
        // dimension (MetalKVCacheManager::onAlloc), so batch>1 prefill writes
        // past the cache buffer; the engine itself only runs batch=1.
        pass &= runPrefillCase(16, 8, 128, 1, 192, 8, false, 0.0f, 0.0f, "fa g2 hd128");
        pass &= runPrefillCase(16, 8, 128, 1, 128, 8, false, 0.0f, 0.0f, "fa g2 seq128");
        pass &= runPrefillCase(16, 8, 128, 1, 511, 8, false, 0.0f, 0.0f, "fa g2 seq511");
        pass &= runPrefillCase(16, 4, 128, 1, 192, 8, false, 0.0f, 0.0f, "fa g4");
        pass &= runPrefillCase(32, 4, 128, 1, 192, 8, false, 0.0f, 0.0f, "fa g8");
        pass &= runPrefillCase(16, 16, 128, 1, 192, 8, false, 0.0f, 0.0f, "fa g1 mqa");
        pass &= runPrefillCase(16, 8, 64, 1, 192, 8, false, 0.0f, 0.0f, "fa hd64");
        pass &= runPrefillCase(8, 2, 256, 1, 192, 8, false, 0.0f, 0.0f, "fa hd256");
        // hint=0: deterministic three-stage (hd 96 also falls back there).
        pass &= runPrefillCase(16, 8, 128, 1, 192, 0, false, 0.0f, 0.0f, "three-stage seq192");
        pass &= runPrefillCase(16, 8, 128, 1, 511, 0, false, 0.0f, 0.0f, "three-stage seq511");
        pass &= runPrefillCase(16, 8, 96, 1, 192, 8, false, 0.0f, 0.0f, "prefill hd96");
        // C4 output on the fused prefill.
        pass &= runPrefillCase(16, 8, 128, 1, 192, 8, true, 0.0f, 0.0f, "fa out c4");
        // KV quantization on prefill (relaxed tolerance).
        pass &= runPrefillCase(16, 8, 128, 1, 192, 9, false, 0.03f, 0.12f, "prefill quant k");
        pass &= runPrefillCase(16, 8, 128, 1, 192, 10, false, 0.03f, 0.12f, "prefill quant kv");
        pass &= runPrefillCase(16, 8, 128, 1, 192, 11, false, 0.03f, 0.12f, "prefill static quant");

        // Env-forced prefill paths (Metal-only; MetalEnv reads the vars once).
        if ((MNNForwardType)MNNTestSuite::get()->pStaus.forwardType == MNN_FORWARD_METAL) {
            const bool forceSg = _envInt("MNN_METAL_PREFILL_FA_SG", -1) == 1;
            const bool forceFa = _envInt("MNN_ENABLE_FLASH_ATTN_PREFILL", -1) == 1;
            const bool forceFaOff = _envInt("MNN_ENABLE_FLASH_ATTN_PREFILL", -1) == 0;
            if (forceSg) {
                // FA-SG force-on: seq>=64, hd 64/128, group 1/2/4/8.
                pass &= runPrefillCase(16, 8, 128, 1, 64, 8, false, 0.0f, 0.0f, "fa-sg seq64");
                pass &= runPrefillCase(16, 8, 64, 1, 192, 8, false, 0.0f, 0.0f, "fa-sg hd64");
                pass &= runPrefillCase(16, 4, 128, 1, 128, 8, false, 0.0f, 0.0f, "fa-sg g4");
            }
            if (forceFa) {
                // Legacy FA force-on (overrides the M4-class demotion).
                pass &= runPrefillCase(16, 8, 128, 1, 192, 8, false, 0.0f, 0.0f, "fa forced");
            }
            if (forceFaOff) {
                // FA force-off: hint=8 falls back to three-stage.
                pass &= runPrefillCase(16, 8, 128, 1, 192, 8, false, 0.0f, 0.0f, "three-stage forced");
            }
        }
        NumHead = savedNumHead;
        KvNumHead = savedKvNumHead;
        HeadDim = savedHeadDim;
        return pass;
    }

private:
    // quantMode: 0 float, 9/10 dynamic quant, 11 static quant (see runDecodeCase).
    bool runPrefillCase(int numHead, int kvNumHead, int headDim, int batch, int seq, int quantMode,
                        bool outputC4, float tolAbs, float tolRel, const char* tag) {
        NumHead = numHead;
        KvNumHead = kvNumHead;
        HeadDim = headDim;
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        generateInput(seq, mPrecision);
        if (quantMode == 11) {
            // Static scales put K/V in [-1,1] inside the int8 cache; full-range
            // data would saturate there while the unquantized reference stays
            // full-range. Regenerate K/V at small magnitude so quantization
            // stays exact (<= half an int8 step).
            key   = generateRandTensor(seq, KvNumHead, HeadDim, 2);
            value = generateRandTensor(seq, KvNumHead, HeadDim, 2);
        }
        generateMask(seq, seq);
        expected_result = naiveAttention->onExecute(query, key, value, mask, seq);
        std::shared_ptr<Module> module;
        if (quantMode == 11) {
            module = _makeAttentionModuleQuant(8, outputC4, 1.0f, 1.0f / 127.0f, 1.0f, 1.0f / 127.0f);
        } else {
            module = _makeAttentionModule(quantMode, outputC4);
        }
        gMeta.previous = 0;
        gMeta.remove = 0;
        gMeta.add = seq;
        Output = module->onForward({_rowsToVar(query, batch), _rowsToVar(key, batch), _rowsToVar(value, batch),
                                    _scalarMask()})[0];
        gMeta.sync();
        if (outputC4) {
            if (!compareC4Result(seq, tag)) {
                MNN_PRINT("Error: %s (seq=%d, %dq/%dkv/hd%d, batch=%d) unit test failed!\n", tag, seq,
                          numHead, kvNumHead, headDim, batch);
                return false;
            }
            return true;
        }
        const float* resultPtr = Output->readMap<float>();
        if (nullptr == resultPtr) {
            MNN_PRINT("Error: %s failed to map output\n", tag);
            return false;
        }
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < NumHead; ++j) {
                for (int k = 0; k < HeadDim; ++k) {
                    const float got = resultPtr[(i * NumHead + j) * HeadDim + k];
                    const float exp = expected_result[i][j][k];
                    if (got != got) {
                        MNN_PRINT("Error: %s NaN at [%d][%d][%d]\n", tag, i, j, k);
                        Output->unMap();
                        return false;
                    }
                    const float diff = fabsf(got - exp);
                    if (diff > (tolAbs > 0.0f ? tolAbs : diff_threshold) &&
                        diff > (tolRel > 0.0f ? tolRel : diff_percent_threshold) * fabsf(exp)) {
                        MNN_PRINT("Error: %s (seq=%d, %dq/%dkv/hd%d, batch=%d) mismatch at [%d][%d][%d]: "
                                  "got=%f exp=%f\n",
                                  tag, seq, numHead, kvNumHead, headDim, batch, i, j, k, got, exp);
                        Output->unMap();
                        return false;
                    }
                }
            }
        }
        Output->unMap();
        return true;
    }

    int mPrecision = 2;
};

MNNTestSuiteRegister(AttentionDecodeTest, "op/attention_decode");
MNNTestSuiteRegister(AttentionDecodeEnvTest, "op/attention_decode_env");
MNNTestSuiteRegister(AttentionPrefillPathTest, "op/attention_prefill_paths");
#endif
