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
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <vector>
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;
using MNN::KVMeta;

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;
const int pastLength = 101;
#define GENERATE_TOKENS 128

static KVMeta gMeta;
static std::shared_ptr<Module> _makeAttentionModule(int attentionMode = 8, bool outputC4 = false,
                                                    bool forceOpenCLBuffer = false, int numThread = 1) {
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
    config.numThread = forceOpenCLBuffer && status.forwardType == MNN_FORWARD_OPENCL
                           ? MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE
                           : numThread;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
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
        for (int v=0; v<pastLength; ++v) {
            pastMask->writeMap<float>()[v] = std::numeric_limits<float>::lowest();
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
        mask = (_Scalar<float>(1.0) - _Cast<float>(mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());
    }

    Q = _Reshape(Q, {batch, seqLength, kvNumHead,group, headDim});
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
    for (int y=0; y<kvNumHead; ++y) {
        ::memcpy(cache.pastK->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, K->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
        ::memcpy(cache.pastV->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, V->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
    }
    for (int i=0; i<seqLength; ++i) {
        cache.pastMask->writeMap<float>()[i+cache.current] = 0.0f;
    }
    cache.current += seqLength;
    return O;
}

static std::vector< std::vector< std::vector<float> > > generateRandTensor(int C, int H, int W, int precision) {
    std::vector< std::vector< std::vector<float> > > a;
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

VARP vector_to_var(std::vector< std::vector< std::vector<float> > > & a) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    VARP var = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    float * ptr = var->writeMap<float>();
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

VARP vector_to_c4_value(std::vector< std::vector< std::vector<float> > > & a) {
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

VARP vector_to_var(std::vector< std::vector<int> > & a) {
    int H = a.size();
    int W = a[0].size();
    VARP var = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
    int * ptr = var->writeMap<int>();
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            ptr[i * W + j] = a[i][j];
        }
    }
    var->unMap();
    return var;
}

static std::vector< std::vector< std::vector<float> > >
computeAttention (
    std::vector< std::vector< std::vector<float> > > & query,
    std::vector< std::vector< std::vector<float> > > & key,
    std::vector< std::vector< std::vector<float> > > & value,
    std::vector< std::vector<int> > & mask,
    int seq_len, int kv_seq_len )
{
    int group_size = NumHead / KvNumHead;
    std::vector< std::vector< std::vector<float> > > output(seq_len);
    for (int i = 0; i < seq_len; i++) {
        output[i].resize(NumHead);
        for (int j = 0; j < NumHead; j++) {
            output[i][j].resize(HeadDim);
        }
    }
    for (int h = 0; h < NumHead; h++) {
        int kv_h = h / group_size;
        /*---- Q * K ----*/
        std::vector< std::vector<float> > qk(seq_len, std::vector<float>(kv_seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = 0.0f;
                for (int k = 0; k < HeadDim; k++) {
                    qk[i][j] += query[i][h][k] * key[j][kv_h][k];
                }
            }
        }
        /*---- Mask QK ----*/
        if(mask.size() > 0) {
            float scale = 1.0 / sqrt(HeadDim);
            if (mask[0].size() == seq_len) {
                auto diff = kv_seq_len - seq_len;
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        qk[i][j+diff] = qk[i][j+diff] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
                    }
                }
            } else {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < kv_seq_len; j++) {
                        qk[i][j] = qk[i][j] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
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
        std::vector< std::vector< std::vector<float> > >  mPastKey, mPastValue;
        int mPastLen;
    public:
        NaiveAttention() : mPastLen(0) {}
        ~NaiveAttention() = default;
        // Push prefill K/V into history WITHOUT computing attention. The wide-KV-block boundary
        // test needs kv cache filled past 2048 rows; running onExecute for that prefill would cost
        // O(kv^2) scalar work (~10 GFLOP at kv=2040). Only the decode steps need a reference.
        void appendHistory(
            std::vector< std::vector< std::vector<float> > > & key,
            std::vector< std::vector< std::vector<float> > > & value,
            int seq_len )
        {
            for (int i = 0; i < seq_len; i++) {
                mPastKey.push_back(key[i]);
                mPastValue.push_back(value[i]);
            }
            mPastLen += seq_len;
        }
        int pastLen() const {
            return mPastLen;
        }
        std::vector< std::vector< std::vector<float> > > onExecute (
            std::vector< std::vector< std::vector<float> > > & query,
            std::vector< std::vector< std::vector<float> > > & key,
            std::vector< std::vector< std::vector<float> > > & value,
            std::vector< std::vector<int> > & mask,
            int seq_len )
        {
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
    std::vector< std::vector< std::vector<float> > > query;
    std::vector< std::vector< std::vector<float> > > key;
    std::vector< std::vector< std::vector<float> > > value;
    std::vector< std::vector<int> > mask;
    std::vector< std::vector< std::vector<float> > > expected_result;
    VARP Query, Key, Value, Mask, Output;
    VARP Query1, Key1, Value1, Mask1;
public:
    AttentionTest() = default;
    virtual ~AttentionTest() = default;
    void generateInput(int seq_len, int precision, bool genDecodeInput = false) {
        query = generateRandTensor(seq_len, NumHead, HeadDim, precision);
        key   = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        value = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        Query = vector_to_var(query);
        Key   = vector_to_var(key);
        Value = vector_to_var(value);
        if (genDecodeInput) {
            auto vecquery = generateRandTensor(1, NumHead, HeadDim, precision);
            auto veckey   = generateRandTensor(1, KvNumHead, HeadDim, precision);
            auto vecvalue = generateRandTensor(1, KvNumHead, HeadDim, precision);
            Query1 = vector_to_var(vecquery);
            Key1   = vector_to_var(veckey);
            Value1 = vector_to_var(vecvalue);
        }
    }
    void generateChunkMask(int seq_len, int kv_seq_len, int chunk_size, bool genDecodeInput = false) {
        // 防止除以0
        if (chunk_size <= 0) chunk_size = 1;

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
        Mask = (_Scalar<float>(1.0) - _Cast<float>(Mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());

        // Decode Input 部分通常保持全 1 (即看清所有历史)，或者根据需求修改
        if (genDecodeInput) {
            std::vector<std::vector<int>> vecmask;
            vecmask.resize(1);
            vecmask[0].resize(gMeta.previous + 1);
            for (int i = 0; i < gMeta.previous + 1; ++i) {
                vecmask[0][i] = 1;
            }
            Mask1 = vector_to_var(vecmask);
            Mask1 = (_Scalar<float>(1.0) - _Cast<float>(Mask1)) * _Scalar<float>(std::numeric_limits<float>::lowest());
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
        const float * resultPtr = Output->readMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < NumHead; j++) {
                for (int k = 0; k < HeadDim; k++) {
                    float diff = fabs(resultPtr[i * NumHead * HeadDim + j * HeadDim + k] - expected_result[i][j][k]);
                    float diff_percent = fabs(diff / expected_result[i][j][k]);
                    if (diff > diff_threshold && diff_percent > diff_percent_threshold) {
                        printf("Result Mismatch: expected %lf but got %lf in CPU Attention Test\n", expected_result[i][j][k], resultPtr[i * NumHead * HeadDim + j * HeadDim + k]);
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
            MaskExpr = (_Scalar<float>(1.0) - _Cast<float>(MaskExpr)) * _Scalar<float>(std::numeric_limits<float>::lowest());
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
            if (diff >= 0.01f) {                 FUNC_PRINT_ALL(diff, f);
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
            if (diff >= 0.01f) {
                FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // unit test 3
        {
            auto rtInfo = ExecutorScope::Current()->getRuntime().first;
            bool cpuInfer = true;
            for(auto &rt : rtInfo) {
                if(rt.first != MNN_FORWARD_CPU) {
                    cpuInfer = false;
                    break;
                }
            }
            if(cpuInfer) {
                // TODO: CPU support kv_cache == false
                return true;
            }
            // MNN: kv_cache=false also falls back to CPU on OpenCL with
            // MNN_GPU_MEMORY_IMAGE (no IMAGE-memtype Attention creator) and
            // on Vulkan, so it hits the same CPUAttention "kv_cache == false"
            // TODO and crashes. Skip until the CPU fallback is completed.
            for(auto &rt : rtInfo) {
                if(rt.first == MNN_FORWARD_OPENCL || rt.first == MNN_FORWARD_VULKAN) {
                    return true;
                }
            }
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
        return true;
    }
};

class SpeedAttentionTest : public AttentionTest {
    protected:
        std::vector< std::vector< std::vector<float> > > query;
        std::vector< std::vector< std::vector<float> > > key;
        std::vector< std::vector< std::vector<float> > > value;
        std::vector< std::vector<int> > mask;
        std::vector< std::vector< std::vector<float> > > expected_result;

public:
SpeedAttentionTest() = default;
    virtual ~SpeedAttentionTest() = default;

    virtual bool run(int precision) {
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
            MNN_PRINT("[attention_nocache_mask] seq=%4d allvisible(3d/4d)=%.6f/%.6f rowvarying(3d/4d)=%.6f/%.6f "
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
        for (auto& x : q) x = next();
        for (auto& x : k) x = next();
        for (auto& x : v) x = next();
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
        NumHead = 16; KvNumHead = 8; HeadDim = 128;
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
        NumHead = savedNumHead; KvNumHead = savedKvNumHead; HeadDim = savedHeadDim;
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
    static Tensor3 sliceHead(const Tensor3& src, int len) {
        return Tensor3(src.begin(), src.begin() + len);
    }
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
            Output = module->onForward({vector_to_var(q1), vector_to_var(k1), vector_to_var(v1),
                                        scalarMask()})[0];
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
                auto out = module->onForward({vector_to_var(q1), vector_to_var(k1),
                                              vector_to_var(v1), scalarMask()})[0];
                gMeta.sync();
                const float* ptr = out->readMap<float>();
                if (pass == 0) {
                    captured[s].assign(ptr, ptr + outSize);
                } else {
                    for (int i = 0; i < outSize; ++i) {
                        float diff = fabsf(ptr[i] - captured[s][i]);
                        float rel = fabsf(diff / (captured[s][i] == 0.f ? 1e-20f : captured[s][i]));
                        if (diff > diff_threshold && rel > diff_percent_threshold) {
                            MNN_PRINT("Error: %s flash-on/off mismatch at step %d (kvLen=%d), "
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
        NumHead = 16; KvNumHead = 8; HeadDim = 128;

        // Single thread: physical V chunk 2048, logical block ALIMIN(2048, kvLen).
        if (!runAgainstReference(8, 1, 250, 10, "t1 short kv")) return false;
        if (!runAgainstReference(8, 1, 2040, 12, "t1 kv crossing 2048")) return false;
        // Prefill past the physical chunk boundary: the chunk gate has no insertLen term while the
        // logical-block gate does, so this prefills with 64-row blocks into 2048-row chunks and the
        // following decode must still read both chunks correctly.
        if (!runAgainstReference(8, 1, 2100, 10, "t1 prefill crossing chunk")) return false;

        // Multi thread: physical V chunk 64, logical block ALIMIN(256, kvLen) + sub-chunk addTile.
        if (!runAgainstReference(8, 4, 60, 10, "t4 kv crossing 64")) return false;
        if (!runAgainstReference(8, 4, 250, 12, "t4 kv crossing 256")) return false;
        if (!runAgainstReference(8, 4, 2040, 12, "t4 wide kv")) return false;

        // K-int8 KV cache: wide block is gated separately, use the flash on/off differential.
        if (!runFlashOnOffDiff(1, 1, 2040, 10, "quantK t1 kv crossing 2048")) return false;
        if (!runFlashOnOffDiff(1, 4, 250, 10, "quantK t4 kv crossing 256")) return false;

        // kvSplit > 1 needs few kv heads: numUnits = 2 gives kvSplit = 2 at 2 threads.
        NumHead = 8; KvNumHead = 2; HeadDim = 128;
        if (!runAgainstReference(8, 2, 250, 12, "t2 kvSplit merge")) return false;
        if (!runAgainstReference(8, 4, 2040, 10, "t4 kvSplit merge wide kv")) return false;
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
        srand(2024);
        return runOne(10, precision) && runOne(32, precision);
    }
};

class AttentionC4TailTest : public AttentionC4Test {
public:
    virtual bool run(int precision) {
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

MNNTestSuiteRegister(AttentionC4Test, "op/attention_c4");
MNNTestSuiteRegister(AttentionC4TailTest, "op/attention_c4_tail");
MNNTestSuiteRegister(SpeedAttentionTest, "speed/attention");
#endif
