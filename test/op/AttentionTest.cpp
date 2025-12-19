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

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;
const int pastLength = 101;
#define GENERATE_TOKENS 128
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
static std::shared_ptr<Module> _makeAttentionModule(int quant_qkv = 8) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
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
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, quant_qkv);
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
        Mask  = vector_to_var(mask);
        Mask = (_Scalar<float>(1.0) - _Cast<float>(Mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());
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
                printf("Error: Attention with kv_cache unit test failed!\n");
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
        // unit test 2
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
MNNTestSuiteRegister(SpeedAttentionTest, "speed/attention");
#endif
