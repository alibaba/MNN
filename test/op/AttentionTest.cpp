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
#include "core/OpCommonUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <vector>

using namespace MNN::Express;

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;

static std::vector< std::vector< std::vector<float> > > generateRandTensor(int C, int H, int W) {
    std::vector< std::vector< std::vector<float> > > a;
    a.resize(C);
    for (int i = 0; i < C; i++) {
        a[i].resize(H);
        for (int j = 0; j < H; j++) {
            a[i][j].resize(W);
            for (int k = 0; k < W; k++) {
                a[i][j][k] = (float)rand() / (float)RAND_MAX * 10.0 * (rand() % 2 ? 1 : -1);
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
        float scale = 1.0 / sqrt(HeadDim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                if (mask[i][j] == 1) {
                    qk[i][j] *= scale;
                } else {
                    qk[i][j] = std::numeric_limits<float>::lowest();
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
public:
    AttentionTest() = default;
    virtual ~AttentionTest() = default;

    void generateInput(int seq_len) {
        query = generateRandTensor(seq_len, NumHead, HeadDim);
        key   = generateRandTensor(seq_len, KvNumHead, HeadDim);
        value = generateRandTensor(seq_len, KvNumHead, HeadDim);
        Query = vector_to_var(query);
        Key   = vector_to_var(key);
        Value = vector_to_var(value);
    }

    void generateMask(int seq_len, int kv_seq_len) {
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
        auto rt = ExecutorScope::Current()->getRuntime();
        MNN::KVMeta meta;
        for (auto& iter : rt.first) {
            iter.second->pMeta = &meta;
        }
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = true;
        int seq_len = 10;
        meta.add = seq_len;
        generateInput(seq_len);
        generateMask(seq_len, seq_len);
        expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
        Output = Variable::create(Expr::create(attention.get(), {Query, Key, Value, Mask}));
        bool pass = compareResult(seq_len);
        if (pass) {
            printf("CPU attention unit test passed!\n");
        } else {
            printf("Error: CPU attention unit test failed!\n");
        }
        return pass;
    }
};

MNNTestSuiteRegister(AttentionTest, "op/cpu_attention");
#endif
