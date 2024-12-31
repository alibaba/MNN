//
//  CPUAttention.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <limits>
#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

namespace MNN {

template <typename T>
void CPUAttention::pack_query(Tensor* query, char* pack_q, char* sum_q, int seq_len, int h, float q_scale) {
    if (mUseGemmInt8) { // Shape of Query: numhead, [seqlen/eP8, headdim/lP8, eP8, lP8]
        mMinQ[h] = query->host<T>()[h * mHeadDim];
        mMaxQ[h] = query->host<T>()[h * mHeadDim];
        for (int i = 0; i < seq_len; i++) {
            T * query_src = query->host<T>() + i * mNumHead * mHeadDim + h * mHeadDim;
            for (int j = 0; j < mHeadDim; j++) {
                mMinQ[h] = ALIMIN(mMinQ[h], query_src[j]);
                mMaxQ[h] = ALIMAX(mMaxQ[h], query_src[j]);
            }
        }
        mQueryScale[h] = (mMaxQ[h] - mMinQ[h]) / 255.0f;
        mQueryZeroPoint[h] = -255.0f * mMinQ[h] / (mMaxQ[h] - mMinQ[h]) - 128.0;
        for (int i = 0; i < seq_len; i++) {
            T * query_src = query->host<T>() + i * mNumHead * mHeadDim + h * mHeadDim;
            float sumQ = 0;
            int out_index = i / eP8;
            int in_index  = i % eP8;
            for (int j = 0; j < mHeadDim; j++) {
                int a = j / lP8;
                int b = j % lP8;
                int quant_res = (int)roundf(query_src[j] / mQueryScale[h] + mQueryZeroPoint[h]);
                sumQ += quant_res;
                *((int8_t*)pack_q + out_index * UP_DIV(mHeadDim, lP8) * eP8 * lP8 + a * eP8 * lP8 + in_index * lP8 + b) = quant_res;
            }
            *((float*)sum_q + out_index * eP8 + in_index) = sumQ * mQueryScale[h];
        }
    }
    else {
        T * query_src = query->host<T>();
        T * query_dst = reinterpret_cast<T*>(pack_q);    
        for (int i = 0; i < seq_len; i++) {
            int out_index = i / eP;
            int in_index  = i % eP;
            for (int j = 0; j < mHeadDim; j++) {
                query_dst[out_index * mHeadDim * eP + j * eP + in_index] = query_src[i * mNumHead * mHeadDim + h * mHeadDim + j] * q_scale;
            }
        }
    }
}

template <typename T>
void CPUAttention::unpack_QK(float * unpack_qk_dst, char * pack_qk_src, int seq_len, int kv_seq_len) {
    float * dst = unpack_qk_dst;
    T * src = (T *)(pack_qk_src);    
    // [kv_seq_len/unit, seq_len, unit] -> [seq_len, kv_seq_len]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_seq_len; j++) {
            int out_index = j / unit;
            int in_index  = j % unit;
            dst[i * kv_seq_len + j] = src[out_index * seq_len * unit + i * unit + in_index];
        }
    }
}

template <typename T>
static void pack_QK(char * pack_qk_dst, float * qk_src, int seq_len, int kv_seq_len, int eP) {
    T * dst = reinterpret_cast<T*>(pack_qk_dst);
    float * src = reinterpret_cast<float*>(qk_src);
    // [seq_len, kv_seq_len] -> [seq_len/eP, kv_seq_len, eP]
    for (int i = 0; i < seq_len; i++) {
        int out_index = i / eP;
        int in_index  = i % eP;
        for (int j = 0; j < kv_seq_len; j++) {
            dst[out_index * kv_seq_len * eP + j * eP + in_index] = src[i * kv_seq_len + j];
        }
    }
}

template <typename T>
static void mask_QK(float * unpack_qk, int seq_len, int kv_seq_len, float mScale, float min_val, int * mask_ptr, bool float_mask) {
    if (seq_len == 1) {
        for (int i = 0; i < kv_seq_len; i++) {
            unpack_qk[i] = unpack_qk[i] * mScale;
        }
    } else if (float_mask) {
        // float mask
        T* fpmask_ptr = reinterpret_cast<T*>(mask_ptr);
        for (int i = 0; i < seq_len * kv_seq_len; i++) {
            unpack_qk[i] = unpack_qk[i] * mScale + fpmask_ptr[i];
        }
    } else {
        // int mask
        for (int i = 0; i < seq_len * kv_seq_len; i++) {
            if (mask_ptr[i]) {
                unpack_qk[i] = unpack_qk[i] * mScale;
            } else {
                unpack_qk[i] = min_val;
            }
        }
    }
}

static void softmax_QK(float* softmax_qk_addr, float* unpack_qk_addr, int seq_len, int kv_seq_len) {
    for (int i = 0; i < seq_len; i++) {  // softmax each row
        MNNSoftmax(softmax_qk_addr + i * kv_seq_len, unpack_qk_addr + i * kv_seq_len, kv_seq_len);
    }
}

template <typename T>
static void unpack_QKV(char* pack_qkv, char* unpack_qkv, int mNumHead, int mHeadDim, int unit, int seq_len) {
    auto src_ptr = reinterpret_cast<T*>(pack_qkv);
    auto dst_ptr = reinterpret_cast<T*>(unpack_qkv);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < mHeadDim; j++) {
            int a = j / unit;
            int b = j % unit;
            dst_ptr[i * mNumHead * mHeadDim + j] = src_ptr[a * seq_len * unit + i * unit + b];
        }
    }
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    unit  = core->pack;
    bytes = core->bytes;
    int qkvQuantOptions = static_cast<CPUBackend *>(backend())->getRuntime()->hint().qkvQuantOption;
    mUseGemmInt8 = (qkvQuantOptions == 4);
    if (mUseGemmInt8) {
        static_cast<CPUBackend*>(backend())->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);
    }
    auto query = inputs[0];
    auto key   = inputs[1];
    int seq_len = query->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mKVCacheManager->onResize(mKvNumHead, mHeadDim);
    if (mUseGemmInt8) {
        mPackQ.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seq_len, eP8), UP_DIV(mHeadDim, lP8), eP8 * lP8}));
        mSumQ.reset(Tensor::createDevice<int32_t>({mThreadNum, UP_DIV(seq_len, eP8), eP8}));
        mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mHeadDim, unit), seq_len, unit}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mSumQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mSumQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);
        mMinQ.resize(mNumHead);
        mMaxQ.resize(mNumHead);
        mQueryScale.resize(mNumHead);
        mQueryZeroPoint.resize(mNumHead);
    } else {
        mPackQ.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), mHeadDim, eP}));
        mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mHeadDim, unit), seq_len, unit}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core  = static_cast<CPUBackend *>(backend())->functions();
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    bool float_mask = (mask->getType() == halide_type_of<float>());
    int mask_seqlen = mask->length(2);
    int mask_kvlen  = mask->length(3);
    int seq_len = query->length(1);
    MNN_ASSERT(seq_len == mask_seqlen);
    int tileCount = UP_DIV(mNumHead, mThreadNum);
    int group_size = mNumHead / mKvNumHead;
    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = 1.0 / sqrt(mHeadDim);
    float q_scale = 1.0;
    if (bytes == 2) {
        // reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
        FLOAT16_T minValue;
        FLOAT16_T maxValue;
        core->MNNCountMaxMinValue(query->host<float>(), (float*)(&minValue), (float*)(&maxValue), query->elementSize());
        float maxV = maxValue;
        float minV = minValue;
        float absMax = ALIMAX(fabsf(maxV), fabsf(minV));
        if (absMax > 1.0f) {
            q_scale = 1.0f / absMax;
        }
        mScale /= q_scale;
    }

    if (mMeta->previous == mMeta->remove) {
        mKVCacheManager->onClear();
        mKVCacheManager->onAlloc(mMeta->add);
    } else {
        MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
        mKVCacheManager->onRealloc(mMeta);
    }
    // Add the new kv to the kvcache
    mKVCacheManager->onPushBack(key, value);
    int kv_seq_len  = mKVCacheManager->kvLength();
    int max_len = mKVCacheManager->maxLength();
    bool quant_key = mKVCacheManager->config()->mQuantKey;
    bool quant_value = mKVCacheManager->config()->mQuantValue;
    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> packQK(Tensor::createDevice<float>({mThreadNum, UP_DIV(kv_seq_len, unit), seq_len, unit}));
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> softmMaxQ(Tensor::createDevice<int32_t>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> newPackQK(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), kv_seq_len, eP}));
    std::shared_ptr<Tensor> dequantV(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim, hP), kv_seq_len, hP}));
    backend()->onAcquireBuffer(packQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC);
    if (quant_value) {
        backend()->onAcquireBuffer(dequantV.get(), Backend::STATIC);
        mKVCacheManager->onDequantValue(dequantV.get());
    }

    std::function<void(int)> mCompute = [=](int tId) {
        auto pack_q      = mPackQ->host<char>() + tId * UP_DIV(seq_len, eP) * mHeadDim * eP * bytes;
        auto pack_qk     = packQK->host<char>() + tId * UP_DIV(kv_seq_len, unit) * seq_len * unit * bytes;
        char * sum_q     = nullptr;
        auto unpack_qk   = unpackQK->host<float>() + tId * seq_len * kv_seq_len;
        auto softmax_qk  = softmMaxQ->host<float>() + tId * seq_len * kv_seq_len;
        auto new_pack_qk = newPackQK->host<char>() + tId * UP_DIV(seq_len, eP) * kv_seq_len * eP * bytes;
        auto pack_qkv    = mPackQKV->host<char>() + tId * UP_DIV(mHeadDim, unit) * seq_len * unit * bytes;
        auto QxK         = quant_key ? core->MNNPackedMatMul_int8 : core->MNNPackedMatMul;
        auto QxK_remain  = quant_key ? core->MNNPackedMatMulRemain_int8 : core->MNNPackedMatMulRemain;
        int  head_index  = tId * tileCount;
        if (mUseGemmInt8) {
            pack_q  = mPackQ->host<char>() + tId * UP_DIV(seq_len, eP8) * UP_DIV(mHeadDim, lP8) * eP8 * lP8;
            sum_q   = mSumQ->host<char>() + tId * UP_DIV(seq_len, eP8) * eP8 * 4;
        }
        for (int h = head_index; h < head_index + tileCount && h < mNumHead; h++) {
            int    kv_h            = h / group_size;
            char * key_addr        = mKVCacheManager->addrOfKey(kv_h);
            char * scale_addr      = mKVCacheManager->addrOfScale(kv_h);
            char * zero_point_addr = mKVCacheManager->addrOfZeroPoint(kv_h);
            char * key_sum_addr    = mKVCacheManager->addrOfKeySum(kv_h);
            char * value_addr      = quant_value ? (dequantV->host<char>() + kv_h * UP_DIV(mHeadDim, hP) * kv_seq_len * hP * bytes) : mKVCacheManager->addrOfValue(kv_h);
            if (bytes == 2) {
                pack_query<FLOAT16_T>(query, pack_q, sum_q, seq_len, h, q_scale);
            } else {
                pack_query<float>(query, pack_q, sum_q, seq_len, h, q_scale);
            }
            // query @ key
            if (mUseGemmInt8) {
                auto GemmInt8Kernel = static_cast<CPUBackend*>(backend())->int8Functions()->Int8GemmKernel;
                if (bytes == 2 && unit == 8) {
                    GemmInt8Kernel = static_cast<CPUBackend*>(backend())->int8Functions()->MNNGemmInt8AddBiasScale_Unit_FP16;
                }
                std::vector<float> postScale(ROUND_UP(kv_seq_len, hP8), 0.0f);
                for (int i = 0; i < kv_seq_len; i++) {
                    postScale[i] = ((float*)scale_addr)[i] * mQueryScale[h] * q_scale;
                }
                std::vector<float> weightQuantBias(ROUND_UP(kv_seq_len, hP8), 0.0f);
                for (int i = 0; i < kv_seq_len; i++) {
                    weightQuantBias[i] = -((float*)scale_addr)[i] * ((float*)zero_point_addr)[i] * q_scale;
                }
                std::vector<float> biasFloat(ROUND_UP(kv_seq_len, hP8), 0.0f);
                for (int i = 0; i < kv_seq_len; i++) {
                    biasFloat[i] = -mQueryScale[h] * mQueryZeroPoint[h] * ((float*)key_sum_addr)[i] * q_scale;
                }
                QuanPostTreatParameters post;
                post.bias = nullptr;
                post.biasFloat = biasFloat.data();
                post.blockNum = 1;
                post.extraBias = nullptr;
                post.extraScale = nullptr;
                post.fp32minmax = nullptr;
                post.scale = postScale.data();
                post.useInt8 = false;
                post.weightQuanBias = weightQuantBias.data();
                int N = UP_DIV(seq_len, eP8);
                for (int i = 0; i < N; i++) {
                    int realcount = ALIMIN(eP8, seq_len - i * eP8);
                    post.srcKernelSum = (float*)((char*)sum_q + i * eP8 * 4);
                    GemmInt8Kernel(
                        (int8_t*)pack_qk + i * eP8 * unit * bytes,
                        (int8_t*)pack_q + i * ROUND_UP(mHeadDim, lP8) * eP8,
                        (int8_t*)key_addr,
                        UP_DIV(mHeadDim, lP8),
                        seq_len * unit * bytes,
                        UP_DIV(kv_seq_len, unit),
                        &post,
                        realcount
                    );
                }
            }
            else {
                int loop_e = seq_len / eP;
                int remain = seq_len % eP;
                size_t shapeParameters[7] = {(size_t)eP * bytes, (size_t)mHeadDim, (size_t)kv_seq_len, (size_t)seq_len * unit * bytes, 0, 0, 0};
                for (int i = 0 ; i < loop_e; i++) {
                    QxK((float*)(pack_qk + (i * eP * unit) * bytes), (float*)(pack_q + (i * mHeadDim * eP) * bytes), (float*)key_addr, shapeParameters, nullptr, nullptr, (float*)scale_addr, (float*)zero_point_addr);
                }
                QxK_remain((float*)(pack_qk + (loop_e * eP * unit) * bytes), (float*)(pack_q + (loop_e * mHeadDim * eP) * bytes), (float*)key_addr, remain, shapeParameters, nullptr, nullptr, (float*)scale_addr, (float*)zero_point_addr);
            }
            // qk: [kv_seq_len/unit, seq_len, unit] -> [seq_len, kv_seq_len] -> [seq_len/eP, kv_seq_len, eP]
            if(bytes == 2) {
                unpack_QK<FLOAT16_T>(unpack_qk, pack_qk, seq_len, kv_seq_len);
                mask_QK<FLOAT16_T>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                pack_QK<FLOAT16_T>(new_pack_qk, softmax_qk, seq_len, kv_seq_len, eP);
            } else {
                unpack_QK<float>(unpack_qk, pack_qk, seq_len, kv_seq_len);
                mask_QK<float>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                pack_QK<float>(new_pack_qk, softmax_qk, seq_len, kv_seq_len, eP);
            }
            // qk @ v
            size_t shapeParameters[7] = {(size_t)eP * bytes, (size_t)kv_seq_len, (size_t)mHeadDim, (size_t)seq_len * unit * bytes, 0, 0, 0};
            shapeParameters[5] = quant_value ? 0 : (max_len - kv_seq_len) * hP * bytes;
            int loop_e = seq_len / eP;
            int remain = seq_len % eP;
            for (int i = 0 ; i < loop_e; i++) {
                core->MNNPackedMatMul((float*)(pack_qkv + (i * eP * unit) * bytes), (float*)(new_pack_qk + (i * kv_seq_len * eP) * bytes), (float*)value_addr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
            }
            core->MNNPackedMatMulRemain((float*)(pack_qkv + (loop_e * eP * unit) * bytes), (float*)(new_pack_qk + (loop_e * kv_seq_len * eP) * bytes), (float*)value_addr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);
            // unpack: [head_dim/unit, seq_len, unit] -> [seq_len, num_head, head_dim]
            auto dst_ptr = outputs[0]->host<char>() + h * mHeadDim * bytes;
            if (bytes == 2) {
                unpack_QKV<int16_t>(pack_qkv, dst_ptr, mNumHead, mHeadDim, unit, seq_len);
            } else {
                unpack_QKV<float>(pack_qkv, dst_ptr, mNumHead, mHeadDim, unit, seq_len);
            }
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    backend()->onReleaseBuffer(packQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    if (quant_value){
        backend()->onReleaseBuffer(dequantV.get(), Backend::STATIC);
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    tmp->mKVCacheManager = mKVCacheManager;
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend *backend, bool kv_cache) : Execution(backend), mKVCache(kv_cache) {
    mMeta = (KVMeta*)(backend->getRuntime()->pMeta);
    if (mKVCache) {
        mPackQ.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
        mPackQKV.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
        MNN::KVCacheManager::KVCacheConfig kvconfig;
        int qkvQuantOptions = static_cast<CPUBackend *>(backend)->getRuntime()->hint().qkvQuantOption;
        kvconfig.mUseInt8Kernel = (qkvQuantOptions == 4);
        kvconfig.mQuantKey   = (qkvQuantOptions == 4) || (qkvQuantOptions & 1);
        kvconfig.mQuantValue = (qkvQuantOptions == 4) || ((qkvQuantOptions >> 1) & 1);
        kvconfig.mKVCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheDirPath;
        kvconfig.mKVCacheSizeLimit = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheSizeLimit;
        kvconfig.mExpandChunk = 64;
        mKVCacheManager.reset(new KVCacheManager(backend, kvconfig));
    }
}

CPUAttention::~CPUAttention() {
    
}

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        return new CPUAttention(backend, param->kv_cache());
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_Attention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
