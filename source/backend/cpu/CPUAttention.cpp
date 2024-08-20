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

// reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
#define FP16_QSCALE 0.5

#define FP8_E5M2

namespace MNN {

#if defined FP8_E5M2  // E5M2 : [S E E E E E M M]
typedef uint8_t fp8_t;
static inline fp8_t fp16_to_fp8(FLOAT16_T x) {
    return *((fp8_t *)(&x) + 1);
}
static FLOAT16_T fp8_to_fp16(fp8_t x) {
    uint16_t rawData = 0;
    rawData |= (uint16_t)x << 8;
    return *((FLOAT16_T *)(&rawData));
}
static inline fp8_t float_to_fp8(float x) {
    uint32_t rawData = *((uint32_t *)(&x));
    int sign = (rawData >> 31) & 1U;
    int exp = (int)((rawData >> 23) & 0x0ffU) - 127;
    if (exp < -16)
        exp = -16;
    if (exp > 15)
        exp = 15;
    exp += 16;    // exp [-16, 15] ==> [0, 31]
    int mant = (rawData >> 21) & 3U;
    return (sign << 7) | (exp << 2) | mant;
}
static inline float fp8_to_float(fp8_t x) {
    uint32_t sign = (x >> 7) & 1U;
    uint32_t exp = (int)((x >> 2) & 0x1fU) - 16 + 127;
    uint32_t mant = (x & 3U) << 21;
    uint32_t rawData = (sign << 31) | (exp << 23) | mant;
    return *((float *)(&rawData));
}
#elif defined FP8_E4M3  // E4M3: [S E E E E M M M]
typedef uint8_t fp8_t;
static inline fp8_t fp16_to_fp8(FLOAT16_T x) {
    uint16_t rawData = *((uint16_t *)(&x));
    int sign = (rawData >> 15) & 1U;
    int exp  = (int)((rawData >> 10) & 0x1fU) - 15;
    if (exp < -8)
        exp = -8;
    if (exp > 7)
        exp = 7;
    exp += 8;     // exp [-8, 7] ==> [0, 15]
    int mant = (rawData >> 7) & 7U;
    return (sign << 7) | (exp << 3) | mant;
}
static FLOAT16_T fp8_to_fp16(fp8_t x) {
    uint32_t sign = (x >> 7) & 1U;
    uint32_t exp = (int)((x >> 3) & 0x0fU) - 8 + 15;
    uint32_t mant = (x & 7U) << 7;
    uint16_t rawData = (sign << 15) | (exp << 10) | mant;
    return *((FLOAT16_T *)(&rawData));
}
static inline fp8_t float_to_fp8(float x) {
    uint32_t rawData = *((uint32_t *)(&x));
    int sign = (rawData >> 31) & 1U;
    int exp = (int)((rawData >> 23) & 0x0ffU) - 127;
    if (exp < -8)
        exp = -8;
    if (exp > 7)
        exp = 7;
    exp += 8;     // exp [-8, 7] ==> [0, 15]
    int mant = (rawData >> 20) & 7U;
    return (sign << 7) | (exp << 3) | mant;
}
static inline float fp8_to_float(fp8_t x) {
    uint32_t sign = (x >> 7) & 1U;
    uint32_t exp = (int)((x >> 3) & 0x0fU) - 8 + 127;
    uint32_t mant = (x & 7U) << 20;
    uint32_t rawData = (sign << 31) | (exp<< 23) | mant;
    return *((float *)(&rawData));
}
#else
// Do not support fp8
#endif  // fp8 format definition

static int nearestInt(float x) {
    return x < 0 ? -nearestInt(-x) : (int)(x + 0.5f);
}

template <typename T>
static void pack_query(Tensor* query, char* pack_q, int mNumHead, int mHeadDim, int eP, int seq_len, int h, float q_scale) {
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

template <typename T>
static void pack_key(Tensor* key, char* pack_key, int mPastLength, int seq_len, int mKvNumHead, int mHeadDim, int hP, int kv_h, char* scale, char* zero_point, bool quant) {
    if (quant) {  // Quantize the keys
        auto key_src = key->host<T>();
        auto key_dst = reinterpret_cast<int8_t*>(pack_key);
        auto scale_dst = reinterpret_cast<T*>(scale);
        auto zeroPoint_dst = reinterpret_cast<T*>(zero_point);
        for (int i = 0; i < seq_len; i++) {
            float minKey = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + 0];
            float maxKey = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + 0];
            for (int j = 1; j < mHeadDim; j++) {
                auto key = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
                minKey = ALIMIN(minKey, key);
                maxKey = ALIMAX(maxKey, key);
            }
            int out_index = (mPastLength + i) / hP;
            int in_index  = (mPastLength + i) % hP;
            scale_dst[out_index * hP + in_index] = (maxKey - minKey) / 255.0f;
            zeroPoint_dst[out_index * hP + in_index] = 128.0f * (maxKey - minKey) / 255.0f + minKey;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = nearestInt((key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j] - minKey) / (maxKey - minKey) * 255 - 128);
            }
        }
    }
    else {  // Do not quantize the keys
        auto key_src = key->host<T>();
        auto key_dst = reinterpret_cast<T*>(pack_key);
        for (int i = 0; i < seq_len; i++) {
            int out_index = (mPastLength + i) / hP;
            int in_index  = (mPastLength + i) % hP;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
            }
        }
    }
}



template <typename T>
static void pack_value(Tensor* value, char* pack_value, int mMaxLength, int mPastLength, int seq_len, int mKvNumHead, int mHeadDim, int hP, int kv_h, bool quant) {
    if (quant) {  // Quantize the values to fp8
        T * value_src = value->host<T>();
        fp8_t * value_dst = reinterpret_cast<fp8_t*>(pack_value);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                auto origin = value_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
                if (sizeof(T) == 2)
                    value_dst[out_index * mMaxLength * hP + (mPastLength + i) * hP + in_index] = fp16_to_fp8(origin);
                else
                    value_dst[out_index * mMaxLength * hP + (mPastLength + i) * hP + in_index] = float_to_fp8(origin);
            }
        }
    }
    else {  // Do not quantize the values
        T * value_src = value->host<T>();
        T * value_dst = reinterpret_cast<T*>(pack_value);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                value_dst[out_index * mMaxLength * hP + (mPastLength + i) * hP + in_index] = value_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
            }
        }
    }
}

void dequant_value_float(char * dst, char * src, int mHeadDim, int kv_seq_len, int hP, int mMaxLength) {
    fp8_t * qv = (fp8_t *)src;
    float * dqv = (float *)dst;
    for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
        for (int j = 0; j < kv_seq_len; j++) {
            for (int k = 0; k < hP; k++) {
                dqv[i * kv_seq_len * hP + j * hP + k] = fp8_to_float(qv[i * mMaxLength * hP + j * hP + k]);
            }
        }
    }
}

void dequant_value_fp16(char * dst, char * src, int mHeadDim, int kv_seq_len, int hP, int mMaxLength) {
    fp8_t * qv = (fp8_t *)src;
    FLOAT16_T * dqv = (FLOAT16_T *)dst;
    for (int i = 0; i < UP_DIV(mHeadDim, hP); i++) {
        for (int j = 0; j < kv_seq_len; j++) {
            for (int k = 0; k < hP; k++) {
                dqv[i * kv_seq_len * hP + j * hP + k] = fp8_to_fp16(qv[i * mMaxLength * hP + j * hP + k]);
            }
        }
    }
}

template <typename T>
static void unpack_QK(float * unpack_qk_dst, char * pack_qk_src, int seq_len, int kv_seq_len, int unit) {
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

void CPUAttention::allocKVCache(int kv_seq_len, bool quantKey, bool quantValue) {
    if (!mKVCache) {
        return;
    }
    mResource->mMaxLength = kv_seq_len + mResource->mExpandChunk;
    if (quantKey) {
        mResource->mPastKey.reset(Tensor::createDevice<int8_t>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP}));
        mResource->mDequantKeyScale.reset(Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), 1, hP}));
        mResource->mDequantKeyZeroPoint.reset(Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), 1, hP}));
        backend()->onAcquireBuffer(mResource->mPastKey.get(), Backend::STATIC);
        backend()->onAcquireBuffer(mResource->mDequantKeyScale.get(), Backend::STATIC);
        backend()->onAcquireBuffer(mResource->mDequantKeyZeroPoint.get(), Backend::STATIC);
    } else {
        mResource->mPastKey.reset(Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP}));
        backend()->onAcquireBuffer(mResource->mPastKey.get(), Backend::STATIC);
    }
    if (quantValue) {
        mResource->mPastValue.reset(Tensor::createDevice<fp8_t>({mResource->mKvNumHead, UP_DIV(mResource->mHeadDim, hP), mResource->mMaxLength, hP}));
        backend()->onAcquireBuffer(mResource->mPastValue.get(), Backend::STATIC);
    } else {
        mResource->mPastValue.reset(Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mHeadDim, hP), mResource->mMaxLength, hP}));
        backend()->onAcquireBuffer(mResource->mPastValue.get(), Backend::STATIC);
    }
}

void CPUAttention::reallocKVCache(int kv_seq_len, bool quantKey, bool quantValue) {
    if (!mKVCache || kv_seq_len <= mResource->mMaxLength) {
        return;
    }
    int oldMaxLength = mResource->mMaxLength;
    mResource->mMaxLength = kv_seq_len + mResource->mExpandChunk;
    if (quantKey) {
        auto new_key = Tensor::createDevice<int8_t>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP});
        auto new_scale = Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), 1, hP});
        auto new_zeroPoint = Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), 1, hP});
        backend()->onAcquireBuffer(new_key, Backend::STATIC);
        backend()->onAcquireBuffer(new_scale, Backend::STATIC);
        backend()->onAcquireBuffer(new_zeroPoint, Backend::STATIC);
        for (int h = 0; h < mResource->mKvNumHead; h++) {
            memcpy(new_key->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP,
                    mResource->mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mResource->mHeadDim * hP,
                    UP_DIV(oldMaxLength, hP) * mResource->mHeadDim * hP);
            memcpy(new_scale->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * hP * bytes,
                    mResource->mDequantKeyScale->host<char>() + h * UP_DIV(oldMaxLength, hP) * hP * bytes,
                    UP_DIV(oldMaxLength, hP) * hP * bytes);
            memcpy(new_zeroPoint->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * hP * bytes,
                    mResource->mDequantKeyZeroPoint->host<char>() + h * UP_DIV(oldMaxLength, hP) * hP * bytes,
                    UP_DIV(oldMaxLength, hP) * hP * bytes);
        }
        mResource->mPastKey.reset(new_key);
        mResource->mDequantKeyScale.reset(new_scale);
        mResource->mDequantKeyZeroPoint.reset(new_zeroPoint);
    }
    else {
        auto new_key = Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP});
        backend()->onAcquireBuffer(new_key, Backend::STATIC);
        for (int h = 0; h < mResource->mKvNumHead; h++) {
            memcpy(new_key->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes,
                    mResource->mPastKey->host<char>() + h * UP_DIV(oldMaxLength, hP) * mResource->mHeadDim * hP * bytes,
                    UP_DIV(oldMaxLength, hP) * mResource->mHeadDim * hP * bytes);
        }
        mResource->mPastKey.reset(new_key);
    }
    if (quantValue) {
        auto new_value = Tensor::createDevice<fp8_t>({mResource->mKvNumHead, UP_DIV(mResource->mHeadDim, hP), mResource->mMaxLength, hP});
        backend()->onAcquireBuffer(new_value, Backend::STATIC);
        for (int h = 0; h < mResource->mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mResource->mHeadDim, hP); i++) {
                memcpy(new_value->host<char>() + (h * UP_DIV(mResource->mHeadDim, hP) + i) * mResource->mMaxLength * hP,
                        mResource->mPastValue->host<char>() + (h * UP_DIV(mResource->mHeadDim, hP) + i) * oldMaxLength * hP,
                        oldMaxLength * hP);
            }
        }
        mResource->mPastValue.reset(new_value);
    }
    else {
        auto new_value = Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mHeadDim, hP), mResource->mMaxLength, hP});
        backend()->onAcquireBuffer(new_value, Backend::STATIC);
        for (int h = 0; h < mResource->mKvNumHead; h++) {
            for (int i = 0; i < UP_DIV(mResource->mHeadDim, hP); i++) {
                memcpy(new_value->host<char>() + (h * UP_DIV(mResource->mHeadDim, hP) + i) * mResource->mMaxLength * hP * bytes,
                        mResource->mPastValue->host<char>() + (h * UP_DIV(mResource->mHeadDim, hP) + i) * oldMaxLength * hP * bytes,
                        oldMaxLength * hP * bytes);
            }
        }
        mResource->mPastValue.reset(new_value);
    }
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    unit  = core->pack;
    bytes = core->bytes;
    auto query = inputs[0];
    auto key   = inputs[1];
    int seq_len = query->shape()[1];
    mResource->mNumHead = query->shape()[2];
    mResource->mHeadDim = query->shape()[3];
    mResource->mKvNumHead = key->shape()[2];
    mPackQ.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), mResource->mHeadDim, eP}));
    mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
    backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto mask_shape = mask->shape();
    bool float_mask = (mask->getType() == halide_type_of<float>());
    int mask_seqlen = mask_shape[2];
    int mask_kvlen  = mask_shape[3];
    int seq_len = query->shape()[1];
    MNN_ASSERT(seq_len == mask_seqlen);
    mIsPrefill = (seq_len > 1);
    // isPrefill and mask is Square Matrix, is FirstPrefill
    mIsFirstPrefill = mIsPrefill && (mask_kvlen == mask_seqlen);
    int tileCount = UP_DIV(mResource->mNumHead, mThreadNum);
    int group_size = mResource->mNumHead / mResource->mKvNumHead;

    // 0: do not quant kv
    // 1: only quant k
    // 2: only quant v
    // 3: quant kv
    int quantKV = static_cast<CPUBackend *>(backend())->getRuntime()->hint().kvcacheQuantOption;
    bool quantKey = (quantKV & 1) == 1;
    bool quantValue = ((quantKV >> 1) & 1) == 1;

    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = 1.0 / sqrt(mResource->mHeadDim);
    float q_scale = 1.0;
    if (bytes == 2) {
        q_scale = FP16_QSCALE;
        mScale /= q_scale;
    }

    if (mIsPrefill) {
        // Only reset the kvcache in the first prefill, but keep the kvcache in subsequent prefill
        if (mIsFirstPrefill) {
            mResource->mPastLength = 0;
            allocKVCache(seq_len, quantKey, quantValue);
        } else {
            reallocKVCache(mResource->mPastLength + seq_len, quantKey, quantValue);
        }
    } else { // Decode
        reallocKVCache(mResource->mPastLength + 1, quantKey, quantValue);
    }
    int kv_seq_len  = mResource->mPastLength + seq_len;

    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> packQK(Tensor::createDevice<float>({mThreadNum, UP_DIV(kv_seq_len, unit), seq_len, unit}));
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> softmaxQK(Tensor::createDevice<int>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> newPackQK(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), kv_seq_len, eP}));
    std::shared_ptr<Tensor> dequantV(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, hP), kv_seq_len, hP}));
    backend()->onAcquireBuffer(packQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmaxQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC);
    if (quantValue) {
        backend()->onAcquireBuffer(dequantV.get(), Backend::STATIC);
    }

    std::function<void(int)> mCompute = [=](int tId) {
        auto pack_q      = mPackQ->host<char>() + tId * UP_DIV(seq_len, eP) * mResource->mHeadDim * eP * bytes;
        auto pack_qk     = packQK->host<char>() + tId * UP_DIV(kv_seq_len, unit) * seq_len * unit * bytes;
        auto unpack_qk   = unpackQK->host<float>() + tId * seq_len * kv_seq_len;
        auto softmax_qk   = softmaxQK->host<float>() + tId * seq_len * kv_seq_len;
        auto new_pack_qk = newPackQK->host<char>() + tId * UP_DIV(seq_len, eP) * kv_seq_len * eP * bytes;
        auto pack_qkv    = mPackQKV->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;
        int head_index   = tId * tileCount;
        for (int h = head_index; h < head_index + tileCount && h < mResource->mNumHead; h++) {
            int    kv_h                 = h / group_size;
            char * key_dst              = nullptr;
            char * key_scale_dst        = nullptr;
            char * key_zero_point_dst   = nullptr;
            char * value_dst            = nullptr;
            if (quantKey) {
                key_dst = mResource->mPastKey->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP;
                key_scale_dst = mResource->mDequantKeyScale->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * 1 * hP * bytes;
                key_zero_point_dst = mResource->mDequantKeyZeroPoint->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * 1 * hP * bytes;
            } else {
                key_dst   = mResource->mPastKey->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes;
            }
            if (quantValue) {
                value_dst = mResource->mPastValue->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * mResource->mMaxLength * hP;
            } else {
                value_dst = mResource->mPastValue->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * mResource->mMaxLength * hP * bytes;
            }
            // pack for matmul
            if (bytes == 2) {
                pack_query<FLOAT16_T>(query, pack_q, mResource->mNumHead, mResource->mHeadDim, eP, seq_len, h, q_scale);
                pack_key<FLOAT16_T>(key, key_dst, mResource->mPastLength, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, key_scale_dst, key_zero_point_dst, quantKey);
                pack_value<FLOAT16_T>(value, value_dst, mResource->mMaxLength, mResource->mPastLength, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, quantValue);
            } else {
                pack_query<float>(query, pack_q, mResource->mNumHead, mResource->mHeadDim, eP, seq_len, h, q_scale);
                pack_key<float>(key, key_dst, mResource->mPastLength, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, key_scale_dst, key_zero_point_dst, quantKey);
                pack_value<float>(value, value_dst, mResource->mMaxLength, mResource->mPastLength, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, quantValue);
            }
            // query @ key
            int loop_e = seq_len / eP;
            int remain = seq_len % eP;
            for (int i = 0 ; i < loop_e; i++) {
                size_t shapeParameters[7];
                size_t* parameters = shapeParameters;
                parameters[0] = eP * bytes;
                parameters[1] = mResource->mHeadDim;
                parameters[2] = kv_seq_len;
                parameters[3] = seq_len * unit * bytes;
                parameters[4] = 0;
                parameters[5] = 0;
                parameters[6] = 0;
                if (quantKey) {
                    core->MNNPackedMatMul_int8(
                        (float*)(pack_qk + (i * eP * unit) * bytes),
                        (float*)(pack_q + (i * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        parameters, nullptr, nullptr,
                        (float*)key_scale_dst, (float*)key_zero_point_dst
                    );
                } else {
                    core->MNNPackedMatMul(
                        (float*)(pack_qk + (i * eP * unit) * bytes),
                        (float*)(pack_q + (i * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        parameters, nullptr, nullptr,
                        nullptr, nullptr
                    );
                }
            }
            {
                size_t shapeParameters[7];
                size_t* parameters = shapeParameters;
                parameters[0] = eP * bytes;
                parameters[1] = mResource->mHeadDim;
                parameters[2] = kv_seq_len;
                parameters[3] = seq_len * unit * bytes;
                parameters[4] = 0;
                parameters[5] = 0;
                parameters[6] = 0;
                if (quantKey) {
                    core->MNNPackedMatMulRemain_int8(
                        (float*)(pack_qk + (loop_e * eP * unit) * bytes),
                        (float*)(pack_q + (loop_e * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        remain, parameters, nullptr, nullptr,
                        (float*)key_scale_dst, (float*)key_zero_point_dst
                    );
                } else {
                    core->MNNPackedMatMulRemain(
                        (float*)(pack_qk + (loop_e * eP * unit) * bytes),
                        (float*)(pack_q + (loop_e * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        remain, parameters, nullptr, nullptr,
                        nullptr, nullptr
                    );
                }
            }
            if(bytes == 2) {
                // unpack qk: [kv_seq_len/unit, seq_len, unit] -> [seq_len, kv_seq_len]
                unpack_QK<FLOAT16_T>(unpack_qk, pack_qk, seq_len, kv_seq_len, unit);
                mask_QK<FLOAT16_T>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                // pack qk for qk @ v : [seq_len, kv_seq_len] -> [seq_len/eP, kv_seq_len, eP]
                pack_QK<FLOAT16_T>(new_pack_qk, softmax_qk, seq_len, kv_seq_len, eP);
            } else {
                unpack_QK<float>(unpack_qk, pack_qk, seq_len, kv_seq_len, unit);
                mask_QK<float>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                pack_QK<float>(new_pack_qk, softmax_qk, seq_len, kv_seq_len, eP);
            }
            // Dequantize values from fp8 to float
            if (quantValue) {
                char * qv = value_dst;
                char * dqv = dequantV->host<char>() + tId * UP_DIV(mResource->mHeadDim, hP) * kv_seq_len * hP * bytes;
                if (bytes == 2) {
                    dequant_value_fp16(dqv, qv, mResource->mHeadDim, kv_seq_len, hP, mResource->mMaxLength);
                } else {
                    dequant_value_float(dqv, qv, mResource->mHeadDim, kv_seq_len, hP, mResource->mMaxLength);
                }
                value_dst = dqv;
            }
            // qk @ v
            for (int i = 0 ; i < loop_e; i++) {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = kv_seq_len;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = quantValue ? 0 : (mResource->mMaxLength - kv_seq_len) * hP * bytes;
                core->MNNPackedMatMul(
                    (float*)(pack_qkv + (i * eP * unit) * bytes),
                    (float*)(new_pack_qk + (i * kv_seq_len * eP) * bytes),
                    (float*)value_dst, parameters,
                    nullptr, nullptr, nullptr, nullptr
                );
            }
            {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = kv_seq_len;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = quantValue ? 0 : (mResource->mMaxLength - kv_seq_len) * hP * bytes;
                core->MNNPackedMatMulRemain(
                    (float*)(pack_qkv + (loop_e * eP * unit) * bytes),
                    (float*)(new_pack_qk + (loop_e * kv_seq_len * eP) * bytes),
                    (float*)value_dst, remain, parameters,
                    nullptr, nullptr, nullptr, nullptr
                );
            }
            // unpack: [head_dim/unit, seq_len, unit] -> [seq_len, num_head, head_dim]
            auto dst_ptr = outputs[0]->host<char>() + h * mResource->mHeadDim * bytes;
            if (bytes == 2) {
                unpack_QKV<int16_t>(pack_qkv, dst_ptr, mResource->mNumHead, mResource->mHeadDim, unit, seq_len);
            } else {
                unpack_QKV<float>(pack_qkv, dst_ptr, mResource->mNumHead, mResource->mHeadDim, unit, seq_len);
            }
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    mResource->mPastLength += seq_len;
    backend()->onReleaseBuffer(packQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmaxQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    if (quantValue){
        backend()->onReleaseBuffer(dequantV.get(), Backend::STATIC);
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    tmp->mResource = mResource;
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend *backend, bool kv_cache) : Execution(backend) {
    mKVCache = kv_cache;
    mResource.reset(new Resource);
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

#endif