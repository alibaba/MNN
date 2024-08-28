//
//  CPUAttention.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <cstdio>
#include <initializer_list>
#include <limits>
#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include <MNN/StateCacheManager.hpp>
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

// key: [seq_len, kvnum_heads, head_dim], quantize over axis=-1, per-channel quantization.
template <typename T>
static void pack_key(Tensor* key, const std::vector<std::shared_ptr<StateCacheBlock>>& pack_key, int seq_len, int mKvNumHead, int mHeadDim, int hP, int kv_h, int first_dst_block_id, MNNStateCacheQuantType quantType) {
    int current_block_id = first_dst_block_id;
    int slot_num = pack_key[current_block_id]->getSlotNum();
    int block_size = pack_key[current_block_id]->getBlockSize();
    T* key_src = key->host<T>();
    if (quantType == MNNStateCacheQuantType::QuantKeyInt8 || \
        quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8 || \
        quantType == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
        // Quantize the keys
        int8_t*  key_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K)->host<int8_t>()[kv_h *  UP_DIV(block_size, hP) * mHeadDim * hP]);
        float* scale_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_SCALES)->host<float>()[kv_h * UP_DIV(block_size, hP) * hP]);
        float* zeroPoint_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_ZERO_POINTS)->host<float>()[kv_h * UP_DIV(block_size, hP) * hP]);
        for (int i = 0; i < seq_len; i++) {
            if (slot_num == block_size) {
                current_block_id++;
                slot_num = pack_key[current_block_id]->getSlotNum();
                key_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K)->host<int8_t>()[kv_h *  UP_DIV(block_size, hP) * mHeadDim * hP]);
                scale_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_SCALES)->host<float>()[kv_h * UP_DIV(block_size, hP) * hP]);
                zeroPoint_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_ZERO_POINTS)->host<float>()[kv_h * UP_DIV(block_size, hP) * hP]);
            } 
            float minKey = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + 0];
            float maxKey = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + 0];
            // per-head-token asymmetric quantization
            for (int j = 1; j < mHeadDim; j++) {
                auto key = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
                minKey = ALIMIN(minKey, key);
                maxKey = ALIMAX(maxKey, key);
            }
            int out_index = slot_num / hP;
            int in_index  = slot_num % hP;
            float scale = (maxKey - minKey) / 255.0f;
            float zero_point = 128.0f * scale + minKey;
            scale_dst[out_index * hP + in_index] = scale;
            zeroPoint_dst[out_index * hP + in_index] = zero_point;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = nearestInt((key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j] - minKey) / (maxKey - minKey) * 255 - 128);
            }
            slot_num++;
        }
    }
    else {  
        // Do not quantize the keys
        float* key_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::NoQuant::PAST_K)->host<float>()[kv_h *  UP_DIV(block_size, hP) * mHeadDim * hP]);
        for (int i = 0; i < seq_len; i++) {
            if (slot_num == block_size) {
                current_block_id++;
                slot_num = pack_key[current_block_id]->getSlotNum();
                key_dst = &(pack_key[current_block_id]->getTensor((int)StateCacheBlock::LAYOUT::NoQuant::PAST_K)->host<float>()[kv_h *  UP_DIV(block_size, hP) * mHeadDim * hP]);
            } 
            int out_index = slot_num / hP;
            int in_index  = slot_num % hP;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[out_index * mHeadDim * hP + j * hP + in_index] = key_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
            }
            slot_num++;
        }
    }
}

template <typename T>
static void pack_value(Tensor* value, const std::vector<std::shared_ptr<StateCacheBlock>>& pack_value, int seq_len, int mKvNumHead, int mHeadDim, int hP, int kv_h, int first_dst_block_id, MNNStateCacheQuantType quantType) {
    int current_block_id = first_dst_block_id;
    int slot_num = pack_value[current_block_id]->getSlotNum();
    int block_size = pack_value[current_block_id]->getBlockSize();
    T * value_src = value->host<T>();
    if (quantType == MNNStateCacheQuantType::QuantValueFp8 || \
        quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) {  // Quantize the values to fp8
        int tId = (quantType == MNNStateCacheQuantType::QuantValueFp8) ? ((int)StateCacheBlock::LAYOUT::QuantValueFp8::PAST_V) : 0;
        tId = (quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) ? ((int)StateCacheBlock::LAYOUT::QuantKeyInt8ValueFp8::PAST_V) : tId;
        fp8_t * value_dst = reinterpret_cast<fp8_t*>(&(pack_value[current_block_id]->getTensor(tId)->host<int8_t>()[kv_h *  UP_DIV(mHeadDim, hP) * block_size * hP]));
        for (int i = 0; i < seq_len; i++) {
            if (slot_num == block_size) {
                current_block_id++;
                slot_num = pack_value[current_block_id]->getSlotNum();
                value_dst = (fp8_t*)&(pack_value[current_block_id]->getTensor(tId)->host<int8_t>()[kv_h *  UP_DIV(mHeadDim, hP) * block_size * hP]);
            } 
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                auto origin = value_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
                if (sizeof(T) == 2)
                    value_dst[out_index * block_size * hP + slot_num * hP + in_index] = fp16_to_fp8(origin);
                else
                    value_dst[out_index * block_size * hP + slot_num * hP + in_index] = float_to_fp8(origin);
            }
            slot_num++;
        }
    } else if (quantType == MNNStateCacheQuantType::QuantValueInt8 || \
                quantType == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
        // int 8 quantization
        // Not Supported Yet!!!
    } else {  
        // Do not quantize the values
        int tId = (quantType == MNNStateCacheQuantType::NoQuant) ? ((int)StateCacheBlock::LAYOUT::NoQuant::PAST_V) : 0;
        tId = (quantType == MNNStateCacheQuantType::QuantKeyInt8) ? ((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_V) : tId;
        float* value_dst = &(pack_value[current_block_id]->getTensor(tId)->host<float>()[kv_h *  UP_DIV(mHeadDim, hP) * block_size * hP]);
        for (int i = 0; i < seq_len; i++) {
            if (slot_num == block_size) {
                current_block_id++;
                slot_num = pack_value[current_block_id]->getSlotNum();
                value_dst = &(pack_value[current_block_id]->getTensor(tId)->host<float>()[kv_h *  UP_DIV(mHeadDim, hP) * block_size * hP]);
            } 
            for (int j = 0; j < mHeadDim; j++) {
                int out_index = j / hP;
                int in_index  = j % hP;
                value_dst[out_index * block_size * hP + slot_num * hP + in_index] = value_src[i * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
            }
            slot_num++;
        }
    }
}



void dequant_value_float(char* dst, char* src, int size) {
    fp8_t * qv = (fp8_t *)src;
    float * dqv = (float *)dst;
    for (int i = 0; i < size; i++)
        dqv[i] = fp8_to_float(qv[i]);
}

void dequant_value_fp16(char* dst, char* src, int size) {
    fp8_t * qv = (fp8_t *)src;
    FLOAT16_T * dqv = (FLOAT16_T *)dst;
    for (int i = 0; i < size; i++)
        dqv[i] = fp8_to_fp16(qv[i]);
}

template <typename T>
static void unpack_QK(float * unpack_qk_dst, const std::vector<std::shared_ptr<Tensor>>& pack_qk, int seq_len, int kv_seq_len, int tId, int block_size, int unit) {
    float * dst = unpack_qk_dst;
    // UP_DIV(kv_seq_len, block_size): [block_size/unit, seq_len, unit] -> [seq_len, kv_seq_len]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < kv_seq_len; j++) {
            T* src = &(pack_qk[j/block_size]->host<T>()[tId*UP_DIV(block_size, unit)*seq_len*unit]);
            int out_index = (j%block_size)/unit;
            int in_index  = (j%block_size)%unit;
            dst[i * kv_seq_len + j] = src[out_index * seq_len * unit + i * unit + in_index];
            std::cout << dst[i * kv_seq_len + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// [mThreadNum, UP_DIV(seq_len, eP), block_size, eP]
template <typename T>
static void pack_QK(const std::vector<std::shared_ptr<Tensor>>& pack_qk_dst, float * qk_src, int seq_len, int kv_seq_len, int tId, int block_size, int last_block_slot_num, int eP) {
    float * src = reinterpret_cast<float*>(qk_src);
    // [seq_len, kv_seq_len] -> [seq_len/eP, block_size, eP]
    for (int j = 0; j < kv_seq_len; j++) {
        int block = j / block_size;
        int num = (block==pack_qk_dst.size()-1) ? last_block_slot_num : block_size;
        T * dst = reinterpret_cast<T*>(&(pack_qk_dst[block]->host<T>()[tId*UP_DIV(seq_len,eP)*num*eP]));
        for (int i = 0; i < seq_len; i++) {
            int out_index = i / eP;
            int in_index  = i % eP;
            dst[out_index * num * eP + (j % num) * eP + in_index] = src[i * kv_seq_len + j];
            std::cout << src[i * kv_seq_len + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
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
            std::cout << dst_ptr[i * mNumHead * mHeadDim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<std::vector<int>> CPUAttention::getKVshape(MNNStateCacheQuantType type) {
    // the kv_seq_len dim is indicated by 0, which is then handled by manager.
    std::vector<std::vector<int>> shape;
    if (type == MNNStateCacheQuantType::NoQuant || type == MNNStateCacheQuantType::QuantValueFp8) {
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, mResource->mHeadDim});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, mResource->mHeadDim, 0});
    }
    if (type == MNNStateCacheQuantType::QuantKeyInt8 || type == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) {
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, mResource->mHeadDim});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, 1});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, 1});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, mResource->mHeadDim, 0});
    }
    if (type == MNNStateCacheQuantType::QuantValueInt8) {
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, mResource->mHeadDim});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, mResource->mHeadDim, 0});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 1, 0});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 1, 0});
    }
    if (type == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, mResource->mHeadDim});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, 1});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 0, 1});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, mResource->mHeadDim, 0});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 1, 0});
        shape.emplace_back((std::initializer_list<int>){mResource->mKvNumHead, 1, 0});
    }
    return shape;
}

void CPUAttention::allocKVCache(int new_seq_len) {
    if (!mKVCache) {
        return;
    }
    StateCacheManager* manager = backend()->getStateCacheManager();
    std::vector<std::vector<int>> shape = getKVshape(manager->getQuantType());
    manager->onAllocateCache(mIdentifier, backend(), new_seq_len, shape, hP);
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    unit  = core->pack;
    bytes = core->bytes;
    MNN_ASSERT(bytes == 4);
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
    // int quantKV = static_cast<CPUBackend *>(backend())->getRuntime()->hint().kvcacheQuantOption;
    // bool quantKey = (quantKV & 1) == 1;
    // bool quantValue = ((quantKV >> 1) & 1) == 1;

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
            allocKVCache(seq_len);
        } else {
            allocKVCache(seq_len);
        }
    } else { // Decode
        allocKVCache(1);
    }
    int kv_seq_len  = mResource->mPastLength + seq_len;

    StateCacheManager* manager = backend()->getStateCacheManager();
    MNNStateCacheQuantType quantType = manager->getQuantType();
    int block_size = manager->getBlockSize();
    std::vector<std::shared_ptr<StateCacheBlock>> pastKV;
    int first_dst_block_id  = manager->prepareAttn(mIdentifier, mResource->mPastLength, pastKV);
    int last_block_slot_num = (kv_seq_len % block_size != 0) ? (kv_seq_len % block_size) : block_size;
 
    // Temporary tensors for intermediate results
    std::vector<std::shared_ptr<Tensor>> packQK;
    std::vector<std::shared_ptr<Tensor>> newPackQK;
    for (int i = 0; i < pastKV.size(); ++i) {
        packQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(block_size, unit), seq_len, unit}));
        backend()->onAcquireBuffer(packQK.back().get(), Backend::STATIC);
    }
    for (int i = 0; i < pastKV.size(); ++i) {
        if (i==pastKV.size()-1)
            newPackQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), last_block_slot_num, eP}));
        else
            newPackQK.emplace_back(Tensor::createDevice<float>({mThreadNum, UP_DIV(seq_len, eP), block_size, eP}));
        backend()->onAcquireBuffer(newPackQK.back().get(), Backend::STATIC);
    }
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> softmaxQK(Tensor::createDevice<int>({mThreadNum, seq_len, kv_seq_len}));
    std::shared_ptr<Tensor> dequantV(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, hP), block_size, hP}));
    // will be removed in the future
    std::shared_ptr<Tensor> QKVBuffer(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmaxQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(QKVBuffer.get(), Backend::STATIC);
    if (quantType == MNNStateCacheQuantType::QuantValueFp8 || \
        quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) {
        backend()->onAcquireBuffer(dequantV.get(), Backend::STATIC);
    }

    /* 1. Prepare inputs: concatenate them together.
       2. Calculation: perform tiled CPU Matmul.
       3. Prepare outputs: disperse the outputs to separate outputs Tensors.
    */
    /*
    StateCacheBlock Tensor shape: 
        K: [kvnum_heads, block_size / hp, head_dim, hp]
        V: [kvnum_heads, head_dim / hp, block_size, hp]
    */ 
    /*
    query, key, value Tensor shape:
        query: [num_heads, seq_len / eP, head_dim, eP], [kvnum_heads, group_size, ...]
        qk: [num_heads, kv_seq_len / unit, seq_len, unit]
        unpack_qk: [num_heads, seq_len, kv_seq_len]
        new_pack_qk: [num_heads, seq_len / eP, kv_seq_len, eP]
        pack_qkv: [num_heads, head_dim / unit, seq_len, unit]
    */

    std::function<void(int)> mCompute = [=](int tId) {
        auto pack_q      = mPackQ->host<char>() + tId * UP_DIV(seq_len, eP) * mResource->mHeadDim * eP * bytes;
        // auto pack_qk     = packQK->host<char>() + tId * UP_DIV(kv_seq_len, unit) * seq_len * unit * bytes;
        auto unpack_qk   = unpackQK->host<float>() + tId * seq_len * kv_seq_len;
        auto softmax_qk   = softmaxQK->host<float>() + tId * seq_len * kv_seq_len;
        // auto new_pack_qk = newPackQK->host<char>() + tId * UP_DIV(seq_len, eP) * kv_seq_len * eP * bytes;
        auto pack_qkv    = mPackQKV->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;
        int head_index   = tId * tileCount;
        int loop_e = seq_len / eP;
        int remain = seq_len % eP;
        for (int h = head_index; h < head_index + tileCount && h < mResource->mNumHead; h++) {
            std::cout << "head" << h << std::endl;
            // ----------need revision----------
            int    kv_h                 = h / group_size;
            // pack for matmul
            if (bytes == 2) {
                pack_query<FLOAT16_T>(query, pack_q, mResource->mNumHead, mResource->mHeadDim, eP, seq_len, h, q_scale);
                pack_key<FLOAT16_T>(key, pastKV, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, first_dst_block_id, quantType);
                pack_value<FLOAT16_T>(value, pastKV, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, first_dst_block_id, quantType);
            } else {
                pack_query<float>(query, pack_q, mResource->mNumHead, mResource->mHeadDim, eP, seq_len, h, q_scale);
                pack_key<float>(key, pastKV, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, first_dst_block_id, quantType);
                pack_value<float>(value, pastKV, seq_len, mResource->mKvNumHead, mResource->mHeadDim, hP, kv_h, first_dst_block_id, quantType);
            }
            // query @ key
            {
            size_t shapeParameters[7];
            size_t* parameters = shapeParameters;
            parameters[0] = eP * bytes;
            parameters[1] = mResource->mHeadDim;
            parameters[2] = block_size;
            parameters[3] = seq_len * unit * bytes;
            parameters[4] = 0;
            parameters[5] = 0;
            parameters[6] = 0;
            for (int j = 0; j < packQK.size(); ++j) {
                parameters[2] = (j == packQK.size()-1) ? last_block_slot_num : block_size;
                auto pack_qk = packQK[j]->host<char>() + tId * UP_DIV(block_size, unit) * seq_len * unit * bytes;
                if (quantType == MNNStateCacheQuantType::QuantKeyInt8 || \
                    quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8 || \
                    quantType == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
                    auto key_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K)->host<char>() + kv_h * UP_DIV(block_size, hP) * mResource->mHeadDim * hP * 1;
                    auto key_scale_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_SCALES)->host<char>() + kv_h * UP_DIV(block_size, hP) * 1 * hP * bytes;
                    auto key_zero_point_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_K_ZERO_POINTS)->host<char>() + kv_h * UP_DIV(block_size, hP) * 1 * hP * bytes;
                    for (int i = 0 ; i < loop_e; i++) {
                        core->MNNPackedMatMul_int8(
                            (float*)(pack_qk + (i * eP * unit) * bytes),
                            (float*)(pack_q + (i * mResource->mHeadDim * eP) * bytes),
                            (float*)key_dst,
                            parameters, nullptr, nullptr,
                            (float*)key_scale_dst, (float*)key_zero_point_dst
                        );
                    } 
                    core->MNNPackedMatMulRemain_int8(
                        (float*)(pack_qk + (loop_e * eP * unit) * bytes),
                        (float*)(pack_q + (loop_e * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        remain, parameters, nullptr, nullptr,
                        (float*)key_scale_dst, (float*)key_zero_point_dst
                    );
                }
                else {
                    auto key_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::NoQuant::PAST_K)->host<char>() + kv_h * UP_DIV(block_size, hP) * mResource->mHeadDim * hP * bytes;
                    for (int i = 0 ; i < loop_e; i++) {
                        core->MNNPackedMatMul(
                            (float*)(pack_qk + (i * eP * unit) * bytes),
                            (float*)(pack_q + (i * mResource->mHeadDim * eP) * bytes),
                            (float*)key_dst,
                            parameters, nullptr, nullptr,
                            nullptr, nullptr
                        );
                    }
                    core->MNNPackedMatMulRemain(
                        (float*)(pack_qk + (loop_e * eP * unit) * bytes),
                        (float*)(pack_q + (loop_e * mResource->mHeadDim * eP) * bytes),
                        (float*)key_dst,
                        remain, parameters, nullptr, nullptr,
                        nullptr, nullptr
                    );
                }
            }
            }
            // mask and softmax
            if(bytes == 2) {
                // unpack qk: [kv_seq_len/unit, seq_len, unit] -> [seq_len, kv_seq_len]
                unpack_QK<FLOAT16_T>(unpack_qk, packQK, seq_len, kv_seq_len, tId, block_size, unit);
                mask_QK<FLOAT16_T>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                // pack qk for qk @ v : [seq_len, kv_seq_len] -> [seq_len/eP, kv_seq_len, eP]
                pack_QK<FLOAT16_T>(newPackQK, softmax_qk, seq_len, kv_seq_len, tId, block_size, last_block_slot_num, eP);
            } else {
                unpack_QK<float>(unpack_qk, packQK, seq_len, kv_seq_len, tId, block_size, unit);
                mask_QK<float>(unpack_qk, seq_len, kv_seq_len, mScale, std::numeric_limits<float>::lowest(), mask->host<int>(), float_mask);
                softmax_QK(softmax_qk, unpack_qk, seq_len, kv_seq_len);
                pack_QK<float>(newPackQK, softmax_qk, seq_len, kv_seq_len, tId, block_size, last_block_slot_num, eP);
            }
            // prepare v and perform qk @ v
            {
            char* qkv_buffer = QKVBuffer->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;
            for (int j = 0; j < pastKV.size(); ++j) {
                if (quantType == MNNStateCacheQuantType::QuantValueInt8 || \
                    quantType == MNNStateCacheQuantType::QuantKeyInt8ValueInt8) {
                    // Not Implemented
                }
                int qk_block_size = (j == pastKV.size()-1) ? last_block_slot_num : block_size;
                char* value_dst = nullptr;
                char* new_pack_qk = newPackQK[j]->host<char>() + tId * UP_DIV(seq_len, eP) * qk_block_size * eP * bytes;
                // Dequantize values from fp8 to float
                if (quantType == MNNStateCacheQuantType::QuantValueFp8 || \
                    quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8) {
                    if (quantType == MNNStateCacheQuantType::QuantValueFp8)
                        value_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantValueFp8::PAST_V)->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * block_size * hP * 1;
                    if (quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8)
                        value_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8ValueFp8::PAST_V)->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * block_size * hP * 1;
                    char * qv = value_dst;
                    char * dqv = dequantV->host<char>() + tId * UP_DIV(mResource->mHeadDim, hP) * block_size * hP * bytes;
                    if (bytes == 2) {
                        dequant_value_fp16(dqv, qv, UP_DIV(mResource->mHeadDim, hP) * block_size * hP);
                    } else {
                        dequant_value_float(dqv, qv, UP_DIV(mResource->mHeadDim, hP) * block_size * hP);
                    }
                    value_dst = dqv;
                } else {
                    // No Quant for PastV
                    if (quantType == MNNStateCacheQuantType::NoQuant)
                        value_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::NoQuant::PAST_V)->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * block_size * hP * bytes;
                    if (quantType == MNNStateCacheQuantType::QuantKeyInt8)
                        value_dst = pastKV[j]->getTensor((int)StateCacheBlock::LAYOUT::QuantKeyInt8::PAST_V)->host<char>() + kv_h * UP_DIV(mResource->mHeadDim, hP) * block_size * hP * bytes;
                }
                // qk @ v
                size_t shapeParameters[7];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = qk_block_size;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = (block_size - qk_block_size) * hP * bytes;
                parameters[6]          = 0;
                for (int i = 0 ; i < loop_e; i++) {
                    core->MNNPackedMatMul(
                        (float*)(qkv_buffer + (i * eP * unit) * bytes),
                        (float*)(new_pack_qk + (i * qk_block_size * eP) * bytes),
                        (float*)value_dst, parameters,
                        nullptr, nullptr, nullptr, nullptr
                    );
                }
                {
                    core->MNNPackedMatMulRemain(
                        (float*)(qkv_buffer + (loop_e * eP * unit) * bytes),
                        (float*)(new_pack_qk + (loop_e * qk_block_size * eP) * bytes),
                        (float*)value_dst, remain, parameters,
                        nullptr, nullptr, nullptr, nullptr
                    );
                }
                // add the qkv_buffer to pack_qkv, first one copy, the followings are added.
                // the addition is performed element-wise unit-at-a-time.
                if (j==0){
                    core->MNNCopyC4WithStride(
                        (float*)qkv_buffer, (float*)pack_qkv,
                        unit, unit, UP_DIV(mResource->mHeadDim, unit) * seq_len
                    );
                }
                else{
                    core->MNNAddC4WithStride(
                        (float*)qkv_buffer, (float*)pack_qkv,
                        unit, unit, UP_DIV(mResource->mHeadDim, unit) * seq_len
                    );
                }
            }
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

    std::cout << "Finish Calculation!\n" << std::endl;
    // for (int i = 0; i < kv_seq_len; ++i) {
    //     for (int j = 0; j < mResource->mNumHead; ++j){
    //         for (int k = 0; k < mResource->mHeadDim; ++k){
    //             std::cout << outputs[0]->host<float>()[i*mResource->mNumHead*mResource->mHeadDim + j*mResource->mHeadDim + k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;


    // update slot_num of all blocks.
    manager->postAttn(mIdentifier, last_block_slot_num);
    mResource->mPastLength += seq_len;
    for (int i = 0; i < pastKV.size(); ++i) {
        backend()->onReleaseBuffer(packQK[i].get(), Backend::STATIC);
        backend()->onReleaseBuffer(newPackQK[i].get(), Backend::STATIC);
    }
    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmaxQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(QKVBuffer.get(), Backend::STATIC);
    if (quantType == MNNStateCacheQuantType::QuantValueFp8 || \
        quantType == MNNStateCacheQuantType::QuantKeyInt8ValueFp8){
        backend()->onReleaseBuffer(dequantV.get(), Backend::STATIC);
    }
    // std::vector<int> no;
    // std::cout << no[10] << std::endl;
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    tmp->mIdentifier = bn->getStateCacheManager()->onCreateIdentifier(mIdentifier);
    tmp->mResource = mResource;
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend *backend, bool kv_cache) : Execution(backend) {
    mIdentifier = backend->getStateCacheManager()->onCreateIdentifier();
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