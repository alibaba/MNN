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
static void prefill_pack(Tensor* query, Tensor* key, Tensor* value, char* query_ptr, char* key_ptr, char* value_ptr,
                         int mMaxLength, int mNumHead, int mKvNumHead, int mHeadDim, int mValueH,
                         int eP, int hP, int query_e, int key_h, int seq_len, int h, int kv_h) {
    auto query_src = query->host<T>();
    auto key_src = key->host<T>();
    auto value_src = value->host<T>();
    auto query_dst = reinterpret_cast<T*>(query_ptr);
    auto key_dst = reinterpret_cast<T*>(key_ptr);
    auto value_dst = reinterpret_cast<T*>(value_ptr);
    // transpose query: [seq_len, num_head, head_dim] -> numhead, [seq_len/eP, head_dim, eP]
    for (int i = 0; i < query_e; i++) {
        for (int j = 0; j < mHeadDim; j++) {
            for (int k = 0; k < eP; k++) {
                int s = i * eP + k;
                if (s < seq_len) {
                    query_dst[i * mHeadDim * eP + j * eP + k] = query_src[s * mNumHead * mHeadDim + h * mHeadDim + j];
                }
            }
        }
    }
    // transpose key: [seq_len, num_head, head_dim] -> numhead, [seq_len/hP, head_dim, hP]
    for (int i = 0; i < key_h; i++) {
        for (int j = 0; j < mHeadDim; j++) {
            for (int k = 0; k < hP; k++) {
                int s = i * hP + k;
                if (s < seq_len) {
                    key_dst[i * mHeadDim * hP + j * hP + k] = key_src[s * mKvNumHead * mHeadDim + kv_h * mHeadDim + j];
                }
            }
        }
    }
    // transpose value: [seq_len, num_head, head_dim] -> numhead, [head_dim/hP, seq_len, hP]
    for (int i = 0; i < mValueH; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < hP; k++) {
                int hd = i * hP + k;
                if (hd < mHeadDim) {
                    value_dst[i * mMaxLength * hP + j * hP + k] = value_src[j * mKvNumHead * mHeadDim + kv_h * mHeadDim + hd];
                }
            }
        }
    }
}

template <typename T>
static void decode_pack(Tensor* query, Tensor* key, Tensor* value, char* query_ptr, char* key_ptr, char* value_ptr,
                         int mMaxLength, int mPastLength, int mHeadDim, int mValueH, int eP, int hP, int h, int kv_h) {
    auto query_src = query->host<T>();
    auto key_src = key->host<T>();
    auto value_src = value->host<T>();
    auto query_dst = reinterpret_cast<T*>(query_ptr);
    auto key_dst = reinterpret_cast<T*>(key_ptr);
    auto value_dst = reinterpret_cast<T*>(value_ptr);
    for (int i = 0; i < mHeadDim; i++) {
        query_dst[i * eP] = query_src[h * mHeadDim + i];
    }
    // transpose key: [1, num_head, head_dim] -> numhead, [kv_seq_len/hP, head_dim, hP]
    int outside_offset = UP_DIV(mPastLength, hP);
    int inside_offset = mPastLength % hP;
    for (int i = 0; i < mHeadDim; i++) {
        key_dst[(outside_offset - (inside_offset != 0)) * mHeadDim * hP + i * hP + inside_offset] = key_src[kv_h * mHeadDim + i];
    }
    // transpose value: [1, num_head, head_dim] -> numhead, [head_dim/hP, kv_seq_len, hP]
    for (int i = 0; i < mValueH; i++) {
        for (int j = 0; j < hP; j++) {
            value_dst[i * mMaxLength * hP + mPastLength * hP + j] = value_src[kv_h * mHeadDim + i * hP + j];
        }
    }
}

template <typename T>
static void prefill_unpack(char* pack_qkv, char* unpack_qkv, int mNumHead, int mHeadDim, int unit, int seq_len) {
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

template <typename T>
static void prefill_softmax(int* mask_ptr, float* mask_qk, float* softmax_qk, char* unpack_qk, char* pack_qk,
                            float mScale, int eP, int query_e, int seq_len,  float min_val, bool float_mask) {
    T* qk_src = reinterpret_cast<T*>(unpack_qk);
    T* qk_dst = reinterpret_cast<T*>(pack_qk);
    if (float_mask) {
        T* fpmask_ptr = reinterpret_cast<T*>(mask_ptr);
        // float mask
        for (int i = 0; i < seq_len * seq_len; i++) {
            mask_qk[i] = qk_src[i] * mScale + fpmask_ptr[i];
        }
    } else {
        // int mask
        for (int i = 0; i < seq_len * seq_len; i++) {
            if (mask_ptr[i]) {
                mask_qk[i] = qk_src[i] * mScale;
            } else {
                mask_qk[i] = min_val;
            }
        }
    }
    for (int i = 0; i < seq_len; i++) {
        MNNSoftmax(softmax_qk + i * seq_len, mask_qk + i * seq_len, seq_len);
    }
    for (int i = 0; i < query_e; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < eP; k++) {
                int s = i * eP + k;
                if (s < seq_len) {
                    qk_dst[i * seq_len * eP + j * eP + k] = softmax_qk[s * seq_len + j];
                }
            }
         }
    }
}

template <typename T>
static void decode_softmax(float* mask_qk, float* softmax_qk, char* unpack_qk, char* pack_qk,
                           float mScale, int eP, int kv_seq_len) {
    T* qk_src = reinterpret_cast<T*>(unpack_qk);
    T* qk_dst = reinterpret_cast<T*>(pack_qk);
    for (int i = 0; i < kv_seq_len; i++) {
        mask_qk[i] = qk_src[i] * mScale;
    }
    // softmax
    MNNSoftmax(softmax_qk, mask_qk, kv_seq_len);
    // pack qk
    for (int i = 0; i < kv_seq_len; i++) {
        qk_dst[i * eP] = softmax_qk[i];
    }
}

void CPUAttention::allocKVCache() {
    if (!mKVCache || mResource->mPastLength < mResource->mMaxLength) {
        return;
    }
    mResource->mMaxLength = mResource->mPastLength + mResource->mExpandChunk;
    // past_key: [1, numhead, headdim, maxlen] -> numhead, [headdim, maxlen] -> pack_b -> numhead, [maxlen/hP, head_dim, hP]
    mResource->mPastKey.reset(Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP}));
    // past_value: [1, numhead, maxlen, headdim] -> numhead, [maxlen, headdim] -> pack_b -> numhead, [head_dim/hP, max_len, hP]
    mResource->mPastValue.reset(Tensor::createDevice<float>({mResource->mKvNumHead, mResource->mValueH, mResource->mMaxLength, hP}));
    backend()->onAcquireBuffer(mResource->mPastKey.get(), Backend::STATIC);
    backend()->onAcquireBuffer(mResource->mPastValue.get(), Backend::STATIC);
}

void CPUAttention::reallocKVCache() {
    if (!mKVCache || mResource->mPastLength < mResource->mMaxLength) {
        return;
    }
    mResource->mMaxLength = mResource->mPastLength + mResource->mExpandChunk;
    // past_key: [1, numhead, headdim, maxlen] -> numhead, [headdim, maxlen] -> pack_b -> numhead, [maxlen/hP, head_dim, hP]
    auto new_key = Tensor::createDevice<float>({mResource->mKvNumHead, UP_DIV(mResource->mMaxLength, hP), mResource->mHeadDim, hP});
    // past_value: [1, numhead, maxlen, headdim] -> numhead, [maxlen, headdim] -> pack_b -> numhead, [head_dim/hP, max_len, hP]
    auto new_value = Tensor::createDevice<float>({mResource->mKvNumHead, mResource->mValueH, mResource->mMaxLength, hP});
    backend()->onAcquireBuffer(new_key, Backend::STATIC);
    backend()->onAcquireBuffer(new_value, Backend::STATIC);
    // copy
    for (int h = 0; h < mResource->mKvNumHead; h++) {
        ::memset(new_key->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes, 0, UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes);
        ::memset(new_value->host<char>() + h * mResource->mValueH * mResource->mMaxLength * hP * bytes, 0, mResource->mValueH * mResource->mMaxLength * hP * bytes);
        ::memcpy(new_key->host<char>() + h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes,
                 mResource->mPastKey->host<char>() + h * UP_DIV(mResource->mPastLength, hP) * mResource->mHeadDim * hP * bytes,
                 UP_DIV(mResource->mPastLength, hP) * mResource->mHeadDim * hP * bytes);
        for (int i = 0; i < mResource->mValueH; i++) {
            ::memcpy(new_value->host<char>() + (h * mResource->mValueH + i) * mResource->mMaxLength * hP * bytes,
                     mResource->mPastValue->host<char>() + (h * mResource->mValueH + i) * mResource->mPastLength * hP * bytes,
                     mResource->mPastLength * hP * bytes);
        }
    }
    mResource->mPastKey.reset(new_key);
    mResource->mPastValue.reset(new_value);
    mTempQK.reset(Tensor::createDevice<float>({mThreadNum, eP + 2, mResource->mMaxLength}));
    backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC);
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    int unit  =  core->pack;
    bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto shape = query->shape();
    int seq_len = shape[1];
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    mIsDecode = seq_len == 1;
    if (mResource->mPastLength == 0 || seq_len > 1) {
        mResource->mPastLength = seq_len;
    }
    mResource->mNumHead = shape[2];
    mResource->mKvNumHead = key->shape()[2];
    mResource->mHeadDim = shape[3];
    mResource->mScale = 1.0 / sqrt(mResource->mHeadDim);
    mResource->mValueH = UP_DIV(mResource->mHeadDim, hP);
    int query_e = UP_DIV(seq_len, eP);
    int key_h = UP_DIV(seq_len, hP);
    // mPastLength = 10;
    // alloc kv cache
    allocKVCache();

    int tileCount = UP_DIV(mResource->mNumHead, mThreadNum);

    // temp_query
    mPackQ.reset(Tensor::createDevice<float>({mThreadNum, query_e, mResource->mHeadDim, eP}));
    mPackQKV.reset(Tensor::createDevice<float>({mThreadNum, UP_DIV(mResource->mHeadDim, unit), seq_len, unit}));
    if (mIsDecode) {
        mTempQK.reset(Tensor::createDevice<float>({mThreadNum, eP + 2, mResource->mMaxLength}));
        backend()->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    } else {
        mTempQK.reset(Tensor::createDevice<float>({mThreadNum, 4, seq_len, seq_len}));
        backend()->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    }
    backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    int unit  =  core->pack;
    bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto matmulUnit   = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    bool float_mask = (mask->getType() == halide_type_of<float>());
    auto shape = query->shape();
    int seq_len = shape[1];
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    mIsDecode = seq_len == 1;
    if (mResource->mPastLength == 0 || seq_len > 1) {
        mResource->mPastLength = seq_len;
    }
    mResource->mNumHead = shape[2];
    mResource->mKvNumHead = key->shape()[2];
    int group_size = mResource->mNumHead / mResource->mKvNumHead;
    mResource->mHeadDim = shape[3];
    mResource->mScale = 1.0 / sqrt(mResource->mHeadDim);
    mResource->mValueH = UP_DIV(mResource->mHeadDim, hP);
    int query_e = UP_DIV(seq_len, eP);
    int key_h = UP_DIV(seq_len, hP);
    // mPastLength = 10;

    int tileCount = UP_DIV(mResource->mNumHead, mThreadNum);

    // try calloc kv cache
    mPrefill = [=](int tId){
        auto pack_q     = mPackQ->host<char>() + tId * query_e * mResource->mHeadDim * eP * bytes;
        auto pack_qk    = mTempQK->host<char>() + tId * 4 * seq_len * seq_len * bytes;
        auto unpack_qk  = pack_qk + seq_len * seq_len * 2 * bytes;
        auto mask_qk    = reinterpret_cast<float*>(pack_qk);
        auto softmax_qk = reinterpret_cast<float*>(unpack_qk);
        auto pack_qkv   = mPackQKV->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * seq_len * unit * bytes;

        int head_index = tId * tileCount;
        for (int h = head_index; h < head_index + tileCount && h < mResource->mNumHead; h++) {
            // pack for matmul
            int kv_h = h / group_size;
            auto key_dst = mResource->mPastKey->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes;
            auto value_dst = mResource->mPastValue->host<char>() + kv_h * mResource->mValueH * mResource->mMaxLength * hP * bytes;
            if (bytes == 2) {
                prefill_pack<int16_t>(query, key, value, pack_q, key_dst, value_dst, mResource->mMaxLength, mResource->mNumHead, mResource->mKvNumHead, mResource->mHeadDim, mResource->mValueH, eP, hP, query_e, key_h, seq_len, h, kv_h);
            } else {
                prefill_pack<float>(query, key, value, pack_q, key_dst, value_dst, mResource->mMaxLength, mResource->mNumHead, mResource->mKvNumHead, mResource->mHeadDim, mResource->mValueH, eP, hP, query_e, key_h, seq_len, h, kv_h);
            }
            // query @ key
            int loop_e = seq_len / eP;
            int remain = seq_len % eP;
            for (int i = 0 ; i < loop_e; i++) {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = mResource->mHeadDim;
                parameters[2]          = seq_len;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = 0;
                matmulUnit((float*)(pack_qk + (i * eP * unit) * bytes), (float*)(pack_q + (i * mResource->mHeadDim * eP) * bytes), (float*)key_dst, parameters, nullptr, nullptr, nullptr, nullptr);
            }
            {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = mResource->mHeadDim;
                parameters[2]          = seq_len;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = 0;
                matmulRemain((float*)(pack_qk + (loop_e * eP * unit) * bytes), (float*)(pack_q + (loop_e * mResource->mHeadDim * eP) * bytes), (float*)key_dst, remain, parameters, nullptr, nullptr, nullptr, nullptr);
            }
            int area_offset[1] {seq_len};
            core->MNNUnpackCUnitTranspose((float*)unpack_qk, (float*)pack_qk, seq_len, seq_len, area_offset);
            // div scale and mask
            auto mask_ptr = mask->host<int>();
            if (bytes == 2) {
                prefill_softmax<FLOAT16_T>(mask_ptr, mask_qk, softmax_qk, unpack_qk, pack_qk, mResource->mScale, eP, query_e, seq_len, -65504.0, float_mask);
            } else {
                prefill_softmax<float>(mask_ptr, mask_qk, softmax_qk, unpack_qk, pack_qk, mResource->mScale, eP, query_e, seq_len, std::numeric_limits<float>::lowest(), float_mask);
            }
            // qk @ v
            for (int i = 0 ; i < loop_e; i++) {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = seq_len;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = (mResource->mMaxLength - seq_len) * hP * bytes;
                matmulUnit((float*)(pack_qkv + (i * eP * unit) * bytes), (float*)(pack_qk + (i * seq_len * eP) * bytes), (float*)value_dst, parameters, nullptr, nullptr, nullptr, nullptr);
            }
            {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = seq_len;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = seq_len * unit * bytes;
                parameters[4]          = 0;
                parameters[5]          = (mResource->mMaxLength - seq_len) * hP * bytes;
                matmulRemain((float*)(pack_qkv + (loop_e * eP * unit) * bytes), (float*)(pack_qk + (loop_e * seq_len * eP) * bytes), (float*)value_dst, remain, parameters, nullptr, nullptr, nullptr, nullptr);
            }
            // transpose: [head_dim/unit, seq_len, unit] -> [seq_len, num_head, head_dim]
            auto dst_ptr = outputs[0]->host<char>() + h * mResource->mHeadDim * bytes;
            if (bytes == 2) {
                prefill_unpack<int16_t>(pack_qkv, dst_ptr, mResource->mNumHead, mResource->mHeadDim, unit, seq_len);
            } else {
                prefill_unpack<float>(pack_qkv, dst_ptr, mResource->mNumHead, mResource->mHeadDim, unit, seq_len);
            }
        }
    };

    mDecode = [=](int tId) {
        int kv_seq_len  = mResource->mPastLength + 1;
        auto pack_q     = mPackQ->host<char>() + tId * mResource->mHeadDim * eP * bytes;
        auto pack_qk    = mTempQK->host<char>() + tId * (eP + 2) * kv_seq_len * bytes;
        auto unpack_qk  = pack_qk + kv_seq_len * eP * bytes;
        auto mask_qk    = reinterpret_cast<float*>(pack_qk);
        auto softmax_qk = reinterpret_cast<float*>(unpack_qk);
        auto pack_qkv   = mPackQKV->host<char>() + tId * UP_DIV(mResource->mHeadDim, unit) * unit * bytes;

        int head_index = tId * tileCount;
        for (int h = head_index; h < head_index + tileCount && h < mResource->mNumHead; h++) {
            int kv_h = h / group_size;
            auto key_dst = mResource->mPastKey->host<char>() + kv_h * UP_DIV(mResource->mMaxLength, hP) * mResource->mHeadDim * hP * bytes;
            auto value_dst = mResource->mPastValue->host<char>() + kv_h * mResource->mValueH * mResource->mMaxLength * hP * bytes;
            // pack for matmul
            if (bytes == 2) {
                decode_pack<int16_t>(query, key, value, pack_q, key_dst, value_dst, mResource->mMaxLength, mResource->mPastLength, mResource->mHeadDim, mResource->mValueH, eP, hP, h, kv_h);
            } else {
                decode_pack<float>(query, key, value, pack_q, key_dst, value_dst, mResource->mMaxLength, mResource->mPastLength, mResource->mHeadDim, mResource->mValueH, eP, hP, h, kv_h);
            }
            // query @ key: [1, head_dim] @ [head_dim, kv_seq_len] -> [1, kv_seq_len]
            size_t shapeParameters[6];
            size_t* parameters = shapeParameters;
            parameters[0]          = eP * bytes;
            parameters[1]          = mResource->mHeadDim;
            parameters[2]          = kv_seq_len;
            parameters[3]          = seq_len * unit * bytes;
            parameters[4]          = 0;
            parameters[5]          = 0;
            matmulRemain((float*)pack_qk, (float*)pack_q, (float*)key_dst, seq_len, parameters, nullptr, nullptr, nullptr, nullptr);
            int area_offset[1] {seq_len};
            core->MNNUnpackCUnitTranspose((float*)unpack_qk, (float*)pack_qk, seq_len, kv_seq_len, area_offset);
            if (bytes == 2) {
                decode_softmax<FLOAT16_T>(mask_qk, softmax_qk, unpack_qk, pack_qk, mResource->mScale, eP, kv_seq_len);
            } else {
                decode_softmax<float>(mask_qk, softmax_qk, unpack_qk, pack_qk, mResource->mScale, eP, kv_seq_len);
            }
            // qk @ v: [1, kv_seq_len] @ [kv_seq_len, head_dim] -> [1, head_dim]
            {
                size_t shapeParameters[6];
                size_t* parameters = shapeParameters;
                parameters[0]          = eP * bytes;
                parameters[1]          = kv_seq_len;
                parameters[2]          = mResource->mHeadDim;
                parameters[3]          = 1 * unit * bytes;
                parameters[5]          = (mResource->mMaxLength - kv_seq_len) * hP * bytes;
                matmulRemain((float*)pack_qkv, (float*)pack_qk, (float*)value_dst, 1, parameters, nullptr, nullptr, nullptr, nullptr);
            }
            // transpose: [head_dim/unit, 1, unit] -> [1, num_head, head_dim]
            auto dst_ptr = outputs[0]->host<char>() + h * mResource->mHeadDim * bytes;
            core->MNNUnpackCUnitTranspose((float*)dst_ptr, (float*)pack_qkv, 1, mResource->mHeadDim, area_offset);
        }
    };
    mFunction = mIsDecode ? mDecode : mPrefill;
    reallocKVCache();
    // compute
    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mFunction((int)tId);
    }
    MNN_CONCURRENCY_END();
    mResource->mPastLength += mIsDecode;
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
