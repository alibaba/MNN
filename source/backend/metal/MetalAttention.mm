//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#import "core/Macro.h"
#import "MetalCast.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

static const char* gMatMulDivMask = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int head_dim;
    float scale;
};

kernel void main0(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_key [[buffer(3)]],
    const device int* mask [[buffer(4)]],
    constant Param& param [[buffer(5)]],
    uint3 gid[[thread_position_in_grid]]) {
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num
    const int z = gid.z; // key_seq_len
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.key_seq_len) {
        return;
    }
    int query_seq_len = param.query_seq_len;
    int key_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    
    const int offset = head_num * head_dim;
    const int offset_head = y * head_dim;
    const device T* A_offset = input0 + x * offset + offset_head;
    device T* Pastkey_offset = past_key + z * offset + offset_head;
    float Vscale = (float)param.scale;
#ifdef FOR_PREFILL
    device const T* B_offset = input1 + z * offset + offset_head;
    const int output_offset = y * query_seq_len * key_seq_len;
    float out0 = 0.0;
    
    for(int i = 0; i < head_dim; ++i){
        float A = (float)(A_offset[i]);
        float B = (float)(B_offset[i]);
        out0 += B * A;
        Pastkey_offset[i] = (T)B;
    }
    
    out0 *= Vscale;
    
    out0 = mask[((x + 0) * key_seq_len + (z + 0))] == 0 ? -FLT_MAX : out0;
    output[output_offset + x * key_seq_len + z] = (T)out0;
#else
    const device T *B_offset = input1 + offset_head;
    float out = 0.0;
    if (z == key_seq_len - 1) {
        for(int i = 0; i < head_dim; ++i){
            float A = (float)(A_offset[i]);
            float B = (float)(B_offset[i]);
            out += B * A;
            Pastkey_offset[i] = (T)B;
        }
    } else {
        for(int i = 0; i < head_dim; ++i){
            float A = A_offset[i];
            float B = (float)Pastkey_offset[i];
            
            out += A * B;
        }
    }
    out *= Vscale;
    output[y + z * head_num] = (T)out;
#endif
}

)metal";


static const char* gMatMulQKV = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int head_dim;
    float scale;
};
kernel void main0(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_value [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    uint3 gid[[thread_position_in_grid]]) {
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num
    const int z = gid.z; // head_dim
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.head_dim) {
        return;
    }
    int qk_seq_len = param.query_seq_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    const int offset = head_num * head_dim;
    const int offset_head = y * head_dim + z;
#ifdef FOR_PREFILL
    device const T *A_offset = input0 + (y * qk_seq_len + x) * value_seq_len;
    device const T *B_offset = input1 + offset_head;
    device T *Pastvalue_offset = past_value + offset_head;
    float out = 0.0;
    
    for(int i = 0; i < value_seq_len; ++i){
        float A0 = (float)A_offset[i];
        float B = (float)B_offset[i*offset];
        out += A0 * B;
        Pastvalue_offset[i*offset] = B;
    }
    output[ x * offset + (y * head_dim + z)] = out;
#else
    device const T *A_offset = input0 + y;
    device const T *B_offset = input1 + offset_head;
    device T *Pastvalue_offset = past_value + offset_head;
    float out = 0;
    
    for(int i = 0; i < value_seq_len - 1; ++i){
        float A = (float)A_offset[i * head_num];
        float B = (float)Pastvalue_offset[i * offset];
        
        out += A * B;
    }
    out += (float)A_offset[(value_seq_len - 1)*head_num] * (float)B_offset[0];
    Pastvalue_offset[(value_seq_len - 1)*offset] = B_offset[0];
    output[(y * head_dim + z)] = (T)out;
#endif

}
)metal";

namespace MNN {
class AttentionBufExecution : public MetalExecution {
public:
    struct SharedCache {
        std::shared_ptr<Tensor> mPastKey;
        std::shared_ptr<Tensor> mPastValue;
        int mPastLength = 0, mMaxLength = 0, mKv_seq_len = 0;
    };
    AttentionBufExecution(Backend *backend, bool kv_cache);

    virtual ~AttentionBufExecution() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache);
        exe->mCache = mCache;
        *dst = exe;
        return true;
    }

private:
    void _init();
    void reallocKVCache();
    bool mKVCache;
    std::shared_ptr<SharedCache> mCache;
    float mScale;
    const int mExpandChunk = 64;
    bool mIsDecode = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0;
    id<MTLComputePipelineState> mKernel_softmax;
    
    id<MTLComputePipelineState> mKernel_qk;
    id<MTLComputePipelineState> mKernel_qkv;
    id<MTLComputePipelineState> mKernelPrefill_qk;
    id<MTLComputePipelineState> mKernelPrefill_qkv;
    id<MTLBuffer> mParamQKV;
    id<MTLBuffer> mParamSoftmax;
};

struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int head_dim;
    float scale;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce){
    _init();
}
void AttentionBufExecution::_init() {
    mCache.reset(new SharedCache);
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mKernel_softmax = [context pipelineWithName:@"softmax_plane" fp16:mtbn->useFp16InsteadFp32()];
    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    auto rt = mtbn->runtime();
    std::string T = "float";
    if (mtbn->useFp16InsteadFp32()) {
        T = "half";
    }
    std::vector<std::string> keys = {
        "matmul_qk_div_mask",
        T
    };
    auto pipeline = rt->findPipeline(keys);
    if (nil == pipeline) {
        // Rebuild Pipeline
        MTLCompileOptions *decodeOption = [[MTLCompileOptions alloc] init];
        decodeOption.preprocessorMacros = @{
            @"T" : @(T.c_str()),
        };

        MTLCompileOptions* encodeOption = [[MTLCompileOptions alloc] init];
        encodeOption.preprocessorMacros = @{
            @"T" : @(T.c_str()),
            @"FOR_PREFILL": @"1"
        };
        mKernel_qk = mtbn->makeComputePipelineWithSourceOption(gMatMulDivMask, "main0", decodeOption);
        mKernelPrefill_qk = mtbn->makeComputePipelineWithSourceOption(gMatMulDivMask, "main0", encodeOption);
        mKernel_qkv = mtbn->makeComputePipelineWithSourceOption(gMatMulQKV, "main0", decodeOption);
        mKernelPrefill_qkv = mtbn->makeComputePipelineWithSourceOption(gMatMulQKV, "main0", encodeOption);

        rt->insertPipeline({"matmul_qk_div_mask", T}, mKernel_qk);
        rt->insertPipeline({"matmul_qk_div_mask", T, "PREFILL"}, mKernelPrefill_qk);
        rt->insertPipeline({"matmul_qkv", T}, mKernel_qkv);
        rt->insertPipeline({"matmul_qkv", T, "PREFILL"}, mKernelPrefill_qkv);
    } else {
        mKernel_qk = rt->findPipeline({"matmul_qk_div_mask", T});
        mKernelPrefill_qk = rt->findPipeline({"matmul_qk_div_mask", T, "PREFILL"});
        mKernel_qkv = rt->findPipeline({"matmul_qkv", T});
        mKernelPrefill_qkv = rt->findPipeline({"matmul_qkv", T, "PREFILL"});
    }
    MNN_ASSERT(nil != mKernel_qk);
    MNN_ASSERT(nil != mKernel_qkv);
    MNN_ASSERT(nil != mKernelPrefill_qk);
    MNN_ASSERT(nil != mKernelPrefill_qkv);
}

void AttentionBufExecution::reallocKVCache() {
    if (mCache->mPastLength < mCache->mMaxLength || nullptr == mTempQK || (!mIsDecode)) {
        if (mIsDecode) {
            mTempQK.reset(Tensor::createDevice<float>({mNumHead, mCache->mMaxLength}));
            mTempSoftMax.reset(Tensor::createDevice<float>({mNumHead, mCache->mMaxLength}));
        } else {
            mTempQK.reset(Tensor::createDevice<float>({mNumHead, mCache->mPastLength, mCache->mPastLength}));
            mTempSoftMax.reset(Tensor::createDevice<float>({mNumHead, mCache->mPastLength, mCache->mPastLength}));
        }
        backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC);
        backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
    }
    if (!mKVCache || mCache->mPastLength < mCache->mMaxLength) {
        return;
    }
    auto mtbn = static_cast<MetalBackend *>(backend());
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    bool needCopy = mCache->mMaxLength > 0;

    size_t old_size = mNumHead * mCache->mMaxLength * mHeadDim * byte;
    mCache->mMaxLength = mCache->mPastLength + mExpandChunk;
    // past_key: [1, numhead, headdim, maxlen]
    auto new_key = Tensor::createDevice<float>({mCache->mMaxLength, mNumHead, mHeadDim});
    // past_value: [1, numhead, maxlen, headdim]
    auto new_value = Tensor::createDevice<float>({mCache->mMaxLength, mNumHead, mHeadDim});
    size_t size = mNumHead * mCache->mMaxLength * mHeadDim * byte;
    backend()->onAcquireBuffer(new_key, Backend::STATIC);
    backend()->onAcquireBuffer(new_value, Backend::STATIC);
    if (needCopy) {
        auto newKeyBuf = MetalBackend::getBuffer(new_key);
        auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
        auto keyBuf = MetalBackend::getBuffer(mCache->mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;;
        ::memcpy(new_key_ptr, key_ptr, old_size);
        
        auto newValueBuf = MetalBackend::getBuffer(new_value);
        auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
        auto valueBuf = MetalBackend::getBuffer(mCache->mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        ::memcpy(new_value_ptr, value_ptr, old_size);
    }
    mCache->mPastKey.reset(new_key);
    mCache->mPastValue.reset(new_value);
}


void AttentionBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto shape = query->shape();
    
    int seq_len = shape[1];
    mNumHead = shape[2];
    mHeadDim = shape[3];
    mScale = 1.0 / sqrt(mHeadDim);
    mIsDecode = seq_len == 1;
    if (mCache->mPastLength == 0 || seq_len > 1) {
        mCache->mPastLength = seq_len;
    }
    mCache->mKv_seq_len = mCache->mPastLength;
    if(mIsDecode){
        mCache->mKv_seq_len = mCache->mPastLength + 1;
    }
    reallocKVCache();

    // Update Parameters
    {
        auto param = (Param*)mParamQKV.contents;
        param->scale = mScale;
        param->head_dim = mHeadDim;
        param->key_seq_len = mCache->mKv_seq_len;
        param->head_num = mNumHead;
        param->query_seq_len = seq_len;
    }
    // For softmax parameter
    int inside, outside;
    if (mIsDecode) {
        inside = mNumHead;
        outside = 1;
    } else {
        inside = 1;
        outside = mCache->mKv_seq_len * mNumHead;
    }
    int axis = mCache->mKv_seq_len;
    {
        auto softmax = (int*)mParamSoftmax.contents;
        // Inside, axis, outside, plane(invalid)
        softmax[0] = inside;
        softmax[1] = axis;
        softmax[2] = outside;
        softmax[3] = 0;
    }
    // Run QK Kernel
    {
        id<MTLComputePipelineState> pipeline;
        if (mIsDecode) {
            pipeline = mKernel_qk;
        } else {
            pipeline = mKernelPrefill_qk;
        }
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(query, encoder, 0);
        MetalBackend::setTensor(key, encoder, 1);
        MetalBackend::setTensor(mTempQK.get(), encoder, 2);
        MetalBackend::setTensor(mCache->mPastKey.get(), encoder, 3);
        MetalBackend::setTensor(mask, encoder, 4);
        [encoder setBuffer:mParamQKV offset:0 atIndex:5];
        auto gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mCache->mKv_seq_len)];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Run Softmax Kernel
    {
        [encoder setComputePipelineState:mKernel_softmax];
        MetalBackend::setTensor(mTempQK.get(), encoder, 0);
        MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
        [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];
        auto gl = [context computeBestGroupAndLocal: mKernel_softmax threads:MTLSizeMake(inside, outside, 1)];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Run QKV Kernel
    {
        id<MTLComputePipelineState> pipeline;
        if (mIsDecode) {
            pipeline = mKernel_qkv;
        } else {
            pipeline = mKernelPrefill_qkv;
        }
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        MetalBackend::setTensor(outputs[0], encoder, 2);
        MetalBackend::setTensor(mCache->mPastValue.get(), encoder, 3);
        [encoder setBuffer:mParamQKV offset:0 atIndex:4];
        auto gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mHeadDim)];
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Update status
    if(mIsDecode){
        mCache->mPastLength += 1;
        mCache->mKv_seq_len = mCache->mPastLength + 1;
    }

    return;
}

class AttentionBufCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const override {
        auto param = op->main_as_AttentionParam();
        return new AttentionBufExecution(backend, param->kv_cache());
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif

