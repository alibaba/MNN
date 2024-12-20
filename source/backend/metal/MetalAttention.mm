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
#import "MetalAttentionShader.hpp"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

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
    void reallocKVCache(int history_len);
    void compilerShader(const std::vector<Tensor *> &inputs);
    bool mKVCache;
    std::shared_ptr<SharedCache> mCache;
    float mScale;
    const int mExpandChunk = 64;
    bool mIsDecode = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    id<MTLComputePipelineState> mKernel_softmax = nil;
    
    id<MTLComputePipelineState> mKernel_qk = nil;
    id<MTLComputePipelineState> mKernel_qkv = nil;
    id<MTLComputePipelineState> mKernel_copy = nil;
    id<MTLComputePipelineState> mKernelPrefill_qk = nil;
    id<MTLComputePipelineState> mKernelPrefill_qkv = nil;
    id<MTLBuffer> mParamQKV;
    id<MTLBuffer> mParamSoftmax;
    id<MTLBuffer> mParamCopy;
    
private:
    bool mQkSimdReduce = false;
    bool mQkSimdMatrix = false;
    bool mSftmSimdReduce = false;
    bool mQkvSimdReduce = false;
    bool mQkvSimdMatrix = false;
    bool mUseHeadNum2 = false;
};

struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce) {
    _init();
}
void AttentionBufExecution::_init() {
    mCache.reset(new SharedCache);
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:5 * sizeof(int) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));
}

void AttentionBufExecution::reallocKVCache(int history_len) {
    /*
     when kv-cache
     decoding: past_len > max_len, realloc and copy past_len cache
     prefill : max_len == 0 (first prefill), realloc and copy history_len cache
     prefill : max_len > 0 (not first prefill) && past_len >= max_len, realloc and copy history_len cache. copy current prompt in copy shader(not this function)
     prefill : max_len > 0 (not first prefill) && past_len < max_len, not realloc and no need copy history_len cache, just copy current prompt in copy shader(not this function)

     */
    if (!mKVCache) {
        return;
    }
    
    if (mIsDecode && mCache->mPastLength < mCache->mMaxLength) {
        return;
    }

    // not first prefill (do reuse_kvcache) and total past_len < max_len
    if(!mIsDecode && mCache->mMaxLength > 0 && mCache->mPastLength < mCache->mMaxLength && history_len != 0) {
        return;
    }
    auto mtbn = static_cast<MetalBackend *>(backend());
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    bool needCopy = history_len > 0;

    size_t old_size = mKvNumHead * history_len * mHeadDim * byte;
    size_t old_piece_size = history_len * byte;
    size_t old_piece_stride = mCache->mMaxLength * byte;

    mCache->mMaxLength = mCache->mPastLength + mExpandChunk;
    // past_key: [1, numhead, headdim, maxlen]
    auto new_key = Tensor::createDevice<float>({mCache->mMaxLength, mKvNumHead, mHeadDim});
    // past_value: [1, numhead, maxlen, headdim]
    auto new_value = Tensor::createDevice<float>({mKvNumHead, mHeadDim, mCache->mMaxLength});
    size_t size = mKvNumHead * mCache->mMaxLength * mHeadDim * byte;
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
        for(int i = 0; i <  mKvNumHead * mHeadDim; i++) {
            ::memcpy(new_value_ptr + i * mCache->mMaxLength * byte, value_ptr + i * old_piece_stride, old_piece_size);
        }
    }
    mCache->mPastKey.reset(new_key);
    mCache->mPastValue.reset(new_value);
}

void AttentionBufExecution::compilerShader(const std::vector<Tensor *> &inputs) {
    auto mask = inputs[3];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    // Init Kernel
    bool float_mask = (mask->getType() == halide_type_of<float>());
    std::string T = "float";
    if (mtbn->useFp16InsteadFp32()) {
        T = "half";
    }
    std::vector<std::string> qkKeys = {
        {"matmul_qk_div_mask", T}
    };
    if(mQkSimdReduce) {
        qkKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    
    // QK matmul total thread is large
    mUseHeadNum2 = mIsDecode && mCache->mKv_seq_len > 1024;
    if(mUseHeadNum2) {
        qkKeys.emplace_back("HEAD_NUM_2");
    }
    std::vector<std::string> qkvKeys = {
        {"matmul_qkv", T}
    };
    if(mQkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {
        {"matmul_qk_div_mask", T, "FOR_PREFILL"}
    };
    if (float_mask) {
        qkPrefillKeys.emplace_back("FLOAT_MASK");
    }
    if(mQkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> qkvPrefillKeys = {
        {"matmul_qkv", T, "FOR_PREFILL"}
    };
    if(mQkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> copyPastKeys = {
        {"pastkv_copy", T}
    };
    std::vector<std::vector<std::string>> keys = {
        qkKeys,
        qkvKeys,
        qkPrefillKeys,
        qkvPrefillKeys,
        copyPastKeys
    };
    std::vector<const char*> sources = {
        gMatMulDivMask,
        gMatMulQKV,
        gMatMulDivMask,
        gMatMulQKV,
        gCopyPastKV
    };
    std::vector<std::string> shaders = {
        "decode_qk",
        "decode_qkv",
        "prefill_qk",
        "prefill_qkv",
        "copy"
    };
    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i=0; i<keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline) {
            // Rebuild Pipeline
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"T"];
            for (int j=2; j<keys[i].size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[i][j].c_str())];;
            }
            option.preprocessorMacros = dic;
            
            pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], shaders[i].c_str(), option);
            rt->insertPipeline(keys[i], pipeline);
        }
        pipelines[i] = pipeline;
    }
    mKernel_qk = pipelines[0];
    mKernel_qkv = pipelines[1];
    mKernelPrefill_qk = pipelines[2];
    mKernelPrefill_qkv = pipelines[3];
    mKernel_copy = pipelines[4];
    MNN_ASSERT(nil != mKernel_qk);
    MNN_ASSERT(nil != mKernel_qkv);
    MNN_ASSERT(nil != mKernelPrefill_qk);
    MNN_ASSERT(nil != mKernelPrefill_qkv);
    MNN_ASSERT(nil != mKernel_copy);

    if(mSftmSimdReduce) {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane_sg" fp16:mtbn->useFp16InsteadFp32()];
    } else {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane" fp16:mtbn->useFp16InsteadFp32()];
    }

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

    int history_len = mCache->mPastLength;
    // first prefill set history_len to 0
    if(!mIsDecode && mask->length(2) == mask->length(3)) {
        history_len = 0;
    }
    if (!mIsDecode) {
        mCache->mPastLength = mask->length(3);
    }
    mCache->mKv_seq_len = mCache->mPastLength;
    if(mIsDecode){
        mCache->mKv_seq_len = mCache->mPastLength + 1;
    }
    mKvNumHead = key->shape()[2];
    
    auto rt = (MetalRuntime*)mtbn->runtime();
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();

    // decode and thread number not too large
    mQkSimdReduce = supportSimdReduce && seq_len == 1;
    // loop_k can divide 8, thus avoid branch
    mQkSimdMatrix = supportSimdMatrix && seq_len >= 16 && mHeadDim % 8 == 0;

    mSftmSimdReduce = supportSimdReduce;
    mQkvSimdReduce = supportSimdReduce && seq_len == 1 && mHeadDim * mNumHead < mCache->mKv_seq_len * 32;
    mQkvSimdMatrix = supportSimdMatrix && seq_len >= 16;
    
    // start to compile attention shaders
    compilerShader(inputs);
    
    int group_size = mNumHead / mKvNumHead;

    // kv-cache realloc function
    reallocKVCache(history_len);
    
    // temp tensor alloc memory
    bool needMalloc = mTempQK->length(0) != mNumHead;
    if (mIsDecode) {
        if (mTempQK->length(1) != mCache->mMaxLength) {
            needMalloc = true;
        }
        mTempQK->setLength(0, mNumHead);
        mTempQK->setLength(1, mCache->mMaxLength);
        mTempSoftMax->setLength(0, mNumHead);
        mTempSoftMax->setLength(1, mCache->mMaxLength);

    } else {
        if (mTempQK->length(1) != seq_len * mCache->mPastLength) {
            needMalloc = true;
        }
        mTempQK->setLength(0, mNumHead);
        mTempQK->setLength(1, seq_len * mCache->mPastLength);
        mTempSoftMax->setLength(0, mNumHead);
        mTempSoftMax->setLength(1, seq_len * mCache->mPastLength);
    }
    if (needMalloc) {
        auto res = backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC) && backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal %d\n", mCache->mPastLength);
            return;
        }
    }

    // Update Parameters
    {
        auto param = (Param*)mParamQKV.contents;
        param->scale = mScale;
        param->head_dim = mHeadDim;
        param->key_seq_len = mCache->mKv_seq_len;
        param->head_num = mNumHead;
        param->group = group_size;
        param->query_seq_len = seq_len;
        param->max_kv_len = mCache->mMaxLength;
    }
    // For softmax parameter
    int inside, outside;
    if (mIsDecode) {
        inside = 1;
        outside = mNumHead;
    } else {
        inside = 1;
        outside = seq_len * mNumHead;
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
    
    // Run Copy Kernel
    {
        auto copyp = (int*)mParamCopy.contents;
        copyp[0] = mKvNumHead * mHeadDim;
        
        int copy_line;
        if(mIsDecode) {
            /*
             each decode fill one kv_seq.
             Key -> K-Cache :   [1, mKvNumHead, mHeadDim] -> [mCache->mKv_seq_len + 1, mKvNumHead, mHeadDim]
             Value -> V-Cache : [1, mKvNumHead, mHeadDim] -> [mKvNumHead, mHeadDim, mCache->mKv_seq_len + 1]
             */
            copyp[1] = 1;
            copyp[2] = mCache->mMaxLength;
            copyp[3] = (mCache->mKv_seq_len - 1) * copyp[0];
            copyp[4] = mCache->mKv_seq_len - 1;
            copy_line = 1;
        } else {
            /*
             first time copy.
             Key -> K-Cache :   [mCache->mKv_seq_len, mKvNumHead, mHeadDim] -> [mCache->mKv_seq_len, mKvNumHead, mHeadDim]
             Value -> V-Cache : [mCache->mKv_seq_len, mKvNumHead, mHeadDim] -> [mKvNumHead, mHeadDim, mCache->mMaxLength (fill when decode)]
             */
            copyp[1] = seq_len;
            copyp[2] = mCache->mMaxLength;
            copyp[3] = history_len * copyp[0];
            copyp[4] = history_len;
            copy_line = seq_len;
        }

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        MetalBackend::setTensor(mCache->mPastKey.get(), encoder, 2);
        MetalBackend::setTensor(mCache->mPastValue.get(), encoder, 3);
        [encoder setBuffer:mParamCopy offset:0 atIndex:4];
        
        std::pair<MTLSize, MTLSize> gl;
        gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim, copy_line, 1)];

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

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
        MetalBackend::setTensor(mTempQK.get(), encoder, 1);
        MetalBackend::setTensor(mCache->mPastKey.get(), encoder, 2);
        MetalBackend::setTensor(mask, encoder, 3);
        [encoder setBuffer:mParamQKV offset:0 atIndex:4];

        int decode_grid_y = mNumHead;
        if(mUseHeadNum2) {
            decode_grid_y = (decode_grid_y + 1) / 2;
        }
        std::pair<MTLSize, MTLSize> gl;
        if(mQkSimdReduce) {
            gl = std::make_pair(MTLSizeMake(seq_len, decode_grid_y, mCache->mKv_seq_len), MTLSizeMake(32, 1, 1));
        } else if(mQkSimdMatrix) {
            gl = std::make_pair(MTLSizeMake(UP_DIV(seq_len, 16), UP_DIV(mCache->mKv_seq_len, 16), mNumHead), MTLSizeMake(32, 1, 1));
        } else if(mIsDecode){
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, decode_grid_y, mCache->mKv_seq_len)];
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mCache->mKv_seq_len)];
        }
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    }
    // Run Softmax Kernel
    {
        [encoder setComputePipelineState:mKernel_softmax];
        MetalBackend::setTensor(mTempQK.get(), encoder, 0);
        MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
        [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];

        int thread_group_size = 32;
        std::pair<MTLSize, MTLSize> gl;
        if(mSftmSimdReduce) {
            gl = std::make_pair(MTLSizeMake(inside, outside, 1), MTLSizeMake(thread_group_size, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal: mKernel_softmax threads:MTLSizeMake(inside, outside, 1)];
        }

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
        MetalBackend::setTensor(outputs[0], encoder, 1);
        MetalBackend::setTensor(mCache->mPastValue.get(), encoder, 2);
        [encoder setBuffer:mParamQKV offset:0 atIndex:3];
        std::pair<MTLSize, MTLSize> gl;
        if(mQkvSimdReduce) {
            gl = std::make_pair(MTLSizeMake(seq_len, mNumHead, mHeadDim), MTLSizeMake(32, 1, 1));
        } else if(mQkvSimdMatrix){
            gl = std::make_pair(MTLSizeMake(UP_DIV(seq_len, 16), UP_DIV(mHeadDim, 16), mNumHead), MTLSizeMake(32, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mHeadDim)];
        }
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

