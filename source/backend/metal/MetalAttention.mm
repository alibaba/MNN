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
#include "core/OpCommonUtils.hpp"

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
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

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
    void compilerShader(const std::vector<Tensor *> &inputs);
    void handleKVAllocMemory();
    bool mKVCache;
    std::shared_ptr<SharedCache> mCache;
    float mScale;
    const int mExpandChunk = 64;
    bool mShortSeq = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    int mSeqLen;
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
    KVMeta* mMeta;
    bool mQkSimdReduce = false;
    bool mQkSimdMatrix = false;
    bool mSftmSimdReduce = false;
    bool mQkvSimdReduce = false;
    bool mQkvSimdMatrix = false;
private:
    bool mHasMask = false;
    bool mIsAddMask = false;
    int mBatch, mKvSeqLen, mKvMaxLen;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempK, mTempV;
};

struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce) {
    _init();
}
void AttentionBufExecution::_init() {
    mCache.reset(new SharedCache);
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mMeta = (KVMeta*)(mtbn->getRuntime()->pMeta);

    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));
}

void AttentionBufExecution::reallocKVCache() {
    if (!mKVCache) {
        return;
    }
    auto kv_seq_len = mMeta->previous + mMeta->add - mMeta->remove + mMeta->computeReverseSize();
    
    auto mtbn = static_cast<MetalBackend *>(backend());
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    
    auto start = mCache->mPastLength - mMeta->remove;
    // latest length larger than maxLen
    if (kv_seq_len > mCache->mMaxLength) {

        auto copy_len = mCache->mPastLength - mMeta->remove + mMeta->computeReverseSize();
        bool needCopy = copy_len > 0;
        
        size_t old_size = mKvNumHead * start * mHeadDim * byte;
        size_t old_piece_size = start * byte;
        size_t old_piece_stride = mCache->mMaxLength * byte;

        mCache->mMaxLength = kv_seq_len + mExpandChunk;
        // past_key: [1, numhead, headdim, maxlen]
        auto new_key = Tensor::createDevice<float>({mCache->mMaxLength, mKvNumHead, mHeadDim});
        // past_value: [1, numhead, maxlen, headdim]
        auto new_value = Tensor::createDevice<float>({mKvNumHead, mHeadDim, mCache->mMaxLength});
        size_t size = mKvNumHead * mCache->mMaxLength * mHeadDim * byte;
        auto res = backend()->onAcquireBuffer(new_key, Backend::STATIC);
        res = res && backend()->onAcquireBuffer(new_value, Backend::STATIC);
        if(!res) {
            MNN_ERROR("attition kv cache realloc memory error:%d\n", res);
        }
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
        mCache->mPastLength = (int)start;

        mCache->mPastKey.reset(new_key);
        mCache->mPastValue.reset(new_value);
    }
    
    // Remove
    {
        if (0 == mMeta->n_reserve) {
            mCache->mPastLength = start;
            return;
        }
        
        auto keyBuf = MetalBackend::getBuffer(mCache->mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;
        auto valueBuf = MetalBackend::getBuffer(mCache->mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        
        // TODO: need to ensure reserve info is sorted
        for (int n = 0; n < mMeta->n_reserve; ++n) {
            auto begin = mMeta->reserve[2 * n];
            auto length = mMeta->reserve[2 * n + 1];
            // past_key   : [mCache->mPastLength, mKvNumHead, mHeadDim]
            // past_value : [mKvNumHead, mHeadDim, mCache->mMaxLength]

            auto copy_src_index = mCache->mPastLength - mMeta->remove + begin;
            auto copy_dst_index = start;
            for(int i = 0; i < length; i++) {
                ::memcpy(key_ptr + (copy_dst_index + i) * mKvNumHead * mHeadDim * byte, key_ptr + (copy_src_index + i) * mKvNumHead * mHeadDim * byte, mKvNumHead * mHeadDim * byte);
            }
            for(int j = 0; j <  mKvNumHead * mHeadDim; j++) {
                for(int i = 0; i < length; i++) {
                    ::memcpy(value_ptr + (j * mCache->mMaxLength + copy_dst_index + i) * byte, value_ptr + (j * mCache->mMaxLength + copy_src_index + i) * byte, byte);
                }
            }
            start += length;
        }
        mCache->mPastLength = (int)start;
    }
}

void AttentionBufExecution::compilerShader(const std::vector<Tensor *> &inputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext *)mtbn->context();

    // Init Kernel
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
    
    std::vector<std::string> qkvKeys = {
        {"matmul_qkv", T}
    };
    if(mQkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {
        {"matmul_qk_div_mask", T, "FOR_PREFILL"}
    };
    if(mHasMask) {
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
        }
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
        // basic marco info
        std::string ftype = "float";
        std::string ftype4 = "float4";
        if (mtbn->useFp16InsteadFp32()) {
            ftype = "half";
            ftype4 = "half4";
        }

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        option.preprocessorMacros = @{
            @"ftype" : @(ftype.c_str()),
            @"ftype4" : @(ftype4.c_str()),
        };
        std::vector<std::string> keys = {"softmax_sg_reduce", ftype};
        keys.emplace_back("softmax_plane_sg");
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gSoftmaxSgReduce, keys.back().c_str(), option);
            rt->insertPipeline(keys, pipeline);
        }
        mKernel_softmax = pipeline;
    } else {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane" fp16:mtbn->useFp16InsteadFp32()];
    }

}

void AttentionBufExecution::handleKVAllocMemory() {
    if(mKVCache) {
        mCache->mPastLength = mMeta != nullptr ? mMeta->previous : 0;
        // kv-cache realloc function
        reallocKVCache();
        mCache->mKv_seq_len = mCache->mPastLength + mSeqLen;
        mKvSeqLen = mCache->mKv_seq_len;
        mKvMaxLen = mCache->mMaxLength;
        
        float useMemorySize = 1.0 * mKvMaxLen / 1024.0 * mSeqLen / 1024.0 * mBatch * mNumHead;
        // elementSize larger than 32M
        mQseqSplitNum = 1;
        if(useMemorySize > 32.0) {
            mQseqSplitNum = useMemorySize >= 256.0 ? 16 : ((useMemorySize < 128.0) ? 4 : 8);
        }

        int qSeqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
        // temp tensor alloc memory
        bool needMalloc = mTempQK->length(0) != mBatch * mNumHead;
        if (mTempQK->length(1) != qSeqLenPiece * mKvMaxLen) {
            needMalloc = true;
        }
        mTempQK->setLength(0, mBatch * mNumHead);
        mTempQK->setLength(1, qSeqLenPiece * mKvMaxLen);
        mTempSoftMax->setLength(0, mBatch * mNumHead);
        mTempSoftMax->setLength(1, qSeqLenPiece * mKvMaxLen);

        if (needMalloc) {
            auto res = backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC) && backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
            if (!res) {
                MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal %d\n", res);
                return;
            }
        }
    }
}
ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mHasMask = inputs.size() > 3;
    if(mHasMask) {
        mIsAddMask = (inputs[3]->getType() == halide_type_of<float>());
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto shape = query->shape();
    mBatch = shape[0];
    mSeqLen = shape[1];
    mNumHead = shape[2];
    mHeadDim = shape[3];
    mScale = 1.0 / sqrt(mHeadDim);
    // currently only mSeqLen=1 means short_seq
    mShortSeq = mSeqLen == 1;
    mKvNumHead = key->shape()[2];
    mKvSeqLen = key->shape()[1];
    mKvMaxLen = ROUND_UP(mKvSeqLen, 4);
    
    if(mKVCache) {
        return NO_ERROR;
    }
    
    float useMemorySize = 1.0 * mKvMaxLen / 1024.0 * mSeqLen / 1024.0 * mBatch * mNumHead;
    // elementSize larger than 32M
    mQseqSplitNum = 1;
    if(useMemorySize > 32.0) {
        mQseqSplitNum = useMemorySize >= 256.0 ? 8 : ((useMemorySize < 128.0) ? 2 : 4);
    }
    
    // no kv_cache memory, should create temp q/k memory
    mTempK.reset(Tensor::createDevice<float>({mKvMaxLen * mHeadDim * mBatch * mKvNumHead}));
    mTempV.reset(Tensor::createDevice<float>({mKvMaxLen * mHeadDim * mBatch * mKvNumHead}));
    mTempQK.reset(Tensor::createDevice<float>({mKvMaxLen * UP_DIV(mSeqLen, mQseqSplitNum) * mBatch * mNumHead}));
    mTempSoftMax.reset(Tensor::createDevice<float>({mKvMaxLen * UP_DIV(mSeqLen, mQseqSplitNum) * mBatch * mNumHead}));
    
    backend()->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
void AttentionBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    if(mKVCache) {
        // if has kv_cache, default has mask
        MNN_ASSERT(inputs.size() > 3);
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto rt = (MetalRuntime*)mtbn->runtime();

    int group_size = mNumHead / mKvNumHead;
    
    // temp memory alloc, handle variable set
    Tensor* tempTensorK;
    Tensor* tempTensorV;
    handleKVAllocMemory();
    
    if(mKVCache) {
        tempTensorK = mCache->mPastKey.get();
        tempTensorV = mCache->mPastValue.get();
    } else {
        tempTensorK = mTempK.get();
        tempTensorV = mTempV.get();
    }
    
    // whether use simdgroup
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();

    // decode and thread number not too large
    mQkSimdReduce = supportSimdReduce && mSeqLen == 1;
    // loop_k can divide 8, thus avoid branch
    mQkSimdMatrix = supportSimdMatrix && mSeqLen >= 16 && mHeadDim % 8 == 0;

    mSftmSimdReduce = supportSimdReduce;
    mQkvSimdReduce = supportSimdReduce && mSeqLen == 1 && mHeadDim * mNumHead < mKvSeqLen * 32;
    mQkvSimdMatrix = supportSimdMatrix && mSeqLen >= 16;
    
    // start to compile attention shaders
    compilerShader(inputs);
    
    // Run Copy and Format-Convert Kernel
    {
        auto copyp = (int*)mParamCopy.contents;
        /*
         Key -> K-Cache :   [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mKvMaxLen, mBatch, mKvNumHead, mHeadDim]
         Value -> V-Cache : [mBatch, mKvSeqLen, mKvNumHead, mHeadDim] -> [mBatch, mKvNumHead, mHeadDim, mKvMaxLen (fill when decode)]
         */
        copyp[0] = mKvNumHead * mHeadDim;
        // current new kv_len
        copyp[1] = key->shape()[1];
        copyp[2] = mKvMaxLen;
        copyp[3] = mCache->mPastLength * copyp[0];
        copyp[4] = mCache->mPastLength;
        copyp[5] = mBatch;
        int copy_line = key->shape()[1];

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        MetalBackend::setTensor(tempTensorK, encoder, 2);
        MetalBackend::setTensor(tempTensorV, encoder, 3);
        [encoder setBuffer:mParamCopy offset:0 atIndex:4];
        
        std::pair<MTLSize, MTLSize> gl;
        gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim, copy_line, mBatch)];

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    }
    
    // Update Parameters
    int seqLenPiece = UP_DIV(mSeqLen, mQseqSplitNum);
    {
        auto param = (Param*)mParamQKV.contents;
        param->scale = mScale;
        param->head_dim = mHeadDim;
        param->key_seq_len = mKvSeqLen;
        param->head_num = mNumHead;
        param->group = group_size;
        param->query_seq_len = mSeqLen;
        param->q_seq_piece_len = seqLenPiece;
        param->max_kv_len = mKvMaxLen;
        param->batch = mBatch;
    }
    
    for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        // Run QK Kernel
        {
            id<MTLComputePipelineState> pipeline;
            if (mShortSeq) {
                pipeline = mKernel_qk;
            } else {
                pipeline = mKernelPrefill_qk;
            }
            [encoder setComputePipelineState:pipeline];
            // [mBatch, mSeqLen, mNumHead, mHeadDim]
            MetalBackend::setTensor(query, encoder, 0);
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 1);
            // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            if(mHasMask) {
                MetalBackend::setTensor(inputs[3], encoder, 5);
            }
            
            int decode_grid_y = mBatch * mNumHead;
            std::pair<MTLSize, MTLSize> gl;
            if(mQkSimdReduce) {
                gl = std::make_pair(MTLSizeMake(seqLenPiece, decode_grid_y, mKvSeqLen), MTLSizeMake(32, 1, 1));
            } else if(mQkSimdMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mKvSeqLen, 16), decode_grid_y), MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, decode_grid_y, mKvSeqLen)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            
        }
        // Run Softmax Kernel
        {
            // For softmax parameter
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            int inside = 1;
            int outside = mBatch * mNumHead * seqLenPiece;
            int axis = mKvSeqLen;
            {
                auto softmax = (int*)mParamSoftmax.contents;
                // Inside, axis, outside, plane(invalid)
                softmax[0] = inside;
                softmax[1] = axis;
                softmax[2] = outside;
                softmax[3] = 0;
            }
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
            if (mShortSeq) {
                pipeline = mKernel_qkv;
            } else {
                pipeline = mKernelPrefill_qkv;
            }
            [encoder setComputePipelineState:pipeline];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0);
            // [mBatch, mSeqLen, mNumHead, mHeadDim]
            MetalBackend::setTensor(outputs[0], encoder, 1);
            // [mBatch, mKvNumHead, mHeadDim, mMaxSeqLen]
            MetalBackend::setTensor(tempTensorV, encoder, 2);
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            std::pair<MTLSize, MTLSize> gl;
            if(mQkvSimdReduce) {
                gl = std::make_pair(MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim), MTLSizeMake(32, 1, 1));
            } else if(mQkvSimdMatrix){
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead), MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim)];
            }
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            
        }
    }
    // Update status
    if(mKVCache) {
        mCache->mPastLength += mSeqLen;
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

