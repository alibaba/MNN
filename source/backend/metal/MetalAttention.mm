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
#include "MetalKVCacheManager.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {
class AttentionBufExecution : public MetalExecution {
public:
    AttentionBufExecution(Backend *backend, bool kv_cache);
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache);
        exe->mKVCacheManager = mKVCacheManager;
        *dst = exe;
        return true;
    }

private:
    void _init();
    void compilerShader(const std::vector<Tensor *> &inputs);
    void handleKVAllocMemory();
    bool mKVCache;
    std::shared_ptr<MetalKVCacheManager> mKVCacheManager = nullptr;
    float mScale;
    bool mShortSeq = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    int mSeqLen;
    // for simd/tensor maxtrix load alignment
    int mKvAlignNum = 32;
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
    bool mQkTensorMatrix = false;
    bool mSftmSimdReduce = false;
    bool mQkvSimdReduce = false;
    bool mQkvSimdMatrix = false;
private:
    bool mHasMask = false;
    bool mIsAddMask = false;
    int mBatch, mKvSeqLen, mKvMaxLen;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempK, mTempV;
    bool mKvInDisk;
    
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
    int kv_align_len;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce) {
    _init();
}
void AttentionBufExecution::_init() {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mMeta = (KVMeta*)(mtbn->getMetaPtr());

    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));
    
    MNN::MetalKVCacheManager::KVCacheConfig kvconfig;
    kvconfig.mKVCacheDir = mtbn->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = mtbn->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mKvAlignNum = mKvAlignNum;

    mKVCacheManager.reset(new MetalKVCacheManager(backend(), kvconfig));
    mKvInDisk = !kvconfig.mKVCacheDir.empty();
}

void AttentionBufExecution::compilerShader(const std::vector<Tensor *> &inputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    
    auto seq_len = inputs[0]->length(1);
    int group_size = inputs[0]->length(2) / inputs[1]->length(2);
    std::string group_str = std::to_string(group_size);
    
    // Init Kernel
    std::string ftype = "float";
    std::string ftype4 = "float4";
    if (mtbn->useFp16InsteadFp32()) {
        ftype = "half";
        ftype4 = "half4";
    }
    std::vector<std::string> qkKeys = {
        {"matmul_qk_div_mask", ftype, group_str}
    };
    if(mHeadDim % 4 != 0) {
        qkKeys.emplace_back("HEAD_DIM_UNALIGNED_4");
    }
    
    std::vector<std::string> qkvKeys = {
        {"matmul_qkv", ftype, group_str}
    };
    if(mQkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {
        {"matmul_qk_div_mask", ftype, group_str, "FOR_PREFILL"}
    };
    if(mHasMask) {
        if (mIsAddMask) {
            qkPrefillKeys.emplace_back("ADD_MASK");
            if(seq_len > 1) {
                qkKeys.emplace_back("ADD_MASK");
            }
        } else {
            qkPrefillKeys.emplace_back("SET_MASK");
            if(seq_len > 1) {
                qkKeys.emplace_back("SET_MASK");
            }
        }
    }
    if(mQkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> qkvPrefillKeys = {
        {"matmul_qkv", ftype, group_str, "FOR_PREFILL"}
    };
    if(mQkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    if (mtbn->useFp16InsteadFp32()) {
        qkPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        qkvPrefillKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    std::vector<std::string> copyPastKeys = {
        {"pastkv_copy", ftype, group_str}
    };
    std::vector<std::string> shaders = {
        "decode_qk",
        "decode_qkv",
        "prefill_qk",
        "prefill_qkv",
        "copy"
    };
    if(mQkTensorMatrix) {
        shaders[2] = "prefill_qk_tensor";
        shaders[3] = "prefill_qkv_tensor";
        qkPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
        qkvPrefillKeys.emplace_back("USE_METAL_TENSOR_OPS");
    }
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

    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i=0; i<keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline) {
            // Rebuild Pipeline
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"ftype"];
            [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
            [dic setValue:@(keys[i][2].c_str()) forKey:@"GROUP_SIZE"];
            for (int j=3; j<keys[i].size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[i][j].c_str())];
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
        mKVCacheManager->setPastLength(mMeta != nullptr ? mMeta->previous : 0);

        if (mMeta->previous == mMeta->remove) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, mSeqLen);
        } else {
            MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
            mKVCacheManager->onRealloc(mMeta);
        }
        
        mKvSeqLen = mKVCacheManager->kvLength() + mSeqLen;
        mKvMaxLen = mKVCacheManager->maxLength();
        
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

        if (needMalloc) {
            mTempQK->setLength(0, mBatch * mNumHead);
            mTempQK->setLength(1, qSeqLenPiece * mKvMaxLen);
            mTempSoftMax->setLength(0, mBatch * mNumHead);
            mTempSoftMax->setLength(1, qSeqLenPiece * mKvMaxLen);
            
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
    // TODO : define short_seq more accurately
    mShortSeq = mSeqLen <= 10;
    mKvNumHead = key->shape()[2];
    mKvSeqLen = key->shape()[1];
    // Align to mKvAlignNum, for simd/tensor matrix load
    mKvMaxLen = ROUND_UP(mKvSeqLen, mKvAlignNum);
    
    if(mKVCache) {
        mKVCacheManager->onResize(mKvNumHead, mHeadDim);
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
    
    id<MTLBuffer> tempBufferK;
    id<MTLBuffer> tempBufferV;
    if(mKvInDisk) {
        tempBufferK = mKVCacheManager->getKeyBuffer();
        tempBufferV = mKVCacheManager->getValueBuffer();
    } else if(mKVCache) {
        tempTensorK = mKVCacheManager->getKeyTensor();
        tempTensorV = mKVCacheManager->getValueTensor();
    } else {
        tempTensorK = mTempK.get();
        tempTensorV = mTempV.get();
    }
    
    // whether use simdgroup
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();
    bool supportTensorMatrix = rt->supportTensorOps();

    // decode and thread number not too large
    mQkSimdReduce = supportSimdReduce && mShortSeq;
    // loop_k can divide 8, thus avoid branch
    mQkSimdMatrix = supportSimdMatrix && mSeqLen >= 16 && mHeadDim % 8 == 0;
    // 32x32x32 tensor block
    mQkTensorMatrix = supportTensorMatrix && mSeqLen >= 128 && mHeadDim % 32 == 0;

    mSftmSimdReduce = supportSimdReduce;
    mQkvSimdReduce = supportSimdReduce && mShortSeq && mHeadDim * mNumHead < mKvSeqLen * 32;
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
        copyp[3] = mKVCacheManager->kvLength() * copyp[0];
        copyp[4] = mKVCacheManager->kvLength();
        copyp[5] = mBatch;
        int copy_line = key->shape()[1];

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        if(mKvInDisk) {
            MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
            MetalBackend::setBuffer(tempBufferV, 0, encoder, 3);
        } else {
            MetalBackend::setTensor(tempTensorK, encoder, 2);
            MetalBackend::setTensor(tempTensorV, encoder, 3);
        }
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
        param->kv_align_len = mKvAlignNum;
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
            if(mKvInDisk) {
                MetalBackend::setBuffer(tempBufferK, 0, encoder, 2);
            } else {
                MetalBackend::setTensor(tempTensorK, encoder, 2);
            }
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            if(mHasMask) {
                MetalBackend::setTensor(inputs[3], encoder, 5);
            }
            
            int decode_grid_y = mBatch * mNumHead;
            std::pair<MTLSize, MTLSize> gl;
            if(mShortSeq) {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, decode_grid_y / group_size, mKvSeqLen)];
            } else if(mQkTensorMatrix) {
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mKvSeqLen, 32), decode_grid_y), MTLSizeMake(128, 1, 1));
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
            int axis_align = ROUND_UP(axis, mKvAlignNum);
            {
                auto softmax = (int*)mParamSoftmax.contents;
                // Inside, axis, outside, plane(invalid)
                softmax[0] = inside;
                softmax[1] = axis;
                softmax[2] = outside;
                softmax[3] = axis_align;
            }
            [encoder setComputePipelineState:mKernel_softmax];
            // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
            MetalBackend::setTensor(mTempQK.get(), encoder, 0);
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
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
            // [mBatch, mNumHead, mSeqLen, ROUND_UP(mKvSeqLen, mKvAlignNum)]
            MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0);
            // [mBatch, mSeqLen, mNumHead, mHeadDim]
            MetalBackend::setTensor(outputs[0], encoder, 1);
            // [mBatch, mKvNumHead, mHeadDim, mMaxSeqLen]
            if(mKvInDisk) {
                MetalBackend::setBuffer(tempBufferV, 0, encoder, 2);
            } else {
                MetalBackend::setTensor(tempTensorV, encoder, 2);
            }
            [encoder setBytes:&seq_idx length:sizeof(seq_idx) atIndex:3];
            [encoder setBuffer:mParamQKV offset:0 atIndex:4];
            std::pair<MTLSize, MTLSize> gl;
            if(mQkvSimdReduce) {
                gl = std::make_pair(MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim), MTLSizeMake(32, 1, 1));
            } else if(mQkTensorMatrix){
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 32), UP_DIV(mHeadDim, 32), mBatch * mNumHead), MTLSizeMake(128, 1, 1));
            } else if(mQkvSimdMatrix){
                gl = std::make_pair(MTLSizeMake(UP_DIV(seqLenPiece, 16), UP_DIV(mHeadDim, 16), mBatch * mNumHead), MTLSizeMake(32, 1, 1));
            } else {
                gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seqLenPiece, mBatch * mNumHead, mHeadDim)];
            }
//            printf("mBatch:%d, mNumHead:%d, mSeqLen:%d, mKvSeqLen:%d, mHeadDim:%d\n", mBatch, mNumHead, mSeqLen, mKvSeqLen, mHeadDim);
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            
        }
    }
    // Update status
    if(mKVCache) {
        mKVCacheManager->setPastLength(mKVCacheManager->kvLength() + mSeqLen);
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

