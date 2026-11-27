//
//  MetalConvolution1x1.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolution1x1.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "ConvSimdGroupShader.hpp"

#if MNN_METAL_ENABLED

#if MNN_METAL_OP_PROFILE
#define CONV1X1_SET_TAG(name) mProfileTag = (name)
#else
#define CONV1X1_SET_TAG(name) do {} while(0)
#endif

namespace MNN {

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    loadWeight(op, false);
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op,
                                         std::shared_ptr<MNN::Tensor> weight,
                                         std::shared_ptr<MNN::Tensor> bias)
    : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
    mBias = bias;
}

bool MetalConvolution1x1::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        // Tied weight-quant gather needs dequant metadata, which the float executor never builds.
        return false;
    }
    *dst = new MetalConvolution1x1(bn, op, mWeight, mBias);
    MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
    return true;
}

ErrorCode MetalConvolution1x1::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    auto previousPipeline = mPipeline;
    auto clearPreparation = [&]() {
        mPrepared = false;
        mFusedRowScale = nullptr;
        mPipeline = nil;
        mThreads = std::make_pair(MTLSizeMake(0, 0, 0), MTLSizeMake(0, 0, 0));
        CONV1X1_SET_TAG("");
    };
    auto ready = [&](id<MTLComputePipelineState> pipeline, MTLSize threads) {
        const NSUInteger count = threads.width * threads.height * threads.depth;
        return pipeline != nil && count > 0 && pipeline.maxTotalThreadsPerThreadgroup >= count &&
               pipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength;
    };
    clearPreparation();
    auto code = resizeImpl(inputs, outputs);
    if (code == NO_ERROR && !ready(mPipeline, mThreads.second)) {
        code = NOT_SUPPORT;
    }
    if (code == NO_ERROR) {
        mPrepared = true;
        if (mPipeline != previousPipeline) {
            mReplayEvents.clear();
            mReplayState = 0;
            mReplayStable = 0;
        }
        return NO_ERROR;
    }
    clearPreparation();
    invalidateReplay();
    mRecordedStructure = 0;
    MNN_ERROR("Metal Conv1x1 pipeline preparation failed: %d.\n", code);
    return code;
}

ErrorCode MetalConvolution1x1::resizeImpl(const std::vector<Tensor *> &inputs,
                                          const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int icDiv4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto ocDiv4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    // Allocate once and rewrite in place: this buffer is bound raw, so encode-replay
    // cannot see a swap and a live recording would keep emitting the previous Param.
    if (nil == mConstBuffer) {
        mConstBuffer = backend->getConstBuffer(sizeof(Param));
    }
    auto param = (Param *)mConstBuffer.contents;
    param->inputSize = is;
    param->inputDepthQuad = icDiv4;
    param->outputWidth = ow;
    param->outputHeight = oh;
    param->outputSize = os;
    param->outputDepthQuad = ocDiv4;
    param->outputChannel = oc;
    param->batch = ob;
    // Float kernels never read blockCount/scaleCoef; neutral placeholders keep the
    // Param ABI deterministic without touching the uninitialized quant members.
    param->blockCount = 1;
    param->activation = mActivationType;
    param->scaleCoef = 1.0f;
    return prepareFloatConv(inputs, outputs);
}

ErrorCode MetalConvolution1x1::prepareFloatConv(const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    auto backend = static_cast<MetalBackend*>(this->backend());
    auto context = (__bridge MNNMetalContext*)backend->context();
    auto rt = const_cast<MetalRuntime*>(backend->runtime());
    auto input = inputs[0];
    auto output = outputs[0];
    const auto param = (const Param*)mConstBuffer.contents;
    const int ow = param->outputWidth, oh = param->outputHeight, ob = param->batch;
    const int area = ob * ow * oh;
    const int icDiv4 = param->inputDepthQuad, ocDiv4 = param->outputDepthQuad, oc = param->outputChannel;
    const bool fp16 = backend->useFp16InsteadFp32();
    auto baseDic = conv1x1Macros(fp16);
    std::vector<std::string> baseKeys = {fp16 ? "half4" : "float4", "MNN_METAL_FLOAT32_COMPUTER"};
    const std::string basicShaderPrefix = gBasicConvPrefix;
    const std::string sgmWfpShader = std::string(gConv1x1WfpSgMatrix) + gConv1x1WfpSgMatrixM64;
    const std::string sgrWfpShader = gConv1x1WfpSgReduce;
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    option.preprocessorMacros = baseDic;

    if(rt->supportSimdGroupMatrix()) {
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_matrix");
        // total computation not too small
        if(area >= 16 && icDiv4 >= 4 && icDiv4 % 2 == 0 && ocDiv4 >= 4 &&
           (int64_t)area * icDiv4 * ocDiv4 >= 64 * 64 * 64) {
            // Enough threads
            if((int64_t)area * ocDiv4 / icDiv4 >= 1024) {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_32x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_16x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_16x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            }
            return NO_ERROR;
        }
    }
    if(rt->supportSimdGroupReduce()) {
        std::string sgrWfpStr = basicShaderPrefix + sgrWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_reduce");
        // do input_channel reduce
        auto magic_num = 4.0; // total threads pretty small and loop pretty large
        if(icDiv4 >= 32 && icDiv4 % 2 == 0 && 1.0 * area * ocDiv4 / icDiv4 < magic_num) {
            auto keys = baseKeys;
            if (mRowScaleOffer != nullptr) {
                [baseDic setValue:@"1" forKey:@"CONV_ROW_MUL"];
                keys.emplace_back("CONV_ROW_MUL");
                mFusedRowScale = mRowScaleOffer;
            }
            keys.emplace_back("conv1x1_z4_sg");
            mThreads = std::make_pair(MTLSizeMake(ow * oh, ocDiv4, ob), MTLSizeMake(32, 1, 1));
            mPipeline = loadConv1x1Pipeline(backend, keys, sgrWfpStr, "conv1x1_z4_sg", baseDic);
            CONV1X1_SET_TAG(keys.back());
            return NO_ERROR;
        }
    }
    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = ocDiv4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z8" fp16:backend->useFp16InsteadFp32()];
            if (mPipeline == nil) {
                return NOT_SUPPORT;
            }

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];

            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z8";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
        } else {
            NSUInteger gid_x = UP_DIV(ow * oh, 4);
            NSUInteger gid_y = ocDiv4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z4" fp16:backend->useFp16InsteadFp32()];
            if (mPipeline == nil) {
                return NOT_SUPPORT;
            }

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0,  TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z4";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
        }
    } else {
        NSString* shaderName[] = {@"conv1x1_g1z8", @"conv1x1_g1z4", @"conv1x1_w4h4",  @"conv1x1_w2c2", @"conv1x1_w4c2"};
        int itemW[] = {8, 4, 16, 2, 4};
        int itemC[] = {4, 4, 4, 8, 8};
        int actual_kernel = 5;
        if (ocDiv4 % 2 != 0) {
            // Don't unrool c for avoid memory exceed
            actual_kernel = 3;
        }
        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)

        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};

        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx] fp16:backend->useFp16InsteadFp32()];
            if (pipeline == nil) {
                return NOT_SUPPORT;
            }
            NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
            NSUInteger gid_y = UP_DIV(oc, itemC[knl_idx]);
            NSUInteger gid_z = 1;

            std::string name = [shaderName[knl_idx] UTF8String];
            auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset queue:backend->queue()];

            if(min_cost.first > std::get<2>(ret)) {
                min_cost.first = std::get<2>(ret);
                min_cost.second = knl_idx;
                mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            }
        }
        mPipeline = [context pipelineWithName:shaderName[min_cost.second] fp16:backend->useFp16InsteadFp32()];
        CONV1X1_SET_TAG(std::string([shaderName[min_cost.second] UTF8String]));
    }

    return NO_ERROR;
}

uint64_t MetalConvolution1x1::replayStructure(const std::vector<Tensor*>& inputs) const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    auto mixPtr = [&mix](const void* p) { mix((uint64_t)(uintptr_t)p); };
    auto mixPipeline = [&mixPtr](id<MTLComputePipelineState> p) { mixPtr((__bridge void*)p); };
    auto mixBuffer = [&mixPtr](id<MTLBuffer> b) { mixPtr((__bridge void*)b); };
    auto mixSize = [&mix](MTLSize s) {
        mix((uint64_t)s.width);
        mix((uint64_t)s.height);
        mix((uint64_t)s.depth);
    };
    auto mixGrid = [&mixSize](const std::pair<MTLSize, MTLSize>& g) {
        mixSize(g.first);
        mixSize(g.second);
    };

    mixPipeline(mPipeline);
    mixGrid(mThreads);
    mix((uint64_t)mPrepared);
    mixBuffer(mConstBuffer);
    mixPtr(mFusedRowScale);
    mix(inputs.empty() ? 0ull : (uint64_t)inputs[0]->elementSize());
    return h;
}

bool MetalConvolution1x1::onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mPrepared && replayStructure(inputs) == mRecordedStructure;
}

void MetalConvolution1x1::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    if (!mPrepared) {
        mEncodeStatus = NOT_SUPPORT;
        return;
    }
    mRecordedStructure = replayStructure(inputs);
#if MNN_METAL_OP_PROFILE
    static_cast<MetalBackend *>(backend())->setProfileSubtag(mProfileTag);
#endif
    auto input = inputs[0];
    auto output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    MetalBackend::setTensor(mWeight.get(), encoder, 3);
    MetalBackend::setTensor(mBias.get(), encoder, 4);
    if (mFusedRowScale != nullptr) {
        MetalBackend::setTensor(mFusedRowScale, encoder, 24);
    }
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */