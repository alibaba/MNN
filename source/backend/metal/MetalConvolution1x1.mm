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

namespace MNN {
bool MetalConvolution1x1::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    auto dx = common->dilateX(), dy = common->dilateY();
    auto sx = common->strideX(), sy = common->strideY();
    auto px = common->padX(), py = common->padY();
    return kx == 1 && ky == 1 && dx == 1 && dy == 1 && px == 0 && py == 0 && sx == 1 && sy == 1;
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    auto conv2D = op->main_as_Convolution2D();
    bool ldInt8Weight = false;
    if(static_cast<MetalBackend*>(backend)->getMemoryMode() == BackendConfig::Memory_Low) {
        if (conv2D->quanParameter() && (conv2D->external() || conv2D->quanParameter()->buffer())) {
            ldInt8Weight = true;
        }
    }
    loadWeight(op, ldInt8Weight);
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> weight, std::shared_ptr<MNN::Tensor> bias, std::shared_ptr<MNN::Tensor> dequantScale, int dequantBits, float scaleCoef) : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
    mBias = bias;
    mDequantScaleBias = dequantScale;
    mDequantBits = dequantBits;
    mScaleCoef = scaleCoef;
}


bool MetalConvolution1x1::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalConvolution1x1(bn, op, mWeight, mBias, mDequantScaleBias, mDequantBits, mScaleCoef);
    return true;
}

ErrorCode MetalConvolution1x1::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int ic_4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto oc_4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int blockSize = 1;
    if (mDequantScaleBias.get()) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockSize = (int)(mDequantScaleBias->usize() / bytes / oc_4 / 2 / 4);
    }
    // create const buffer
    mConstBuffer = backend->getConstBuffer(sizeof(Param));
    auto param = (Param *)mConstBuffer.contents;
    param->input_size = is;
    param->input_slice = ic_4;
    param->output_width = ow;
    param->output_height = oh;
    param->output_size = os;
    param->output_slice = oc_4;
    param->output_channel = oc;
    param->batch = ob;
    param->block_size = blockSize;
    param->activation = mActivationType;
    param->scale_coef = mScaleCoef;

    // basic marco info
    std::string ftype2 = "float2";
    std::string ftype4 = "float4";
    std::string ftype4x4 = "float4x4";
    if (backend->useFp16InsteadFp32()) {
        ftype2 = "half2";
        ftype4 = "half4";
        ftype4x4 = "half4x4";
    }

    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
    option.preprocessorMacros = @{
        @"ftype2" : @(ftype2.c_str()),
        @"ftype4" : @(ftype4.c_str()),
        @"ftype4x4" : @(ftype4x4.c_str()),
        @"MNN_METAL_FLOAT32_COMPUTER": @"1"
    };

    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    if (mDequantScaleBias.get()) {
        NSUInteger gid_x = UP_DIV(ow * oh, 4);
        NSUInteger gid_y = oc_4;
        NSUInteger gid_z = ob;
        std::string name = "conv1x1_g1z4_w8";
        mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];
        
        if (mDequantBits == 4) {
            if(rt->supportSimdGroupReduce() && ob * ow * oh == 1) {
                std::vector<std::string> baseKeys = {"conv1x1_w4_sg_reduce", ftype4, "MNN_METAL_FLOAT32_COMPUTER"};

                // unrool c for avoid memory exceed
                if(oc > 16384 && oc_4 % 2 == 0) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g16_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgReduce, "conv1x1_gemv_g16_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), 1, 1), MTLSizeMake(64, 1, 1));
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g8_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgReduce, "conv1x1_gemv_g8_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
//                    MNN_PRINT("g8  ic: %d oc: %d\n", input->channel(), oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(64, 1, 1));
                }
                return NO_ERROR;
            } else if(rt->supportSimdGroupMatrix()  && ob * ow * oh > 8 && oc > 8 && ic_4 % 8 == 0) {
                std::vector<std::string> baseKeys = {"conv1x1_w4_sg_matrix", ftype4, "MNN_METAL_FLOAT32_COMPUTER"};

                // Generally threadgroup memory >= 16KB
                auto smem_size = [[context device] maxThreadgroupMemoryLength];
                // choose different tile for different computation
                if(ob * ow * oh >= 128 && oc >= 512 && ob * ow * oh * oc > 512 * 2048 && smem_size >= 8192) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x64_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgMatrix, "conv1x1_gemm_32x64_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));
                                        
                } else if(ob * ow * oh >= 32 && ob * ow * oh * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x16_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgMatrix, "conv1x1_gemm_32x16_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 512 && ob * ow * oh * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x32_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgMatrix, "conv1x1_gemm_16x32_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 16), UP_DIV(oc, 32), 1), MTLSizeMake(32, 1, 1));
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x16_w4_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1W4SgMatrix, "conv1x1_gemm_16x16_w4_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
//                                    MNN_PRINT("gemm M: %d N: %d\n", ob * ow * oh, oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                }
                return NO_ERROR;
            } else {
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w4" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w4";
            }
        }
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(),
                        ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(),
                        (((MetalRuntimeAllocator::MetalBufferAlloc *)mDequantScaleBias->deviceId()))->getBuffer(),
                        nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {
            TensorUtils::getDescribe(input)->extra.offset,
            TensorUtils::getDescribe(output)->extra.offset,
            0,
            TensorUtils::getDescribe(weight)->extra.offset,
            TensorUtils::getDescribe(bias)->extra.offset,
            TensorUtils::getDescribe(mDequantScaleBias.get())->extra.offset,
            0};

        MetalRuntime *rt = (MetalRuntime *)backend->runtime();
        auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset  queue:backend->queue()];
        mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        return NO_ERROR;
    }
    
    if(rt->supportSimdGroupMatrix()) {
        std::vector<std::string> baseKeys = {"conv1x1_float_sg_matrix", ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
        // total computation not too small
        if(ob * ow * oh >= 16 && ic_4 >= 4 && ic_4 % 2 == 0 && oc_4 >= 4 && ob * ow * oh * ic_4 * oc_4 >= 64 * 64 * 64) {
            // Enough threads
            if(ob * ow * oh * oc_4 / ic_4 >= 1024) {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_32x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1SgMatrix, "conv1x1_gemm_32x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_16x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1SgMatrix, "conv1x1_gemm_16x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(ob * ow * oh, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            }
            return NO_ERROR;
        }
    }
    if(rt->supportSimdGroupReduce()) {
        std::vector<std::string> baseKeys = {"conv1x1_float_sg_reduce", ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
        // do input_channel reduce
        auto magic_num = 4.0; // total threads pretty small and loop pretty large
        if(ic_4 >= 32 && ic_4 % 2 == 0 && 1.0 * ob * ow * oh * oc_4 / ic_4 < magic_num) {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_z4_sg");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(gConv1x1SgReduce, "conv1x1_z4_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline;
            mThreads = std::make_pair(MTLSizeMake(ow * oh, oc_4, ob), MTLSizeMake(32, 1, 1));
            return NO_ERROR;
        }
    }
//    printf("lora: %d %d %d %d %d\n", ob, oh, ow, oc, input->channel());
    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z8" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];

            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribe(input)->extra.offset, TensorUtils::getDescribe(output)->extra.offset, 0, TensorUtils::getDescribe(weight)->extra.offset, TensorUtils::getDescribe(bias)->extra.offset, 0};
            std::string name = "conv1x1_g1z8";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        } else {
            NSUInteger gid_x = UP_DIV(ow * oh, 4);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z4" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribe(input)->extra.offset, TensorUtils::getDescribe(output)->extra.offset, 0,  TensorUtils::getDescribe(weight)->extra.offset, TensorUtils::getDescribe(bias)->extra.offset, 0};
            std::string name = "conv1x1_g1z4";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            //printf("conv1x1_z4, %d %d %d %d\n", ow, oh, oc_4, ic_4);
        }
    } else {
        NSString* shaderName[] = {@"conv1x1_g1z8", @"conv1x1_g1z4", @"conv1x1_w4h4",  @"conv1x1_w2c2", @"conv1x1_w4c2"};
        int itemW[] = {8, 4, 16, 2, 4};
        int itemC[] = {4, 4, 4, 8, 8};
        int actual_kernel = 5;
        if (oc_4 % 2 != 0) {
            // Don't unrool c for avoid memory exceed
            actual_kernel = 3;
        }
        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)

        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {TensorUtils::getDescribe(input)->extra.offset, TensorUtils::getDescribe(output)->extra.offset, 0, TensorUtils::getDescribe(weight)->extra.offset, TensorUtils::getDescribe(bias)->extra.offset, 0};

        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx] fp16:backend->useFp16InsteadFp32()];
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
            //printf("conv1x1 idx:%d, global:%d %d %d, local:%d %d %d, min_cost:%d\n", knl_idx, (int)retTune.second.first.width, (int)retTune.second.first.height, (int)retTune.second.first.depth, (int)retTune.second.second.width, (int)retTune.second.second.height, (int)retTune.second.second.depth, (int)retTune.first);
        }
        //printf("conv1x1 idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);
        mPipeline = [context pipelineWithName:shaderName[min_cost.second] fp16:backend->useFp16InsteadFp32()];
    }

    return NO_ERROR;
}

void MetalConvolution1x1::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0];
    auto output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    MetalBackend::setTensor(mWeight.get(), encoder, 3);
    MetalBackend::setTensor(mBias.get(), encoder, 4);
    if (mDequantScaleBias) {
        MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
    }
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
