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
            // quant type equal to 3 means fp16, fallback to float weight
            if(conv2D->quanParameter()->type() != 3) {
            	ldInt8Weight = true;
            }
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
    int ic  = input->channel();
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
    int area = ob * ow * oh;
    // basic marco info
    std::string ftype = "float";
    std::string ftype2 = "float2";
    std::string ftype4 = "float4";
    std::string ftype2x4 = "float2x4";
    std::string ftype4x4 = "float4x4";
    if (backend->useFp16InsteadFp32()) {
        ftype = "half";
        ftype2 = "half2";
        ftype4 = "half4";
        ftype2x4 = "half2x4";
        ftype4x4 = "half4x4";
    }

    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto baseDic = [NSMutableDictionary dictionaryWithCapacity:0];
    [baseDic setValue:@(ftype.c_str()) forKey:@"ftype"];
    [baseDic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
    [baseDic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
    [baseDic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
    [baseDic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
    [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
    if (backend->useFp16InsteadFp32()) {
        [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
    }
    std::vector<std::string> baseKeys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};

    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    std::string basicShaderPrefix = gBasicConvPrefix;
    
    // if M is small, dequant weight in shader
    // if device not support simdgroup matrix, only support dequant in shader
    bool dequantInShader = (area < 128) || !(rt->supportSimdGroupMatrix());
    mPreDequantWeight = false;
    
#ifdef MNN_LOW_MEMORY
    if (mDequantScaleBias.get() && dequantInShader) {
        //printf("inner dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);
        
        std::string sgmWqShader  = gConv1x1WqSgMatrix;
        std::string sgrWqShader  = gConv1x1WqSgReduce;

        NSMutableDictionary *dic = [baseDic mutableCopy];
        if(mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if(mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        option.preprocessorMacros = dic;
        
        NSUInteger gid_x = UP_DIV(ow * oh, 4);
        NSUInteger gid_y = oc_4;
        NSUInteger gid_z = ob;
        std::string name = "conv1x1_g1z4_w8";
        mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];
        
        if (mDequantBits == 4 || mDequantBits == 8) {
            // TODO: define short_seq more accurately
            int short_seq = 6;
            
            if(mDequantBits == 4) {
                baseKeys.emplace_back("conv1x1_wquant_4");
            } else if(mDequantBits == 8) {
                baseKeys.emplace_back("conv1x1_wquant_8");
            }
            if(rt->supportSimdGroupReduce() && area <= short_seq) {
                baseKeys.emplace_back("conv1x1_wquant_sg_reduce");
                
                std::string sgrWqStr = basicShaderPrefix + sgrWqShader;
                if(area > 1) {
                    auto keys = baseKeys;
                    int piece = 1;
                    // memory bound not so seriously, can add more thread to reduce computation in each thread
                    float ratio = 1.0 * ic_4 / 2048.0 * oc / 2048.0;
                    bool heavyMemory = ratio > 1.0;
                    if(area > 5 && !heavyMemory) {
                        if(area % 2 != 0) {
                            keys.emplace_back("MNN_METAL_SRC_PROTECT");
                            [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];;
                            option.preprocessorMacros = dic;
                        }
                        area = UP_DIV(area, 2);
                        piece = 2;
                    }
//                    MNN_PRINT("Conv1x1 Oc:%d Ic:%d\n", oc, ic_4*4);
                    std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                    keys.emplace_back(kernel_name);
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), piece, 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 16384 && oc_4 % 2 == 0) {
                    // unrool c for avoid memory exceed
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), area, 1), MTLSizeMake(64, 1, 1));
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g8_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g8_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
//                    MNN_PRINT("g8  ic: %d oc: %d\n", input->channel(), oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), area, 1), MTLSizeMake(64, 1, 1));
                }
                return NO_ERROR;
            } else if(rt->supportSimdGroupMatrix()  && area > short_seq && oc > 8 && ic_4 % 8 == 0) {
                baseKeys.emplace_back("conv1x1_wquant_sg_matrix");

                std::string sgmWqStr = basicShaderPrefix + sgmWqShader;

                // Generally threadgroup memory >= 16KB
                auto smem_size = [[context device] maxThreadgroupMemoryLength];
                // choose different tile for different computation
                if(area >= 128 && oc >= 512 && area * oc > 512 * 2048 && smem_size >= 8192) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x64_wquant_split_k_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x64_wquant_split_k_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));
                                        
                } else if(area >= 32 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 512 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x32_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x32_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 32), 1), MTLSizeMake(32, 1, 1));
                } else if(area < 16) {
                    // TODO: define useMatrix more accurate
                    bool useMatrix = area > 6 && oc > 2048 && ic*2 < oc;
                    if(useMatrix) {
                        auto keys = baseKeys;
                        int oc_block = (oc > 4096) ? 32 : 16;
                        std::string kernel_name = "conv1x1_gemm_8x" + std::to_string(oc_block) + "_wquant_sg";

                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline;
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 8), UP_DIV(oc, oc_block), 1), MTLSizeMake(32, 1, 1));
                    } else {
                        std::string sgrWqStr = basicShaderPrefix + sgrWqShader;

                        auto keys = baseKeys;
                        std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline;
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), 1, 1), MTLSizeMake(32, 1, 1));
                    }
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
//                                    MNN_PRINT("gemm M: %d N: %d\n", area, oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                }
                return NO_ERROR;
            } else if(mDequantBits == 4) {
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w4" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w4";
            } else {
                // mDequantBits == 8
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w8";
            }
        } else {
            MNN_ERROR("metal conv weight quant not support %d bits yet!\n", mDequantBits);
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
#endif

    std::string sgmWfpShader = gConv1x1WfpSgMatrix;
    std::string sgrWfpShader = gConv1x1WfpSgReduce;
    
    // Dequant using single shader
    if (mDequantScaleBias.get()) {
        baseKeys.emplace_back("conv1x1_dequant_weight_outter");
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;
        
        mPreDequantWeight = true;
        {
            NSMutableDictionary *dic = [baseDic mutableCopy];
            
            if(mDequantBits == 4) {
                [dic setValue:@"1" forKey:@"W_QUANT_4"];
            } else if(mDequantBits == 8) {
                [dic setValue:@"1" forKey:@"W_QUANT_8"];
            }
            if(ic % 16 != 0) {
                [dic setValue:@"1" forKey:@"W_ALIGN_K16_PROTECT"];
            }
            option.preprocessorMacros = dic;
            
            int bytes = backend->useFp16InsteadFp32() ? 2 : 4;
            // accquire space
            mTempWeight.reset(Tensor::createDevice<uint8_t>(std::vector<int>{ROUND_UP(oc, 4) * ROUND_UP(ic, 16) * bytes}));
            backend->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
            backend->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
            
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_w_dequant");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_w_dequant", option);
                rt->insertPipeline(keys, pipeline);
            }
            mDequantPipeline = pipeline;
            
            mDequantThreads = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(UP_DIV(oc, 1),  UP_DIV(ic, 16), 1)];
        }
        
        {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_gemm_32x64_split_k_sg");
            
            NSMutableDictionary *dic = [baseDic mutableCopy];
            if(backend->isSupportTensorApi() == true) {
                [dic setValue:@"1" forKey:@"USE_METAL_TENSOR_OPS"];
                keys.emplace_back("USE_METAL_TENSOR_OPS");
                if(ic > oc && ic > 2048 && (ic / blockSize) % 64 == 0) {
                    [dic setValue:@"1" forKey:@"LOOP_K64"];
                    keys.emplace_back("LOOP_K64");
                }
            }
            option.preprocessorMacros = dic;

            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x64_split_k_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline;
            mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));
            //printf("out dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);
        }
            
        return NO_ERROR;
    }
    
    option.preprocessorMacros = baseDic;

    if(rt->supportSimdGroupMatrix()) {
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_matrix");
        // total computation not too small
        if(area >= 16 && ic_4 >= 4 && ic_4 % 2 == 0 && oc_4 >= 4 && area * ic_4 * oc_4 >= 64 * 64 * 64) {
            // Enough threads
            if(area * oc_4 / ic_4 >= 1024) {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_32x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_16x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_16x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
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
        if(ic_4 >= 32 && ic_4 % 2 == 0 && 1.0 * area * oc_4 / ic_4 < magic_num) {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_z4_sg");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgrWfpStr.c_str(), "conv1x1_z4_sg", option);
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
    if(mPreDequantWeight) {
        // pre dequant weight pipeline
        {
            [encoder setComputePipelineState:mDequantPipeline];
            MetalBackend::setTensor(mWeight.get(), encoder, 0);
            MetalBackend::setTensor(mTempWeight.get(), encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 3);
            [encoder dispatchThreadgroups:mDequantThreads.first threadsPerThreadgroup:mDequantThreads.second];
        }
        // convolution pipeline
        {
            [encoder setComputePipelineState:mPipeline];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(mTempWeight.get(), encoder, 3);
            MetalBackend::setTensor(mBias.get(), encoder, 4);
            MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
            [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        }
    } else {
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
#ifdef MNN_METAL_DEBUG_INFO
    if(!static_cast<MetalBackend*>(backend())->useFp16InsteadFp32()) {
        {
            static_cast<MetalBackend*>(backend())->flushEncoder();
            static_cast<MetalBackend*>(backend())->commit_net();
            static_cast<MetalBackend*>(backend())->wait();
            
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(input);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(mWeight.get());
            auto ptr = (int8_t*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%d ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(mDequantScaleBias.get());
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
        
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(output);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
    }
#endif
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
