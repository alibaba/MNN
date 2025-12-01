//
//  MetalLayerNorm.mm
//  MNN
//
//  Created by MNN on 2022/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalLayerNorm.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"
#import "LayerNormSimdGroupShader.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalLayerNorm::MetalLayerNorm(Backend *backend, std::shared_ptr<Resource> res)
    : MetalExecution(backend) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    mShapeBuffer = [context newDeviceBuffer:3 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    mResource = res;
}

bool MetalLayerNorm::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalLayerNorm(bn, mResource);
    return true;
}

std::shared_ptr<MetalLayerNorm::Resource> MetalLayerNorm::makeResource(Backend *backend, const LayerNorm *layernorm) {
    std::shared_ptr<MetalLayerNorm::Resource> res(new Resource);
    res->mGroup = layernorm->group();
    res->mEps = layernorm->epsilon();
    res->mRMSNorm = layernorm->useRMSNorm();
    int axis_size = 0;
    if (nullptr != layernorm->axis()) {
        axis_size = layernorm->axis()->size();
    }
    res->mAxisSize = axis_size;
    int gamma_size = 0;
    if (layernorm->gamma() && layernorm->beta()) {
        gamma_size = layernorm->gamma()->size();
    }
    if (layernorm->external() != nullptr && layernorm->external()->size() >= 2) {
        auto externalInfo = layernorm->external()->data();
        auto externalSize = layernorm->external()->size();
        gamma_size = static_cast<int32_t>(externalInfo[1]) / sizeof(float);
    }
    if (gamma_size > 0) {
        res->mHasGammaBeta = true;
        res->mGammaBuffer.reset(Tensor::createDevice<uint8_t>({(int)(gamma_size * sizeof(float))}));
        res->mBetaBuffer.reset(Tensor::createDevice<uint8_t>({(int)(gamma_size * sizeof(float))}));
        auto allocRes = backend->onAcquireBuffer(res->mGammaBuffer.get(), Backend::STATIC);
        allocRes = allocRes && backend->onAcquireBuffer(res->mBetaBuffer.get(), Backend::STATIC);
        if (!allocRes) {
            MNN_ERROR("MetalLayerNorm: Alloca gamma and beta buffer error!\n");
            return nullptr;
        }
    }
    auto useCache = backend->getRuntime()->hint().useCachedMmap > 1;
    if (layernorm->gamma() && layernorm->beta() && (!useCache)) {
        const float* gamma_data = layernorm->gamma()->data();
        auto gammaPtr = MetalBackend::getBuffer(res->mGammaBuffer.get());
        memcpy((uint8_t*)gammaPtr.first.contents + gammaPtr.second, (const void *)gamma_data, gamma_size * sizeof(float));
        
        if (layernorm->beta()->size() != gamma_size) {
            MNN_ERROR("Size of gamma and beta are not match in MetalLayerNorm.\n");
        }

        const float* beta_data = layernorm->beta()->data();
        auto betaPtr = MetalBackend::getBuffer(res->mBetaBuffer.get());
        memcpy((uint8_t*)betaPtr.first.contents + betaPtr.second, (const void *)beta_data, gamma_size * sizeof(float));
    }
    return res;
}

ErrorCode MetalLayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    auto input = inputs[0], output = outputs[0];
    
    mOutside = 1;
    mInside = 1;
    int rank = input->dimensions();
    if (mResource->mGroup > 1) {
        mOutside = input->length(0) * mResource->mGroup;
        for (int i = 1; i < rank; i++) {
            mInside *= input->length(i);
        }
        mInside /= mResource->mGroup;
    } else {
        for (int i = 0; i < rank - mResource->mAxisSize; ++i) {
            mOutside *= input->length(i);
        }
        for (int i = rank - mResource->mAxisSize; i < rank; ++i) {
            mInside *= input->length(i);
        }
    }

    ((int *)mShapeBuffer.contents)[0]   = mInside;
    ((int *)mShapeBuffer.contents)[1]   = mOutside;
    ((float *)mShapeBuffer.contents)[2] = mResource->mEps;
    ((int *)mShapeBuffer.contents)[3]   = (int)mResource->mHasGammaBeta;

    bool parallel = (mInside > 32) && ((mInside & 3) == 0);
    auto inside = parallel ? mInside/4 : mInside;
    auto rt = (MetalRuntime *)backend->runtime();
    if(rt->supportSimdGroupReduce()) {
        // basic marco info
        std::string ftype = "float";
        std::string ftype4 = "float4";
        if (backend->useFp16InsteadFp32()) {
            ftype = "half";
            ftype4 = "half4";
        }

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        option.preprocessorMacros = @{
            @"ftype" : @(ftype.c_str()),
            @"ftype4" : @(ftype4.c_str()),
        };
        std::vector<std::string> baseKeys = {"layernorm_sg_reduce", ftype};
        if(mResource->mRMSNorm) {
            // pretty much threads compute all inside dims in a threadgroup
            if(mOutside / 512.0 * mInside / 512.0 > 1.0) {
                auto keys = baseKeys;
                keys.emplace_back("layernorm_in_all_rms_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_in_all_rms_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(1, mOutside, 1), MTLSizeMake(32, 1, 1));
            } else if(parallel) {
                if(inside >= 16 && inside * mOutside >= 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("layernorm_x16_rms_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_x16_rms_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(inside, 4), mOutside, 1), MTLSizeMake(32, 1, 1));
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("layernorm_x4_rms_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_x4_rms_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline;
                    mThreads = std::make_pair(MTLSizeMake(inside, mOutside, 1), MTLSizeMake(32, 1, 1));
                }
            } else {                    
                auto keys = baseKeys;
                keys.emplace_back("layernorm_x1_rms_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_x1_rms_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(inside, mOutside, 1), MTLSizeMake(32, 1, 1));
            }
        } else {
            if(mOutside / 512.0 * mInside / 512.0 > 1.0) {
                auto keys = baseKeys;
                keys.emplace_back("layernorm_in_all_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_in_all_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(1, mOutside, 1), MTLSizeMake(32, 1, 1));
            } else if(parallel) {
                auto keys = baseKeys;
                keys.emplace_back("layernorm_x4_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_x4_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(inside, mOutside, 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("layernorm_x1_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(gLayerNormSgReduce, "layernorm_x1_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline;
                mThreads = std::make_pair(MTLSizeMake(inside, mOutside, 1), MTLSizeMake(32, 1, 1));
            }
        }
    } else {
        if(mResource->mRMSNorm){
            mPipeline = [context pipelineWithName:parallel ? @"layernorm_x4_rms" : @"layernorm_x1_rms" fp16:backend->useFp16InsteadFp32()];
        }else{
            mPipeline = [context pipelineWithName:parallel ? @"layernorm_x4" : @"layernorm_x1" fp16:backend->useFp16InsteadFp32()];
        }
        mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)inside, (NSUInteger)mOutside, 1)];
    }
    return NO_ERROR;
}

void MetalLayerNorm::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mShapeBuffer offset:0 atIndex:2];
    if (!mResource->mHasGammaBeta) {
        // Set fake buffer to avoid validate
        MetalBackend::setTensor(input, encoder, 3);
        MetalBackend::setTensor(input, encoder, 4);
    } else {
        MetalBackend::setTensor(mResource->mGammaBuffer.get(), encoder, 3);
        MetalBackend::setTensor(mResource->mBetaBuffer.get(), encoder, 4);
    }

    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    MNN_PRINT_ENCODER(context, encoder);
}

class MetalLayerNormCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const {
        auto res = MetalLayerNorm::makeResource(backend, op->main_as_LayerNorm());
        if (nullptr == res) {
            return nullptr;
        }
        return new MetalLayerNorm(backend, res);
    }
};
REGISTER_METAL_OP_CREATOR(MetalLayerNormCreator, OpType_LayerNorm);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
