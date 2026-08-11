//
//  MetalGatedRMSNorm.mm
//  MNN
//
//  Metal execution for OpType_GatedRMSNorm: out = RMSNorm(x) * silu(z).
//
//  Replaces what used to be discovered at runtime by
//  MetalBackend::matchLinearAttnGatedNormFolds, which walked the
//  Raster/Cast/RMSNorm/SILU/MUL/Raster chain and claimed its six executions.
//  The exporter now emits this op directly, so the backend just runs it.
//
//  The kernel is unchanged (MetalGatedNormShader.hpp): its RMS reduction mirrors
//  layernorm_c4_rms_sg line for line, which is what makes the result bit-identical
//  to the unfused chain in fp32 builds. Do not "optimize" the grid or the
//  reduction order — see skills/metal-optimize/kernel-dev-and-optimize.md §2.4.4.
//
//  Layout: x is NC4HW4 [outside, inside] with the head as the batch axis; z and
//  the output are NC4HW4 [1, outside*inside] and contiguous. Reading z and
//  writing out at the flattened index absorbs the two C4 repacks that used to
//  bracket the chain (they are exact inverses).
//

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalExecution.hpp"
#import "backend/metal/MetalGatedNormShader.hpp"
#import "MNN_generated.h"
#import "core/TensorUtils.hpp"
#import "core/OpCommonUtils.hpp"
#import "core/Macro.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {
#ifdef MNN_GATED_RMS_NORM

class MetalGatedRMSNorm : public MetalExecution {
public:
    struct Resource {
        std::shared_ptr<Tensor> mGamma;
        std::shared_ptr<Tensor> mBeta;
        float mEps = 0.f;
        int mGammaSize = 0;
    };

    MetalGatedRMSNorm(Backend *backend, std::shared_ptr<Resource> res)
        : MetalExecution(backend), mResource(res) {}
    virtual ~MetalGatedRMSNorm() = default;

    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override {
        if (nullptr == dst) {
            return true;
        }
        *dst = new MetalGatedRMSNorm(bn, mResource);
        return true;
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override {
        auto mtbn = static_cast<MetalBackend *>(backend());
        auto rt   = (MetalRuntime *)mtbn->runtime();
        auto x    = inputs[0];
        mOutside  = x->length(0);
        mInside   = x->length(1);
        if (mOutside <= 0 || mInside <= 0 || (mInside % 4) != 0) {
            return NOT_SUPPORT;
        }
        if (mResource->mGammaSize != mInside) {
            return NOT_SUPPORT;
        }

        const bool fp16   = mtbn->useFp16InsteadFp32();
        std::string ftype  = fp16 ? "half" : "float";
        std::string ftype4 = fp16 ? "half4" : "float4";
        // One simdgroup per head keeps the RMS reduction inside a simdgroup,
        // exactly as layernorm_c4_rms_sg does.
        const int sgsPerTG = 1;

        std::vector<std::string> keys = {"linear_attn_gated_norm", ftype,
                                         "sgs" + std::to_string(sgsPerTG)};
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            option.preprocessorMacros = @{
                @"ftype" : @(ftype.c_str()),
                @"ftype4" : @(ftype4.c_str()),
                @"SGS_PER_TG" : @(std::to_string(sgsPerTG).c_str()),
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedNorm,
                                                                 "linear_attn_gated_norm", option);
            if (nil == pipeline) {
                return NOT_SUPPORT;
            }
            rt->insertPipeline(keys, pipeline);
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(sgsPerTG * 32)) {
            return NOT_SUPPORT;
        }
        mPipeline = pipeline;

        mParam     = mtbn->getConstBuffer(4 * sizeof(int));
        auto param = (int *)mParam.contents;
        param[0]   = mInside;
        param[1]   = mOutside;
        ((float *)param)[2] = mResource->mEps;
        param[3]   = 1; // gamma/beta are required, see the creator

        mThreads = std::make_pair(MTLSizeMake(1, UP_DIV(mOutside, sgsPerTG), 1),
                                  MTLSizeMake(sgsPerTG * 32, 1, 1));
        return NO_ERROR;
    }

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);   // x, [outside, inside]
        MetalBackend::setTensor(inputs[1], encoder, 1);   // z, [1, outside*inside]
        MetalBackend::setTensor(outputs[0], encoder, 2);
        [encoder setBuffer:mParam offset:0 atIndex:3];
        MetalBackend::setTensor(mResource->mGamma.get(), encoder, 4);
        MetalBackend::setTensor(mResource->mBeta.get(), encoder, 5);
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }

private:
    std::shared_ptr<Resource> mResource;
    id<MTLComputePipelineState> mPipeline = nil;
    id<MTLBuffer> mParam = nil;
    std::pair<MTLSize, MTLSize> mThreads;
    int mOutside = 0;
    int mInside  = 0;
};

class MetalGatedRMSNormCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                const std::vector<Tensor *> &outputs) const override {
        auto mtbn = static_cast<MetalBackend *>(backend);
        auto rt   = (MetalRuntime *)mtbn->runtime();
        // Same predicate the geometry gate uses: if it fails there the op was
        // already decomposed, so reaching here with a false means the two drifted.
        if (!OpCommonUtils::gatedRMSNormFusable(op, inputs, outputs, rt->supportSimdGroupReduce())) {
            return nullptr;
        }
        auto param = op->main_as_LayerNorm();
        const bool inlineGammaBeta = (param->gamma() != nullptr && param->beta() != nullptr);
        int gammaSize = 0;
        if (inlineGammaBeta) {
            gammaSize = (int)param->gamma()->size();
        } else if (param->external() != nullptr && param->external()->size() >= 2) {
            gammaSize = (int)(param->external()->data()[1] / sizeof(float));
        }
        // External gamma/beta are inlined by createExecutionWithExternal before
        // this creator runs; under cached mmap the STATIC buffers are restored
        // from the cache instead, so only allocation happens here (same pattern
        // as MetalLayerNorm::makeResource).
        const bool useCachedMmap = mtbn->getRuntime()->hint().useCachedMmap > 1;
        if (gammaSize <= 0 || (!inlineGammaBeta && !useCachedMmap)) {
            return nullptr;
        }

        auto res = std::make_shared<MetalGatedRMSNorm::Resource>();
        res->mEps       = param->epsilon();
        res->mGammaSize = gammaSize;
        res->mGamma.reset(Tensor::createDevice<uint8_t>({(int)(gammaSize * sizeof(float))}));
        res->mBeta.reset(Tensor::createDevice<uint8_t>({(int)(gammaSize * sizeof(float))}));
        if (!backend->onAcquireBuffer(res->mGamma.get(), Backend::STATIC) ||
            !backend->onAcquireBuffer(res->mBeta.get(), Backend::STATIC)) {
            MNN_ERROR("MetalGatedRMSNorm: failed to allocate gamma/beta\n");
            return nullptr;
        }
        if (inlineGammaBeta && !useCachedMmap) {
            auto gammaPtr = MetalBackend::getBuffer(res->mGamma.get());
            ::memcpy((uint8_t *)gammaPtr.first.contents + gammaPtr.second, param->gamma()->data(),
                     gammaSize * sizeof(float));
            auto betaPtr = MetalBackend::getBuffer(res->mBeta.get());
            ::memcpy((uint8_t *)betaPtr.first.contents + betaPtr.second, param->beta()->data(),
                     gammaSize * sizeof(float));
        }
        return new MetalGatedRMSNorm(backend, res);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalGatedRMSNormCreator, OpType_GatedRMSNorm);
#else
void ___MetalGatedRMSNormCreator__OpType_GatedRMSNorm__() {
}
#endif

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
