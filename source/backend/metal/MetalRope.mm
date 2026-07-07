//
//  MetalRope.mm
//  MNN
//
//  Fused RoPE (Rotary Positional Embedding) kernel for Metal backend.
//
//  Inputs:  x, cos, sin
//  Output:  same shape as x
//
//  For last dimension D (must be even), let halfD = D/2 and split x as
//    even = x[..., 0:halfD]
//    odd  = x[..., halfD:]
//  Then compute
//    q0 = even * cos[i] - odd * sin[i]
//    q1 = odd  * cos[i + halfD] + even * sin[i + halfD]
//  and concatenate [q0, q1] along the last dimension.
//

#define MNN_UNUSED(x)
#import "MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"
#import "MetalExecution.hpp"
#import "MetalLayerNorm.hpp"
#import "core/TensorUtils.hpp"
#import "core/Macro.h"
#include "MNN_generated.h"
#include <cstring>
#include <vector>

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

static std::shared_ptr<MetalLayerNorm::Resource> makeRopeNormResource(Backend* backend, const LayerNorm* layerNorm) {
    if (nullptr == layerNorm || nullptr == layerNorm->gamma()) {
        return nullptr;
    }
    int gammaSize = layerNorm->gamma()->size();
    if (gammaSize <= 0) {
        return nullptr;
    }
    std::shared_ptr<MetalLayerNorm::Resource> res(new MetalLayerNorm::Resource);
    res->mGroup = layerNorm->group();
    res->mEps = layerNorm->epsilon();
    res->mAxisSize = layerNorm->axis() == nullptr ? 1 : layerNorm->axis()->size();
    res->mHasGammaBeta = true;
    res->mRMSNorm = layerNorm->useRMSNorm();
    res->mGammaSize = gammaSize;
    res->mGammaBuffer.reset(Tensor::createDevice<uint8_t>({gammaSize * (int)sizeof(float)}));
    if (!backend->onAcquireBuffer(res->mGammaBuffer.get(), Backend::STATIC)) {
        MNN_ERROR("MetalRope: alloc q/k norm gamma buffer error.\n");
        return nullptr;
    }
    auto gammaPtr = MetalBackend::getBuffer(res->mGammaBuffer.get());
    ::memcpy((uint8_t*)gammaPtr.first.contents + gammaPtr.second, layerNorm->gamma()->data(),
             gammaSize * sizeof(float));
    return res;
}

static bool validRopeC4Input(const Tensor* q, const Tensor* k, int numHead, int kvNumHead, int headDim) {
    if (q == nullptr || k == nullptr || numHead <= 0 || kvNumHead <= 0 || headDim <= 0) {
        return false;
    }
    if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
    }
    if (q->dimensions() < 2 || k->dimensions() < 2) {
        return false;
    }
    return q->length(1) == numHead * headDim && k->length(1) == kvNumHead * headDim;
}

struct RopeParam {
    int outerSize;
    int halfD;
    int ropeHalfD;
    int D;
    int numHead;
    int kvnumHead;
    int fullHead;
    float qEps;
    float kEps;
};

// Metal kernel source. ftype is float / half selected by MNN_METAL_FLOAT16_STORAGE.
static const char* gMetalRopeKernelSource = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
#ifdef MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct RopeParam {
    int outerSize;
    int halfD;
    int ropeHalfD;
    int D;
    int numHead;
    int kvnumHead;
    int fullHead;
    float qEps;
    float kEps;
};

static inline int c4Offset(int token, int channel, int seqLen) {
    return (channel / 4) * seqLen * 4 + token * 4 + (channel % 4);
}

static inline ftype loadC4(const device ftype* tensor, int token, int base, int offset, int seqLen) {
    if (seqLen == 1) {
        return tensor[base + offset];
    }
    return tensor[c4Offset(token, base + offset, seqLen)];
}

#if defined(Q_NORM) || defined(K_NORM)
kernel void rope_kernel(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cos         [[ buffer(2) ]],
                        const device ftype* sin         [[ buffer(3) ]],
                        device ftype* qo                 [[ buffer(4) ]],
                        device ftype* ko                 [[ buffer(5) ]],
                        constant RopeParam& p           [[ buffer(6) ]],
#ifdef Q_NORM
                        const device float* qGamma      [[ buffer(7) ]],
#endif
#ifdef K_NORM
                        const device float* kGamma      [[ buffer(8) ]],
#endif
#ifdef USE_SG
                        uint3 gid                      [[ threadgroup_position_in_grid]],
                        uint tiisg                     [[ thread_index_in_simdgroup]],
                        uint sgitg                     [[ simdgroup_index_in_threadgroup ]]
#else
                        uint3 gid                      [[ thread_position_in_grid]]
#endif
) {
#ifdef USE_SG
    uint actual_z = gid.z * 2 + sgitg;
    if (gid.y >= (uint)p.outerSize || actual_z >= p.fullHead) {
        return;
    }
    int step = 32;
    int start = tiisg;
#else
    uint actual_z = gid.z;
    if (gid.x >= 1 || gid.y >= (uint)p.outerSize || actual_z >= p.fullHead) {
        return;
    }
    int step = 1;
    int start = 0;
#endif

    bool isQ = true;
    const device ftype* xTensor = q;
    int xBase = actual_z * p.D;
    int xSeq = p.outerSize;
    device ftype* y = qo + actual_z * p.D + gid.y * p.D * p.numHead;
    if (actual_z >= p.numHead) {
        xTensor = k;
        xBase = (actual_z - p.numHead) * p.D;
        y = ko + (actual_z - p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
        isQ = false;
    }
    
    float square_sum = 0.0f;
#ifdef Q_NORM
    if (isQ) {
        for (int i = start; i < p.D; i += step) {
            float val = loadC4(xTensor, gid.y, xBase, i, xSeq);
            square_sum += val * val;
        }
#ifdef USE_SG
        square_sum = simd_sum(square_sum);
#endif
    }
#endif
#ifdef K_NORM
    if (!isQ) {
        for (int i = start; i < p.D; i += step) {
            float val = loadC4(xTensor, gid.y, xBase, i, xSeq);
            square_sum += val * val;
        }
#ifdef USE_SG
        square_sum = simd_sum(square_sum);
#endif
    }
#endif

    float var = 0;
#ifdef Q_NORM
    if (isQ) {
        var = 1.0 / sqrt(square_sum / p.D + p.qEps);
    }
#endif
#ifdef K_NORM
    if (!isQ) {
        var = 1.0 / sqrt(square_sum / p.D + p.kEps);
    }
#endif

    for (int i = start; i < p.halfD; i += step) {
        ftype evenVal = loadC4(xTensor, gid.y, xBase, i, xSeq);
        ftype oddVal  = loadC4(xTensor, gid.y, xBase, i + p.halfD, xSeq);
#ifdef Q_NORM
        if (isQ) {
            evenVal = evenVal * var * qGamma[i];
            oddVal  = oddVal * var * qGamma[i + p.halfD];
        }
#endif
#ifdef K_NORM
        if (!isQ) {
            evenVal = evenVal * var * kGamma[i];
            oddVal  = oddVal * var * kGamma[i + p.halfD];
        }
#endif

        if (i < p.ropeHalfD) {
            int cosIndex = gid.y * p.D + i;
            ftype cEven = cos[cosIndex];
            ftype cOdd  = cos[cosIndex + p.halfD];
            ftype sEven = sin[cosIndex];
            ftype sOdd  = sin[cosIndex + p.halfD];

            y[i]           = evenVal * cEven - oddVal * sEven;
            y[i + p.halfD] = oddVal  * cOdd  + evenVal * sOdd;
        } else {
            y[i]           = evenVal;
            y[i + p.halfD] = oddVal;
        }
    }
}
#else
kernel void rope_kernel(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cos         [[ buffer(2) ]],
                        const device ftype* sin         [[ buffer(3) ]],
                        device ftype* qo                 [[ buffer(4) ]],
                        device ftype* ko                 [[ buffer(5) ]],
                        constant RopeParam& p           [[ buffer(6) ]],
                        uint3 gid                      [[ thread_position_in_grid]]) {
    if (gid.x >= (uint)p.halfD || gid.y >= (uint)p.outerSize || gid.z >= p.fullHead) {
        return;
    }
    const device ftype* xTensor = q;
    int xBase = gid.z * p.D;
    int xSeq = p.outerSize;
    device ftype* y = qo + gid.z * p.D + gid.y * p.D * p.numHead;
    if (gid.z >= p.numHead) {
        xTensor = k;
        xBase = (gid.z - p.numHead) * p.D;
        y = ko + (gid.z-p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
    }
    ftype evenVal = loadC4(xTensor, gid.y, xBase, gid.x, xSeq);
    ftype oddVal  = loadC4(xTensor, gid.y, xBase, gid.x + p.halfD, xSeq);

    if (gid.x < (uint)p.ropeHalfD) {
        int cosIndex = gid.y * p.D + gid.x;

        ftype cEven = cos[cosIndex];
        ftype cOdd  = cos[cosIndex + p.halfD];
        ftype sEven = sin[cosIndex];
        ftype sOdd  = sin[cosIndex + p.halfD];

        ftype q0 = evenVal * cEven - oddVal * sEven;
        ftype q1 = oddVal  * cOdd  + evenVal * sOdd;

        y[gid.x]           = q0;
        y[gid.x + p.halfD] = q1;
    } else {
        y[gid.x]           = evenVal;
        y[gid.x + p.halfD] = oddVal;
    }
}
#endif
)metal";

class MetalRopeExecution : public MetalExecution {
public:
    explicit MetalRopeExecution(Backend *backend, int ropeCutHeadDim, std::shared_ptr<MetalLayerNorm::Resource> qNorm,
                                std::shared_ptr<MetalLayerNorm::Resource> kNorm, int numHead, int kvNumHead,
                                int headDim)
        : MetalExecution(backend),
          mRopeCutHeadDim(ropeCutHeadDim),
          mNumHead(numHead),
          mKvNumHead(kvNumHead),
          mHeadDim(headDim),
          mQNorm(qNorm),
          mKNorm(kNorm) {
        auto mtbn = static_cast<MetalBackend *>(backend);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(RopeParam) access:CPUWriteOnly];
        auto rt = static_cast<MetalRuntime*>(mtbn->getRuntime());
        std::vector<std::string> keys = {"rope_kernel"};
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        NSMutableDictionary *macros = [NSMutableDictionary dictionary];
        if (mtbn->useFp16InsteadFp32()) {
            macros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
            keys.emplace_back("fp16");
        }
        if (mQNorm) {
            macros[@"Q_NORM"] = @"1";
            keys.emplace_back("q_norm");
        }
        if (mKNorm) {
            macros[@"K_NORM"] = @"1";
            keys.emplace_back("k_norm");
        }
        if ((mQNorm || mKNorm) && rt->supportSimdGroupReduce()) {
            macros[@"USE_SG"] = @"1";
            keys.emplace_back("sg");
            mUseSG = true;
        } else {
            mUseSG = false;
        }
        option.preprocessorMacros = macros;
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline) {
            pipeline = mtbn->makeComputePipelineWithSourceOption(gMetalRopeKernelSource, "rope_kernel", option);
            rt->insertPipeline(keys, pipeline);
        }
        mPipeline = pipeline;
        if (nil == mPipeline) {
            MNN_ERROR("MetalRope: failed to compile rope_kernel.\n");
        }
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        MNN_ASSERT(6 == inputs.size());
        MNN_ASSERT(2 == outputs.size());
        auto q       = inputs[0];
        auto k       = inputs[1];
        if (!validRopeC4Input(q, k, mNumHead, mKvNumHead, mHeadDim)) {
            MNN_ERROR("MetalRope: invalid C4 input, numHead=%d, kvNumHead=%d, headDim=%d.\n", mNumHead, mKvNumHead,
                      mHeadDim);
            return NOT_SUPPORT;
        }
        int headDim = mHeadDim;
        int batch = 1;
        int seqLen = q->length(0);
        int numHead = mNumHead;
        int kvnumHead = mKvNumHead;

        RopeParam* p = (RopeParam*)(mParam.contents);
        p->outerSize  = static_cast<int>(batch * seqLen);
        p->halfD      = headDim / 2;
        int ropeDim = mRopeCutHeadDim;
        if (ropeDim <= 0 || ropeDim > headDim) {
            ropeDim = headDim;
        }
        ropeDim = (ropeDim / 2) * 2;
        p->ropeHalfD  = ropeDim / 2;
        p->D          = headDim;
        p->numHead    = numHead;
        p->kvnumHead  = kvnumHead;
        p->fullHead  = kvnumHead + numHead;
        p->qEps       = mQNorm ? mQNorm->mEps : 0.0f;
        p->kEps       = mKNorm ? mKNorm->mEps : 0.0f;
        auto mtbn = static_cast<MetalBackend *>(backend());
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        if (mQNorm || mKNorm) {
            if (mUseSG) {
                mThreads = std::make_pair(MTLSizeMake(1, p->outerSize, (NSUInteger)(numHead + kvnumHead + 1) / 2), MTLSizeMake(64, 1, 1));
            } else {
                mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(1, p->outerSize, (NSUInteger)(numHead + kvnumHead))];
            }
        } else {
            mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake((NSUInteger)p->halfD, p->outerSize, (NSUInteger)(numHead + kvnumHead))];
        }
        return NO_ERROR;
    }

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        if (nil == mPipeline) {
            return;
        }
        
        auto backend = static_cast<MetalBackend *>(this->backend());

        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(inputs[1], encoder, 1);
        MetalBackend::setTensor(inputs[2], encoder, 2);
        MetalBackend::setTensor(inputs[3], encoder, 3);
        MetalBackend::setTensor(outputs[0], encoder, 4);
        MetalBackend::setTensor(outputs[1], encoder, 5);
        [encoder setBuffer:mParam offset:0 atIndex:6];
        if (mQNorm && mQNorm->mGammaBuffer) {
            MetalBackend::setTensor(mQNorm->mGammaBuffer.get(), encoder, 7);
        }
        if (mKNorm && mKNorm->mGammaBuffer) {
            MetalBackend::setTensor(mKNorm->mGammaBuffer.get(), encoder, 8);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
    
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto rope = new MetalRopeExecution(bn, mRopeCutHeadDim, mQNorm, mKNorm, mNumHead, mKvNumHead, mHeadDim);
        *dst = rope;
        return true;
    }

private:
    int mRopeCutHeadDim = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    bool mUseSG = false;
    std::shared_ptr<MetalLayerNorm::Resource> mQNorm;
    std::shared_ptr<MetalLayerNorm::Resource> mKNorm;
    id<MTLBuffer> mParam = nil;
    id<MTLComputePipelineState> mPipeline = nil;
    std::pair<MTLSize, MTLSize> mThreads;
};

class MetalRoPECreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        int ropeCutHeadDim = 0;
        std::shared_ptr<MetalLayerNorm::Resource> qNorm;
        std::shared_ptr<MetalLayerNorm::Resource> kNorm;
        int numHead = 0;
        int kvNumHead = 0;
        int headDim = 0;
        auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
        if (param != nullptr) {
            ropeCutHeadDim = param->rope_cut_head_dim();
            numHead = param->num_head();
            kvNumHead = param->kv_num_head();
            headDim = param->head_dim();
            qNorm = makeRopeNormResource(backend, param->q_norm());
            kNorm = makeRopeNormResource(backend, param->k_norm());
        }
        return new MetalRopeExecution(backend, ropeCutHeadDim, qNorm, kNorm, numHead, kvNumHead, headDim);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRoPECreator, OpType_RoPE);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif // MNN_METAL_ENABLED
