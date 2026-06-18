//
//  MetalRope.mm
//  MNN
//
//  Fused RoPE (Rotary Positional Embedding) kernel for Metal backend via Extra op path.
//
//  Inputs:  x, cosEven, cosOdd, sinEven, sinOdd
//  Output:  same shape as x
//
//  For last dimension D (must be even), let halfD = D/2 and split x as
//    even = x[..., 0:halfD]
//    odd  = x[..., halfD:]
//  Then compute
//    q0 = even * cosEven - odd * sinEven
//    q1 = odd  * cosOdd + even * sinOdd
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
#include <vector>

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

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

#if defined(Q_NORM) || defined(K_NORM)
kernel void rope_kernel(
                        const device ftype* q           [[ buffer(0) ]],
                        const device ftype* k           [[ buffer(1) ]],
                        const device ftype* cosEven     [[ buffer(2) ]],
                        const device ftype* cosOdd      [[ buffer(3) ]],
                        const device ftype* sinEven     [[ buffer(4) ]],
                        const device ftype* sinOdd      [[ buffer(5) ]],
                        device ftype* qo                 [[ buffer(6) ]],
                        device ftype* ko                 [[ buffer(7) ]],
                        constant RopeParam& p           [[ buffer(8) ]],
#ifdef Q_NORM
                        const device float* qGamma      [[ buffer(9) ]],
#endif
#ifdef K_NORM
                        const device float* kGamma      [[ buffer(10) ]],
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

    const device ftype* x = q + actual_z * p.D + gid.y * p.D * p.numHead;
    device ftype* y = qo + actual_z * p.D + gid.y * p.D * p.numHead;
    bool isQ = true;
    if (actual_z >= p.numHead) {
        x = k + (actual_z-p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
        y = ko + (actual_z-p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
        isQ = false;
    }
    
    float square_sum = 0.0f;
#ifdef Q_NORM
    if (isQ) {
        for (int i = start; i < p.D; i += step) {
            float val = x[i];
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
            float val = x[i];
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
        ftype evenVal = x[i];
        ftype oddVal  = x[i + p.halfD];
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
            int cosIndex = gid.y * p.halfD + i;
            ftype cEven = cosEven[cosIndex];
            ftype cOdd  = cosOdd[cosIndex];
            ftype sEven = sinEven[cosIndex];
            ftype sOdd  = sinOdd[cosIndex];

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
                        const device ftype* cosEven     [[ buffer(2) ]],
                        const device ftype* cosOdd      [[ buffer(3) ]],
                        const device ftype* sinEven     [[ buffer(4) ]],
                        const device ftype* sinOdd      [[ buffer(5) ]],
                        device ftype* qo                 [[ buffer(6) ]],
                        device ftype* ko                 [[ buffer(7) ]],
                        constant RopeParam& p           [[ buffer(8) ]],
                        uint3 gid                      [[ thread_position_in_grid]]) {
    if (gid.x >= (uint)p.halfD || gid.y >= (uint)p.outerSize || gid.z >= p.fullHead) {
        return;
    }
    const device ftype* x = q + gid.z * p.D + gid.y * p.D * p.numHead;
    device ftype* y = qo + gid.z * p.D + gid.y * p.D * p.numHead;
    if (gid.z >= p.numHead) {
        x = k + (gid.z-p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
        y = ko + (gid.z-p.numHead) * p.D + gid.y * p.D * p.kvnumHead;
    }
    ftype evenVal = x[gid.x];
    ftype oddVal  = x[gid.x + p.halfD];

    if (gid.x < (uint)p.ropeHalfD) {
        int cosIndex = gid.y * p.halfD + gid.x;

        ftype cEven = cosEven[cosIndex];
        ftype cOdd  = cosOdd[cosIndex];
        ftype sEven = sinEven[cosIndex];
        ftype sOdd  = sinOdd[cosIndex];

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
    explicit MetalRopeExecution(Backend *backend, int ropeCutHeadDim, std::shared_ptr<MetalLayerNorm::Resource> qNorm, std::shared_ptr<MetalLayerNorm::Resource> kNorm)
        : MetalExecution(backend), mRopeCutHeadDim(ropeCutHeadDim), mQNorm(qNorm), mKNorm(kNorm) {
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
        int headDim = q->length(3);
        int batch = q->length(0);
        int seqLen = q->length(1);
        int numHead = q->length(2);
        int kvnumHead = k->length(2);

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
        MetalBackend::setTensor(inputs[4], encoder, 4);
        MetalBackend::setTensor(inputs[5], encoder, 5);
        MetalBackend::setTensor(outputs[0], encoder, 6);
        MetalBackend::setTensor(outputs[1], encoder, 7);
        [encoder setBuffer:mParam offset:0 atIndex:8];
        if (mQNorm && mQNorm->mGammaBuffer) {
            MetalBackend::setTensor(mQNorm->mGammaBuffer.get(), encoder, 9);
        }
        if (mKNorm && mKNorm->mGammaBuffer) {
            MetalBackend::setTensor(mKNorm->mGammaBuffer.get(), encoder, 10);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
    
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto rope = new MetalRopeExecution(bn, mRopeCutHeadDim, mQNorm, mKNorm);
        *dst = rope;
        return true;
    }

private:
    int mRopeCutHeadDim = 0;
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
        if (nullptr != op && OpParameter_Extra == op->main_type()) {
            auto extra = op->main_as_Extra();
            if (nullptr != extra && nullptr != extra->attr()) {
                for (int i = 0; i < extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (nullptr == attr || nullptr == attr->key()) {
                        continue;
                    }
                    if (attr->key()->str() == "rope_cut_head_dim") {
                        ropeCutHeadDim = attr->i();
                        continue;
                    }
                    if (attr->key()->str() == "q_norm") {
                        auto qLayernorm = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                        qNorm = MetalLayerNorm::makeResource(backend, qLayernorm->main_as_LayerNorm());
                        continue;
                    }
                    if (attr->key()->str() == "k_norm") {
                        auto kLayernorm = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                        kNorm = MetalLayerNorm::makeResource(backend, kLayernorm->main_as_LayerNorm());
                        continue;
                    }
                }
            }
        }
        return new MetalRopeExecution(backend, ropeCutHeadDim, qNorm, kNorm);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRoPECreator, OpType_RoPE);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif // MNN_METAL_ENABLED
