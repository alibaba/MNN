//
//  GatedRMSNormBufExecution.hpp
//  MNN
//
//  OpenCL (buffer mode) execution for OpType_GatedRMSNorm:
//  out = (RMSNorm(x) * gamma + beta) * silu(z).
//
//  See GatedRMSNormBufExecution.cpp for why this op is kept whole here.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)
#ifndef GatedRMSNormBufExecution_hpp
#define GatedRMSNormBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

struct GatedRMSNormResource {
    // Bound as a pair or not at all, matching the kernels' GAMMA_BETA option.
    std::shared_ptr<cl::Buffer> mGammaBuffer;
    std::shared_ptr<cl::Buffer> mBetaBuffer;
    bool hasGammaBeta = false;
    float epsilon = 0.0f;
};

class GatedRMSNormBufExecution : public CommonExecution {
public:
    GatedRMSNormBufExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    GatedRMSNormBufExecution(std::shared_ptr<GatedRMSNormResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~GatedRMSNormBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<GatedRMSNormResource> mResource;
    OpenCLBackend* mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN

#endif /* GatedRMSNormBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE && MNN_GATED_RMS_NORM */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
