//
//  RopeBufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of RoPE (Rotary Positional Embedding).
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef RopeBufExecution_hpp
#define RopeBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class RopeBufExecution : public CommonExecution {
public:
    RopeBufExecution(const MNN::Op* op, Backend* backend);
    RopeBufExecution(const MNN::Op* op, Backend* backend, int ropeCutHeadDim, std::shared_ptr<cl::Buffer> qGamma,
                     float qEps, std::shared_ptr<cl::Buffer> kGamma, float kEps);
    virtual ~RopeBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    OpenCLBackend* mOpenCLBackend = nullptr;
    uint32_t mMaxWorkGroupSize = 0;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize = {1, 1, 1};
    int mRopeCutHeadDim = 0;
    std::shared_ptr<cl::Buffer> mQGamma;
    std::shared_ptr<cl::Buffer> mKGamma;
    float mQEps = 0.0f;
    float mKEps = 0.0f;
};

} // namespace OpenCL
} // namespace MNN

#endif /* RopeBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
