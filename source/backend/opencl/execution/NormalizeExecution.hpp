//
//  NormalizeExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NormalizeExecution_hpp
#define NormalizeExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"
#include "core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class NormalizeExecution : public Execution {
public:
    NormalizeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~NormalizeExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> normalizeLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    float mEps;
    const Normalize *mNormalizeParams;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mScale;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* NormalizeExecution_hpp */
