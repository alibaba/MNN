//
//  LrnExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LrnExecution_hpp
#define LrnExecution_hpp

#include <MNN_generated.h>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

#include <vector>

namespace MNN {
namespace OpenCL {

class LrnExecution : public Execution {
public:
    LrnExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~LrnExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cl::Kernel mKernel;
    cl::Kernel mBufferToImageKernel;
    cl::Kernel mImageToBufferKernel;
    int mRegionType;
    int mLocalSize;
    float mAlpha;
    float mBeta;
    OpenCLBackend *mOpenCLBackend;
    uint32_t mMaxWorkGroupSize;

    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LrnExecution_hpp */
