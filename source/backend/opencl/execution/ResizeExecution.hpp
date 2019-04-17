//
//  ResizeExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ResizeExecution_hpp
#define ResizeExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class ResizeExecution : public Execution {
public:
    ResizeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ResizeExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mXScale;
    float mYScale;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    cl::Kernel mImageToBufferKernel;
    cl::Kernel mBufferToImageKernel;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ResizeExecution_hpp */
