//
//  DepthwiseConvInt8Execution.hpp
//  MNN
//
//  Created by MNN on 2019/6/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DepthwiseConvInt8Execution_hpp
#define DepthwiseConvInt8Execution_hpp

#include "CommonExecution.hpp"
#include <MNN_generated.h>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class DepthwiseConvInt8Execution : public Execution {
public:
    DepthwiseConvInt8Execution(Backend *backend, const MNN::Op *param);
    virtual ~DepthwiseConvInt8Execution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<cl::Buffer> mScaleBuffer;
    std::shared_ptr<cl::Buffer> mFilterBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
    OpenCLBackend *mOpenCLBackend;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};
}
} // namespace MNN

#endif /* DepthwiseConvInt8Execution_hpp */
