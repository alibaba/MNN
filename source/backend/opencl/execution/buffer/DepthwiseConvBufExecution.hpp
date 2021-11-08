//
//  DepthwiseConvBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef DepthwiseConvBufExecution_hpp
#define DepthwiseConvBufExecution_hpp

#include "ConvBufExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvBufExecution : public ConvBufCommonExecution {
public:
    DepthwiseConvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DepthwiseConvBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::pair<std::vector<uint32_t>,  int> DepthwiseConvBufLwsTune(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize);
private:
    const Convolution2DCommon *mConv2dCommonParams;
    const Convolution2D *mCon2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    bool mStride_1 = false;
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
