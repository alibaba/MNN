//
//  DepthwiseConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DepthwiseConvExecution_hpp
#define DepthwiseConvExecution_hpp

#include "ConvExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvExecution : public ConvCommonExecution {
public:
    DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DepthwiseConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> depthwiseConvLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

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
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvExecution_hpp */
