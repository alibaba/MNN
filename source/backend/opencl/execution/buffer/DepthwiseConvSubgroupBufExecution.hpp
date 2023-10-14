//
//  DepthwiseConvSubgroupBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#ifndef DepthwiseConvSubgroupBufExecution_hpp
#define DepthwiseConvSubgroupBufExecution_hpp

#include "ConvBufExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvSubgroupBufExecution : public ConvBufCommonExecution {
public:
    DepthwiseConvSubgroupBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DepthwiseConvSubgroupBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void transformWeight(const Tensor *weightDest, const Tensor *source);
    const Convolution2DCommon *mConv2dCommonParams;
    const Convolution2D *mCon2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    std::shared_ptr<Tensor> mSource;
    cl::Kernel mTranseKernel;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::vector<uint32_t> mTranseGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mTranseLocalWorkSize{1, 1, 1, 1};
    bool mNeedTranse = false;
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvSubgroupBufExecution_hpp */
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
