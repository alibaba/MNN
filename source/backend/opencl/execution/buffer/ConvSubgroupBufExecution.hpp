//
//  ConvSubgroupBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#ifndef ConvSubgroupBufExecution_hpp
#define ConvSubgroupBufExecution_hpp

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

class ConvSubgroupBuf : public Execution {
public:
    virtual ~ConvSubgroupBuf();

    ConvSubgroupBuf(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                    Backend* backend);

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth,
                      int maxHeight, int limit = 8192);

private:
    void transformWeight(const Tensor* weightDest, const Tensor* source);

    OpenCLBackend* mOpenCLBackend;
    const Convolution2DCommon* mConv2dCommonParams;
    const Convolution2D* mConv2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::vector<uint32_t> mTranseGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mTranseLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mFilter;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mSource;
    cl::Kernel mKernel;
    cl::Kernel mTranseKernel;
    uint32_t mMaxWorkGroupSize;
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;
    int mKernelWidth;
    int mKernelHeight;
    int mOutputChannel;
    int mInputChannel;
    std::set<std::string> mBuildOptions;
    bool mNeedTranse = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvSubgroupBufExecution_hpp */
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
