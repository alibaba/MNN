//
//  ConvBufWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef __CONVBUF_WINOGRAD__
#define __CONVBUF_WINOGRAD__

#include "core/Execution.hpp"

#include <array>
#include <memory>
#include <vector>
#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {
class ConvBufWinograd : public Execution {
public:
    ConvBufWinograd(const MNN::Convolution2D* op, Backend* backend);
    virtual ~ConvBufWinograd();

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, int limit = 8192);
    std::vector<uint32_t> getLocalWS(std::string kernelName, int index, std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, cl::Kernel mKernel);

private:
    OpenCLBackend* mOpenCLBackend;
    const Convolution2DCommon* mCommon;
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;

    std::shared_ptr<Tensor> mSource;
    std::shared_ptr<Tensor> mDest;

    std::vector<cl::Kernel> mSourceTransform;
    std::vector<cl::Kernel> mDestTransform;
    std::vector<cl::Kernel> mMatMul;

    std::vector<uint32_t> mMaxWGS_S;
    std::vector<uint32_t> mMaxWGS_D;
    std::vector<uint32_t> mMaxWGS_M;

    std::vector<std::vector<uint32_t> > mGWS_S;
    std::vector<std::vector<uint32_t> > mGWS_D;
    std::vector<std::vector<uint32_t> > mGWS_M;
    
    std::vector<std::vector<uint32_t> > mLWS_S;
    std::vector<std::vector<uint32_t> > mLWS_D;
    std::vector<std::vector<uint32_t> > mLWS_M;
};

} // namespace OpenCL
} // namespace MNN

#endif /* __CONVBUF_WINOGRAD__ */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
