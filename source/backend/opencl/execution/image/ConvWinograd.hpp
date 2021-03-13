//
//  ConvWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef conv_winograd_hpp
#define conv_winograd_hpp

#include "core/Execution.hpp"

#include <array>
#include <memory>
#include <vector>
#include "backend/opencl/execution/image/ConvExecution.hpp"
namespace MNN {
namespace OpenCL {
class ConvWinograd : public Execution {
public:
    virtual ~ConvWinograd() = default;

    ConvWinograd(const MNN::Convolution2D* op, Backend* backend);

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, int limit = 8192);

private:
    OpenCLBackend* mOpenCLBackend;
    const Convolution2DCommon* mCommon;
    int mKernelX;
    int mKernelY;
    int mPadX;
    int mPadY;
    int mStrideX;
    int mStrideY;
    MNN::PadMode mPadMode;
    std::shared_ptr<cl::Image2D> mWeight;
    std::shared_ptr<cl::Image2D> mBias;

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

    int mSliceNumber;
};

} // namespace OpenCL
} // namespace MNN

#endif /* conv_winograd_hpp */
