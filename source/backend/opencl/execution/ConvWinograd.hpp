//
//  ConvWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef conv_winograd_hpp
#define conv_winograd_hpp

#include "Execution.hpp"

#include <array>
#include <memory>
#include <vector>
#include "execution/ConvExecution.hpp"
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

    cl::Kernel mSourceTransform;
    cl::Kernel mDestTransform;
    cl::Kernel mMatMul;

    int mSliceNumber;
};

} // namespace OpenCL
} // namespace MNN

#endif /* conv_winograd_hpp */
