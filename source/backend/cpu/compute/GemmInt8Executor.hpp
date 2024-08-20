//
//  GemmInt8Executor.hpp
//  MNNCPU
//
//  Created by jbyang on 2023/3/16.
//

#ifndef GemmInt8Executor_hpp
#define GemmInt8Executor_hpp

#include "Int8FunctionsOpt.h"
#include "backend/cpu/CPUConvolution.hpp"

namespace MNN {
class GemmInt8Executor : public CPUConvolution {
public:
    GemmInt8Executor(Backend* bn, std::shared_ptr<ResourceInt8> resource, const Convolution2D *conv2D, decltype(CoreInt8Functions::Int8GemmKernel), 
                     std::vector<int32_t> bias);
    virtual ~GemmInt8Executor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
protected:
    int mThreadNums;
    int mTileCnt;
    int mKernelX;
    int mKernelY;
    std::shared_ptr<Tensor> mInputCol;
    std::vector<float> mScaleData;
    std::vector<float> mKernelSum;
    std::vector<int32_t> mQuantBias;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResourceInt8;
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    CPUConvolution::MutableResourceInt8 mMutableResource;
    decltype(CoreInt8Functions::Int8GemmKernel) mGemmKernel;
    MemChunk mBlitInfo;
    std::pair<size_t, size_t> mBlitInfoStride;
    std::shared_ptr<CPUConvolution::Resource> mResource;
};
} // namespace MNN
#endif /* DeconvInt8Executor_hpp */
