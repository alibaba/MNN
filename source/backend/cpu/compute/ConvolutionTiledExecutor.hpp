//
//  ConvolutionTiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionTiledExecutor_hpp
#define ConvolutionTiledExecutor_hpp

#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
class ConvolutionTiledExecutorBasic : public CPUConvolution {
public:
    ConvolutionTiledExecutorBasic(const Convolution2DCommon *common, Backend *b) : CPUConvolution(common, b) {
        // Do nothing
    }
    virtual ~ConvolutionTiledExecutorBasic() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    Tensor mTempBufferTranspose;
    std::pair<int, std::function<void(int)>> mFunction;
};
class ConvolutionTiledExecutor : public Execution {
public:
    ConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize);
    ConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~ConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mProxy->onExecute(inputs, outputs);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
        return mProxy->onResize(mInputs, outputs);
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

protected:
    std::shared_ptr<ConvolutionTiledExecutorBasic> mProxy;
    std::vector<Tensor *> mInputs;
    std::shared_ptr<CPUConvolution::Resource> mResource;
};
} // namespace MNN

#endif /* ConvolutionTiledExecutor_hpp */
