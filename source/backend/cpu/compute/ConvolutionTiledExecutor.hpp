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
#include "../CPUConvolution.hpp"

// Tiled Slide Window Algorithm
namespace MNN {
class ConvolutionTiledExecutorBasic : public CPUConvolution {
public:
    ConvolutionTiledExecutorBasic(const Convolution2DCommon *common, Backend *b) : CPUConvolution(common, b) {
    }
    virtual ~ConvolutionTiledExecutorBasic() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    Tensor mTempBuffer;
    std::vector<std::pair<int, std::function<void(int)>>> mFunctions;
};
class ConvolutionTiledExecutorMultiInput : public Execution {
public:
    ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new ConvolutionTiledExecutorBasic(common, b));
    }
    virtual ~ConvolutionTiledExecutorMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<ConvolutionTiledExecutorBasic> mProxy;
    std::vector<Tensor *> mInputs;
};
class ConvolutionTiledExecutor : public Execution {
public:
    ConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize);
    virtual ~ConvolutionTiledExecutor();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mProxy->onExecute(inputs, outputs);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mWeight.get(), mBias.get()};
        return mProxy->onResize(mInputs, outputs);
    }

protected:
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<ConvolutionTiledExecutorBasic> mProxy;
    std::vector<Tensor *> mInputs;
};
} // namespace MNN

#endif /* ConvolutionTiledExecutor_hpp */
