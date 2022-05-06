//
//  DenseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DenseConvolutionTiledExecutor_hpp
#define DenseConvolutionTiledExecutor_hpp


#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
#include "ConvolutionTiledExecutor.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
class DenseConvolutionTiledImpl : public ConvolutionTiledImpl {
public:
    DenseConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b) : ConvolutionTiledImpl(common, b) {
        // Do nothing
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    virtual ~DenseConvolutionTiledImpl() = default;
    void getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) override;
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b);
protected:

};
class DenseConvolutionTiledExecutor : public ConvolutionTiledExecutor {
public:
    DenseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                             size_t originWeightSize, const float *bias, size_t biasSize);

    DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~DenseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mProxy->onExecute(inputs, outputs);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
        return mProxy->onResize(mInputs, outputs);
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function);
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
        return DenseConvolutionTiledImpl::bestTileConvolutionConfig(common, inputTensor, outputTensor, threadNumber, b);
    }
protected:
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
};

class ConvolutionTiledExecutorMultiInput : public Execution {
public:
    ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new DenseConvolutionTiledImpl(common, b));
    }
    virtual ~ConvolutionTiledExecutorMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
    std::vector<Tensor *> mInputs;
};

} // namespace MNN

#endif /* DenseConvolutionTiledExecutor_hpp */
