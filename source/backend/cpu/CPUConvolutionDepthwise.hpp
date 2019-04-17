//
//  CPUConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolutionDepthwise_hpp
#define CPUConvolutionDepthwise_hpp

#include "AutoStorage.h"
#include "CPUConvolution.hpp"
#include "compute/ConvolutionIntFactory.hpp"

namespace MNN {
class CPUConvolutionDepthwise : public Execution {
public:
    class FloatExecution : public CPUConvolution {
    public:
        FloatExecution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                       size_t originWeightSize, const float *bias, size_t biasSize);
        virtual ~FloatExecution();
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    private:
        std::shared_ptr<Tensor> mWeight;
        std::shared_ptr<Tensor> mBias;
        std::function<void(const float *, float *, int)> mExecutor;
        int mNumber = 1;
    };

    class Int8Execution : public CPUConvolution {
    public:
        Int8Execution(const Convolution2DCommon *convOp, Backend *b, const ConvolutionIntFactory::Int8Common *common,
                      const float *bias, size_t biasSize);
        virtual ~Int8Execution() = default;
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    private:
        AutoStorage<int8_t> mWeight;
        AutoStorage<float> mBias;
        AutoStorage<float> mAlpha;

        Tensor mInputTempBuffer;
        const IDSTQuan *mQuan;
    };

    CPUConvolutionDepthwise(const Op *convOp, Backend *b);
    virtual ~CPUConvolutionDepthwise() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::unique_ptr<Execution> mSubExecution;
};
} // namespace MNN

#endif /* CPUConvolutionDepthwise_hpp */
