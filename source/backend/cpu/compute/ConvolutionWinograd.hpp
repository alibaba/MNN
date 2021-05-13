//
//  ConvolutionWinograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinograd_hpp
#define ConvolutionWinograd_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionFloatFactory.h"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
class ConvolutionWinograd : public CPUConvolution {
public:
    ConvolutionWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        int unit);
    virtual ~ConvolutionWinograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool canUseWinograd(const Convolution2DCommon *convOp);
    static int bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber, Backend* b);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvolutionWinograd(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *convOp, Backend* b) : CPUConvolution(convOp, b) {
        mResource = resource;
    }
    std::shared_ptr<CPUConvolution::Resource> mResource;
    std::shared_ptr<Tensor> mA;
    std::shared_ptr<Tensor> mB;

    std::shared_ptr<Tensor> mTempBuffer;
    std::shared_ptr<Tensor> mTransformMidBuffer;
    std::shared_ptr<Tensor> mGemmMidBuffer;

    CoreFunctions::WinoTransFunc mSourceTransform;
    CoreFunctions::WinoTransFunc mDestTransform;
    std::vector<float> mPostParameters;
};
} // namespace MNN
#endif /* ConvolutionWinograd_hpp */
