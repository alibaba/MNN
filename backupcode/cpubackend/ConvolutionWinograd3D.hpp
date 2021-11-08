//
//  ConvolutionWinograd3D.hpp
//  MNN
//
//  Created by MNN on 2018/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionWinograd3d_hpp
#define ConvolutionWinograd3d_hpp

#include "backend/cpu/CPUConvolution3D.hpp"
#include "backend/cpu/compute/WinogradOptFunction.hpp"

namespace MNN {
class ConvolutionWinograd3D : public Execution {
public:
    ConvolutionWinograd3D(const Convolution3DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        int unit);
    virtual ~ConvolutionWinograd3D();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool canUseWinograd(const Convolution3DCommon *convOp);
    static int bestWinogradUnit(const Convolution3DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber);

private:
    int mUnit;
    int mAlpha;
    PadMode mPadMode;
    std::vector<int> mKernels;
    std::vector<int> mPads;
    CPUConvolution3D::POSTFUNCTION mPostFunction;
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mSourceBuffer;
    std::shared_ptr<Tensor> mDestBuffer;
    std::shared_ptr<Tensor> mTempBuffer;

    WinogradFunction::TransformFunc mSourceTransform;
    WinogradFunction::TransformFunc mDestTransform;
};
} // namespace MNN
#endif /* ConvolutionWinograd3d_hpp */
