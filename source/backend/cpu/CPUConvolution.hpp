//
//  CPUConvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolution_hpp
#define CPUConvolution_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUConvolution : public Execution {
public:
    CPUConvolution(const Convolution2DCommon *convOp, Backend *b);
    virtual ~CPUConvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    typedef void (*POSTFUNCTION)(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);

    POSTFUNCTION getPostFunction() const;
    struct Im2ColParameter {
        int32_t padX;
        int32_t padY;
        int32_t dilateX;
        int32_t dilateY;
        int32_t strideX;
        int32_t strideY;
        int32_t kernelX;
        int32_t kernelY;
        int32_t icDiv4;
        int32_t kernelCountUnit;
        int32_t iw;
        int32_t ih;
        int32_t ow;
        int32_t oh;
    };
    static int reorderWeightSize(int depth, int outputCount, int kernelSize, int unit);
    static void reorderWeight(float *destBuffer, const float *source, int depth, int outputCount, int kernelSize,
                              float *cache);

protected:
    const Convolution2DCommon *mCommon;

    // In execute, use pad from mPadX and mPadY, don't use mCommon's pad
    mutable int mPadX;
    mutable int mPadY;
    CPUConvolution::POSTFUNCTION mPostFunction;
};

} // namespace MNN

#endif /* CPUConvolution_hpp */
