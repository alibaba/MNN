//
//  MetalConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolutionCommon_hpp
#define MetalConvolutionCommon_hpp

#import "core/ConvolutionCommon.hpp"
#import "MetalBackend.hpp"
#import "MetalExecution.hpp"
#import "MNNMetalContext.h"
#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolutionCommon : public MetalExecution {
public:
    MetalConvolutionCommon(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> bias);
    virtual ~MetalConvolutionCommon() = default;

protected:
    void loadWeight(const MNN::Op *op, bool loadWeightInt8 = false);

    virtual std::shared_ptr<MNN::Tensor> weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight = false, bool int4Weight = false);

protected:
    struct Param {
        int input_size;
        int input_slice;
        int output_width;
        int output_height;
        int output_size;
        int output_slice;
        int output_channel;
        int batch;
        int block_size;
        int activation;
        float scale_coef;
    };
    
protected:
    int mKernelX        = 0;
    int mKernelY        = 0;
    int mStrideX        = 0;
    int mStrideY        = 0;
    int mDilateX        = 0;
    int mDilateY        = 0;
    int mActivationType = 0;
    const MNN::Op *mOp  = nullptr;

    std::shared_ptr<MNN::Tensor> mWeight;
    std::shared_ptr<MNN::Tensor> mBias;
    std::shared_ptr<MNN::Tensor> mDequantScaleBias;
    int mDequantBits;
    float mScaleCoef;
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionCommon_hpp */
