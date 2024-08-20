//
//  MetalConvolutionWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolutionWinograd_hpp
#define MetalConvolutionWinograd_hpp

#import "MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolutionWinograd : public MetalConvolutionCommon {
public:
    static bool isValid(const Convolution2D *conv, const Tensor *input, const Tensor* output);
    MetalConvolutionWinograd(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolutionWinograd() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

protected:
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual std::shared_ptr<MNN::Tensor> weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight=false, bool int4Weight=false) override;

private:
    MetalConvolutionWinograd(Backend *backend, const MNN::Op *op, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias);

    
    id<MTLBuffer> mShapeBuffer = nil;

    int mSrcUnit;
    int mDstUnit;

    std::shared_ptr<Tensor> mTempSrc;
    std::shared_ptr<Tensor> mTempDst;

    MTLSize mInputTransformThreads;
    MTLSize mMatMulThreads;
    MTLSize mOutputTransformThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalConvolutionWinograd_hpp */
