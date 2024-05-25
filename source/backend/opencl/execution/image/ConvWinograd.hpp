//
//  ConvWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef conv_winograd_hpp
#define conv_winograd_hpp

#include "backend/opencl/execution/image/ConvExecution.hpp"
namespace MNN {
namespace OpenCL {
struct ConvWinoResource {
    const Convolution2DCommon* mCommon;
    std::shared_ptr<cl::Image2D> mWeight;
    std::shared_ptr<cl::Image2D> mBias;
};

class ConvWinograd : public CommonExecution {
public:
    virtual ~ConvWinograd() = default;

    ConvWinograd(const MNN::Op *op, Backend* backend);
    ConvWinograd(std::shared_ptr<ConvWinoResource> resource, const MNN::Op* op, Backend* backend);

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth, int maxHeight, int limit = 8192);

private:
    OpenCLBackend* mOpenCLBackend;
    std::shared_ptr<ConvWinoResource> mResource;
    int mKernelX;
    int mKernelY;
    int mPadX;
    int mPadY;
    int mStrideX;
    int mStrideY;
    MNN::PadMode mPadMode;

    std::shared_ptr<Tensor> mSource;
    std::shared_ptr<Tensor> mDest;

    std::vector<uint32_t> mMaxWGS_S;
    std::vector<uint32_t> mMaxWGS_D;

    std::vector<std::vector<uint32_t> > mGWS_S;
    std::vector<std::vector<uint32_t> > mGWS_D;
    std::vector<std::vector<uint32_t> > mGWS_M;
    
    std::vector<std::vector<uint32_t> > mLWS_S;
    std::vector<std::vector<uint32_t> > mLWS_D;
    std::vector<std::vector<uint32_t> > mLWS_M;

};

} // namespace OpenCL
} // namespace MNN

#endif /* conv_winograd_hpp */
