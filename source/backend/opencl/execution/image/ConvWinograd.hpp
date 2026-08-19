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
    // maxWidth / maxHeight bound the image dimensions the driver accepts; fpBytes / memory let
    // valid() additionally price the transform images this conv would need, which the dimension
    // bounds do not constrain tightly. Under Memory_Low a conv over budget falls back to direct
    // convolution. See ConvBufWinograd::valid for the buffer-mode equivalent.
    static bool valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth,
                      int maxHeight, int limit = 8192, int fpBytes = 4,
                      BackendConfig::MemoryMode memory = BackendConfig::Memory_Normal);
    // Bytes onEncode will allocate for the mSource / mDest image pair.
    static size_t transformImageBytes(int alpha, int wUnit, int hUnit, int inputChannel, int outputChannel,
                                     int fpBytes);

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
