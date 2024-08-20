//
//  ConvBufWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef __CONVBUF_WINOGRAD__
#define __CONVBUF_WINOGRAD__

#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

struct ConvBufWinoResource {
    const Convolution2DCommon* mCommon;
    bool mUseSubgroup{false};
    std::shared_ptr<Tensor> mWeight;
    std::shared_ptr<Tensor> mBias;
};

class ConvBufWinograd : public CommonExecution {
public:
    ConvBufWinograd(const MNN::Op* op, Backend* backend);
    ConvBufWinograd(std::shared_ptr<ConvBufWinoResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~ConvBufWinograd();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, bool isIntel = false, int limit = 8192);
    std::vector<uint32_t> getLocalWS(std::string kernelName, int index, std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, cl::Kernel mKernel);
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    ErrorCode SubgroupOnResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    
private:
    void convertWeightFormat(cl::Buffer& buffer, const int tileK, const int tileN);
private:
    OpenCLBackend* mOpenCLBackend;
    std::shared_ptr<ConvBufWinoResource> mResource;
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    int mCi;
    int mCo;

    std::shared_ptr<Tensor> mSource;
    std::shared_ptr<Tensor> mDest;

    std::vector<uint32_t> mMaxWGS_S;
    std::vector<uint32_t> mMaxWGS_D;
    std::vector<uint32_t> mMaxWGS_M;

    std::vector<std::vector<uint32_t> > mGWS_S;
    std::vector<std::vector<uint32_t> > mGWS_D;
    std::vector<std::vector<uint32_t> > mGWS_M;
    
    std::vector<std::vector<uint32_t> > mLWS_S;
    std::vector<std::vector<uint32_t> > mLWS_D;
    std::vector<std::vector<uint32_t> > mLWS_M;
};

} // namespace OpenCL
} // namespace MNN

#endif /* __CONVBUF_WINOGRAD__ */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
