//
//  DeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2021/04/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef DeconvBufExecution_hpp
#define DeconvBufExecution_hpp

#include "ConvBufExecution.hpp"
namespace MNN {
namespace OpenCL {

class DeconvBufExecution : public ConvBufCommonExecution {
public:
    DeconvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DeconvBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    std::vector<int> mStrides{0, 0};
    std::vector<int> mDilations{0, 0};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DeconvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
