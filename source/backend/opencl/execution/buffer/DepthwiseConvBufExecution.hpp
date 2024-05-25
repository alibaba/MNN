//
//  DepthwiseConvBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef DepthwiseConvBufExecution_hpp
#define DepthwiseConvBufExecution_hpp

#include "ConvBufExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvBufExecution : public ConvBufCommonExecution, public CommonExecution {
public:
    DepthwiseConvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    DepthwiseConvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~DepthwiseConvBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    std::pair<std::vector<uint32_t>,  int> DepthwiseConvBufLwsTune(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize);
private:
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    bool mStride_1 = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
