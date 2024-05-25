//
//  DepthwiseConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DepthwiseConvExecution_hpp
#define DepthwiseConvExecution_hpp

#include "ConvExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvExecution : public ConvCommonExecution, public CommonExecution {
public:
    DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    DepthwiseConvExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend* backend);
    virtual ~DepthwiseConvExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::vector<int> mPaddings{0, 0};
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvExecution_hpp */
