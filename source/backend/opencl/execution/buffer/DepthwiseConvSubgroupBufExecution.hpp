//
//  DepthwiseConvSubgroupBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#ifndef DepthwiseConvSubgroupBufExecution_hpp
#define DepthwiseConvSubgroupBufExecution_hpp

#include "ConvBufExecution.hpp"
namespace MNN {
namespace OpenCL {

class DepthwiseConvSubgroupBufExecution : public ConvBufCommonExecution, public CommonExecution {
public:
    DepthwiseConvSubgroupBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    DepthwiseConvSubgroupBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~DepthwiseConvSubgroupBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    void transformWeight(const Tensor *weightDest, const Tensor *source);
    std::vector<int> mPaddings{0, 0};
    std::shared_ptr<Tensor> mSource;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::vector<uint32_t> mTranseGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mTranseLocalWorkSize{1, 1, 1, 1};
    bool mNeedTranse = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvSubgroupBufExecution_hpp */
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
