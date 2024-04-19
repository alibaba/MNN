//
//  ReluBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef ReluBufExecution_hpp
#define ReluBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class ReluBufExecution : public CommonExecution {
public:
    ReluBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ReluBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    ErrorCode SubgrouponResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    std::shared_ptr<Tensor> mPreluParam;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReluExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
