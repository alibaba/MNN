//
//  ScaleBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef ScaleBufExecution_hpp
#define ScaleBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ScaleBufExecution : public CommonExecution {
public:
    ScaleBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ScaleBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScale;
    std::shared_ptr<Tensor> mBias;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1};
    OpenCLBackend *mOpenCLBackend;
    bool mHasBias = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ScaleBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
