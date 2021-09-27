//
//  RasterBufExecution.hpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef RaterBufExecution_hpp
#define RaterBufExecution_hpp
#include <array>
#include <memory>
#include <vector>
#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class RasterBufExecution : public CommonExecution {
public:
    RasterBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~RasterBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::map<Tensor*, cl::Buffer *> mTempInput;
    cl::Buffer *mTempOutput;
    OpenCLBackend *mOpenCLBackend;
    bool mNeedZero = false;
    bool mFast = false;
};

} // namespace OpenCL
} // namespace MNN

#endif
#endif /* MNN_OPENCL_BUFFER_CLOSED */
