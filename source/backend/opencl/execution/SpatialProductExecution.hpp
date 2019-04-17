//
//  SpatialProductExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SpatialProductExecution_hpp
#define SpatialProductExecution_hpp

#include <MNN_generated.h>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class SpatialProductExecution : public Execution {
public:
    SpatialProductExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~SpatialProductExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* SpatialProductExecution_hpp */
