//
//  SpaceToBatchExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SpaceToBatchExecution_hpp
#define SpaceToBatchExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class SpaceToBatchExecution : public Execution {
public:
    SpaceToBatchExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~SpaceToBatchExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mPaddings[2]   = {0, 0};
    int mBlockShape[2] = {0, 0};
    cl::Kernel mKernel;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* SpaceToBatchExecution_hpp */
