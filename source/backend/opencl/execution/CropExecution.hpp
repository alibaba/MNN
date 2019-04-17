//
//  CropExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CropExecution_hpp
#define CropExecution_hpp

#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

#include <array>
#include <memory>
#include <vector>

namespace MNN {
namespace OpenCL {

class CropExecution : public Execution {
public:
    CropExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~CropExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    int mAxis = 2;
    std::vector<int> mOffsets;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* CropExecution_hpp */
