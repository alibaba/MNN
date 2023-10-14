//
//  InterpExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpExecution_hpp
#define InterpExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/execution/image/CommonExtension.hpp"

namespace MNN {
namespace OpenCL {

class InterpExecution : public Execution, public CommonExtension {
public:
    InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~InterpExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    cl::Kernel mKernel;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[4];
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpExecution_hpp */
