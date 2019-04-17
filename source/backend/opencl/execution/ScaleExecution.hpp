//
//  ScaleExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScaleExecution_hpp
#define ScaleExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class ScaleExecution : public Execution {
public:
    ScaleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ScaleExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScale;
    std::shared_ptr<Tensor> mBias;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mAreadySetArg;
    OpenCLBackend *mOpenCLBackend;
    bool mHasBias = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ScaleExecution_hpp */
