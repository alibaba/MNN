//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class SoftmaxExecution : public Execution {
public:
    SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend);

    virtual ~SoftmaxExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildSoftmaxKernel();
    std::vector<uint32_t> softmaxLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SoftmaxExecution_hpp */
