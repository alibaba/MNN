//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SoftmaxExecution : public CommonExecution {
public:
    SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, const MNN::Op *op, Backend *backend);

    virtual ~SoftmaxExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildSoftmaxKernel(int localSize);
private:
    int getLocalSize(int size, int maxGroupSize);
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SoftmaxExecution_hpp */
