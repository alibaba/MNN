//
//  SliceExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SliceExecution_hpp
#define SliceExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SliceExecution : public CommonExecution {
public:
    SliceExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend) : CommonExecution(backend) {
        mAxis = axis;
        if (inputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            int axisMap[] = {0, 3, 1, 2};
            mAxis         = axisMap[axis];
        }
    }
    virtual ~SliceExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};
class SliceBufferExecution : public CommonExecution {
public:
    SliceBufferExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend) : CommonExecution(backend) {
        mAxis = axis;
        if (inputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            int axisMap[] = {0, 3, 1, 2};
            mAxis         = axisMap[axis];
        }
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cl::Buffer *mTempInput;
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SliceExecution_hpp */
