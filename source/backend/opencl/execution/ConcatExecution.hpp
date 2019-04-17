//
//  ConcatExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConcatExecution_hpp
#define ConcatExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {
class ConcatImageExecution : public CommonExecution {
public:
    ConcatImageExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend) : CommonExecution(backend) {
        mAxis = axis;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
};
class ConcatBufferExecution : public CommonExecution {
public:
    ConcatBufferExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend) : CommonExecution(backend) {
        mAxis = axis;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<cl::Buffer *> mTempInputs;
    cl::Buffer *mTempOutput;
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* ConcatExecution_hpp */
