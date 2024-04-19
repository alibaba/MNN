//
//  ReluExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReluExecution_hpp
#define ReluExecution_hpp

#include "CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class ReluExecution : public CommonExecution {
public:
    ReluExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ReluExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mPreluParam;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReluExecution_hpp */
