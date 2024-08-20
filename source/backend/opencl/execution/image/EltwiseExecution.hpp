//
//  EltwiseExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef EltwiseExecution_hpp
#define EltwiseExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class EltwiseExecution : public CommonExecution {
public:
    EltwiseExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string &compute, const MNN::Op *op, Backend *backend);
    virtual ~EltwiseExecution() = default;

    uint32_t realSize(const Tensor* tensor);
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::string mCompute;
    std::shared_ptr<Tensor> mTempOutput;
    
    std::vector<uint32_t> mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* EltwiseExecution_hpp */
