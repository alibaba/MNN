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
    EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend);
    virtual ~EltwiseExecution() = default;

    uint32_t realSize(const Tensor* tensor);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mBroadCast;
    float mOperatorData;
    std::string mCompute;
    std::set<std::string> mBuildOptions;
    std::shared_ptr<Tensor> mTempOutput;
    
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize = {1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* EltwiseExecution_hpp */
