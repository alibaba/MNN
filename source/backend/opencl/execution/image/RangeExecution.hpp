//
//  RangeBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RangeExecution_hpp
#define RangeExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class RangeExecution : public CommonExecution {
public:
    RangeExecution(const std::string &compute, const MNN::Op *op, Backend *backend);
    virtual ~RangeExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    uint32_t mMaxWorkGroupSize;
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* RangeExecution_hpp */
