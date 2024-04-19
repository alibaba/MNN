//
//  SelectBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/12/1.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SelectExecution_hpp
#define SelectExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SelectExecution : public CommonExecution {
public:
    SelectExecution(const MNN::Op *op, Backend *backend);
    virtual ~SelectExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalSize      = {1, 1, 1};
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* SelectExecution_hpp */
