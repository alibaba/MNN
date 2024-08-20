//
//  ReductionExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReductionExecution_hpp
#define ReductionExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ReductionExecution : public CommonExecution {
public:
    ReductionExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op* op, Backend* backend);
    virtual ~ReductionExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int getLocalSize(int size, int maxGroupSize);
    OpenCLBackend *mOpenCLBackend;
    MNN::DataType mdataType;
    int mReductType;
    int mAxis;
    bool mUseLocal = false;
    uint32_t mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReductionExecution_hpp */
