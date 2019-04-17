//
//  CPUStridedSlice.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUStridedSlice_hpp
#define CPUStridedSlice_hpp

#include "Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUStridedSlice : public Execution {
public:
    CPUStridedSlice(Backend *b, const MNN::Op *op);
    virtual ~CPUStridedSlice() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    template <typename T>
    ErrorCode execute(Tensor *input, Tensor *output);

protected:
    const Op *mOp;
    std::vector<int32_t> mBeginShape;
    std::vector<int32_t> mEndShape;
    std::vector<int32_t> mStrideShape;
    std::vector<int32_t> mOutputShape;
    DataType mDataType;
};
} // namespace MNN
#endif /* CPUStridedSlice_hpp */
