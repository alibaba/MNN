//
//  NPUStridedSlice.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUStridedSlice_HPP
#define MNN_NPUStridedSlice_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {
class NPUStridedSlice : public NPUCommonExecution {
public:
    NPUStridedSlice(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUStridedSlice() = default;

private:
    hiai::op::Const mConst_b;
    hiai::op::Const mConst_e;
    hiai::op::Const mConst_a;
    hiai::op::Const mConst_s;
    bool isConst1;
    bool isConst2;
    bool isConst3;
    bool isConst4;
    int32_t beginMask;
    int32_t endMask;
    int32_t ellipsisMask;
    int32_t newAxisMask;
    int32_t shrinkAxisMask;
};

} // namespace MNN

#endif // MNN_NPUStridedSlice_HPP
