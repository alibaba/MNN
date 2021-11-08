//
//  CPUDequantize.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDequantize_hpp
#define CPUDequantize_hpp

#include "core/Execution.hpp"
#include "TFQuantizeOp_generated.h"

namespace MNN {

template <typename T>
class CPUDequantize : public Execution {
public:
    CPUDequantize(Backend *backend, QuantizeMode mode, const Op *op);
    virtual ~CPUDequantize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mHalfRange;
    QuantizeMode mMode;
    bool mIsLiteDequantize;
    int mZeroPoint;
    float mScale;
};

} // namespace MNN

#endif /* CPUDequantize_hpp */
