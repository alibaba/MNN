//
//  CPUQuantizedAdd.hpp
//  MNN
//
//  Created by MNN on 2018/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedAdd_hpp
#define CPUQuantizedAdd_hpp

#include "core/Execution.hpp"
#include "TFQuantizeOp_generated.h"

// have to include after Marco.h
#include "CPUFixedPoint.hpp"

namespace MNN {

class CPUQuantizedAdd : public Execution {
public:
    CPUQuantizedAdd(Backend *backend, const Op *op);
    virtual ~CPUQuantizedAdd() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const QuantizedAdd *mQuantizedAddParam;
    int mInput1Offset;
    int mInput2Offset;
    int mOutputOffset;
    int mInput1Multiplier;
    int mInput2Multiplier;
    int mOutputMultiplier;
    int mInput1Shift;
    int mInput2Shift;
    int mOutputShift;
    int mOutputActivationMin, mOutputActivationMax;
    int mLeftShiftResult1, mLeftShiftResult2;
    int mRightShift1, mRightShift2;
    int mLeftShiftOut, mRightShiftOut;    
};

} // namespace MNN
#endif /* CPUQuantizedAdd_hpp */
