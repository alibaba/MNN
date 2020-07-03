//
//  CPUQuantizedLogistic.hpp
//  MNN
//
//  Created by MNN on 2018/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedLogistic_hpp
#define CPUQuantizedLogistic_hpp

#include "core/Execution.hpp"
#include "TFQuantizeOp_generated.h"

namespace MNN {

class CPUQuantizedLogistic : public Execution {
public:
    CPUQuantizedLogistic(Backend *backend, const Op *op);
    virtual ~CPUQuantizedLogistic() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const QuantizedLogistic *mLogisticParam;
    int mInputMultiplier;
    int mInputZeroPoint;
    int mInputLeftShift;
    int mInputRangeRadius;
};

} // namespace MNN
#endif /* CPUQuantizedLogistic_hpp */
