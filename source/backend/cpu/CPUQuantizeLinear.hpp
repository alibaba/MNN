//
//  CPUQuantizeLinear.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizeLinear_hpp
#define CPUQuantizeLinear_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUQuantizeLinear : public Execution {
public:
    CPUQuantizeLinear(Backend *b, int size = 1, int axis = 0);
    virtual ~CPUQuantizeLinear() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    std::vector<float> mQuantScales;
    std::vector<int8_t> mQuantZeroPoints;
    int mSize = 1;
    int mAxis = 0;
};

} // namespace MNN

#endif /* CPUQuantizeLinear_hpp */
