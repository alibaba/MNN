//
//  CPUScale.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUScale_hpp
#define CPUScale_hpp

#include "AutoStorage.h"
#include "Execution.hpp"

namespace MNN {
class CPUScale : public Execution {
public:
    CPUScale(const Op *op, Backend *bn);
    virtual ~CPUScale() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    AutoStorage<float> mScale;
    AutoStorage<float> mBias;
};

} // namespace MNN
#endif /* CPUScale_hpp */
