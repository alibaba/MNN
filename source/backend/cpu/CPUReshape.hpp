//
//  CPUReshape.hpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReshape_hpp
#define CPUReshape_hpp

#include "Execution.hpp"
#include "Tensor_generated.h"

namespace MNN {
class CPUReshape : public Execution {
public:
    CPUReshape(Backend *b, MNN_DATA_FORMAT dimType);
    virtual ~CPUReshape() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    Tensor mStorage;
    MNN_DATA_FORMAT mDimType;
    Tensor mWrapTensorForInput;
    Tensor mWrapTensorForOutput;
};
} // namespace MNN
#endif /* CPUReshape_hpp */
