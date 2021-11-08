//
//  CPUSelu.hpp
//  MNN
//
//  Created by MNN on 2018/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSelu_hpp
#define CPUSelu_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUSelu : public Execution {
public:
    CPUSelu(Backend *b, const MNN::Op *op);
    virtual ~CPUSelu() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mScale = 0.0f;
    float mAlpha = 0.0f;
};
} // namespace MNN
#endif /* CPUSelu_hpp */
