//
//  NPUSqueeze.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUSqueeze_HPP
#define MNN_NPUSqueeze_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {
class NPUSqueeze : public NPUCommonExecution {
public:
    NPUSqueeze(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUSqueeze() = default;
private:
    hiai::op::Const shapeConst;
};

} // namespace MNN

#endif // MNN_NPUSqueeze_HPP
