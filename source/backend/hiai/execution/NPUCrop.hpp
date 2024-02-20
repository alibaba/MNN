//
//  NPUCrop.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUCrop_HPP
#define MNN_NPUCrop_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUCrop : public NPUCommonExecution {
public:
    NPUCrop(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUCrop() = default;
};

} // namespace MNN

#endif // MNN_NPUCrop_HPP
