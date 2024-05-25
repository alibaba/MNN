//
//  NPUPooling3D.hpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUPooling3D_HPP
#define MNN_NPUPooling3D_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUPooling3D : public NPUCommonExecution {
public:
    NPUPooling3D(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUPooling3D() = default;
};

} // namespace MNN

#endif // MNN_NPUPooling3D_HPP
