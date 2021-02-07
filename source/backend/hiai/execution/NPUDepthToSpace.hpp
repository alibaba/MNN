//
//  NPUDepthToSpace.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUDepthToSpace_HPP
#define NPUDEMO_NPUDepthToSpace_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUDepthToSpace : public NPUCommonExecution {
public:
    NPUDepthToSpace(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUDepthToSpace() = default;
   
private:
};
} // namespace MNN

#endif // NPUDEMO_NPUDepthToSpace_HPP
