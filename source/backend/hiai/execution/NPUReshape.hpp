//
//  NPUReshape.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPURESHAPE_HPP
#define NPUDEMO_NPURESHAPE_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUReshape : public NPUCommonExecution {
public:
    NPUReshape(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUReshape() = default;
private:
    hiai::op::Const shapeConst;
    hiai::op::Const nhwshapeConst;
};

} // namespace MNN

#endif // NPUDEMO_NPURESHAPE_HPP
