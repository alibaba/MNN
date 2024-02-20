//
//  NPUTile.hpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUTile_HPP
#define NPUDEMO_NPUTile_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUTile : public NPUCommonExecution {
public:
    NPUTile(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUTile() = default;
private:
    hiai::op::Const mConst_m;
};

} // namespace MNN

#endif // NPUDEMO_NPUTile_HPP
