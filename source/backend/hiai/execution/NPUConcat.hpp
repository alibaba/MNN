//
//  NPUConcat.hpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUCONCAT_HPP
#define MNN_NPUCONCAT_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUConcat : public NPUCommonExecution {
public:
    NPUConcat(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUConcat() = default;
};

} // namespace MNN

#endif // MNN_NPUCONCAT_HPP
