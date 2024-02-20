//
//  NPUPermute.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUPermute_HPP
#define NPUDEMO_NPUPermute_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUPermute : public NPUCommonExecution {
public:
    NPUPermute(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUPermute() = default;
private:
    vector<int32_t> mNHWC{0, 1, 2, 3};
    vector<int32_t> mNCHW{0, 2, 3, 1};
};
} // namespace MNN

#endif // NPUDEMO_NPUPermute_HPP
