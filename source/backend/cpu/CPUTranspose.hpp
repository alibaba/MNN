//
//  CPUTranspose.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTranspose_hpp
#define CPUTranspose_hpp

#include "Execution.hpp"
#include "Type_generated.h"

namespace MNN {

template <typename T>
class CPUTranspose : public Execution {
public:
    CPUTranspose(Backend *backend, const Op *op);
    virtual ~CPUTranspose() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    DataType permDateType;
};

} // namespace MNN
#endif /* CPUTranspose_hpp */
