//
//  CoreMLRelu6.hpp
//  MNN
//
//  Created by MNN on 2024/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLRelu6_HPP
#define MNN_COREMLRelu6_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLRelu6 : public CoreMLCommonExecution {
public:
    CoreMLRelu6(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLRelu6() = default;
private:
    float mMinValue = 0.0f;
    float mMaxValue = 6.0f;
};
} // namespace MNN

#endif // MNN_COREMLRelu6_HPP
