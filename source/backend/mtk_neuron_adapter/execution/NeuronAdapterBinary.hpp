//
//  NeuronAdapterBinary.hpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NeuronAdapterBINARY_HPP
#define MNN_NeuronAdapterBINARY_HPP

#include "NeuronAdapterBackend.hpp"
#include "NeuronAdapterCommonExecution.hpp"

namespace MNN {

class NeuronAdapterBinary : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterBinary() = default;
};
} // namespace MNN

#endif // MNN_NeuronAdapterBINARY_HPP
