//
//  NeuronAdapterQuant.hpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NeuronAdapterQUANT_HPP
#define MNN_NeuronAdapterQUANT_HPP

#include "NeuronAdapterBackend.hpp"
#include "NeuronAdapterCommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NeuronAdapterQuant : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterQuant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterQuant() = default;
};

class NeuronAdapterDequant : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterDequant(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterDequant() = default;
};
} // namespace MNN

#endif // MNN_NeuronAdapterQUANT_HPP
