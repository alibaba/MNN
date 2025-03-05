//
//  NeuronAdapterPool.hpp
//  MNN
//
//  Created by MNN on 2022/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NeuronAdapterPOOL_HPP
#define MNN_NeuronAdapterPOOL_HPP

#include "NeuronAdapterBackend.hpp"
#include "NeuronAdapterCommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NeuronAdapterPool : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterPool(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterPool() = default;
};
} // namespace MNN

#endif // MNN_NeuronAdapterPOOL_HPP
