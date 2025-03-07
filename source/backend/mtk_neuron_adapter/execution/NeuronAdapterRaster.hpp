//
//  NeuronAdapterRaster.hpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NeuronAdapterRASTER_HPP
#define MNN_NeuronAdapterRASTER_HPP

#include "NeuronAdapterBackend.hpp"
#include "NeuronAdapterCommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NeuronAdapterRaster : public NeuronAdapterCommonExecution {
public:
    NeuronAdapterRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NeuronAdapterRaster() = default;
private:
    std::vector<std::vector<int32_t>> mDatas;
    ErrorCode buildReshape(const std::vector<Tensor *> &outputs);
    ErrorCode buildPermute(const std::vector<Tensor *> &outputs);
    ErrorCode buildTile(const std::vector<Tensor *> &outputs);
    ErrorCode buildPad(const std::vector<Tensor *> &outputs);
    ErrorCode buildSlice(const std::vector<Tensor *> &outputs);
    ErrorCode buildDepthToSpace(const std::vector<Tensor *> &outputs);
    ErrorCode buildConcat(const std::vector<Tensor *> &outputs, int axis);
};
} // namespace MNN

#endif // MNN_NeuronAdapterRASTER_HPP
