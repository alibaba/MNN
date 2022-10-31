//
//  NNAPIRaster.hpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NNAPIRASTER_HPP
#define MNN_NNAPIRASTER_HPP

#include "NNAPIBackend.hpp"
#include "NNAPICommonExecution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class NNAPIRaster : public NNAPICommonExecution {
public:
    NNAPIRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NNAPIRaster() = default;
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

#endif // MNN_NNAPIRASTER_HPP
