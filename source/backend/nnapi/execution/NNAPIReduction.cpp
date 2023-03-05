//
//  NNAPIReduction.cpp
//  MNN
//
//  Created by MNN on 2022/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIReduction.hpp"

namespace MNN {


NNAPIReduction::NNAPIReduction(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    std::map<ReductionType, int> reduce_map {
        {ReductionType_SUM, ANEURALNETWORKS_REDUCE_SUM},
        {ReductionType_ASUM, -1},
        {ReductionType_SUMSQ, -1},
        {ReductionType_MEAN, ANEURALNETWORKS_MEAN},
        {ReductionType_MAXIMUM, ANEURALNETWORKS_REDUCE_MAX},
        {ReductionType_MINIMUM, ANEURALNETWORKS_REDUCE_MIN},
        {ReductionType_PROD, ANEURALNETWORKS_REDUCE_PROD},
        {ReductionType_ALL, ANEURALNETWORKS_REDUCE_ALL},
        {ReductionType_ANY, ANEURALNETWORKS_REDUCE_ANY}
    };
    auto param = mOp->main_as_ReductionParam();
    auto operation = param->operation();
    auto dim = param->dim();
    bool keep_dims = param->keepDims();
    auto iter = reduce_map.find(operation);
    if (iter == reduce_map.end() || iter->second < 0) {
        MNN_ERROR("[NNAPI] Reduction not support %s\n", MNN::EnumNameReductionType(operation));
        return NOT_SUPPORT;
    }
    // reduce : [input, dim, keep_dims]
    auto inputIdxs = getTensorIdxs(inputs);
    // inputIdxs.push_back(buildConstant(dim->data(), dim->size() * sizeof(int), ANEURALNETWORKS_TENSOR_INT32, {static_cast<uint32_t>(dim->size())}));
    std::vector<int> rdim(1);
    rdim[0] = dim->data()[0];
    inputIdxs.push_back(buildVector(rdim));
    if (operation == ReductionType_MEAN) {
        // mean arg_2 is `int32_t`
        inputIdxs.push_back(buildScalar(static_cast<int>(keep_dims)));
    } else {
        // other reduce arg_2 is `bool`
        inputIdxs.push_back(buildScalar(keep_dims));
    }
    return buildOperation(iter->second, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIReduction, OpType_Reduction)
} // namespace MNN
