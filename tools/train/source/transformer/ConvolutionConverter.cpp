//
//  ConvolutionConverter.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionConverter.hpp"
#include <MNN/Tensor.hpp>
#include "core/Macro.h"
using namespace MNN;

OpConverter::ReductResult ConvolutionConverter::onReduct(int opIndex, MNN::OpT* op, MNN::NetT* net) {
    OpConverter::ReductResult result;
    if (op->inputIndexes.size() != 3) {
        return result;
    }

    auto conv2D       = op->main.AsConvolution2D();
    auto conv2DCommon = conv2D->common.get();

    auto relu  = conv2DCommon->relu;
    auto relu6 = conv2DCommon->relu6;

    // set output
    auto outputIndex = op->outputIndexes[0];
    if (relu || relu6) {
        auto& reluOp = net->oplists[opIndex + 1];
        outputIndex  = reluOp->outputIndexes[0];
    }

    op->outputIndexes = {outputIndex};

    // add weight
    auto& weightOp                     = net->oplists[opIndex - 2];
    op->main.AsConvolution2D()->weight = std::move(weightOp->main.AsBlob()->float32s);

    // add bias
    auto& biasOp                     = net->oplists[opIndex - 1];
    op->main.AsConvolution2D()->bias = std::move(biasOp->main.AsBlob()->float32s);

    // set input
    op->inputIndexes = {op->inputIndexes[0]};

    result.needDeleteOpIndexes.emplace_back(opIndex - 2);
    result.needDeleteOpIndexes.emplace_back(opIndex - 1);
    if (relu || relu6) {
        result.needDeleteOpIndexes.emplace_back(opIndex + 1);
    }

    return result;
}

static const auto gRegister = []() {
    static ConvolutionConverter _c;
    OpConverter::insert(OpType_Convolution, &_c);
    OpConverter::insert(OpType_ConvolutionDepthwise, &_c);
    return true;
}();
