//
//  TRTBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2021/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTBatchMatMul.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

nvinfer1::MatrixOperation transposeFormat(nvinfer1::ITensor *x, bool transpose) {
    return transpose ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
}

TRTBatchMatMul::TRTBatchMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
#ifdef TRT_LOG
    printf("TRTBatchMatMul in\n");
#endif
}

std::vector<ITensor *> TRTBatchMatMul::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTBatchMatMul in\n");
#endif
    auto param       = mOp->main_as_BatchMatMulParam();
    MNN_ASSERT(mInputs.size() == 2);
    bool isConst0 = TensorUtils::getDescribe(mInputs[0])->usage == Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(mInputs[1])->usage == Tensor::InsideDescribe::Usage::CONSTANT;

    auto dimSize0 = mInputs[0]->dimensions();
    auto dimSize1 = mInputs[1]->dimensions();

    auto transpose_a = transposeFormat(xOp[0], param->adjX());
    auto transpose_b = transposeFormat(xOp[1], param->adjY());

    auto matmul_layer = mTrtBackend->getNetwork()->addMatrixMultiply(*xOp[0], transpose_a, *xOp[1], transpose_b);
    return {matmul_layer->getOutput(0)};

}

TRTCreatorRegister<TypedCreator<TRTBatchMatMul>> __batch_matmul_op(OpType_BatchMatMul);

} // namespace MNN
