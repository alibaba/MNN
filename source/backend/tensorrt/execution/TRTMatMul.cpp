//
//  TRTMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTMatMul.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

inline nvinfer1::Dims squeeze_trailing_dims(nvinfer1::Dims const &dims) {
    nvinfer1::Dims new_dims = dims;

    // for(int i=0; i<new_dims.nbDims; i++)
    //   printf("kk %d ", new_dims.d[i]);
    // printf("\n");

    // Note: TRT requires at least one dimension, so we don't squeeze [1]->[]
    while (new_dims.nbDims > 1 && new_dims.d[new_dims.nbDims - 1] == 1) {
        // printf("%d ", new_dims.d[new_dims.nbDims-1]);
        --new_dims.nbDims;
    }
    return new_dims;
}

nvinfer1::MatrixOperation transpose_format(nvinfer1::ITensor *x, bool transpose) {
    // printf("%d\n", x->getDimensions().nbDims);
    return transpose ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
}

TRTMatMul::TRTMatMul(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
#ifdef TRT_LOG
    printf("TRTMatMul in\n");
#endif
}

std::vector<ITensor *> TRTMatMul::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTMatMul in\n");
#endif
    auto param       = mOp->main_as_MatMul();
    auto transpose_a = transpose_format(xOp[0], param->transposeA());
    auto transpose_b = transpose_format(xOp[1], param->transposeB());

    auto matmul_layer = mTrtBackend->getNetwork()->addMatrixMultiply(*xOp[0], transpose_a, *xOp[1], transpose_b);
    if (xOp.size() == 2) {
        return {matmul_layer->getOutput(0)};
    }
    auto C = matmul_layer->getOutput(0);
    auto shuffle =  mTrtBackend->getNetwork()->addShuffle(*(xOp[2]));
    auto dimReshape = xOp[0]->getDimensions();
    dimReshape.nbDims = 2;
    dimReshape.d[0] = 1;
    dimReshape.d[1] = mInputs[2]->elementSize();    
    shuffle->setReshapeDimensions(dimReshape);
    auto biasReshape = shuffle->getOutput(0);
    auto biasAdd = mTrtBackend->getNetwork()->addElementWise(*C, *biasReshape, ElementWiseOperation::kSUM);
    return {biasAdd->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTMatMul>> __matmul_op(OpType_MatMul);

} // namespace MNN
