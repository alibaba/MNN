//
//  NPUMatmul.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUMatmul.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUMatmul::NPUMatmul(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
    auto opName = mOp->name()->str();

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    Tensor* input = nullptr;
    if (isConst0 && !isConst1){
        input = inputs[0];
    }
    if (!isConst0 && isConst1){
        input = inputs[1];
    }
    if (input != nullptr) {
        mConst = ge::op::Const(opName + "_w_const");
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (input->getType().code == halide_type_int && input->getType().bits == 32) {
            fdesc.SetDataType(ge::DT_INT32);
            filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        } else {
            filter->SetData((uint8_t *)input->host<float>(), input->elementSize() * sizeof(float));
        }
        filter->SetTensorDesc(fdesc);
        mConst.set_attr_value(filter);
    }
        
}

ErrorCode NPUMatmul::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    auto param = mOp->main_as_MatMul();
    if (outputs[0]->buffer().dimensions == 4 || outputs[0]->buffer().dimensions == 3) {
        shared_ptr<hiai::op::BatchMatMul> matmul(new hiai::op::BatchMatMul(opName));
        if (isConst0 && !isConst1) {
            auto inputIndex1 = mOp->inputIndexes()->data()[1];
            auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; 
            auto xOp1        = iops1.back().first;
            (*matmul)
                .set_input_x1(mConst)
                .set_input_x2(*xOp1.get())
                .set_attr_adj_x1(param->transposeA()) 
                .set_attr_adj_x2(param->transposeB());
        } else if (!isConst0 && isConst1) {
            auto inputIndex = mOp->inputIndexes()->data()[0];
            auto iops       = mNpuBackend->mGrapMap[inputIndex]; 
            auto xOp        = iops.back().first;
            (*matmul)
                .set_input_x1(*xOp.get())
                .set_input_x2(mConst)
                .set_attr_adj_x1(param->transposeA()) 
                .set_attr_adj_x2(param->transposeB());
        } else {
            auto inputIndex = mOp->inputIndexes()->data()[0];
            auto iops       = mNpuBackend->mGrapMap[inputIndex]; 
            auto xOp        = iops.back().first;
            auto inputIndex1 = mOp->inputIndexes()->data()[1];
            auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; 
            auto xOp1        = iops1.back().first;
            (*matmul)
                .set_input_x1(*xOp.get())
                .set_input_x2(*xOp1.get())
                .set_attr_adj_x1(param->transposeA()) 
                .set_attr_adj_x2(param->transposeB());
        }
        mNpuBackend->setOutputOps(mOp, {matmul}, outputs);
    } else {
        shared_ptr<ge::op::MatMul> matmul(new ge::op::MatMul(opName));
        if (isConst0 && !isConst1) {
            auto inputIndex1 = mOp->inputIndexes()->data()[1];
            auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; 
            auto xOp1        = iops1.back().first;
            (*matmul)
                .set_input_x1(mConst)
                .set_input_x2(*xOp1.get())
                .set_attr_transpose_x1(param->transposeA()) 
                .set_attr_transpose_x2(param->transposeB());
        } else if (!isConst0 && isConst1) {
            auto inputIndex = mOp->inputIndexes()->data()[0];
            auto iops       = mNpuBackend->mGrapMap[inputIndex]; 
            auto xOp        = iops.back().first;
            (*matmul)
                .set_input_x1(*xOp.get())
                .set_input_x2(mConst)
                .set_attr_transpose_x1(param->transposeA()) 
                .set_attr_transpose_x2(param->transposeB());
        } else {
            auto inputIndex = mOp->inputIndexes()->data()[0];
            auto iops       = mNpuBackend->mGrapMap[inputIndex]; 
            auto xOp        = iops.back().first;
            auto inputIndex1 = mOp->inputIndexes()->data()[1];
            auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; 
            auto xOp1        = iops1.back().first;
            (*matmul)
                .set_input_x1(*xOp.get())
                .set_input_x2(*xOp1.get())
                .set_attr_transpose_x1(param->transposeA()) 
                .set_attr_transpose_x2(param->transposeB());
        }
        mNpuBackend->setOutputOps(mOp, {matmul}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUMatmul>> __matmul_op(OpType_MatMul);

} // namespace MNN