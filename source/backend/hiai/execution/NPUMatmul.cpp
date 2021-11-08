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

    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if(isConst1){
        auto input1 = inputs[1];
        // om input weight const op
        mConst = ge::op::Const(opName + "_w_const");
        {
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            ge::TensorDesc fdesc(ge::Shape({inputs[1]->buffer().dim[0].extent, inputs[1]->buffer().dim[1].extent}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)input1->host<float>(), input1->elementSize() * sizeof(float));
            mConst.set_attr_value(filter);
        }
    }
}

ErrorCode NPUMatmul::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    // 
    auto inputIndex1 = mOp->inputIndexes()->data()[0];
    auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; 
    auto xOp1        = iops1.back().first;

    shared_ptr<ge::op::Reshape> reshape(new ge::op::Reshape(opName + "_reshape"));
    (*reshape).set_input_tensor(*xOp1.get()).set_attr_shape(ge::AttrValue::LIST_INT({inputs[0]->buffer().dim[0].extent, inputs[0]->buffer().dim[1].extent}));

    vector<pair<shared_ptr<ge::Operator>, string>> ops;
    auto param = mOp->main_as_MatMul();

    shared_ptr<ge::op::MatMul> matmul(new ge::op::MatMul(opName));

    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if(isConst1){

        (*matmul)
            .set_input_x1(*reshape)
            .set_input_x2(mConst)
            .set_attr_transpose_x1(param->transposeA()) 
            .set_attr_transpose_x2(param->transposeB());

        shared_ptr<ge::op::Reshape> reshape3(new ge::op::Reshape(opName + "_reshape3"));
        auto shape = tensorShapeFormat(outputs[0]);
        (*reshape3).set_input_tensor(*matmul).set_attr_shape(ge::AttrValue::LIST_INT(shape));

        mNpuBackend->setOutputOps(mOp, {reshape, matmul, reshape3}, outputs);
        
    }else{
//hangxing todo
        
        auto inputIndex2 = mOp->inputIndexes()->data()[1];
        auto iops2       = mNpuBackend->mGrapMap[inputIndex2]; 
        auto xOp2        = iops2.back().first;
        shared_ptr<ge::op::Reshape> reshape2(new ge::op::Reshape(opName + "_reshape2"));
        (*reshape2).set_input_tensor(*xOp2.get()).set_attr_shape(ge::AttrValue::LIST_INT({inputs[1]->buffer().dim[0].extent, inputs[1]->buffer().dim[1].extent}));

        (*matmul)
            .set_input_x1(*reshape)
            .set_input_x2(*reshape2)
            .set_attr_transpose_x1(!param->transposeA())
            .set_attr_transpose_x2(param->transposeB());

        shared_ptr<ge::op::Permute> permute(new ge::op::Permute(opName + "_permute"));
        (*permute).set_input_x(*matmul).set_attr_order(ge::AttrValue::LIST_INT({1,0}));

        shared_ptr<ge::op::Reshape> reshape3(new ge::op::Reshape(opName + "_reshape3"));
        (*reshape3).set_input_tensor(*permute).set_attr_shape(ge::AttrValue::LIST_INT({1, outputs[0]->buffer().dim[1].extent, outputs[0]->buffer().dim[0].extent, 1}));

        mNpuBackend->setOutputOps(mOp, {reshape, reshape2, matmul, permute, reshape3}, outputs);

    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUMatmul>> __matmul_op(OpType_MatMul);

} // namespace MNN