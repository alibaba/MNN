//
//  NPUUnary.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUUnary.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUUnary::NPUUnary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

ErrorCode NPUUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    auto xOp = mNpuBackend->getInputOps(mOp);

    shared_ptr<ge::Operator> xOp2; 

    auto unary_type = mOp->main_as_UnaryOp()->opType();

    if(unary_type == UnaryOpOperation_EXP){
        shared_ptr<ge::op::Exp> unary(new ge::op::Exp(opName));
        (*unary).set_input_x(*xOp.get());
        mNpuBackend->setOutputOps(mOp, {unary}, outputs);
    }else if(unary_type == UnaryOpOperation_NEG){
        shared_ptr<ge::op::Neg> unary(new ge::op::Neg(opName));
        (*unary).set_input_x(*xOp.get());
        mNpuBackend->setOutputOps(mOp, {unary}, outputs);
    }else if(unary_type == UnaryOpOperation_ABS){
        shared_ptr<ge::op::Activation> unary(new ge::op::Activation(opName));
        (*unary).set_input_x(*xOp.get())
                .set_attr_mode(6);
        mNpuBackend->setOutputOps(mOp, {unary}, outputs);
    }else if(unary_type == UnaryOpOperation_SQRT){
        shared_ptr<ge::op::Sqrt> unary(new ge::op::Sqrt(opName));
        (*unary).set_input_x(*xOp.get());
        mNpuBackend->setOutputOps(mOp, {unary}, outputs);
    }else{
        MNN_ERROR("unary not support this case : %d \n", unary_type);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUUnary>> __unary_op(OpType_UnaryOp);

} // namespace MNN