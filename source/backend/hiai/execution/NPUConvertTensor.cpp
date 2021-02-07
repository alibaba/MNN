//
//  NPUConvertTensor.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConvertTensor.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUConvertTensor::NPUConvertTensor(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {
}

ErrorCode NPUConvertTensor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto opName = mOp->name()->str();

    shared_ptr<ge::op::Reshape> convertTensor(new ge::op::Reshape(opName));

    vector<int64_t>  shapeDims = {outputs[0]->batch(), outputs[0]->channel(), outputs[0]->height(), outputs[0]->width()};

    auto xOp = mNpuBackend->getInputOps(mOp);
    int index = mOp->inputIndexes()->data()[0];
    auto iter = mNpuBackend->mSclipMap.find(index);
    if(iter != mNpuBackend->mSclipMap.end()){
        (*convertTensor).SetInput(0, *xOp, mNpuBackend->mSclipMap[index]);
        (*convertTensor).set_attr_shape(
            ge::AttrValue::LIST_INT(shapeDims));
    }else{
        (*convertTensor).set_input_tensor(*xOp).set_attr_shape(
            ge::AttrValue::LIST_INT(shapeDims));
    }
    
    mNpuBackend->setOutputOps(mOp, {convertTensor});

    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvertTensor>> __convert_tensor_op(OpType_ConvertTensor);

} // namespace MNN