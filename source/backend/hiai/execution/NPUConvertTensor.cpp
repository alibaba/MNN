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
    auto xOp = mNpuBackend->getInputOps(mOp);
    //om input weight const op
    std::vector<int32_t> inputShape = inputs[0]->shape();
    std::vector<int32_t> outShape = outputs[0]->shape();
    std::vector<std::vector<int64_t>> dims ={{0,1,2,3}, {0,2,3,1}, {0,3,1,2}, {0,1,2}, {0,2,1}, {0,1}, {1,0}};

    int32_t dimIndex = -1;
    bool flag = true;
    if (inputShape.size() != outShape.size()) {
        std::cout<<"inputsize not equal outputs size" <<std::endl;
        return NOT_SUPPORT;
    }
    for (int32_t i = 0; i < dims.size(); i++) {
        flag = true;
        for (int32_t j = 0; j < inputShape.size(); j++) {
            if (dims[i].size() != inputShape.size() || inputShape[dims[i][j]] != outShape[j]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            dimIndex = i;
            break;
        }
    }
    if (dimIndex == -1) {
        std::cout<<"inputsize cannot tans output" <<std::endl;
        return NOT_SUPPORT;
    }
    shared_ptr<hiai::op::Permute> convertTensor(new hiai::op::Permute(opName));
    int index = mOp->inputIndexes()->data()[0];
    auto iter = mNpuBackend->mSclipMap.find(index);
    if (iter != mNpuBackend->mSclipMap.end()){
        (*convertTensor).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[index]))
                        .set_attr_order(dims[dimIndex]);
    } else {
        (*convertTensor).set_input_x(*xOp).set_attr_order(dims[dimIndex]);
    }
    mNpuBackend->setOutputOps(mOp, {convertTensor}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvertTensor>> __convert_tensor_op(OpType_ConvertTensor);

} // namespace MNN