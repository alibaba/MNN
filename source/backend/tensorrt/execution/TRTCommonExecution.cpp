//
//  TRTCommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTCommonExecution.hpp"
namespace MNN {

TRTCommonExecution::TRTCommonExecution(Backend *backend, const Op *op) : Execution(backend) {
    mTrtBackend = (TRTBackend *)backend;
    mOp         = op;
}

ErrorCode TRTCommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mInputs  = inputs;
    mOutputs = outputs;
    // if(mOp->name() != nullptr){
    //     MNN_PRINT("layer info: Type:%s name:%s \n", EnumNameOpType(mOp->type()), mOp->name()->c_str());
    // }
    // MNN_PRINT(" ===========    layer info: Type:%s     =========== \n", EnumNameOpType(mOp->type()));
    std::vector<ITensor *> nvTensors(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        nvTensors[i] = mTrtBackend->getTensorOps(inputs[i]);
    }

    // printf("inputs size : %d \n", inputs.size());
    // printf("outputs size : %d \n", outputs.size());
    // printf("nvTensors input size : %d \n", nvTensors.size());

    // printf("input : \n");
    // for(int n = 0; n < nvTensors.size(); n++){
    //     auto dims = nvTensors[n]->getDimensions();
    //     for(int i = 0; i < dims.nbDims; i++){
    //         printf("%d ", dims.d[i]);
    //     }
    //     printf("\n");
    //     for(int i = 0; i < dims.nbDims; i++){
    //         printf("%d ", inputs[n]->shape()[i]);
    //     }
    //     printf("\n");
    // }
    
    auto outputsTRT = this->onEncode(nvTensors);
    // printf("output : \n");
    // auto out_dims = outputsTRT[0]->getDimensions();
    // for(int i = 0; i < out_dims.nbDims; i++){
    //     printf("%d ", out_dims.d[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < outputs[0]->dimensions(); i++){
    //     printf("%d ", outputs[0]->shape()[i]);
    // }
    // printf("\n");
    mTrtBackend->setTensorOps(outputs, std::move(outputsTRT));
    return NO_ERROR;
}

ErrorCode TRTCommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

}; // namespace MNN
