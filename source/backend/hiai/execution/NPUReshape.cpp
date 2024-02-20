//
//  NPUReshape.cpp
//  MNN
//
//  Created by MNN on 2019/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUReshape.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUReshape::NPUReshape(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {
}

static bool isSameDims(Tensor * input,Tensor * output)
{
    if(input->buffer().dimensions == output->buffer().dimensions)
    {
        for(auto i =0; i < input->buffer().dimensions; i++) {
            if(input->buffer().dim[i].extent != output->buffer().dim[i].extent) {
                return false;
            }
        }
        return true;
    }
    return false;
}

ErrorCode NPUReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(opName));
    std::vector<int32_t> shape = outputs[0]->shape();
    shapeConst = hiai::op::Const(opName + "_shape_const");
    {
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shape.size())}), 
            ge::FORMAT_NCHW,  ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shape.data(), shape.size() * sizeof(int32_t));
        shapeConst.set_attr_value(filter);
    }
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*reshape).set_input_x(*xOp.get());
    } else {
        (*reshape).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    (*reshape).set_input_shape(shapeConst);
    mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUReshape>> __reshape_op(OpType_Reshape);

} // namespace MNN